import unittest
from fractions import Fraction
from math_posttrain.parsing import *
from math_posttrain.rewards import score_batch, group_advantages


class ParsingTests(unittest.TestCase):
    def test_valid(self):
        for raw, expected in [('0', 0), ('-2', -2), ('1,234', 1234), ('.5', Fraction(1, 2)), ('2/4', Fraction(1, 2)), ('$3', 3)]:
            with self.subTest(raw=raw):
                self.assertEqual(extract_prediction('Work 99\nFinal Answer: ' + raw), expected)
                self.assertTrue(format_ok('Work\nFinal Answer: ' + raw))
        self.assertEqual(extract_prediction(r'\boxed{1/2}'), Fraction(1, 2))
        self.assertEqual(extract_gold('work\n#### 0.5'), Fraction(1, 2))

    def test_rejections(self):
        cases = ['', 'NaN', 'Inf', '1e3', '10%', '3 kg', '1,23', '1/0', '__import__("os")', '1' * 300]
        for raw in cases:
            with self.subTest(raw=raw):
                self.assertIsNone(extract_prediction('Final Answer: ' + raw))
        for raw in ['Question 42', 'Final Answer: 2\nextra', 'Final Answer: 2\nFinal Answer: 2', '#### 2\nFinal Answer: 3', r'\boxed{{2}}', r'\boxed{2', 'x' * 40000]:
            self.assertIsNone(extract_prediction(raw))
        self.assertFalse(format_ok('Work Final Answer: 2'))
        with self.assertRaises(ValueError):
            extract_gold('no label')

    def test_rewards(self):
        self.assertEqual(score_batch(['Final Answer: 2'], [Fraction(3)], 'accuracy_format')[0]['total'], 0.1)
        self.assertEqual(score_batch(['#### 1/2'], [Fraction(1, 2)])[0]['accuracy'], 1)
        with self.assertRaises(ValueError):
            score_batch([''], [])
        self.assertEqual(group_advantages([1, 1, 1, 1]), [0] * 4)
        self.assertEqual(group_advantages([0, 0, 0, 0]), [0] * 4)
        self.assertEqual(group_advantages([0, 1]), [-1, 1])


if __name__ == '__main__':
    unittest.main()

class DataAndMaskTests(unittest.TestCase):
    def test_mask_and_padding(self):
        from math_posttrain.prompting import encode_sft, pad_examples
        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                return list(text.encode())
        a = encode_sft(Tokenizer(), 'Q:', 'A!', 4)
        b = encode_sft(Tokenizer(), 'Q', 'A', 4)
        batch = pad_examples([a, b], 0)
        self.assertEqual(a['labels'], [-100, -100, 65, 33])
        self.assertEqual(batch['labels'][1], [-100, 65, -100, -100])
        with self.assertRaises(ValueError):
            encode_sft(Tokenizer(), 'QQQ', 'AA', 4)

    def test_duplicate_audit(self):
        from math_posttrain.data import audit_rows, normalize
        rows = [dict(sample_id='train:0', split='sft_train', question=normalize(' A  B ')),
                dict(sample_id='test:0', split='official_test', question='a b')]
        self.assertFalse(audit_rows(rows)['passed'])
        rows[1]['question'] = 'entirely different'
        self.assertTrue(audit_rows(rows)['passed'])
        rows[1]['sample_id'] = 'train:0'
        with self.assertRaises(ValueError):
            audit_rows(rows)

class MetricsTests(unittest.TestCase):
    def test_invalid_counts_as_wrong(self):
        from math_posttrain.metrics import evaluate_records
        records = [dict(sample_id='a', completion='Final Answer: 2', answer='#### 2'),
                   dict(sample_id='b', completion='', answer='#### 3')]
        rows, metrics = evaluate_records(records)
        self.assertEqual(metrics['accuracy'], 0.5)
        self.assertEqual(metrics['count'], 2)
        with self.assertRaises(ValueError):
            evaluate_records(records + [records[0]])

class ProvenanceTests(unittest.TestCase):
    def test_no_overwrite(self):
        import tempfile
        from math_posttrain.provenance import create_run
        with tempfile.TemporaryDirectory() as root:
            create_run(root, 'one', {})
            with self.assertRaises(FileExistsError):
                create_run(root, 'one', {})
            with self.assertRaises(ValueError):
                create_run(root, '../escape', {})

class ConfigTests(unittest.TestCase):
    def test_unknown_and_invalid_fields(self):
        import tempfile
        from pathlib import Path
        from math_posttrain.config import load_config
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'config.yaml'
            for text in ['unknown: 2', 'group_size: 0', 'batch_size: 3', 'model: Qwen/Instruct', 'max_steps: true']:
                path.write_text(text)
                with self.assertRaises(ValueError):
                    load_config(path)

class BudgetTests(unittest.TestCase):
    def test_hard_cap(self):
        from math_posttrain.budget import estimate
        self.assertFalse(estimate(600, 590, 0, 5, 10, 3600, 1)['may_start'])
        self.assertTrue(estimate(600, 0, 0, 50, 10, 3600, 1)['may_start'])
        with self.assertRaises(ValueError):
            estimate(600, 0, 0, 0, 0, 1, 1)

class ProtocolTests(unittest.TestCase):
    def test_ab_fairness(self):
        from math_posttrain.protocol import compare_ab
        compare_ab(dict(reward='accuracy', seed=42), dict(reward='accuracy_format', seed=42))
        with self.assertRaises(ValueError):
            compare_ab(dict(reward='accuracy', seed=42), dict(reward='accuracy_format', seed=43))

class CallbackTests(unittest.TestCase):
    def test_callback_order_and_shape(self):
        from math_posttrain.train_grpo import make_reward_callback
        reward = make_reward_callback('accuracy_format')
        self.assertEqual(reward(['Final Answer: 2', '#### 3'], ['3', '3']), [0.1, 1])
        with self.assertRaises(ValueError):
            reward(['Final Answer: 2'], [])
        with self.assertRaises(ValueError):
            reward([[dict(content='2')]], ['2'])
