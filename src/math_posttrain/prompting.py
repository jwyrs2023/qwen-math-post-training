"""Plain-text prompts and completion-only labels, without chat templates."""
from pathlib import Path
from .parsing import extract_gold


def render_prompt(question, path='prompts/math_v1.txt'):
    return Path(path).read_text().rstrip('\n').format(question=question)


def target(answer, eos):
    gold = extract_gold(answer)
    # GSM8K gold decimals are retained rather than converted to fraction notation.
    number = answer.split('####')[1].strip().replace(',', '')
    return answer.split('####')[0].rstrip() + '\nFinal Answer: ' + number + eos


def encode_sft(tokenizer, prompt, completion, max_length):
    # Separate tokenization is deliberate: inference uses exactly the same prompt IDs.
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    completion_ids = tokenizer.encode(completion, add_special_tokens=False)
    if not completion_ids or len(prompt_ids) + len(completion_ids) > max_length:
        raise ValueError('empty or overlong example; never truncate final answer')
    return dict(input_ids=prompt_ids + completion_ids,
                attention_mask=[1] * (len(prompt_ids) + len(completion_ids)),
                labels=[-100] * len(prompt_ids) + completion_ids)


def pad_examples(examples, pad_id):
    if not examples:
        raise ValueError('empty batch')
    length = max(len(x['input_ids']) for x in examples)
    return {key: [x[key] + [fill] * (length - len(x[key])) for x in examples]
            for key, fill in [('input_ids', pad_id), ('attention_mask', 0), ('labels', -100)]}
