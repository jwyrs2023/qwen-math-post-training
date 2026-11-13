"""Pure rewards, independent of trainer callback APIs."""
from .parsing import extract_prediction, format_ok, is_correct


def score_batch(completions, golds, mode='accuracy'):
    if mode not in ('accuracy', 'accuracy_format'):
        raise ValueError('unknown reward mode')
    if len(completions) != len(golds):
        raise ValueError('completion/gold batch length mismatch')
    rows = []
    for completion, gold in zip(completions, golds):
        accuracy = int(is_correct(extract_prediction(completion), gold))
        formatting = int(format_ok(completion))
        rows.append(dict(accuracy=accuracy, format=formatting,
                         total=accuracy + (0.1 * formatting if mode == 'accuracy_format' else 0)))
    return rows


def group_advantages(rewards):
    if not rewards:
        raise ValueError('empty reward group')
    import math
    if not all(math.isfinite(x) for x in rewards):
        raise ValueError('nonfinite reward')
    mean = sum(rewards) / len(rewards)
    std = (sum((x - mean) ** 2 for x in rewards) / len(rewards)) ** 0.5
    return [0.0] * len(rewards) if std == 0 else [(x - mean) / std for x in rewards]
