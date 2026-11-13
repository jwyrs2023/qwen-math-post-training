"""Evaluate saved completions without changing parsing rules."""
from .parsing import extract_gold, prediction_status, format_ok, is_correct


def evaluate_records(records):
    rows = []
    seen = set()
    for record in records:
        sample_id = record['sample_id']
        if sample_id in seen:
            raise ValueError('duplicate prediction ID')
        seen.add(sample_id)
        pred, reason = prediction_status(record['completion'])
        correct = is_correct(pred, extract_gold(record['answer']))
        formatting = format_ok(record['completion'])
        rows.append(dict(record, prediction=str(pred) if pred is not None else None,
                         correct=correct, format_ok=formatting, failure_reason=reason))
    if not rows:
        raise ValueError('no predictions')
    n = len(rows)
    return rows, dict(count=n, accuracy=sum(x['correct'] for x in rows) / n,
                      format_rate=sum(x['format_ok'] for x in rows) / n,
                      parse_rate=sum(x['prediction'] is not None for x in rows) / n,
                      joint_rate=sum(x['correct'] and x['format_ok'] for x in rows) / n)
