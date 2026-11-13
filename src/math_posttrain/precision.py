"""Explicit precision provenance for comparably decoded model groups."""

def validate_eval_precision(config, lineage):
    requested = config['model_dtype']
    trained = lineage.get('model_dtype', 'bfloat16')
    if lineage.get('kind') == 'grpo' and requested != trained:
        raise ValueError('GRPO model_dtype mismatch: evaluate with its recorded precision')
    return requested
