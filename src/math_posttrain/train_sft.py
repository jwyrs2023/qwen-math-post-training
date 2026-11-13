"""SFT dataset preparation with explicit skip accounting and loss masking."""
import json
from pathlib import Path
from .prompting import render_prompt, target, encode_sft
from .protocol import require_clean_audit


def prepare_examples(config, tokenizer, split='sft_train'):
    manifest = require_clean_audit(config['manifest'])
    root = Path(config['manifest']).parent.parent
    train = json.loads((root / 'raw/train.json').read_text())
    if tokenizer.eos_token is None:
        raise ValueError('tokenizer has no EOS token')
    examples, excluded = [], []
    for row in manifest['rows']:
        if row['split'] != split:
            continue
        record = train[int(row['sample_id'].split(':')[1])]
        prompt = render_prompt(record['question'], config['prompt'])
        completion = target(record['answer'], tokenizer.eos_token)
        try:
            encoded = encode_sft(tokenizer, prompt, completion, config['max_length'])
        except ValueError as exc:
            excluded.append(dict(sample_id=row['sample_id'], reason=str(exc)))
            continue
        examples.append(dict(encoded, sample_id=row['sample_id']))
    if not examples:
        raise ValueError('no usable SFT examples')
    return examples, excluded
