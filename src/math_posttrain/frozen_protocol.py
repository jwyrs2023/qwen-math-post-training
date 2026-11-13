"""Bind final evaluation to exact code, data, prompt, configs and checkpoints."""
import json
from pathlib import Path
from .data import digest
from .checkpoints import verify
from .provenance import file_hash


def snapshot(config, models_path):
    models = json.loads(Path(models_path).read_text())
    legacy = {'base', 'sft', 'a', 'b'}
    v2 = {'continue_sft', 'grpo_acc', 'grpo_accfmt'}
    if set(models) not in (legacy, v2):
        raise ValueError('final test requires complete V1 four-model or V2 three-model roster')
    checkpoints = {}
    for name,spec in models.items():
        if name != 'base' and not spec.get('adapter'):
            raise ValueError('missing adapter')
        for key in ('adapter','sft'):
            if spec.get(key): checkpoints[f'{name}:{key}'] = verify(spec[key])
    code = {p.name:file_hash(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    return dict(config=config,models=models,checkpoints=checkpoints,code=code,
                manifest=file_hash(config['manifest']),prompt=file_hash(config['prompt']),
                decode=dict(do_sample=False,max_new_tokens=config['max_completion_length']))


def freeze_protocol(config, models_path, output):
    payload=snapshot(config,models_path)
    with Path(output).open('x') as f:
        json.dump(dict(payload=payload,sha256=digest(payload)),f,indent=2)


def verify_protocol(config,models_path,path):
    document=json.loads(Path(path).read_text())
    if digest(document['payload'])!=document['sha256'] or snapshot(config,models_path)!=document['payload']:
        raise ValueError('frozen evaluation protocol mismatch; do not run test')
    return document['sha256']
