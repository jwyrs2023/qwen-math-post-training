"""Greedy evaluation preserving every completion, including parse failures."""
import json
import time
from pathlib import Path
from .data import digest
from .prompting import render_prompt
from .metrics import evaluate_records
from .protocol import require_clean_audit


def validate_adapter_lineage(spec, lineage, config):
    from .precision import validate_eval_precision
    from .checkpoints import verify
    validate_eval_precision(config, lineage)
    if lineage['model'] != config['model'] or lineage['revision'] != config['revision']:
        raise ValueError('adapter base revision mismatch')
    # Continue-SFT also trains a new adapter on the merged frozen SFT base.
    if lineage['kind'] == 'grpo' or lineage.get('sft_hashes') is not None:
        if not spec.get('sft') or verify(spec['sft']) != lineage.get('sft_hashes'):
            raise ValueError('continued training evaluation requires its exact frozen SFT startpoint')


def evaluate(config, models_path, split, output, limit=None, protocol=None):
    if split not in ('validation', 'official_test'):
        raise ValueError('only validation and official_test are evaluable')
    if split == 'official_test':
        if not protocol:
            raise ValueError('official test requires a frozen evaluation protocol')
        if limit is not None:
            raise ValueError('official test must evaluate the complete split')
        from .frozen_protocol import verify_protocol
        verify_protocol(config, models_path, protocol)
    if limit is not None and limit <= 0:
        raise ValueError('limit must be positive')
    manifest = require_clean_audit(config['manifest'])
    specs = json.loads(Path(models_path).read_text())
    if not specs or not isinstance(specs, dict):
        raise ValueError('models must be a nonempty name-to-spec mapping')
    from .checkpoints import verify
    for name, spec in specs.items():
        if not isinstance(spec, dict) or not name or Path(name).name != name or name in ('.', '..') or set(spec) - {'adapter', 'sft'}:
            raise ValueError('invalid model name or fields')
        for key in ('adapter', 'sft'):
            if spec.get(key):
                verify(spec[key])
        if spec.get('adapter'):
            lineage_path = Path(spec['adapter']) / 'lineage.json'
            if not lineage_path.is_file():
                raise ValueError('adapter missing lineage; freeze a current training artifact')
            lineage = json.loads(lineage_path.read_text())
            validate_adapter_lineage(spec, lineage, config)
    output = Path(output)
    if output.exists():
        raise ValueError('refusing to overwrite evaluation output')
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    from .checkpoints import verify
    if not torch.cuda.is_available():
        raise ValueError('formal evaluation requires CUDA')
    if not config.get('revision') or len(config['revision']) != 40:
        raise ValueError('immutable model revision required')
    raw = json.loads((Path(config['manifest']).parent.parent / ('raw/test.json' if split == 'official_test' else 'raw/train.json')).read_text())
    selected = [r for r in manifest['rows'] if r['split'] == split]
    if limit:
        selected = selected[:limit]
    output.mkdir(parents=True)
    summaries = {}
    for name, spec in specs.items():
        if not name or Path(name).name != name or set(spec) - {'adapter', 'sft'}:
            raise ValueError('invalid model name or fields')
        from huggingface_hub import snapshot_download
        model_path = snapshot_download(config['model'], revision=config['revision'], local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, torch_dtype=getattr(torch, config['model_dtype'])).to('cuda').eval()
        hashes = None
        sft_hashes = None
        if spec.get('sft'):
            sft_hashes = verify(spec['sft'])
            model = PeftModel.from_pretrained(model, spec['sft']).merge_and_unload().eval()
        if spec.get('adapter'):
            hashes = verify(spec['adapter'])
            model = PeftModel.from_pretrained(model, spec['adapter']).eval()
        records = []
        with (output / (name + '.jsonl')).open('x') as stream:
            for row in selected:
                record = raw[int(row['sample_id'].split(':')[1])]
                prompt = render_prompt(record['question'], config['prompt'])
                inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to('cuda')
                torch.cuda.synchronize()
                started = time.perf_counter()
                with torch.inference_mode():
                    generated = model.generate(**inputs, do_sample=False, max_new_tokens=config['max_completion_length'],
                                               pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
                torch.cuda.synchronize()
                tokens = generated[0, inputs['input_ids'].shape[1]:]
                result = dict(sample_id=row['sample_id'], completion=tokenizer.decode(tokens, skip_special_tokens=True),
                    answer=record['answer'], model_revision=config['revision'], checkpoint_hashes=hashes,
                    model_dtype=config['model_dtype'], sft_startpoint_hashes=sft_hashes,
                    prompt_hash=digest(prompt), output_tokens=len(tokens), latency_seconds=time.perf_counter()-started,
                    truncated=len(tokens) == config['max_completion_length'] and tokens[-1].item() != tokenizer.eos_token_id)
                parsed, _ = evaluate_records([result])
                stream.write(json.dumps(parsed[0], ensure_ascii=False) + '\n')
                stream.flush()
                records.append(result)
        _, summaries[name] = evaluate_records(records)
        del model
        torch.cuda.empty_cache()
    (output / 'summary.json').write_text(json.dumps(summaries, indent=2))
    return summaries
