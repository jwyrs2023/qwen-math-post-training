"""Deterministic manifest primitives; official dataset acquisition is pending."""
import hashlib
import json
import random
import copy
import re
import unicodedata
from difflib import SequenceMatcher


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def normalize(question):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', question).casefold()).strip()


def build_manifest(train, test, revision, seed=42):
    from .parsing import extract_gold
    if not isinstance(revision, str) or not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('an immutable dataset revision is required')
    if len(train) != 7473 or len(test) != 1319:
        raise ValueError('unexpected official dataset counts')
    indices = list(range(len(train)))
    random.Random(seed).shuffle(indices)
    splits = {}
    offset = 0
    for name, count in [('sft_train', 3000), ('grpo_train', 2000), ('validation', 500), ('unused', 1973)]:
        splits[name] = indices[offset:offset + count]
        offset += count
    rows = []
    for name, ids in list(splits.items()) + [('official_test', list(range(len(test))))]:
        source = test if name == 'official_test' else train
        for index in ids:
            row = source[index]
            extract_gold(row['answer'])
            rows.append(dict(sample_id=f'{"test" if name == "official_test" else "train"}:{index}',
                             split=name, content_hash=digest(row), question=normalize(row['question'])))
    return dict(schema_version=1, revision=revision, seed=seed, source_hashes=dict(train=digest(train), test=digest(test)), rows=rows)


def audit_rows(rows, near_threshold=0.95):
    """Report normalized cross-split exact/near duplicates, including unused rows."""
    seen_ids = set()
    conflicts = []
    for i, row in enumerate(rows):
        if row['sample_id'] in seen_ids:
            raise ValueError('duplicate sample ID')
        seen_ids.add(row['sample_id'])
        for j in range(i):
            prior = rows[j]
            if row['split'] == prior['split']:
                continue
            a, b = row['question'], prior['question']
            if a == b:
                conflicts.append([prior['sample_id'], row['sample_id'], 'exact'])
            elif min(len(a), len(b)) / max(len(a), len(b), 1) >= near_threshold:
                matcher = SequenceMatcher(None, a, b, autojunk=False)
                if matcher.quick_ratio() >= near_threshold and matcher.ratio() >= near_threshold:
                    conflicts.append([prior['sample_id'], row['sample_id'], 'near'])
    return dict(passed=not conflicts, conflicts=conflicts, near_threshold=near_threshold)


def prepare(output, seed=42, revision=None):
    from pathlib import Path
    from huggingface_hub import HfApi, hf_hub_download
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('output directory is not empty; refusing to overwrite')
    revision = HfApi().dataset_info('openai/gsm8k', revision=revision).sha
    raw = {}
    hashes = {}
    for split in ('train', 'test'):
        path = Path(hf_hub_download('openai/gsm8k', f'main/{split}-00000-of-00001.parquet', repo_type='dataset', revision=revision))
        payload = path.read_bytes()
        hashes[split] = hashlib.sha256(payload).hexdigest()
        import pyarrow.parquet as pq
        raw[split] = pq.read_table(path).to_pylist()
    manifest = build_manifest(raw['train'], raw['test'], revision, seed)
    manifest['raw_file_hashes'] = hashes
    (output / 'raw').mkdir(parents=True)
    (output / 'manifests').mkdir()
    for split, records in raw.items():
        (output / 'raw' / f'{split}.json').write_text(json.dumps(records, ensure_ascii=False))
    path = output / 'manifests/splits.json'
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return path


def audit_manifest(path):
    from pathlib import Path
    path = Path(path)
    manifest = json.loads(path.read_text())
    root = path.parent.parent
    train, test = [json.loads((root / 'raw' / f'{s}.json').read_text()) for s in ('train', 'test')]
    rebuilt = build_manifest(train, test, manifest['revision'], manifest['seed'])
    if 'quarantine_policy' in manifest:
        rebuilt = apply_quarantine(rebuilt, manifest['quarantine_policy']['sample_ids'])
    for key in ('schema_version', 'source_hashes', 'rows'):
        if rebuilt[key] != manifest[key]:
            raise ValueError(f'manifest mismatch: {key}')
    result = audit_rows([r for r in manifest['rows'] if r['split'] != 'quarantine'])
    result['excluded_quarantine_ids'] = [r['sample_id'] for r in manifest['rows'] if r['split'] == 'quarantine']
    result['manifest_hash'] = digest(manifest)
    audit_path(path).write_text(json.dumps(result, indent=2))
    return result


def audit_path(path):
    from pathlib import Path
    path = Path(path)
    return path.with_name(path.stem + '.audit.json')


def apply_quarantine(manifest, sample_ids):
    ids = set(sample_ids)
    if len(ids) != len(sample_ids):
        raise ValueError('duplicate quarantine IDs')
    result = copy.deepcopy(manifest)
    found = set()
    for row in result['rows']:
        if row['sample_id'] in ids:
            if row['split'] not in ('sft_train', 'grpo_train'):
                raise ValueError('only training rows may be quarantined')
            row['original_split'] = row['split']
            row['split'] = 'quarantine'
            found.add(row['sample_id'])
    if found != ids:
        raise ValueError('unknown quarantine ID')
    result['quarantine_policy'] = dict(version=1, sample_ids=sorted(ids),
        rule='isolate both training endpoints; preserve official test; no replacement',
        authorized='user approved in chat on 2026-11-03')
    return result
