"""Guard formal runs against unresolved leakage and unequal A/B settings."""
import json
from pathlib import Path
from .data import digest, audit_path


def require_clean_audit(manifest_path):
    path = Path(manifest_path)
    manifest = json.loads(path.read_text())
    audit = json.loads(audit_path(path).read_text())
    if audit.get('manifest_hash') != digest(manifest) or not audit.get('passed'):
        raise ValueError('formal training blocked: missing, stale, or failed leakage audit')
    for split in ('train', 'test'):
        records = json.loads((path.parent.parent / 'raw' / (split + '.json')).read_text())
        if digest(records) != manifest['source_hashes'][split]:
            raise ValueError('raw dataset hash mismatch: ' + split)
    return manifest


def compare_ab(a, b):
    if a.get('reward') != 'accuracy' or b.get('reward') != 'accuracy_format':
        raise ValueError('A/B reward assignment invalid')
    differences = [key for key in set(a) | set(b) if a.get(key) != b.get(key)]
    if differences != ['reward']:
        raise ValueError(f'unfair A/B config differences: {differences}')
