"""Freeze checkpoint file hashes and verify before reuse."""
import json
import shutil
from pathlib import Path
from .provenance import file_hash


def freeze(source, output):
    source, output = Path(source), Path(output)
    if not (source / 'adapter_config.json').is_file() or not (source / 'adapter_model.safetensors').is_file():
        raise ValueError('SFT adapter checkpoint files missing')
    if output.exists():
        raise ValueError('refusing to overwrite frozen checkpoint')
    shutil.copytree(source, output)
    manifest = seal(output)
    return manifest


def verify(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'checkpoint-manifest.json').read_text())
    actual = {str(p.relative_to(directory)): file_hash(p) for p in directory.rglob('*')
              if p.is_file() and p.name != 'checkpoint-manifest.json'}
    if actual != manifest:
        raise ValueError('checkpoint file hash mismatch')
    return manifest


def seal(directory):
    directory = Path(directory)
    manifest = {str(p.relative_to(directory)): file_hash(p) for p in sorted(directory.rglob('*'))
                if p.is_file() and p.name != 'checkpoint-manifest.json'}
    (directory / 'checkpoint-manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest
