"""Run directories are immutable by default; no fabricated measurements."""
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def create_run(root, run_id, config):
    if not run_id or Path(run_id).name != run_id or run_id in ('.', '..'):
        raise ValueError('run_id must be a single directory name')
    path = Path(root) / run_id
    path.mkdir(parents=True, exist_ok=False)
    def git(*args):
        try:
            result = subprocess.run(['git', *args], capture_output=True, text=True)
        except FileNotFoundError:
            return None
        return result.stdout if result.returncode == 0 else None
    metadata = dict(started_at=datetime.now(timezone.utc).isoformat(),
                    command=sys.argv, python=sys.version, platform=platform.platform(),
                    git_commit=git('rev-parse', 'HEAD'), git_diff=git('diff'),
                    config=config, status='initialized', measurements=None,
                    source_sha256={str(p.relative_to(Path(__file__).parent)): file_hash(p)
                                   for p in sorted(Path(__file__).parent.rglob('*.py'))})
    (path / 'run.json').write_text(json.dumps(metadata, indent=2))
    return path
