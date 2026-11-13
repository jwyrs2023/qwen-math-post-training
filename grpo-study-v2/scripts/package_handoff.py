"""Package only explicit code/data/checkpoint artifacts; exclude access credentials."""
import hashlib,json,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'grpo-study-v2/handoff';OUT.mkdir(exist_ok=True)
paths=['pyproject.toml','requirements-bootstrap.txt','src','tests','configs','prompts','data/raw','data/manifests','runs/sft_frozen','grpo-study-v2/README.md','grpo-study-v2/PROTOCOL.md','grpo-study-v2/GPU_RUNBOOK.md','grpo-study-v2/scripts','grpo-study-v2/configs','grpo-study-v2/checks']
files=[]
for name in paths:
 p=ROOT/name
 if not p.exists():raise FileNotFoundError(p)
 files.extend([p] if p.is_file() else [x for x in p.rglob('*') if x.is_file() and '__pycache__' not in x.parts and not x.name.endswith('.pyc')])
files=sorted(set(files))
manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
archive=OUT/'grpo-v2-gpu-handoff.tar.gz'
with tarfile.open(archive,'w:gz') as t:
 for p in files:t.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
(OUT/'files.sha256.json').write_text(json.dumps(manifest,indent=2))
(OUT/'archive.sha256').write_text(hashlib.sha256(archive.read_bytes()).hexdigest()+'  '+archive.name+'\n')
with tarfile.open(archive) as t:
 assert set(t.getnames())==set(manifest)
 for m in t:
  assert hashlib.sha256(t.extractfile(m).read()).hexdigest()==manifest[m.name]
print(f'Packaged and reverified {len(files)} files, {archive.stat().st_size} bytes; no credentials/cache/venv included.')
