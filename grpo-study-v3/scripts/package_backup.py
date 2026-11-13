"""Whitelist archival; exclude optimizer checkpoints, caches and all access keys."""
import argparse,hashlib,json,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];V3=ROOT/'grpo-study-v3'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def selected_files(root):
    roots=[root/'grpo-study-v3',root/'runs/sft_frozen',root/'src/math_posttrain',root/'prompts',root/'data/manifests',root/'data/raw']
    roots += sorted((root/'runs').glob('v3_*'))
    files=[]
    for base in roots:
        if not base.exists():continue
        for p in base.rglob('*'):
            if not p.is_file():continue
            rel=p.relative_to(root)
            if p.is_symlink() or any(x in ('__pycache__','.pytest_cache','backups') for x in rel.parts):continue
            if any(x.startswith('checkpoint-') for x in rel.parts[:-1]):continue
            if p.suffix in ('.pyc','.tmp') or p.name.endswith('.lock'):continue
            files.append(p)
    return sorted(set(files))
def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True,choices=['training','complete']);a=p.parse_args()
    report=V3/'reports';out=report/'backups';out.mkdir(exist_ok=True)
    if a.stage=='complete' and not (report/'evaluation-complete.json').is_file():raise ValueError('evaluation not complete')
    if a.stage=='training' and not (report/'training-summary.json').is_file():raise ValueError('no completed training artifacts')
    archive=out/f'v3-{a.stage}.tar.gz';manifest=out/f'v3-{a.stage}-manifest.json'
    if archive.exists() or manifest.exists():raise ValueError('refusing to overwrite archive')
    files=selected_files(ROOT)
    if a.stage=='training':
        files=[p for p in files if not (p.relative_to(ROOT).parts[0]=='runs' and p.relative_to(ROOT).parts[1] in ('v3_validation_s43','v3_official_test_s43')) and p.name not in ('validation.log','official-test.log','evaluation-stage.log','evaluation-stage.exit','evaluation-complete.json')]
    members={str(f.relative_to(ROOT)):dict(sha256=sha(f),bytes=f.stat().st_size) for f in files}
    with tarfile.open(archive,'w:gz') as tar:
        for f in files:tar.add(f,arcname=str(f.relative_to(ROOT)),recursive=False)
    # Re-read every archived file rather than only trusting the compressed stream hash.
    with tarfile.open(archive) as tar:
        actual={m.name:dict(sha256=hashlib.sha256(tar.extractfile(m).read()).hexdigest(),bytes=m.size) for m in tar.getmembers() if m.isfile()}
    if actual!=members:raise ValueError('archive member verification failed')
    manifest.write_text(json.dumps(dict(stage=a.stage,archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,members=members,remote_members_verified=True),indent=2))
    print(json.dumps(dict(archive=str(archive),sha256=sha(archive),members=len(members),bytes=archive.stat().st_size)))
if __name__=='__main__':main()
