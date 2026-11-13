"""Archive completed V2 training adapters and evidence; exclude optimizer checkpoints."""
import hashlib,json,tarfile
from pathlib import Path
from math_posttrain.checkpoints import verify
ROOT=Path(__file__).resolve().parents[2]
def main():
 summary=json.loads((ROOT/'grpo-study-v2/reports/formal-training-summary.json').read_text())
 assert len(summary)==3 and all(r['result']['status']=='complete' for r in summary)
 files=[];signals={}
 for row in summary:
  root=ROOT/'runs'/row['run_id'];verify(root/'final')
  files.extend(p for p in root.rglob('*') if p.is_file() and not any(x.startswith('checkpoint-') for x in p.parts))
  if row['branch']!='continue_sft':
   batches=[json.loads(x) for x in (root/'reward-groups.jsonl').read_text().splitlines()]
   groups=[g for b in batches for g in b['groups']];rs=[r for b in batches for r in b['rows']]
   assert len(groups)==640 and len(rs)==2560
   assert all(r['sample_id'] and r['prompt'] and isinstance(r['completion'],str) for r in rs)
   signals[row['branch']]=dict(groups=len(groups),completions=len(rs),zero_variance=sum(g['zero_variance'] for g in groups),format_mixed_groups=sum(len({r['format'] for r in b['rows'][j:j+4]})>1 for b in batches for j in range(0,len(b['rows']),4)),generated_tokens=sum(b['generated_tokens'] for b in batches))
 times=[r['result']['elapsed_seconds'] for r in summary]
 (ROOT/'grpo-study-v2/reports/formal-training-audit.json').write_text(json.dumps(dict(signals=signals,max_min_time_spread=max(times)/min(times)-1),indent=2))
 for directory in ['src','grpo-study-v2/configs']:
  files.extend(p for p in (ROOT/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
 files.extend(p for p in (ROOT/'grpo-study-v2/reports').glob('formal-*.json') if p.is_file())
 files.append(ROOT/'requirements-lock-linux-v2.txt')
 manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(files))}
 m=ROOT/'v2-formal-training-files.sha256.json';m.write_text(json.dumps(manifest,indent=2))
 out=ROOT/'v2-formal-training-backup.tar.gz'
 with tarfile.open(out,'w:gz') as t:
  for p in sorted(set(files)):t.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
  t.add(m,arcname=m.name,recursive=False)
 sha=hashlib.sha256(out.read_bytes()).hexdigest()
 (ROOT/'v2-formal-training-backup.sha256').write_text(sha+'\n')
 print(json.dumps(dict(bytes=out.stat().st_size,sha256=sha,files=len(files),signals=signals),indent=2))
if __name__=='__main__':main()
