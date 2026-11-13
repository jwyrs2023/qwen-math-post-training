"""Restore only verified frozen SFT files; prepare non-spending pilot configs."""
import hashlib,json,tarfile
from pathlib import Path
import yaml
from math_posttrain.checkpoints import verify
from math_posttrain.config import load_config
from math_posttrain.training import execute
ROOT=Path(__file__).resolve().parents[2]
def main():
 archive=ROOT/'reports/validation-base-sft-backup.tar.gz'
 dest=ROOT/'runs/sft_frozen'
 with tarfile.open(archive) as t:
  prefix='runs/sft_frozen/'
  manifest=json.loads(t.extractfile(prefix+'checkpoint-manifest.json').read())
  files={}
  for name,expected in manifest.items():
   if Path(name).name!=name:raise ValueError('unexpected nested checkpoint path')
   data=t.extractfile(prefix+name).read()
   assert hashlib.sha256(data).hexdigest()==expected
   files[name]=data
  if not dest.exists():
   dest.mkdir(parents=True)
   for name,data in files.items():(dest/name).write_bytes(data)
   (dest/'checkpoint-manifest.json').write_text(json.dumps(manifest,indent=2))
 verify(dest)
 base=yaml.safe_load((ROOT/'configs/next-round/grpo_a.yaml').read_text())
 results={}
 for branch in ['continue_sft','grpo_acc','grpo_accfmt']:
  c=dict(base,max_steps=4,save_steps=2,eval_steps=None,seed=43,budget=None)
  kind='sft' if branch=='continue_sft' else 'grpo'
  if kind=='sft':c.update(sft_split='grpo_train',batch_size=1,gradient_accumulation_steps=16)
  c['reward']='accuracy_format' if branch=='grpo_accfmt' else 'accuracy'
  path=ROOT/f'grpo-study-v2/configs/{branch}_pilot.yaml'
  if path.exists() and yaml.safe_load(path.read_text())!=c:raise ValueError('refusing config overwrite')
  path.write_text(yaml.safe_dump(c,sort_keys=False))
  results[branch]=execute(load_config(path),'v2_'+branch+'_pilot',kind,sft=str(dest),dry_run=True)
 out=dict(frozen_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),frozen_checkpoint=manifest,dry_runs=results)
 (ROOT/'grpo-study-v2/reports/local-preflight.json').write_text(json.dumps(out,indent=2))
 print('Frozen SFT verified; three pilot dry-runs passed. No GPU, download or training.')
if __name__=='__main__':main()
