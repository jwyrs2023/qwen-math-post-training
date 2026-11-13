"""Write one run's config with actual account values; never launch training."""
import argparse,math
from pathlib import Path
import yaml
from math_posttrain.config import load_config
p=argparse.ArgumentParser()
p.add_argument('--branch',choices=['continue_sft','grpo_acc','grpo_accfmt'],required=True)
for k in ['limit','spent','non-gpu','reserve','hourly-price']:p.add_argument('--'+k,type=float,required=True)
p.add_argument('--output',required=True)
p.add_argument('--steps',type=int,default=4)
p.add_argument('--seed',type=int,default=43)
a=p.parse_args()
values=[a.limit,a.spent,a.non_gpu,a.reserve,a.hourly_price]
if not all(math.isfinite(v) and v>=0 for v in values) or a.hourly_price<=0 or a.limit>600 or a.spent+a.non_gpu+a.reserve>=a.limit:p.error('invalid/exhausted account budget')
if a.steps<=0 or a.seed<0:p.error('invalid steps/seed')
root=Path(__file__).resolve().parents[1]
c=yaml.safe_load((root/f'configs/{a.branch}_pilot.yaml').read_text())
c.update(max_steps=a.steps,seed=a.seed,budget=dict(limit=a.limit,spent=a.spent,non_gpu=a.non_gpu,reserve=a.reserve,hourly_price=a.hourly_price))
out=Path(a.output)
if out.exists():p.error('refusing overwrite')
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(yaml.safe_dump(c,sort_keys=False))
try:load_config(out)
except Exception:
 out.unlink();raise
print(f'Prepared {out}; no training. Update spent from provider before EACH branch.')
