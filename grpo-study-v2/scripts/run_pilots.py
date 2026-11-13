"""Run only the three four-step pilots after preflight; conservative session cost cap."""
import json,time,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
from math_posttrain.config import load_config
ROOT=Path(__file__).resolve().parents[2]
# Screenshot: instance created at 17:01 Shanghai. Charging from creation conservatively
# includes costs before the user approved 50 CNY additional spending.
START=datetime(2026,11,5,9,1,tzinfo=timezone.utc).timestamp()
PRICE=2.24; LIMIT=50.; RESERVE=5.
def main():
 (ROOT/'grpo-study-v2/reports').mkdir(parents=True,exist_ok=True)
 if (ROOT/'preflight-v2.exit').read_text().strip()!='0':raise ValueError('preflight not successful')
 assert json.loads((ROOT/'grpo-study-v2/reports/gpu-startpoint.json').read_text())['status']=='passed'
 manifest=[]
 for branch,kind in [('continue_sft','sft'),('grpo_acc','grpo'),('grpo_accfmt','grpo')]:
  spent=max(0,time.time()-START)/3600*PRICE
  if spent+RESERVE>=LIMIT:raise ValueError('session cost allowance exhausted')
  run_id='v2_'+branch+'_pilot_s43'
  if (ROOT/'runs'/run_id).exists():raise ValueError(f'run already exists: {run_id}; inspect before restarting')
  out=ROOT/f'grpo-study-v2/configs/account_{branch}_pilot.yaml'
  subprocess.run([sys.executable,str(ROOT/'grpo-study-v2/scripts/prepare_budget.py'),'--branch',branch,'--limit',str(LIMIT),'--spent',str(spent),'--non-gpu','0','--reserve',str(RESERVE),'--hourly-price',str(PRICE),'--output',str(out)],check=True,cwd=ROOT)
  started=time.time()
  with (ROOT/f'{run_id}.log').open('w') as log:
   subprocess.run([sys.executable,'-m','math_posttrain.cli','train-'+kind,'--config',str(out),'--sft','runs/sft_frozen','--run-id',run_id],check=True,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  subprocess.run([sys.executable,str(ROOT/'grpo-study-v2/scripts/check_pilot.py'),'--kind',kind,'--run','runs/'+run_id],check=True,cwd=ROOT)
  result=json.loads((ROOT/'runs'/run_id/'result.json').read_text())
  record=dict(branch=branch,run_id=run_id,outer_elapsed_seconds=time.time()-started,estimated_session_cost=(time.time()-START)/3600*PRICE,cost_basis='conservative elapsed since instance creation at screenshot rate; not provider invoice',result=result)
  manifest.append(record)
  (ROOT/'grpo-study-v2/reports/pilot-summary.json').write_text(json.dumps(manifest,indent=2))
 print('All three four-step pilots passed. No formal long training started.')
if __name__=='__main__':main()
