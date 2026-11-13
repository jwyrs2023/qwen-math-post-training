"""Calibrate three routes, freeze endpoints, then train fresh seed43 branches."""
import json,time,subprocess,sys,math,hashlib
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'grpo-study-v2/reports'
BRANCHES=[('continue_sft','sft'),('grpo_acc','grpo'),('grpo_accfmt','grpo')]
PRICE=2.24;CAP=40.;RESERVE=24.;TARGET=1800

def main():
 REPORT.mkdir(exist_ok=True,parents=True)
 session=REPORT/'formal-session.json'
 with session.open('x') as f:json.dump(dict(started=time.time(),price=PRICE,cap=CAP,reserve=RESERVE,seed=43,target_training_seconds=TARGET),f,indent=2)
 started=json.loads(session.read_text())['started']
 def run(branch,kind,stage,steps):
  c=yaml.safe_load((ROOT/f'grpo-study-v2/configs/{branch}_pilot.yaml').read_text())
  spent=(time.time()-started)/3600*PRICE
  if spent+RESERVE>=CAP:raise ValueError('insufficient session budget')
  c.update(max_steps=steps,save_steps=20 if stage=='calibration' else 50,eval_steps=None,budget=dict(limit=CAP,spent=spent,non_gpu=0,reserve=RESERVE,hourly_price=PRICE))
  cfg=ROOT/f'grpo-study-v2/configs/{branch}_{stage}_s43.yaml'
  with cfg.open('x') as f:yaml.safe_dump(c,f,sort_keys=False)
  rid=f'v2_{branch}_{stage}_s43';cmd=[sys.executable,'-m','math_posttrain.cli','train-'+kind,'--config',str(cfg),'--sft','runs/sft_frozen','--run-id',rid]
  with (ROOT/f'{rid}.log').open('x') as log:subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
  result=json.loads((ROOT/'runs'/rid/'result.json').read_text())
  if result['status']!='complete':raise ValueError(f'{rid} stopped early')
  return rid,result,c
 measurements={}
 for branch,kind in BRANCHES:
  rid,result,c=run(branch,kind,'calibration',20)
  logs=[json.loads(x) for x in (ROOT/'runs'/rid/'metrics.jsonl').read_text().splitlines()]
  points={int(x['step']):x['time'] for x in logs if 'loss' in x and x['step']>=5}
  if 5 not in points or 20 not in points:raise ValueError('missing warm calibration timing')
  rate=(points[20]-points[5])/15
  measurements[branch]=dict(run_id=rid,seconds_per_step=rate,result=result)
  (REPORT/'formal-calibration.json').write_text(json.dumps(measurements,indent=2))
 grpo_rate=(measurements['grpo_acc']['seconds_per_step']+measurements['grpo_accfmt']['seconds_per_step'])/2
 steps=dict(continue_sft=max(1,round(TARGET/measurements['continue_sft']['seconds_per_step'])),grpo_acc=max(1,round(TARGET/grpo_rate)),grpo_accfmt=max(1,round(TARGET/grpo_rate)))
 payload=dict(measurements=measurements,steps=steps,seed=43,target_seconds=TARGET,grpo_pair_same_steps=True,selection='fixed endpoint; no validation or test used to select steps',expected_sft_epochs=steps['continue_sft']*16/1999,expected_grpo_prompt_exposures=steps['grpo_acc']*4,cost_tolerance=.1,limitations=['Calibration timing estimate, not equal-FLOPs guarantee.','Actual >10% time mismatch must be disclosed; do not select best test checkpoint.'])
 with (REPORT/'formal-training-protocol.json').open('x') as f:json.dump(payload,f,indent=2)
 results=[]
 for branch,kind in BRANCHES:
  rid,result,c=run(branch,kind,'formal',steps[branch]);results.append(dict(branch=branch,run_id=rid,result=result,steps=steps[branch],session_cost_estimate=(time.time()-started)/3600*PRICE))
  (REPORT/'formal-training-summary.json').write_text(json.dumps(results,indent=2))
 print('Three formal branches complete; evaluation remains to be run under frozen protocol.')
if __name__=='__main__':main()
