"""Fixed-endpoint validation then three-model official test with session deadline."""
import json,time,subprocess,sys
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[2]
def main():
 session=json.loads((ROOT/'grpo-study-v2/reports/formal-session.json').read_text())
 if (ROOT/'formal-stage.exit').read_text().strip()!='0':raise ValueError('formal training not complete')
 summary=json.loads((ROOT/'grpo-study-v2/reports/formal-training-summary.json').read_text())
 assert len(summary)==3 and all(r['result']['status']=='complete' for r in summary)
 models={r['branch']:dict(adapter=f"runs/{r['run_id']}/final",sft='runs/sft_frozen') for r in summary}
 cfg=yaml.safe_load((ROOT/'grpo-study-v2/configs/grpo_acc_pilot.yaml').read_text())
 cfg['model_dtype']='float32';cfg['budget']=None
 path=ROOT/'grpo-study-v2/configs/formal-eval-s43.yaml'
 with path.open('x') as f:yaml.safe_dump(cfg,f,sort_keys=False)
 def command(args,log):
  remaining=session['started']+(session['cap']-5)/session['price']*3600-time.time()
  if remaining<=0:raise ValueError('session reserve reached; no further evaluation')
  with (ROOT/log).open('x') as f:
   subprocess.run([sys.executable,'-m','math_posttrain.cli',*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=remaining)
 validation=ROOT/'grpo-study-v2/configs/validation-models-s43.json'
 with validation.open('x') as f:json.dump(dict(sft_anchor=dict(adapter='runs/sft_frozen'),**models),f,indent=2)
 command(['evaluate','--config',str(path),'--models',str(validation),'--split','validation','--output','runs/v2_validation_s43'],'formal-validation.log')
 final=ROOT/'grpo-study-v2/configs/test-models-s43.json'
 with final.open('x') as f:json.dump(models,f,indent=2)
 protocol='grpo-study-v2/reports/frozen-eval-protocol-s43.json'
 command(['freeze-protocol','--config',str(path),'--models',str(final),'--output',protocol],'formal-freeze-eval.log')
 command(['evaluate','--config',str(path),'--models',str(final),'--split','official_test','--output','runs/v2_official_test_s43','--protocol',protocol],'formal-official-test.log')
 print('Validation and complete three-model official test finished; reporting/backup remains.')
if __name__=='__main__':main()
