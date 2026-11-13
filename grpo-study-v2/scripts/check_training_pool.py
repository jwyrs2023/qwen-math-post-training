"""Verify shared prompt/target length eligibility using frozen local tokenizer."""
import json
from pathlib import Path
from transformers import AutoTokenizer
from math_posttrain.protocol import require_clean_audit
from math_posttrain.prompting import render_prompt,target,encode_sft
ROOT=Path(__file__).resolve().parents[2]
def main():
 (ROOT/'grpo-study-v2/reports').mkdir(parents=True,exist_ok=True)
 manifest=require_clean_audit(ROOT/'data/manifests/splits_clean.json')
 raw=json.loads((ROOT/'data/raw/train.json').read_text())
 tok=AutoTokenizer.from_pretrained(ROOT/'runs/sft_frozen',local_files_only=True)
 ids=[];excluded=[];lengths=[]
 for row in manifest['rows']:
  if row['split']!='grpo_train':continue
  r=raw[int(row['sample_id'].split(':')[1])];p=render_prompt(r['question'],ROOT/'prompts/math_v1.txt')
  np=len(tok.encode(p,add_special_tokens=False));reason=[]
  if np>256:reason.append('prompt_over_256')
  try: ex=encode_sft(tok,p,target(r['answer'],tok.eos_token),1024)
  except ValueError as e:reason.append(str(e))
  if reason:excluded.append(dict(sample_id=row['sample_id'],reasons=reason))
  else:ids.append(row['sample_id']);lengths.append(dict(sample_id=row['sample_id'],prompt_tokens=np,supervised_tokens=sum(x!=-100 for x in ex['labels'])))
 result=dict(eligible_ids=ids,excluded=excluded,lengths=lengths,identical_pool_without_filter=not excluded,limits=dict(prompt=256,sft_total=1024),tokenizer='verified frozen SFT local tokenizer')
 (ROOT/'grpo-study-v2/reports/training-pool.json').write_text(json.dumps(result,indent=2))
 print(f'Eligible {len(ids)}; excluded {len(excluded)}. '+('No pool change needed.' if not excluded else 'New audited common manifest required.'))
if __name__=='__main__':main()
