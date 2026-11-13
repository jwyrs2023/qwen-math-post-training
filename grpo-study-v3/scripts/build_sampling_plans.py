"""Frozen train-only difficulty strata and reproducible replacement-sampling plans."""
import json,random,hashlib,argparse
from pathlib import Path

def build(scores,train_ids,count=640,seed=43):
 ids=set(train_ids)
 if len(ids)!=len(train_ids) or len(scores)!=len(ids) or {s['sample_id'] for s in scores}!=ids:raise ValueError('scores must cover exactly the unique training pool')
 if count<=0 or count%5:raise ValueError('count must be a positive multiple of five')
 strata={'hard':[],'middle':[],'easy':[]}
 for s in scores:
  k=s['correct_of_four']
  if type(k) is not int or not 0<=k<=4:raise ValueError('invalid correct-of-four score')
  strata['hard' if k==0 else 'easy' if k==4 else 'middle'].append(s['sample_id'])
 if any(not v for v in strata.values()):raise ValueError('empty stratum; revise protocol before training')
 for v in strata.values():v.sort()
 rng=random.Random(seed)
 uniform=rng.choices(sorted(ids),k=count)
 mixture=[]
 quotas=dict(hard=count//5,middle=count*3//5,easy=count//5)
 for k,n in quotas.items():mixture+=rng.choices(strata[k],k=n)
 rng.shuffle(mixture)
 category={sid:k for k,ss in strata.items() for sid in ss}
 return dict(seed=seed,count=count,replacement=True,strata_sizes={k:len(v) for k,v in strata.items()},mixture_quotas=quotas,uniform=uniform,mixture=mixture,category=category,interpretation='Training distribution intervention; no importance correction; scoring overhead must be reported.')

def main():
 p=argparse.ArgumentParser();p.add_argument('--scores',required=True);p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);p.add_argument('--count',type=int,default=640);p.add_argument('--seed',type=int,default=43);a=p.parse_args()
 m=json.loads(Path(a.manifest).read_text());ids=[r['sample_id'] for r in m['rows'] if r['split']=='grpo_train']
 scores=[json.loads(x) for x in Path(a.scores).read_text().splitlines()]
 result=build(scores,ids,a.count,a.seed);result['scores_sha256']=hashlib.sha256(Path(a.scores).read_bytes()).hexdigest();result['manifest_sha256']=hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest()
 with Path(a.output).open('x') as f:json.dump(result,f,indent=2)
 print('Train-only plans frozen. No training started.')
if __name__=='__main__':main()
