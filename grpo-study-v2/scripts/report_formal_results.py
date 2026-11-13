"""Analyze completed real V2 test outputs against each other and historical SFT."""
import json,tarfile,itertools
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'grpo-study-v2/reports'
def main():
 test=ROOT/'runs/v2_official_test_s43'
 summary=json.loads((test/'summary.json').read_text())
 names=['continue_sft','grpo_acc','grpo_accfmt']
 preds={n:[json.loads(x) for x in (test/f'{n}.jsonl').read_text().splitlines()] for n in names}
 archive=ROOT/'reports/official-test-backup.tar.gz'
 with tarfile.open(archive) as t:preds['historical_sft']=[json.loads(x) for x in t.extractfile('runs/official_test_four_groups/sft.jsonl').read().splitlines()]
 indexed={n:{r['sample_id']:r for r in rs} for n,rs in preds.items()}
 ids=sorted(indexed['historical_sft'])
 assert len(ids)==1319
 for n,rs in preds.items():
  assert len(rs)==len(indexed[n])==1319 and set(indexed[n])==set(ids)
  for r in rs:
   old=indexed['historical_sft'][r['sample_id']]
   assert r['answer']==old['answer'] and r['prompt_hash']==old['prompt_hash']
   assert r['model_revision']==old['model_revision'] and r['model_dtype']==old['model_dtype']=='float32'
   if n!='historical_sft':assert r['sft_startpoint_hashes']==old['checkpoint_hashes'],'historical anchor checkpoint mismatch'
 metrics={n:dict(count=len(rs),correct=sum(r['correct'] for r in rs),accuracy=sum(r['correct'] for r in rs)/len(rs),format_rate=sum(r['format_ok'] for r in rs)/len(rs),truncated=sum(r['truncated'] for r in rs),mean_output_tokens=float(np.mean([r['output_tokens'] for r in rs])),generation_seconds=sum(r['latency_seconds'] for r in rs)) for n,rs in preds.items()}
 for n in names:assert abs(metrics[n]['accuracy']-summary[n]['accuracy'])<1e-12
 comparisons={};rng=np.random.default_rng(42)
 pairs=[('historical_sft',n) for n in names]+[('continue_sft','grpo_acc'),('continue_sft','grpo_accfmt'),('grpo_acc','grpo_accfmt')]
 for before,after in pairs:
  delta=np.array([int(indexed[after][i]['correct'])-int(indexed[before][i]['correct']) for i in ids]);lost=int(sum(delta==-1));gained=int(sum(delta==1));unchanged=len(ids)-lost-gained
  draws=rng.multinomial(len(ids),np.array([lost,unchanged,gained])/len(ids),50000)
  ci=np.quantile((draws[:,2]-draws[:,0])/len(ids)*100,[.025,.975]).tolist()
  comparisons[before+'→'+after]=dict(gained=gained,lost=lost,delta_pp=float(delta.mean()*100),paired_ci95_pp=ci)
 training=json.loads((REPORT/'formal-training-summary.json').read_text())
 times={r['branch']:r['result']['elapsed_seconds'] for r in training}
 spread=max(times.values())/min(times.values())-1
 result=dict(metrics=metrics,comparisons=comparisons,training_seconds=times,training_time_spread=spread,time_match_within_10pct=spread<=.1,seed=43,bootstrap=dict(replicates=50000,seed=42),historical_anchor='V1 frozen SFT, verified same checkpoint/revision/prompt/precision; historical inference run',limitations=['Single training seed; question bootstrap does not measure training variability.','Continue-SFT uses reference reasoning; GRPO uses result supervision.','Test was already examined in V1.'])
 (REPORT/'formal-results.json').write_text(json.dumps(result,indent=2)+'\n')
 lines=['# 实验二正式结果','', '单种子43、固定终点；原SFT为校验一致性的V1历史锚点。','', '| 模型 | 正确题数 | 准确率 | 格式率 | 截断题数 |','|---|---:|---:|---:|---:|']
 for n,m in metrics.items():lines.append(f"| {n} | {m['correct']}/1319 | {m['accuracy']:.2%} | {m['format_rate']:.2%} | {m['truncated']} |")
 lines+=['','| 对比 | 新答对 | 改错 | 净变化百分点 | 配对95%区间 |','|---|---:|---:|---:|---|']
 for n,c in comparisons.items():lines.append(f"| {n} | {c['gained']} | {c['lost']} | {c['delta_pp']:+.2f} | {c['paired_ci95_pp']} |")
 lines+=['',f'三分支训练最大/最小耗时差：{spread:.1%}。超过10%时不能宣称严格等时对照。','',*['- '+x for x in result['limitations']]]
 (REPORT/'formal-results.md').write_text('\n'.join(lines)+'\n')
 print('Real complete test outputs validated and analyzed.')
if __name__=='__main__':main()
