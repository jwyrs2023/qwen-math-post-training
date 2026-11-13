"""Diagnose actual V2 rewards; never treat evolving-policy logs as SFT scoring."""
import json,tarfile,hashlib,statistics
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'grpo-study-v3/reports'
def main():
 archive=ROOT/'grpo-study-v2/reports/v2-formal-training-backup.tar.gz'
 assert hashlib.sha256(archive.read_bytes()).hexdigest()=='6877b306f1e43789643a881498f6755fce584bff198385e87e545eefc84f49fd'
 result={}
 with tarfile.open(archive) as t:
  for branch in ['grpo_acc','grpo_accfmt']:
   name=f'runs/v2_{branch}_formal_s43/reward-groups.jsonl'
   batches=[json.loads(x) for x in t.extractfile(name).read().splitlines()]
   groups=[];by_id=defaultdict(list)
   for batch_no,batch in enumerate(batches):
    assert len(batch['rows'])==4*len(batch['groups'])
    for j,g in enumerate(batch['groups']):
     rs=batch['rows'][4*j:4*j+4];ids={r['sample_id'] for r in rs};assert len(ids)==1
     sid=next(iter(ids)); k=sum(r['accuracy'] for r in rs)
     entry=dict(sample_id=sid,batch_index=batch_no,correct_of_four=k,format_mixed=len({r['format'] for r in rs})>1,zero_total_variance=g['zero_variance'])
     groups.append(entry);by_id[sid].append(entry)
   hist=Counter(g['correct_of_four'] for g in groups)
   result[branch]=dict(groups=len(groups),unique_prompts=len(by_id),training_pool_size=1999,coverage=len(by_id)/1999,correct_of_four_histogram={str(k):hist[k] for k in range(5)},mixed_accuracy_fraction=sum(0<g['correct_of_four']<4 for g in groups)/len(groups),zero_total_variance_fraction=sum(g['zero_total_variance'] for g in groups)/len(groups),repeated_prompt_ids=sum(len(v)>1 for v in by_id.values()),groups_by_prompt=dict(by_id))
 (OUT/'previous-signals.json').write_text(json.dumps(result,indent=2)+'\n')
 lines=['# 实验三前置分析','', '基于V2真实训练归档，未新增训练、未使用测试集筛题。','', '| 分支 | 组数 | 唯一题数 | 覆盖训练池 | 全错组 | 部分正确组 | 全对组 |','|---|---:|---:|---:|---:|---:|---:|']
 for n,d in result.items():
  h=d['correct_of_four_histogram'];lines.append(f"| {n} | {d['groups']} | {d['unique_prompts']} | {d['coverage']:.1%} | {h['0']} | {sum(h[str(k)] for k in [1,2,3])} | {h['4']} |")
 lines+=['','## 结论边界','', '历史日志中的模型在持续更新，而且只覆盖训练池的一部分，不能当作冻结SFT对全池的难度标签。没有重复观测的题，也不能据此证明它会反复产生零方差信号。','', '下一步须用同一冻结SFT对全部1999道训练题独立采样打分，保留每次生成和token/成本记录。每题4次仅为有噪声的难度代理；采样概率、前置打分成本与重新采样后的实际有效组比例必须一起报告。']
 (OUT/'previous-signals.md').write_text('\n'.join(lines)+'\n')
 print('\n'.join(lines[:9]))
if __name__=='__main__':main()
