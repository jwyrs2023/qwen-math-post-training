"""Read-only archive audit; write V2 artifacts without changing V1 reports."""
import hashlib,json,tarfile,statistics,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'grpo-study-v2/reports'
def rows(path,name):
 with tarfile.open(path) as t:return [json.loads(x) for x in t.extractfile(name).read().splitlines()]
def advantage(values):
 mean=statistics.mean(values); std=statistics.stdev(values)
 return [(v-mean)/(std+1e-4) for v in values]
def main():
 test=ROOT/'reports/official-test-backup.tar.gz'; train=ROOT/'reports/formal-pair-backup.tar.gz'
 old=json.loads((ROOT/'reports/official-test-analysis.json').read_text())
 hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [test,train]}
 assert hashes==old['archive_sha256'], 'Historical archive hash mismatch'
 preds={n:rows(test,f'runs/official_test_four_groups/{n}.jsonl') for n in ['base','sft','a','b']}
 indexed={n:{r['sample_id']:r for r in rs} for n,rs in preds.items()}
 ids=set(indexed['sft']);assert len(ids)==1319
 for n,rs in preds.items():
  assert len(rs)==len(indexed[n])==1319 and set(indexed[n])==ids
  assert abs(sum(r['correct'] for r in rs)/1319-old['results'][n]['accuracy'])<1e-12
  assert all(r['answer']==indexed['sft'][r['sample_id']]['answer'] for r in rs)
 transfers={}
 for n in ['a','b']:
  groups={k:[] for k in ['both_correct','both_wrong','gained','lost']}
  for i in sorted(ids):
   x=indexed['sft'][i]['correct'];y=indexed[n][i]['correct']
   key=('both_correct' if y else 'lost') if x else ('gained' if y else 'both_wrong')
   groups[key].append(i)
  assert len(groups['gained'])==old['comparisons'][f'sft→{n}']['gained']
  assert len(groups['lost'])==old['comparisons'][f'sft→{n}']['lost']
  transfers[n]=groups
 diagnostics={}
 for n in ['a','b']:
  batches=rows(train,f'runs/grpo_{n}_formal_s42/reward-groups.jsonl')
  totals=dict(groups=0,all_correct=0,all_wrong=0,mixed_accuracy=0,format_mixed=0,zero_variance=0,advantage_sign_changed=0,advantage_changed=0)
  missing=[]
  for batch in batches:
   assert len(batch['rows'])==4*len(batch['groups'])
   for j,g in enumerate(batch['groups']):
    rs=batch['rows'][j*4:(j+1)*4]; a=[r['accuracy'] for r in rs]; b=[r['accuracy']+.1*r['format'] for r in rs]
    actual=advantage(a if n=='a' else b)
    assert all(math.isclose(x,y,abs_tol=1e-9) for x,y in zip(actual,g['advantages']))
    totals['groups']+=1;totals['all_correct']+=all(a);totals['all_wrong']+=not any(a)
    totals['mixed_accuracy']+=len(set(a))>1;totals['format_mixed']+=len({r['format'] for r in rs})>1
    totals['zero_variance']+=g['zero_variance']
    aa,bb=advantage(a),advantage(b)
    totals['advantage_sign_changed']+=any((x>1e-9)-(x< -1e-9)!=(y>1e-9)-(y< -1e-9) for x,y in zip(aa,bb))
    totals['advantage_changed']+=any(abs(x-y)>1e-8 for x,y in zip(aa,bb))
  assert totals['groups']==800
  diagnostics[n]=totals
 result=dict(archive_sha256=hashes,transfers=transfers,reward_diagnostics=diagnostics,limitations=['Archived reward rows lack prompt IDs and raw completions; cannot reconstruct prompt-linked reward case analysis.','Counterfactual uses archived sample standard deviation and epsilon=1e-4, verified against logged advantages; not a simulated training trajectory.','Single training seed; test data already inspected in V1.'])
 OUT.mkdir(exist_ok=True)
 (OUT/'history-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 text=['# V1历史复核与V2机制诊断','', '两份归档SHA256与历史报告一致，四组逐题ID、参考答案与准确率复核通过。','', '## 逐题转移（相对SFT）','']
 for n,g in transfers.items():text.append(f"- {n.upper()}："+'；'.join(f'{k}={len(v)}' for k,v in g.items()))
 text+=['','## 同组样本反事实奖励分析','', '下列比较固定同一组生成，只改变奖励定义；不能解释成重新训练的效果。','']
 for n,d in diagnostics.items():text.append(f'- {n.upper()}：`{json.dumps(d)}`')
 text+=['','## 当前边界','',*['- '+x for x in result['limitations']],'','下一步：审查训练实现与归档resolved配置、制定继续SFT同预算方案，补齐未来日志的prompt ID与原始completion。']
 (OUT/'history-audit.md').write_text('\n'.join(text)+'\n')
 print(json.dumps(diagnostics,indent=2))
if __name__=='__main__':main()
