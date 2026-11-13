"""Report complete V3 outputs; no checkpoint or hyperparameter selection."""
import json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];V3=ROOT/'grpo-study-v3';REPORT=V3/'reports'
def read(p):return json.loads(Path(p).read_text())
def paired(before,after):
    if len(before)!=len(after) or not before:raise ValueError('paired complete records required')
    delta=np.array(after,dtype=int)-np.array(before,dtype=int)
    lost=int(sum(delta==-1));gained=int(sum(delta==1));unchanged=len(delta)-lost-gained
    rng=np.random.default_rng(42);draws=rng.multinomial(len(delta),np.array([lost,unchanged,gained])/len(delta),50000)
    return dict(gained=gained,lost=lost,unchanged=unchanged,delta_pp=float(delta.mean()*100),paired_ci95_pp=np.quantile((draws[:,2]-draws[:,0])/len(delta)*100,[.025,.975]).tolist())
def main():
    summary=read(REPORT/'training-summary.json');formal={r['branch']:r for r in summary if r['stage']=='formal'}
    if set(formal)!={'uniform','mixture'}:raise ValueError('both complete training branches required')
    results={};pair_rows=[]
    for split,count in [('validation',500),('official_test',1319)]:
        root=ROOT/f'runs/v3_{split}_s43';preds={n:[json.loads(x) for x in (root/f'{n}.jsonl').read_text().splitlines()] for n in formal}
        indexed={n:{r['sample_id']:r for r in rs} for n,rs in preds.items()};ids=sorted(indexed['uniform'])
        if any(len(rs)!=count or len(indexed[n])!=count or set(indexed[n])!=set(ids) for n,rs in preds.items()):raise ValueError('incomplete paired evaluation')
        metrics={}
        for name,rs in preds.items():
            for r in rs:
                reference=indexed['uniform'][r['sample_id']]
                for key in ['answer','prompt_hash','model_revision','model_dtype','sft_startpoint_hashes']:
                    if r[key]!=reference[key]:raise ValueError(f'evaluation mismatch: {key}')
                if r['model_dtype']!='float32':raise ValueError('expected FP32 evaluation')
            metrics[name]=dict(count=len(rs),correct=sum(r['correct'] for r in rs),accuracy=sum(r['correct'] for r in rs)/count,format_rate=sum(r['format_ok'] for r in rs)/count,truncated=sum(r['truncated'] for r in rs),mean_tokens=float(np.mean([r['output_tokens'] for r in rs])),generation_seconds=sum(r['latency_seconds'] for r in rs))
            if abs(metrics[name]['accuracy']-read(root/'summary.json')[name]['accuracy'])>1e-12:raise ValueError('summary mismatch')
        results[split]=dict(metrics=metrics,uniform_to_mixture=paired([indexed['uniform'][i]['correct'] for i in ids],[indexed['mixture'][i]['correct'] for i in ids]))
        for sid in ids:
            a,b=indexed['uniform'][sid],indexed['mixture'][sid]
            pair_rows.append(dict(split=split,sample_id=sid,uniform_correct=a['correct'],mixture_correct=b['correct'],change=int(b['correct'])-int(a['correct']),uniform_completion=a['completion'],mixture_completion=b['completion']))
    plan=read(REPORT/'sampling-plan-s43.json');signals={}
    for name,r in formal.items():
        run=ROOT/'runs'/r['run_id'];logs=[json.loads(x) for x in (run/'reward-groups.jsonl').read_text().splitlines()];groups=[g for log in logs for g in log['groups']]
        exposed=[log['rows'][i]['sample_id'] for log in logs for i in range(0,len(log['rows']),4)]
        if exposed!=plan[name] or len(groups)!=640:raise ValueError('exposure audit failed')
        signals[name]=dict(groups=len(groups),unique_questions=len(set(exposed)),zero_variance_groups=sum(g['zero_variance'] for g in groups),mixed_groups=sum(not g['zero_variance'] for g in groups),generated_tokens=sum(log['generated_tokens'] for log in logs),elapsed_seconds=r['result']['elapsed_seconds'],peak_reserved_bytes=r['result']['peak_reserved_bytes'])
    batches=[read(p) for p in (REPORT/'scoring').glob('batch-*.json')]
    if sum(len(b['rows']) for b in batches)!=1999:raise ValueError('scoring archive incomplete')
    score_seconds=sum(b['elapsed_seconds'] for b in batches);session=read(REPORT/'session.json')
    costs=dict(scoring_generation_seconds=score_seconds,scoring_generation_yuan=score_seconds/3600*session['hourly_yuan'],definition='Generation-only scoring excludes model load, waits, retry and idle. Session wall-clock separately accounts for those.',training_yuan={n:s['elapsed_seconds']/3600*session['hourly_yuan'] for n,s in signals.items()},mixed_incremental_scoring_plus_training_yuan=(score_seconds+signals['mixture']['elapsed_seconds'])/3600*session['hourly_yuan'])
    result=dict(results=results,signals=signals,costs=costs,strata_sizes=plan['strata_sizes'],seed=43,bootstrap_replicates=50000,bootstrap_seed=42,limitations=['Single seed; question bootstrap does not measure training variability.','Four draws are a noisy difficulty proxy.','Sampling changes training distribution; no importance correction.','Both branches have equal steps and generations, not guaranteed equal FLOPs or wall time.','GSM8K test was already used in earlier experiments; not a fresh blind benchmark.','Scoring overhead must be included in end-to-end comparisons.'])
    (REPORT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    with (REPORT/'paired-predictions.jsonl').open('w') as f:
        for r in pair_rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    lines=['# 实验三结果','', '种子43；固定160步；两组均从原冻结SFT重新训练。','', '| 数据集 | 模型 | 正确题数 | 准确率 |','|---|---|---:|---:|']
    for split,payload in results.items():
        for name,m in payload['metrics'].items():lines.append(f"| {split} | {name} | {m['correct']}/{m['count']} | {m['accuracy']:.2%} |")
        c=payload['uniform_to_mixture'];lines+=['',f"{split} 混合相对均匀：{c['delta_pp']:+.2f}个百分点；配对95%区间 {c['paired_ci95_pp']}；新答对{c['gained']}题、改错{c['lost']}题。",'']
    lines+=['## 信号与成本','',json.dumps(signals,ensure_ascii=False,indent=2),'',f'全池打分生成耗时 {score_seconds:.1f} 秒，生成成本估算 {costs["scoring_generation_yuan"]:.2f} 元（不含加载、等待、空闲；全部计入session账）。','',*['- '+x for x in result['limitations']]]
    (V3/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n')
    print('Complete paired results and signal/cost report written. No positive result presumed.')
if __name__=='__main__':main()
