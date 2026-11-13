"""Frozen-SFT, train-only difficulty scoring with resumable deterministic batches."""
import argparse, hashlib, json, time, os
from pathlib import Path

REVISION='8faed761d45a263340a0528343f099c05c9a4323'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def train_rows(manifest, raw):
    rows=[]
    for r in manifest['rows']:
        if r['split']=='grpo_train':
            record=raw[int(r['sample_id'].split(':')[1])]
            rows.append(dict(sample_id=r['sample_id'],question=record['question'],gold=record['answer'].split('####')[1].strip()))
    if len(rows)!=1999 or len({x['sample_id'] for x in rows})!=1999: raise ValueError('unexpected training pool')
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--sft',default='runs/sft_frozen');p.add_argument('--session',default='grpo-study-v3/reports/session.json');p.add_argument('--limit',type=int,default=1999);p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    from math_posttrain.protocol import require_clean_audit
    from math_posttrain.checkpoints import verify
    manifest_path='data/manifests/splits_clean.json';manifest=require_clean_audit(manifest_path)
    rows=train_rows(manifest,json.loads(Path('data/raw/train.json').read_text()))
    if not 0<a.limit<=len(rows):raise ValueError('invalid limit')
    hashes=verify(a.sft)
    identity=dict(revision=REVISION,sft_hashes=hashes,manifest_sha256=sha(manifest_path),prompt_sha256=sha('prompts/math_v1.txt'),scorer_sha256=sha(__file__),seed=43,questions_per_batch=2,generations_per_question=4,temperature=.8,top_p=.95,max_new_tokens=512,weights='float32',autocast='bfloat16',rng='seed43+batch_index; interrupted batch regenerated',training_pool_size=len(rows))
    if a.dry_run: print(json.dumps(identity));return
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    idpath=out/'identity.json'
    if idpath.exists():
        if json.loads(idpath.read_text())!=identity:raise ValueError('scoring identity mismatch')
    else:
        with idpath.open('x') as f:json.dump(identity,f,indent=2)
    existing={}
    for file in sorted(out.glob('batch-*.json')):
        batch=json.loads(file.read_text())
        for row in batch['rows']:
            if row['sample_id'] in existing:raise ValueError('duplicate score')
            existing[row['sample_id']]=row
    session=json.loads(Path(a.session).read_text())
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM,set_seed
    from peft import PeftModel
    from huggingface_hub import snapshot_download
    from math_posttrain.prompting import render_prompt
    from math_posttrain.rewards import score_batch
    from math_posttrain.parsing import parse_number
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise ValueError('CUDA BF16 required')
    model_path=snapshot_download('Qwen/Qwen2.5-1.5B',revision=REVISION,local_files_only=True)
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True);tokenizer.pad_token=tokenizer.eos_token;tokenizer.padding_side='left'
    model=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,torch_dtype=torch.float32)
    model=PeftModel.from_pretrained(model,a.sft).merge_and_unload().cuda().eval();model.config.use_cache=True
    torch.cuda.reset_peak_memory_stats();start=time.time()
    for idx in range(0,a.limit,2):
        group=rows[idx:min(idx+2,a.limit)]
        if all(r['sample_id'] in existing for r in group):continue
        if any(r['sample_id'] in existing for r in group):raise ValueError('partial batch; limit must preserve batch boundaries')
        estimate=(time.time()-session['started_at_unix'])/3600*session['hourly_yuan']
        if estimate>=session['budget_yuan']-session['reserve_yuan']:raise RuntimeError('budget reserve reached')
        set_seed(43+idx//2);prompts=[render_prompt(r['question']) for r in group]
        if any(len(tokenizer.encode(x,add_special_tokens=False))>256 for x in prompts):raise ValueError('overlong training prompt')
        inputs=tokenizer(prompts,return_tensors='pt',padding=True,add_special_tokens=False).to('cuda');t=time.time()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            output=model.generate(**inputs,do_sample=True,num_return_sequences=4,temperature=.8,top_p=.95,max_new_tokens=512,pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
        torch.cuda.synchronize();records=[]
        for j,r in enumerate(group):
            completions=[];lengths=[]
            for ids in output[j*4:(j+1)*4,inputs['input_ids'].shape[1]:]:
                tok=ids.tolist();n=tok.index(tokenizer.eos_token_id)+1 if tokenizer.eos_token_id in tok else len(tok)
                completions.append(tokenizer.decode(tok[:n],skip_special_tokens=True));lengths.append(n)
            rewards=score_batch(completions,[parse_number(r['gold'])]*4)
            records.append(dict(sample_id=r['sample_id'],correct_of_four=sum(x['accuracy'] for x in rewards),completions=completions,rewards=rewards,generated_lengths=lengths))
        result=dict(batch_index=idx//2,seed=43+idx//2,elapsed_seconds=time.time()-t,peak_reserved_bytes=torch.cuda.max_memory_reserved(),rows=records)
        dest=out/f'batch-{idx//2:04d}.json';temp=dest.with_suffix('.tmp');temp.write_text(json.dumps(result,ensure_ascii=False));os.replace(temp,dest)
        existing.update({r['sample_id']:r for r in records})
        print(json.dumps(dict(scored=len(existing),total=len(rows),batch_seconds=result['elapsed_seconds'],estimated_session_yuan=estimate)),flush=True)
    with (out/'scores.jsonl').open('w') as f:
        for row in rows[:a.limit]:f.write(json.dumps(existing[row['sample_id']],ensure_ascii=False)+'\n')
    (out/'summary.json').write_text(json.dumps(dict(scored=a.limit,pool=len(rows),complete=a.limit==len(rows),elapsed_this_invocation=time.time()-start,peak_reserved_bytes=torch.cuda.max_memory_reserved()),indent=2))
if __name__=='__main__':main()
