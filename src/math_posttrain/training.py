"""Shared real trainer execution, logging and full-state checkpoint resume."""
import json
import time
from pathlib import Path
from .provenance import create_run, file_hash
from .protocol import require_clean_audit
from .data import digest


def execute(config, run_id, kind, sft=None, dry_run=False, resume=None):
    manifest = require_clean_audit(config['manifest'])
    if kind not in ('sft', 'grpo'):
        raise ValueError('unknown training kind')
    if kind == 'grpo' or sft is not None:
        if not sft:
            raise ValueError('GRPO requires frozen SFT adapter')
        from .checkpoints import verify
        startpoint_hashes = verify(sft)
        lineage = json.loads((Path(sft) / 'lineage.json').read_text())
        if lineage.get('kind') != 'sft' or lineage.get('revision') != config.get('revision') or lineage.get('model') != config['model']:
            raise ValueError('SFT startpoint model/revision mismatch')
    else:
        startpoint_hashes = None
    if not run_id or Path(run_id).name != run_id or run_id in ('.', '..'):
        raise ValueError('invalid run ID')
    if kind == 'grpo' and (config['batch_size'] * config['gradient_accumulation_steps']) % config['group_size']:
        raise ValueError('GRPO generation batch must be divisible by group size')
    if dry_run:
        return dict(kind=kind, config=config, manifest_hash=digest(manifest),
                    gpu_verified=False, action='configuration validated; no training or downloads')
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from trl import SFTConfig, SFTTrainer, GRPOConfig, GRPOTrainer
    from .prompting import pad_examples, render_prompt
    from .train_sft import prepare_examples
    from .train_grpo import make_reward_callback
    from .checkpoints import verify
    from .parsing import extract_gold
    if not torch.cuda.is_available():
        raise ValueError('formal training requires a CUDA server; use --dry-run locally')
    if not config.get('revision') or len(config['revision']) != 40:
        raise ValueError('pin model revision to a 40-character commit before training')
    if not torch.cuda.is_bf16_supported():
        raise ValueError('bf16 unsupported; explicit protocol revision required')
    if not config.get('budget'):
        raise ValueError('formal training requires actual budget and hourly price')
    if int(__import__('os').environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('only single-process training is verified')
    if config['budget']['spent'] + config['budget']['non_gpu'] + config['budget']['reserve'] >= config['budget']['limit']:
        raise ValueError('no budget available; refusing model download')
    set_seed(config['seed'])
    root = Path('runs') / run_id
    fingerprint = digest(dict(config=config, kind=kind, manifest=digest(manifest), sft=startpoint_hashes))
    if resume:
        checkpoint = Path(resume).resolve()
        if checkpoint.parent != root.resolve():
            raise ValueError('resume checkpoint must belong to this run')
        for name in ('optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'trainer_state.json'):
            if not (checkpoint / name).is_file():
                raise ValueError(f'incomplete resume checkpoint: {name}')
        if json.loads((root / 'identity.json').read_text())['fingerprint'] != fingerprint:
            raise ValueError('resume config/data/startpoint mismatch')
    else:
        create_run('runs', run_id, config)
        (root / 'identity.json').write_text(json.dumps(dict(fingerprint=fingerprint)))
    if resume and kind == 'grpo':
        saved_signal = Path(resume) / 'signal-state.json'
        if not saved_signal.is_file():
            raise ValueError('GRPO resume requires reward signal state checkpoint')
        (root / 'signal-state.json').write_text(saved_signal.read_text())
    from .monitoring import RunMonitor
    monitor = RunMonitor(root, config.get('budget'))
    from huggingface_hub import snapshot_download
    model_path = snapshot_download(config['model'], revision=config['revision'], local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left' if kind == 'grpo' else 'right'
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, torch_dtype=getattr(torch, config['model_dtype']))
    lora = LoraConfig(r=config['lora_r'], lora_alpha=config['lora_alpha'], lora_dropout=config['lora_dropout'],
                      target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'], task_type='CAUSAL_LM')
    if kind == 'grpo' or sft is not None:
        if not sft:
            raise ValueError('GRPO requires frozen SFT adapter')
        hashes = verify(sft)
        model = PeftModel.from_pretrained(model, sft).merge_and_unload()
        (root / 'sft-hashes.json').write_text(json.dumps(hashes, indent=2))
    model.config.use_cache = False
    (root / 'tokenizer.json').write_text(json.dumps(dict(eos_id=tokenizer.eos_token_id, pad_id=tokenizer.pad_token_id, padding_side=tokenizer.padding_side)))
    common = dict(output_dir=str(root), bf16=True, fp16=False, max_steps=config['max_steps'],
                  learning_rate=config['learning_rate'], per_device_train_batch_size=config['batch_size'],
                  gradient_accumulation_steps=config['gradient_accumulation_steps'], seed=config['seed'],
                  data_seed=config['seed'], report_to='none', optim='adamw_torch',
                  gradient_checkpointing=True, logging_nan_inf_filter=False, logging_steps=1, save_steps=config['save_steps'],
                  save_strategy='steps', save_only_model=False, save_total_limit=3)
    if kind == 'sft':
        examples, skipped = prepare_examples(config, tokenizer, config.get('sft_split', 'sft_train'))
        (root / 'training-ids.json').write_text(json.dumps([x['sample_id'] for x in examples]))
        (root / 'excluded.json').write_text(json.dumps(skipped, indent=2))
        def collator(rows):
            batch = pad_examples(rows, tokenizer.pad_token_id)
            return {k: torch.tensor(v) for k, v in batch.items()}
        eval_dataset = None
        eval_options = {}
        if config.get('eval_steps'):
            validation, excluded_validation = prepare_examples(config, tokenizer, 'validation')
            (root / 'validation-excluded.json').write_text(json.dumps(excluded_validation, indent=2))
            (root / 'validation-ids.json').write_text(json.dumps([x['sample_id'] for x in validation]))
            eval_dataset = Dataset.from_list([{k:v for k,v in x.items() if k != 'sample_id'} for x in validation])
            eval_options = dict(eval_strategy='steps', eval_steps=config['eval_steps'],
                per_device_eval_batch_size=1, prediction_loss_only=True,
                load_best_model_at_end=True, metric_for_best_model='eval_loss', greater_is_better=False)
        args = SFTConfig(**common, **eval_options, max_length=config['max_length'], packing=False,
                         dataset_kwargs={'skip_prepare_dataset': True}, remove_unused_columns=False)
        trainer = SFTTrainer(model=model, args=args, processing_class=tokenizer,
            train_dataset=Dataset.from_list([{k:v for k,v in x.items() if k != 'sample_id'} for x in examples]),
            data_collator=collator, eval_dataset=eval_dataset, peft_config=lora, callbacks=[monitor])
    else:
        train = json.loads((Path(config['manifest']).parent.parent / 'raw/train.json').read_text())
        rows = []
        for row in manifest['rows']:
            if row['split'] != 'grpo_train':
                continue
            record = train[int(row['sample_id'].split(':')[1])]
            prompt = render_prompt(record['question'], config['prompt'])
            if len(tokenizer.encode(prompt, add_special_tokens=False)) > config['max_prompt_length']:
                raise ValueError('overlong GRPO prompt; revise explicit length policy')
            rows.append(dict(prompt=prompt, sample_id=row['sample_id'], gold_number=record['answer'].split('####')[1].strip()))
        args = GRPOConfig(**common, num_generations=config['group_size'],
            max_completion_length=config['max_completion_length'], temperature=config['temperature'],
            top_p=config['top_p'], beta=config['beta'], use_vllm=False,
            steps_per_generation=config['gradient_accumulation_steps'], loss_type='grpo',
            scale_rewards='group', num_iterations=1, sync_ref_model=False)
        trainer = GRPOTrainer(model=model, args=args, processing_class=tokenizer,
            train_dataset=Dataset.from_list(rows), reward_funcs=make_reward_callback(config['reward'], root / 'reward-groups.jsonl', config['group_size']),
            peft_config=lora, callbacks=[monitor])
    (root / 'resolved-trainer.json').write_text(args.to_json_string())
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    (root / 'attempts.jsonl').open('a').write(json.dumps(dict(started=started, resume=resume)) + '\n')
    try:
        result = trainer.train(resume_from_checkpoint=resume)
        (root / 'selection.json').write_text(json.dumps(dict(best_checkpoint=trainer.state.best_model_checkpoint, best_metric=trainer.state.best_metric, selection='validation completion loss' if config.get('eval_steps') else 'fixed endpoint'), indent=2))
        trainer.save_model(str(root / 'final'))
        tokenizer.save_pretrained(root / 'final')
        from .checkpoints import seal
        (root / 'final/lineage.json').write_text(json.dumps(dict(kind=kind, model=config['model'], revision=config['revision'], sft_hashes=startpoint_hashes, model_dtype=config['model_dtype'])))
        seal(root / 'final')
        torch.cuda.synchronize()
        result = dict(result.metrics, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      elapsed_seconds=time.time()-started, status='stopped' if monitor.reason else 'complete', stop_reason=monitor.reason)
    except BaseException as exc:
        (root / 'failure.json').write_text(json.dumps(dict(error=repr(exc), time=time.time())))
        raise
    (root / 'status.json').write_text(json.dumps(dict(status=result['status'], step=trainer.state.global_step, max_steps=config['max_steps'], stop_reason=monitor.reason)))
    (root / 'result.json').write_text(json.dumps(result, indent=2))
    return result
