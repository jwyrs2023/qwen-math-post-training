"""Train a fresh preference adapter over the verified, merged SFT policy."""
import argparse
import json
import time
from pathlib import Path

from prepare_dpo import PROJECT, ROOT, sha
from runtime import Budget, changed_trainables, trainable_snapshot
from math_posttrain.checkpoints import seal, verify

MODEL = "Qwen/Qwen2.5-1.5B"


def train(args):
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
    from trl import DPOConfig, DPOTrainer

    data = PROJECT / "data/dpo-pairs.jsonl"
    audit = json.loads((PROJECT / "reports/dpo-data-audit.json").read_text())
    identity = audit["scoring_identity"]
    sft = ROOT / "runs/sft_frozen"
    hashes = verify(sft)
    if hashes != identity["sft_hashes"] or sha(data) != audit["pairs_sha256"]:
        raise ValueError("DPO data or SFT hash mismatch")
    rows = [json.loads(line) for line in data.read_text().splitlines()]
    tokenizer = AutoTokenizer.from_pretrained(sft, local_files_only=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    for row in rows:
        enc = DPOTrainer.tokenize_row(row, tokenizer, add_special_tokens=False)
        if len(enc["prompt_input_ids"]) > 256 or any(
            len(enc[k]) > 512 for k in ("chosen_input_ids", "rejected_input_ids")
        ):
            raise ValueError(f"Training pair would be truncated: {row['sample_id']}")
    if args.dry_run:
        print(f"Verified SFT and {len(rows)} preference pairs. No truncation.")
        return
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("DPO training requires CUDA with BF16 support")
    if not 0 < args.minutes <= 35 or args.max_steps < 1:
        raise ValueError("DPO time cap must be within 35 minutes")
    budget = Budget(args.session, args.minutes)
    limit = budget.deadline
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    set_seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=identity["revision"], local_files_only=True, dtype=torch.float32
    )
    model = PeftModel.from_pretrained(model, sft).merge_and_unload()
    model.config.use_cache = False
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                      task_type="CAUSAL_LM")
    config = DPOConfig(
        output_dir=str(out), max_steps=args.max_steps, seed=args.seed,
        bf16=True, learning_rate=5e-6, lr_scheduler_type="constant", beta=0.1,
        max_length=768, max_prompt_length=256, max_completion_length=512,
        per_device_train_batch_size=1, gradient_accumulation_steps=4,
        gradient_checkpointing=True, save_strategy="no", logging_steps=1,
        report_to="none", remove_unused_columns=False,
    )

    class Monitor(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if time.time() >= limit - 45:
                control.should_training_stop = True
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):
            with (out / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps({"step": state.global_step, **(logs or {})}) + "\n")

    trainer = DPOTrainer(model=model, args=config, processing_class=tokenizer,
                         train_dataset=Dataset.from_list(rows), peft_config=lora,
                         callbacks=[Monitor()])
    # TRL 0.26 disables the fresh LoRA adapter for the reference forward pass.
    if trainer.ref_model is not None or not trainer.is_peft_model:
        raise RuntimeError("Expected adapter-disabled reference over merged frozen SFT")
    if time.time() >= limit - 45:
        raise RuntimeError("Insufficient budget after model initialization")
    torch.cuda.reset_peak_memory_stats()
    before = trainable_snapshot(trainer.model) if args.max_steps == 4 else None
    result = trainer.train()
    if before is not None and not changed_trainables(trainer.model, before):
        raise RuntimeError("DPO pilot did not update the policy")
    final = out / "final"
    trainer.save_model(str(final))
    tokenizer.save_pretrained(final)
    (final / "lineage.json").write_text(json.dumps(dict(
        kind="dpo", model=MODEL, revision=identity["revision"], model_dtype="float32",
        sft_hashes=hashes, pairs_sha256=audit["pairs_sha256"],
    ), indent=2))
    seal(final)
    (out / "result.json").write_text(json.dumps(dict(
        steps=trainer.state.global_step, seed=args.seed, metrics=result.metrics,
        pilot_policy_updated=True if before is not None else None,
        elapsed_seconds=time.time()-budget.started, peak_reserved_bytes=torch.cuda.max_memory_reserved(),
    ), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path, default=PROJECT / "runs/dpo-pilot-s43")
    parser.add_argument("--session", type=Path, default=PROJECT / "reports/session.json")
    parser.add_argument("--minutes", type=float, default=10)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=43)
    train(parser.parse_args())
