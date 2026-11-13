"""Run the matched standard and dynamic GRPO branches."""
import argparse
import json
import math
import random
import time

from runtime import Budget, PROJECT, load_policy, policy_lora, protocol, seal_policy, training_rows, trainable_snapshot, changed_trainables


def train(args):
    import torch
    from datasets import Dataset
    from transformers import TrainerCallback, set_seed
    from trl import GRPOConfig, GRPOTrainer

    from dynamic_grpo import DynamicGRPOTrainer
    from math_posttrain.train_grpo import make_reward_callback

    p = protocol()
    pool = training_rows()
    if args.dry_run:
        print(f"Verified {args.branch} GRPO with {len(pool)} training prompts.")
        return
    budget = Budget(PROJECT / "reports/session.json", args.minutes)
    out = PROJECT / "runs" / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    set_seed(args.seed)
    model, tokenizer = load_policy(p)
    model.config.use_cache = False
    if any(len(tokenizer.encode(r["prompt"], add_special_tokens=False)) > 256 for r in pool):
        raise ValueError("Training prompt exceeds frozen length policy")
    planned = random.Random(args.seed).choices(pool, k=4 * args.steps)
    (out / "planned-ids.json").write_text(json.dumps([r["sample_id"] for r in planned]))

    class Monitor(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if any(not math.isfinite(x) for x in (logs or {}).values() if isinstance(x, (int, float))):
                raise RuntimeError("Non-finite GRPO metric")

        def on_step_end(self, args, state, control, **kwargs):
            if time.time() + 90 >= budget.deadline:
                control.should_training_stop = True
            return control

    config = GRPOConfig(
        output_dir=str(out), max_steps=args.steps, seed=args.seed, data_seed=args.seed,
        bf16=True, learning_rate=5e-6, lr_scheduler_type="constant",
        per_device_train_batch_size=4, gradient_accumulation_steps=4,
        steps_per_generation=4, num_generations=4, shuffle_dataset=False,
        max_completion_length=512, temperature=0.8, top_p=1.0, top_k=0,
        use_vllm=False, beta=0.02, num_iterations=1, loss_type="grpo",
        scale_rewards="group", gradient_checkpointing=True,
        logging_steps=1, logging_nan_inf_filter=False, report_to="none",
        save_steps=160, save_only_model=True, remove_unused_columns=False,
    )
    trainer_type = GRPOTrainer
    extra = {}
    if args.branch == "dynamic":
        trainer_type = DynamicGRPOTrainer
        extra = dict(refill_rows=pool, refill_rng=random.Random(f"refill:{args.seed}"),
                     budget=budget, telemetry=out / "refill.jsonl")
    trainer = trainer_type(
        model=model, args=config, processing_class=tokenizer, train_dataset=Dataset.from_list(planned),
        peft_config=policy_lora(), callbacks=[Monitor()],
        reward_funcs=make_reward_callback("accuracy", out / "reward-groups.jsonl", group_size=4), **extra,
    )
    (out / "resolved-trainer.json").write_text(config.to_json_string())
    budget.check()
    torch.cuda.reset_peak_memory_stats()
    from runtime import TimeLimit
    before = trainable_snapshot(trainer.model) if args.steps == 4 else None
    status = "complete"
    try:
        trainer.train()
        if trainer.state.global_step < args.steps:
            status = "time_limited"
    except TimeLimit:
        status = "time_limited"
    if trainer.state.global_step < 1:
        raise RuntimeError("No complete GRPO update")
    if before is not None and not changed_trainables(trainer.model, before):
        raise RuntimeError("GRPO pilot did not update the policy")
    final = out / "final"
    trainer.save_model(str(final))
    tokenizer.save_pretrained(final)
    seal_policy(final, p, "grpo", branch=args.branch, seed=args.seed)
    for checkpoint in out.glob("checkpoint-*"):
        if (checkpoint / "adapter_model.safetensors").is_file():
            seal_policy(checkpoint, p, "grpo", branch=args.branch, seed=args.seed)
    (out / "result.json").write_text(json.dumps(dict(
        status=status, steps=trainer.state.global_step, seed=args.seed,
        pilot_policy_updated=True if before is not None else None,
        elapsed_seconds=time.time()-budget.started, peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        metrics=trainer.state.log_history,
    ), indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("branch", choices=["standard", "dynamic"])
    parser.add_argument("--run-id", default="grpo-pilot")
    parser.add_argument("--minutes", type=float, default=10)
    parser.add_argument("--steps", type=int, default=4, choices=[4, 160, 320, 640])
    parser.add_argument("--seed", type=int, default=43, choices=[42, 43, 44])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.run_id or args.run_id in (".", "..") or "/" in args.run_id:
        parser.error("Invalid run ID")
    train(args)
