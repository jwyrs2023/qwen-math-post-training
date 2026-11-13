"""PPO with a LoRA critic and the same audited numeric verifier as GRPO."""
import argparse
import copy
import json
import math
import time

from runtime import (Budget, PROJECT, TimeLimit, append, load_policy, policy_lora,
                     protocol, seal_policy, training_rows, trainable_snapshot, changed_trainables)


def make_critic(backbone):
    import torch
    from peft import LoraConfig, get_peft_model

    class Critic(torch.nn.Module):
        base_model_prefix = "backbone"

        def __init__(self, backbone):
            super().__init__()
            self.backbone = get_peft_model(backbone, LoraConfig(
                r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                task_type="FEATURE_EXTRACTION",
            ))
            self.score = torch.nn.Linear(backbone.config.hidden_size, 1, bias=False,
                                         dtype=next(backbone.parameters()).dtype)
            torch.nn.init.normal_(self.score.weight, std=0.002)

    return Critic(backbone)


def connect_checkpointing(trainer):
    """Delegate the generation helper's checkpoint toggles to the PPO policy."""
    wrapped = trainer.accelerator.unwrap_model(trainer.model)
    if wrapped.is_gradient_checkpointing:
        wrapped.gradient_checkpointing_disable = wrapped.policy.gradient_checkpointing_disable

        def enable():
            wrapped.policy.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        wrapped.gradient_checkpointing_enable = enable
    return wrapped


def run(args):
    import torch
    from datasets import Dataset
    from transformers import TrainerCallback, set_seed
    from trl.experimental.ppo import PPOConfig, PPOTrainer
    from math_posttrain.parsing import parse_number
    from ppo_reward import AnswerReward

    p = protocol()
    pool = training_rows()
    if args.dry_run:
        print(f"Verified PPO training whitelist of {len(pool)} prompts.")
        return
    budget = Budget(PROJECT / "reports/session.json", args.minutes)
    out = PROJECT / "runs" / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    set_seed(args.seed)
    policy, tokenizer = load_policy(p, pad="<|fim_pad|>")
    if tokenizer.pad_token_id not in tokenizer.all_special_ids:
        raise ValueError("PPO requires a known special pad token")
    mapping, rows = {}, []
    for row in pool:
        ids = tuple(tokenizer.encode(row["prompt"], add_special_tokens=False))
        if ids in mapping or len(ids) > 256:
            raise ValueError("Duplicate or overlength PPO prompt")
        mapping[ids] = parse_number(row["gold_number"])
        rows.append(dict(input_ids=list(ids)))
    reward = AnswerReward(tokenizer, mapping)
    critic = make_critic(copy.deepcopy(policy.model).to(dtype=torch.bfloat16))
    critic.backbone.config.use_cache = False
    policy.generation_config.pad_token_id = tokenizer.pad_token_id
    critic.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    class Monitor(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            record = logs or {}
            if any(not math.isfinite(v) for k, v in record.items()
                   if isinstance(v, (float, int)) and k != "val/ratio_var"):
                raise RuntimeError("Non-finite PPO metric")
            append(out / "metrics.jsonl", dict(step=state.global_step, **{
                k: v for k, v in record.items()
                if not isinstance(v, (float, int)) or math.isfinite(v)
            }))

        def on_step_end(self, args, state, control, **kwargs):
            # TRL 0.26's loop does not honor should_training_stop.
            budget.check(headroom=120)

    config = PPOConfig(
        output_dir=str(out), seed=args.seed, bf16=True,
        per_device_train_batch_size=2, gradient_accumulation_steps=4,
        total_episodes=args.updates * 8, local_rollout_forward_batch_size=2,
        response_length=512, temperature=0.8, stop_token="eos",
        kl_coef=0.02, kl_estimator="k3", missing_eos_penalty=None,
        num_ppo_epochs=1, num_mini_batches=1, num_sample_generations=0,
        learning_rate=5e-6, lr_scheduler_type="constant",
        logging_steps=1, save_strategy="no", report_to="none",
    )
    dataset = Dataset.from_list(rows)
    trainer = PPOTrainer(
        args=config, model=policy, ref_model=None, value_model=critic, reward_model=reward,
        processing_class=tokenizer, train_dataset=dataset, eval_dataset=dataset.select(range(0)),
        peft_config=policy_lora(), callbacks=[Monitor()],
    )
    (out / "resolved-trainer.json").write_text(config.to_json_string())
    budget.check()
    torch.cuda.reset_peak_memory_stats()
    unwrapped = connect_checkpointing(trainer)
    before = trainable_snapshot(unwrapped) if args.updates == 4 else None
    status = "complete"
    try:
        trainer.train()
    except TimeLimit:
        status = "time_limited"
    if trainer.state.global_step < 1:
        raise RuntimeError("No complete PPO update")
    if before is not None:
        changed = changed_trainables(unwrapped, before)
        if not any(n.startswith("policy.") for n in changed) or not any(
            n.startswith("value_model.") for n in changed
        ):
            raise RuntimeError("PPO pilot did not update both policy and critic")
    final = out / "final"
    trainer.save_model(str(final))
    tokenizer.save_pretrained(final)
    seal_policy(final, p, "ppo", seed=args.seed)
    value = trainer.accelerator.unwrap_model(trainer.model).value_model
    value.backbone.save_pretrained(out / "critic")
    from safetensors.torch import save_file
    save_file({k: v.detach().cpu() for k, v in value.score.state_dict().items()},
              out / "critic/value_head.safetensors")
    (out / "result.json").write_text(json.dumps(dict(
        status=status, steps=trainer.state.global_step, seed=args.seed,
        pilot_policy_and_critic_updated=True if before is not None else None,
        elapsed_seconds=time.time()-budget.started, peak_reserved_bytes=torch.cuda.max_memory_reserved(),
    ), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="ppo-pilot")
    parser.add_argument("--minutes", type=float, default=12)
    parser.add_argument("--updates", type=int, default=4)
    parser.add_argument("--seed", type=int, default=43, choices=[42, 43, 44])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.run_id or args.run_id in (".", "..") or "/" in args.run_id or not 1 <= args.updates <= 640:
        parser.error("Invalid run ID or update limit")
    run(args)
