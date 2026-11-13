"""One standard GRPO trajectory per seed, with full checkpoints at optimizer boundaries."""
import json
import math
from pathlib import Path
import random
import tempfile
import time

from transformers import TrainerCallback

from common import PROJECT, RUNS, Budget, checkpoint, completed_checkpoints, immutable, load_policy
from common import policy_lora, read, seal, training_rows, verify, write


def make_trainer(model, tokenizer, planned, config, reward, callback):
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    return GRPOTrainer(model=model, args=GRPOConfig(**config), processing_class=tokenizer,
                       train_dataset=Dataset.from_list(planned), peft_config=policy_lora(),
                       callbacks=[callback], reward_funcs=reward)


class CheckpointMonitor(TrainerCallback):
    def __init__(self, target, milestones, deadline, lineage, log_path, previous, prior_seconds):
        self.target = target
        self.milestones = milestones
        self.deadline = deadline
        self.lineage = lineage
        self.log_path = log_path
        self.previous = previous
        self.prior_seconds = prior_seconds
        self.started = time.monotonic()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if any(not math.isfinite(v) for v in (logs or {}).values() if isinstance(v, (int, float))):
            raise ValueError("Non-finite training metric")

    def on_step_end(self, args, state, control, **kwargs):
        stop = state.global_step >= self.target or time.time() + 180 >= self.deadline
        if stop or state.global_step in self.milestones:
            control.should_save = True
        if stop:
            control.should_training_stop = True
        return control

    def on_save(self, args, state, control, **kwargs):
        path = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        required = ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json",
                    "adapter_model.safetensors", "adapter_config.json")
        if not all((path / name).is_file() for name in required):
            raise ValueError("Incomplete optimizer-boundary checkpoint")
        write(path / "lineage.json", self.lineage | {"step": state.global_step})
        write(path / "v5-state.json", dict(
            updates=state.global_step, next_prompt_index=4 * state.global_step,
            training_seconds=self.prior_seconds + time.monotonic() - self.started,
            segments=self.previous + [{"path": str(self.log_path.relative_to(Path(args.output_dir))),
                                       "bytes": self.log_path.stat().st_size}],
        ))
        seal(path)
        return control


def resume_state(directory, state):
    step, path = state
    verify(path)
    metadata = read(path / "v5-state.json")
    if metadata["updates"] != step or metadata["next_prompt_index"] != step * 4:
        raise ValueError("Checkpoint data cursor mismatch")
    for segment in metadata["segments"]:
        artifact = directory / segment["path"]
        if artifact.stat().st_size < segment["bytes"]:
            raise ValueError("Missing checkpoint reward history")
    return path, metadata


def run(p, v4, seed, target):
    import torch
    from transformers import set_seed
    from math_posttrain.train_grpo import make_reward_callback
    from common import PROTOCOL, sha

    directory = RUNS / f"seed-{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    history = completed_checkpoints(directory)
    if checkpoint(seed, target).is_dir() and any(step == target for step, _ in history):
        resume_state(directory, (target, checkpoint(seed, target)))
        print(f"Seed {seed}, checkpoint {target} already complete.")
        return
    if history and history[-1][0] >= target:
        raise ValueError("Target predates latest complete checkpoint and is missing")
    if any(path.is_dir() and not (path / "checkpoint-manifest.json").is_file()
           for path in directory.glob("checkpoint-*")):
        raise ValueError("Unsealed checkpoint exists. Preserve it separately before retrying.")
    budget = Budget()
    pool = training_rows()
    if len(pool) != p["training_rows"]:
        raise ValueError("Unexpected training pool size")
    set_seed(seed)
    model, tokenizer = load_policy(v4)
    model.config.use_cache = False
    if any(len(tokenizer.encode(row["prompt"], add_special_tokens=False)) > 256 for row in pool):
        raise ValueError("Training prompt exceeds V4 policy")
    planned = random.Random(seed).choices(pool, k=4 * p["max_steps"])
    immutable(directory / "planned-ids.json", [row["sample_id"] for row in planned])
    config = p["training"] | dict(output_dir=str(directory), max_steps=p["max_steps"],
                                  seed=seed, data_seed=seed)
    lineage = {key: v4[key] for key in ("model", "revision", "model_dtype", "sft_hashes")}
    lineage |= dict(kind="grpo", branch="standard", seed=seed, protocol_sha256=sha(PROTOCOL))
    immutable(directory / "run-identity.json", dict(config=config, lineage=lineage))
    previous_path, previous = resume_state(directory, history[-1]) if history else (None, {})
    if previous_path and read(previous_path / "lineage.json") != lineage | {"step": history[-1][0]}:
        raise ValueError("Resume lineage mismatch")
    attempt = tempfile.mkdtemp(prefix="attempt-", dir=directory)
    reward_log = Path(attempt) / "reward-groups.jsonl"
    monitor = CheckpointMonitor(target, set(p["splits"]["gsm8k"]["steps"]), budget.deadline,
                                lineage, reward_log, previous.get("segments", []),
                                previous.get("training_seconds", 0))
    trainer = make_trainer(model, tokenizer, planned, config,
                           make_reward_callback("accuracy", reward_log, group_size=4), monitor)
    budget.check(reserve_seconds=180)
    torch.cuda.reset_peak_memory_stats()
    trainer.train(resume_from_checkpoint=str(previous_path) if previous_path else None)
    step = trainer.state.global_step
    print(json.dumps(dict(seed=seed, step=step, target=target,
                          peak_reserved_bytes=torch.cuda.max_memory_reserved())))
    if step != target:
        raise RuntimeError(f"Budget stop at {step}. Latest sealed checkpoint is retained.")
