"""Run one budgeted, sequential experiment-four screening stage."""
import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from runtime import Budget, PROJECT, ROOT, append, protocol, verify

SCRIPT_DIR = Path(__file__).resolve().parent
RUNS = PROJECT / "runs"
REPORTS = PROJECT / "reports"
SESSION = REPORTS / "session.json"
REQUIRED = {"torch": "2.9.1", "transformers": "4.57.3", "trl": "0.26.0",
            "peft": "0.18.0", "datasets": "4.4.1", "accelerate": "1.12.0"}


def preflight():
    import torch
    from huggingface_hub import snapshot_download
    p = protocol()
    for name, version in REQUIRED.items():
        if importlib.metadata.version(name).split("+")[0] != version:
            raise ValueError(f"{name} differs from locked {version}")
    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise ValueError("Expose one BF16-capable CUDA GPU")
    device = torch.cuda.get_device_properties(0)
    if device.total_memory < 23 * 1024**3:
        raise ValueError("Pilot requires at least 23 GiB of device memory")
    path = snapshot_download(p["model"], revision=p["revision"], local_files_only=True)
    if not list(Path(path).glob("*.safetensors")):
        raise ValueError("Pinned Base weights are missing from local cache")
    print(f"GPU: {device.name}, {device.total_memory / 1024**3:.1f} GiB; pinned weights cached.", flush=True)


def commands(stage):
    if stage == "diagnosis":
        return [("sft-pass8", "diagnose_sft.py", [], None)]
    if stage == "pilot":
        return [
            ("dpo-pilot-s43", "train_dpo.py", [], 4),
            ("standard-pilot-s43", "train_grpo.py", ["standard", "--run-id", "standard-pilot-s43"], 4),
            ("dynamic-pilot-s43", "train_grpo.py", ["dynamic", "--run-id", "dynamic-pilot-s43"], 4),
            ("ppo-pilot-s43", "train_ppo.py", ["--run-id", "ppo-pilot-s43"], 4),
        ]
    if stage == "train":
        return [
            ("dpo-formal-s43", "train_dpo.py", ["--output", str(RUNS / "dpo-formal-s43"),
                                               "--max-steps", "640", "--minutes", "34.9"], 1),
            ("standard-formal-s43", "train_grpo.py", ["standard", "--run-id", "standard-formal-s43",
                                                    "--steps", "640", "--minutes", "135"], 1),
            ("dynamic-formal-s43", "train_grpo.py", ["dynamic", "--run-id", "dynamic-formal-s43",
                                                   "--steps", "640", "--minutes", "135"], 1),
            ("ppo-formal-s43", "train_ppo.py", ["--run-id", "ppo-formal-s43",
                                              "--updates", "640", "--minutes", "135"], 1),
        ]
    raise ValueError("Unknown stage")


def valid_run(name, steps, exact=False):
    result_path = RUNS / name / "result.json"
    result = json.loads(result_path.read_text())
    if (exact and result["steps"] != steps) or result["steps"] < steps:
        raise ValueError(f"{name} did not reach the required update count")
    verify(RUNS / name / "final")
    return result


def execute(stage, dry_run):
    jobs = commands(stage)
    if dry_run:
        for name, script, args, _ in jobs:
            print(name, script, " ".join(args))
        return
    session = json.loads(SESSION.read_text())
    if session["budget_yuan"] > 40:
        raise ValueError("Screening is capped at 40 yuan")
    budget = Budget(SESSION)
    preflight()
    log_dir = REPORTS / "logs"
    log_dir.mkdir(exist_ok=True)
    with (log_dir / f"{stage}.started").open("x") as stream:
        stream.write(str(time.time()))
    env = os.environ | {"PYTHONPATH": str(ROOT / "src"), "HF_HUB_OFFLINE": "1",
                        "TOKENIZERS_PARALLELISM": "false"}
    for name, script, args, steps in jobs:
        budget.check()
        if stage == "train":
            pilot = name.replace("-formal-", "-pilot-")
            try:
                valid_run(pilot, 4, exact=True)
            except (FileNotFoundError, ValueError):
                if name.startswith("ppo-"):
                    append(REPORTS / "screen-stages.jsonl", dict(name=name, status="skipped_pilot"))
                    continue
                raise
        try:
            with (log_dir / f"{name}.log").open("x") as log:
                subprocess.run([sys.executable, str(SCRIPT_DIR / script), *args],
                               cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                               timeout=budget.deadline-time.time(), check=True)
            result = valid_run(name, steps, exact=stage == "pilot") if steps else json.loads(
                (RUNS / name / "summary.json").read_text())
            append(REPORTS / "screen-stages.jsonl", dict(name=name, status="verified", result=result))
        except (subprocess.SubprocessError, FileNotFoundError, ValueError) as exc:
            append(REPORTS / "screen-stages.jsonl", dict(name=name, status="failed", error=str(exc)))
            if not name.startswith("ppo-"):
                raise
    print(f"Stage {stage} ended. Review status and budget before the next stage.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["diagnosis", "pilot", "train"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    execute(args.stage, args.dry_run)
