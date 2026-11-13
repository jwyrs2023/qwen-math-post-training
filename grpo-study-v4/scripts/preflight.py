"""Reproduce all GPU-independent V4 acceptance checks."""
import json
import os
import subprocess
import sys
from pathlib import Path

from runtime import PROJECT, ROOT, protocol, sha

HERE = Path(__file__).resolve().parent


def run():
    p = protocol()
    for split, name in (("validation", "validation.json"), ("holdout", "svamp-heldout.json")):
        if sha(PROJECT / "data" / name) != p[split]["sha256"]:
            raise ValueError(f"Frozen {split} content mismatch")
    if sha(PROJECT / "data/svamp-source.json") != p["holdout"]["raw_sha256"]:
        raise ValueError("Raw SVAMP artifact mismatch")
    jobs = [
        ("prepare_dpo.py", []),
        ("freeze_eval.py", [str(PROJECT / "data/svamp-source.json")]),
        ("train_dpo.py", ["--dry-run"]),
        ("check_ppo_reward.py", []),
        ("diagnose_sft.py", ["--dry-run"]),
        ("train_grpo.py", ["standard", "--dry-run"]),
        ("train_grpo.py", ["dynamic", "--dry-run"]),
        ("train_ppo.py", ["--dry-run"]),
        ("evaluate.py", ["--roster", str(PROJECT / "data/historical-roster.json"), "--dry-run"]),
        *[("check_trainers_cpu.py", [m]) for m in ("dpo", "dynamic", "ppo")],
        *[("run_screen.py", [s, "--dry-run"]) for s in ("diagnosis", "pilot", "train")],
    ]
    env = os.environ | {"PYTHONPATH": str(ROOT / "src"), "HF_HUB_OFFLINE": "1",
                        "ACCELERATE_USE_CPU": "true", "TOKENIZERS_PARALLELISM": "false"}
    report = {}
    for script, args in jobs:
        print(script, " ".join(args), flush=True)
        result = subprocess.run([sys.executable, str(HERE / script), *args],
                                cwd=ROOT, env=env, check=True, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        report[" ".join([script, *args])] = result.stdout
    (PROJECT / "reports/local-preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Passed {len(jobs)} local preflight commands. No CUDA execution.")


if __name__ == "__main__":
    run()
