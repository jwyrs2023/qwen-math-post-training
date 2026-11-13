"""Freeze the screening decision and run a budgeted three-seed confirmation."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from evaluate import snapshot
from run_screen import REPORTS, RUNS, SCRIPT_DIR, SESSION, preflight, valid_run
from runtime import Budget, PROJECT, ROOT, protocol, sha


def write_new(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)


def entry(path):
    return dict(adapter=str(path.relative_to(ROOT)))


def build_screening_roster():
    roster = {"sft": {"adapter": None}}
    for method in ("dpo", "standard", "dynamic", "ppo"):
        name = f"{method}-formal-s43"
        if method == "ppo" and not (RUNS / name / "result.json").is_file():
            continue
        result = valid_run(name, 1)
        directory = RUNS / name
        roster[method] = entry(directory / "final")
        if method == "standard":
            for step in (160, 320):
                path = directory / f"checkpoint-{step}"
                if step < result["steps"] and path.is_dir():
                    roster[f"standard-step{step}"] = entry(path)
    write_new(REPORTS / "screening-roster.json", roster)
    print(f"Screening roster: {', '.join(roster)}")


def select():
    path = RUNS / "validation-screen/summary.json"
    identity = json.loads((path.parent / "identity.json").read_text())
    locked = snapshot(REPORTS / "screening-roster.json")
    if identity["split"] != "validation" or identity["roster"] != locked:
        raise ValueError("Screening results do not match the current sealed models")
    report = json.loads(path.read_text())
    validation_n = protocol()["validation"]["rows"]
    candidates = {name: report["paired"][name]["delta_pp"]
                  for name in ("standard", "dynamic", "dpo", "ppo") if name in locked["models"]}
    if any(m["count"] != validation_n for m in report["summaries"].values()):
        raise ValueError("Incomplete screening predictions")
    ranked = sorted(candidates, key=lambda name: (-candidates[name], name != "standard", name))
    winner = ranked[0] if candidates[ranked[0]] >= protocol()["screening"]["min_delta_pp"] - 1e-9 else None
    write_new(REPORTS / "selection.json", dict(method=winner, deltas_pp=candidates,
                                              validation_summary_sha256=sha(path)))
    print(f"Selected: {winner or 'no candidate reached the threshold'}")


def confirm():
    selected = json.loads((REPORTS / "selection.json").read_text())
    if selected["validation_summary_sha256"] != sha(RUNS / "validation-screen/summary.json"):
        raise ValueError("Validation results changed after selection")
    winner = selected["method"]
    if winner is None:
        raise ValueError("Screening did not qualify for confirmation")
    if winner not in ("standard", "dynamic", "dpo", "ppo"):
        raise ValueError("Unknown selected route")
    session = json.loads(SESSION.read_text())
    if session.get("phase") != "confirm" or session["budget_yuan"] != 90:
        raise ValueError("Use a confirmation session with the total cumulative budget")
    budget = Budget(SESSION)
    preflight()
    roster = {"sft": {"adapter": None}}
    methods = list(dict.fromkeys(("standard", winner)))
    env = os.environ | {"PYTHONPATH": str(ROOT / "src"), "HF_HUB_OFFLINE": "1",
                        "TOKENIZERS_PARALLELISM": "false"}
    for seed in (43, 42, 44):
        for method in methods:
            name = f"{method}-formal-s{seed}"
            if seed != 43 and not (RUNS / name / "result.json").exists():
                budget.check()
                if method == "dpo":
                    script = "train_dpo.py"
                    args = ["--output", str(RUNS / name), "--minutes", "34.9", "--max-steps", "640"]
                elif method == "ppo":
                    script = "train_ppo.py"
                    args = ["--run-id", name, "--minutes", "135", "--updates", "640"]
                else:
                    script = "train_grpo.py"
                    args = [method, "--run-id", name, "--minutes", "135", "--steps", "640"]
                with (REPORTS / "logs" / f"{name}.log").open("x") as log:
                    subprocess.run([sys.executable, str(SCRIPT_DIR / script), *args, "--seed", str(seed)],
                                   cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   timeout=budget.deadline-time.time(), check=True)
            result = valid_run(name, 1)
            if result["seed"] != seed:
                raise ValueError("Confirmation checkpoint has the wrong training seed")
            roster[f"{method}_s{seed}"] = entry(RUNS / name / "final")
    path = REPORTS / "confirmation-roster.json"
    write_new(path, roster)
    write_new(REPORTS / "confirmation-lock.json", snapshot(path))
    print("Confirmation roster locked. Run held-out evaluation once on this roster.")


def report():
    import numpy as np
    from math_posttrain.metrics import evaluate_records

    out = RUNS / "heldout"
    p = protocol()
    roster = snapshot(REPORTS / "confirmation-roster.json")
    identity = json.loads((out / "identity.json").read_text())
    if identity["split"] != "holdout" or identity["roster"] != roster:
        raise ValueError("Unmatched holdout evaluation")
    if roster != json.loads((REPORTS / "confirmation-lock.json").read_text()):
        raise ValueError("Holdout roster lock changed")
    selected = json.loads((REPORTS / "selection.json").read_text())["method"]
    data = json.loads((PROJECT / "data/svamp-heldout.json").read_text())
    expected = [(r["sample_id"], r["answer"]) for r in data]
    predictions = {}
    for name in roster["models"]:
        rows = [json.loads(line) for line in (out / f"{name}.jsonl").read_text().splitlines()]
        if [(r["sample_id"], r["answer"]) for r in rows] != expected:
            raise ValueError("Incomplete or mismatched held-out predictions")
        predictions[name] = np.array([int(r["correct"]) for r in evaluate_records(rows)[0]])
    effects = {}
    rng = np.random.default_rng(42)
    for method in dict.fromkeys(("standard", selected)):
        changes = np.stack([predictions[f"{method}_s{s}"]-predictions["sft"] for s in (42,43,44)])
        per_question_sum = changes.sum(axis=0)
        values, counts = np.unique(per_question_sum, return_counts=True)
        n = len(per_question_sum)
        draws = rng.multinomial(n, counts / n, p["confirmation"]["question_bootstrap_replicates"])
        ci = np.quantile(100 * (draws @ values) / (n * 3), [.025, .975]).tolist()
        by_seed = 100 * changes.mean(axis=1)
        effects[method] = dict(delta_pp_by_seed=dict(zip((42,43,44), by_seed.tolist())),
                               mean_delta_pp=float(by_seed.mean()), question_ci95_pp=ci,
                               practical_target_met=bool(by_seed.mean() >= p["confirmation"]["target_delta_pp"]),
                               positive_all_three=bool((by_seed > 0).all()))
    write_new(REPORTS / "confirmation-results.json", dict(
        heldout_count=len(data), selected=selected, effects=effects,
        interval_scope="Question bootstrap conditional on the three fixed training seeds."
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["roster", "select", "confirm", "report"])
    args = parser.parse_args()
    {"roster": build_screening_roster, "select": select, "confirm": confirm, "report": report}[args.stage]()
