"""Paired effects, simultaneous intervals and phase decisions from complete predictions."""
import json

import numpy as np

from common import PROTOCOL, RUNS, atomic_json, checkpoint, evaluation_path, read, sha
from curve_eval import checked_predictions, expected_identity, names
from math_posttrain.metrics import evaluate_records


def contrasts(p, split):
    steps = p["splits"][split]["steps"]
    return [(0, step) for step in steps] + [(a, b) for a, b in zip(steps, steps[1:])
                                         if a >= 640 and b - a == 320]


def effect(values, repetitions, family_size, rng):
    n_seeds, n_items = values.shape
    totals, counts = np.unique(values.sum(axis=0), return_counts=True)
    draws = rng.multinomial(n_items, counts / n_items, repetitions)
    samples = 100 * (draws @ totals) / (n_items * n_seeds)
    tail = .025 / family_size
    return dict(delta_pp=100 * float(values.mean()),
                by_seed_pp=(100 * values.mean(axis=1)).tolist(),
                ci95_pp=np.quantile(samples, [.025, .975]).tolist(),
                simultaneous_ci95_pp=np.quantile(samples, [tail, 1-tail]).tolist())


def decide(p, target, splits):
    rule = p["decision"]
    output = {}
    plateau_pairs = {}
    for split, data in splits.items():
        onset = None
        previous = 0
        for point in data["points"][1:]:
            delta = point["vs_sft"]
            if delta["delta_pp"] >= rule["onset_pp"] and delta["simultaneous_ci95_pp"][0] > 0:
                onset = [previous, point["step"]]
                break
            previous = point["step"]
        pairs = []
        descriptive = []
        changes = data["marginal"]
        for first, second in zip(changes, changes[1:]):
            if first["end"] != second["start"]:
                continue
            window = [first["start"], first["end"], second["end"]]
            if all(abs(v["delta_pp"]) <= rule["plateau_pp"] for v in (first, second)):
                descriptive.append(window)
            bounds = [v["simultaneous_ci95_pp"] for v in (first, second)]
            if all(-rule["plateau_pp"] <= low <= high <= rule["plateau_pp"] for low, high in bounds):
                pairs.append(window)
        plateau_pairs[split] = pairs
        output[split] = dict(onset_interval=onset, plateau_windows=pairs,
                             descriptive_plateau_windows=descriptive)
    joint = set(map(tuple, plateau_pairs["gsm8k"])) & set(map(tuple, plateau_pairs["svamp"]))
    ending = sorted(list(window) for window in joint if window[-1] == target)
    action = "plateau" if ending else ("cap" if target == p["max_steps"] else "continue")
    return dict(action=action, shared_windows_ending_at_target=ending, splits=output)


def training_summary(seed, step):
    folder = checkpoint(seed, step)
    state = read(folder / "v5-state.json")
    tokens = 0
    zero_groups = 0
    groups = 0
    lengths = 0
    truncated = 0
    ids = []
    for segment in state["segments"]:
        with (folder.parent / segment["path"]).open("rb") as stream:
            data = stream.read(segment["bytes"])
        for line in data.splitlines():
            record = json.loads(line)
            tokens += record["generated_tokens"]
            groups += len(record["groups"])
            zero_groups += sum(group["zero_variance"] for group in record["groups"])
            lengths += len(record["generated_token_lengths"])
            truncated += sum(length >= 512 for length in record["generated_token_lengths"])
            ids.extend(row["sample_id"] for row in record["rows"][::4])
    if groups != 4 * step or lengths != 16 * step or ids != read(folder.parent / "planned-ids.json")[:4*step]:
        raise ValueError(f"Training reward history does not match its exposure plan: {folder}")
    kl = [log["kl"] for log in read(folder / "trainer_state.json")["log_history"] if "kl" in log]
    return dict(updates=step, generated_tokens=tokens, unique_prompts=len(set(ids)),
                cumulative_training_seconds=state["training_seconds"],
                zero_variance_groups=zero_groups, groups=groups,
                mean_completion_tokens=tokens / lengths, at_token_cap=truncated,
                mean_logged_kl=float(np.mean(kl)) if kl else None)


def run(p, v4, target):
    from plots import write_plots

    family_size = sum(len(contrasts(p, split)) for split in p["splits"])
    training = {seed: {step: training_summary(seed, step)
                       for step in p["splits"]["gsm8k"]["steps"] if step <= target}
                for seed in p["seeds"]}
    output = {}
    for split in p["splits"]:
        rows = read(evaluation_path(p, split))
        values, metadata = {}, {}
        for name, seed, step in names(p, split, target):
            dest = RUNS / "curves" / split / name
            if read(dest / "identity.json") != expected_identity(p, v4, split, seed, step):
                raise ValueError(f"Evaluation identity mismatch: {dest}")
            records = checked_predictions(dest / "predictions.jsonl", rows)
            if len(records) != len(rows):
                raise ValueError(f"Incomplete predictions: {dest}")
            results, metrics = evaluate_records(records)
            metadata[(seed, step)] = metrics
            values[(seed, step)] = np.array([row["correct"] for row in results], dtype=np.int8)
        available = [0] + [step for step in p["splits"][split]["steps"] if step <= target]
        anchor = values[(None, 0)]
        matrix = {0: np.tile(anchor, (len(p["seeds"]), 1))}
        for step in available[1:]:
            matrix[step] = np.stack([values[(seed, step)] for seed in p["seeds"]])
        def compare(start, end):
            rng = np.random.default_rng([p["bootstrap_seed"], start, end, len(rows)])
            return effect(matrix[end] - matrix[start], p["bootstrap_draws"], family_size, rng)
        points = [dict(step=0, mean_accuracy=float(anchor.mean()),
                       seed_metrics=[metadata[(None, 0)] for seed in p["seeds"]])]
        points += [dict(step=step, mean_accuracy=float(matrix[step].mean()), vs_sft=compare(0, step),
                        seed_metrics=[metadata[(seed, step)] for seed in p["seeds"]])
                   for step in available[1:]]
        marginal = [dict(start=start, end=end, **compare(start, end))
                    for start, end in contrasts(p, split) if start and end in matrix]
        output[split] = dict(questions=len(rows), points=points, marginal=marginal)
    report = dict(protocol_sha256=sha(PROTOCOL), target=target, seeds=p["seeds"], family_size=family_size,
                  interval_scope="Paired questions, conditional on the three fixed seeds.",
                  splits=output, decision=decide(p, target, output), training=training)
    dest = RUNS / "analysis" / str(target)
    dest.mkdir(parents=True, exist_ok=True)
    atomic_json(dest / "results.json", report)
    write_plots(dest, report)
    print(json.dumps(report["decision"], ensure_ascii=False))
