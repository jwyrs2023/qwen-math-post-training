"""Evaluate a locked V4 model roster on validation or held-out SVAMP."""
import argparse
import json
import time
from pathlib import Path

from runtime import Budget, PROJECT, PROMPT, ROOT, append, load_policy, protocol, sha, verify
from math_posttrain.metrics import evaluate_records
from math_posttrain.prompting import render_prompt
from math_posttrain.evaluate import validate_adapter_lineage


def snapshot(roster):
    p = protocol()
    specs = json.loads(roster.read_text())
    if not isinstance(specs, dict) or not specs or specs.get("sft") != {"adapter": None}:
        raise ValueError("Roster requires a frozen SFT anchor")
    models = {}
    for name, spec in specs.items():
        if not name or Path(name).name != name or name in (".", "..") or set(spec) != {"adapter"}:
            raise ValueError(f"Invalid roster entry: {name}")
        if spec["adapter"] is None:
            if name != "sft":
                raise ValueError("Only SFT anchor can omit an adapter")
            models[name] = None
            continue
        path = (ROOT / spec["adapter"]).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError("Adapter path must be within workspace")
        hashes = verify(path)
        lineage = json.loads((path / "lineage.json").read_text())
        validate_adapter_lineage({"adapter": str(path), "sft": str(ROOT / "runs/sft_frozen")}, lineage, p)
        if lineage.get("sft_hashes") != p["sft_hashes"]:
            raise ValueError("Candidate is not a continuation of the frozen SFT")
        models[name] = dict(adapter=str(path.relative_to(ROOT)), hashes=hashes)
    return dict(protocol_sha256=sha(PROJECT / "reports/evaluation-protocol.json"),
                evaluation_sha256=sha(Path(__file__)), runtime_sha256=sha(Path(__file__).with_name("runtime.py")),
                metrics_sha256=sha(ROOT / "src/math_posttrain/metrics.py"), models=models)


def paired_report(results, repetitions):
    import numpy as np

    anchor = {r["sample_id"]: r["correct"] for r in results["sft"]}
    deltas = {}
    rng = np.random.default_rng(42)
    for name, rows in results.items():
        ids = [r["sample_id"] for r in rows]
        if set(ids) != set(anchor) or len(ids) != len(anchor):
            raise ValueError("Model evaluation IDs do not match SFT")
        if name == "sft":
            continue
        changes = np.array([int(r["correct"]) - int(anchor[r["sample_id"]]) for r in rows])
        lost, gained = int((changes < 0).sum()), int((changes > 0).sum())
        n = len(changes)
        draws = rng.multinomial(n, np.array([lost, n-lost-gained, gained]) / n, repetitions)
        deltas[name] = dict(gained=gained, lost=lost, delta_pp=100 * float(changes.mean()),
                            ci95_pp=np.quantile(100 * (draws[:, 2]-draws[:, 0]) / n, [0.025, 0.975]).tolist())
    return deltas


def run(args):
    p = protocol()
    locked = snapshot(args.roster)
    if args.freeze_roster:
        with args.freeze_roster.open("x") as stream:
            json.dump(locked, stream, indent=2)
        print(f"Locked roster of {len(locked['models'])} models.")
        return
    if args.split == "holdout":
        if args.lock is None or json.loads(args.lock.read_text()) != locked:
            raise ValueError("Held-out evaluation requires a matching locked roster")
    data_path = PROJECT / "data" / ("validation.json" if args.split == "validation" else "svamp-heldout.json")
    if sha(data_path) != p[args.split]["sha256"]:
        raise ValueError("Frozen evaluation data changed")
    rows = json.loads(data_path.read_text())
    if args.dry_run:
        print(f"Verified {args.split}: {len(rows)} items, {len(locked['models'])} models.")
        return
    if args.output is None:
        raise ValueError("Evaluation needs an output directory")
    import torch
    from peft import PeftModel

    budget = Budget(PROJECT / "reports/session.json")
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    identity = dict(split=args.split, roster=locked, rows_sha256=sha(data_path))
    identity_file = out / "identity.json"
    if identity_file.exists():
        if json.loads(identity_file.read_text()) != identity:
            raise ValueError("Evaluation resume identity mismatch")
    else:
        with identity_file.open("x") as stream:
            json.dump(identity, stream, indent=2)
    results, summaries = {}, {}
    for name, spec in locked["models"].items():
        dest = out / f"{name}.jsonl"
        completed = [json.loads(line) for line in dest.read_text().splitlines()] if dest.exists() else []
        if [r["sample_id"] for r in completed] != [r["sample_id"] for r in rows[:len(completed)]]:
            raise ValueError("Evaluation resume is not an exact prefix")
        if len(completed) < len(rows):
            budget.check()
            model, tokenizer = load_policy(p)
            model = model.cuda().eval()
            if spec:
                model = PeftModel.from_pretrained(model, ROOT / spec["adapter"]).eval()
            for row in rows[len(completed):]:
                budget.check()
                prompt = render_prompt(row["question"], PROMPT)
                inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
                started = time.perf_counter()
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs, **p["decode"], pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                response = generated[0, inputs["input_ids"].shape[1]:]
                record = dict(sample_id=row["sample_id"], answer=row["answer"],
                              completion=tokenizer.decode(response, skip_special_tokens=True),
                              output_tokens=len(response), latency_seconds=time.perf_counter()-started)
                append(dest, record)
                completed.append(record)
            del model
            torch.cuda.empty_cache()
        results[name], summaries[name] = evaluate_records(completed)
    report = dict(split=args.split, summaries=summaries, paired=paired_report(results, 50000))
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["validation", "holdout"], default="validation")
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--freeze-roster", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    run(parser.parse_args())
