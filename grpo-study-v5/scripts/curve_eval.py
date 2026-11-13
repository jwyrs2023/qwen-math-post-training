"""Evaluate exact sealed checkpoints, with one resumable record stream per curve point."""
import json
import time

from common import PROTOCOL, PROMPT, RUNS, Budget, atomic_json, checkpoint, evaluation_path
from common import immutable, read, sha, verify
from math_posttrain.metrics import evaluate_records
from math_posttrain.prompting import render_prompt


def names(p, split, target):
    yield "sft", None, 0
    for step in p["splits"][split]["steps"]:
        if step <= target:
            for seed in p["seeds"]:
                yield f"s{seed}-step{step}", seed, step


def expected_identity(p, v4, split, seed, step):
    hashes = v4["sft_hashes"]
    if seed is not None:
        path = checkpoint(seed, step)
        hashes = verify(path)
        expected = {key: v4[key] for key in ("model", "revision", "model_dtype", "sft_hashes")}
        expected |= dict(kind="grpo", branch="standard", seed=seed, step=step,
                         protocol_sha256=sha(PROTOCOL))
        if read(path / "lineage.json") != expected:
            raise ValueError(f"Wrong checkpoint lineage: {path}")
    return dict(protocol_sha256=sha(PROTOCOL), split=split, seed=seed, step=step,
                checkpoint_hashes=hashes, data_sha256=p["splits"][split]["sha256"])


def checked_predictions(path, rows):
    if not path.is_file():
        return []
    records = [json.loads(line) for line in path.read_text().splitlines()]
    keys = ("sample_id", "answer")
    if [tuple(r[k] for k in keys) for r in records] != [
        tuple(r[k] for k in keys) for r in rows[:len(records)]
    ]:
        raise ValueError(f"Prediction file does not match the frozen data prefix: {path}")
    return records


def run(p, v4, split, target):
    import torch
    from peft import PeftModel
    from common import load_policy

    budget = Budget()
    rows = read(evaluation_path(p, split))
    roster = [(name, seed, step, expected_identity(p, v4, split, seed, step))
              for name, seed, step in names(p, split, target)]
    for name, seed, step, identity in roster:
        dest = RUNS / "curves" / split / name
        dest.mkdir(parents=True, exist_ok=True)
        immutable(dest / "identity.json", identity)
        output = dest / "predictions.jsonl"
        completed = checked_predictions(output, rows)
        if len(completed) < len(rows):
            budget.check()
            model, tokenizer = load_policy(v4)
            model = model.cuda().eval()
            if seed is not None:
                model = PeftModel.from_pretrained(model, checkpoint(seed, step)).eval()
            with output.open("a", buffering=1) as stream:
                for row in rows[len(completed):]:
                    budget.check()
                    prompt = render_prompt(row["question"], PROMPT)
                    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
                    started = time.perf_counter()
                    with torch.inference_mode():
                        ids = model.generate(**inputs, **v4["decode"], pad_token_id=tokenizer.pad_token_id,
                                             eos_token_id=tokenizer.eos_token_id)
                    response = ids[0, inputs["input_ids"].shape[1]:]
                    record = dict(sample_id=row["sample_id"], answer=row["answer"],
                                  completion=tokenizer.decode(response, skip_special_tokens=True),
                                  output_tokens=len(response), latency_seconds=time.perf_counter()-started)
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                    completed.append(record)
            del model
            torch.cuda.empty_cache()
        _, metrics = evaluate_records(completed)
        metrics["at_token_cap"] = sum(r["output_tokens"] >= v4["decode"]["max_new_tokens"]
                                      for r in completed)
        atomic_json(dest / "summary.json", metrics)
        print(f"{split}/{name}: {metrics['count']} complete", flush=True)
