"""Measure frozen SFT multi-sample coverage on the frozen validation split."""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

from prepare_dpo import PROJECT, PROMPT, ROOT, sha
from runtime import Budget
from math_posttrain.checkpoints import verify
from math_posttrain.metrics import evaluate_records
from math_posttrain.prompting import render_prompt
from math_posttrain.data import digest
from math_posttrain.parsing import extract_gold
from math_posttrain.rewards import score_batch


def main(args):
    protocol = json.loads((PROJECT / "reports/evaluation-protocol.json").read_text())
    data = PROJECT / "data/validation.json"
    parser = ROOT / "src/math_posttrain/parsing.py"
    sft = ROOT / "runs/sft_frozen"
    if (sha(data) != protocol["validation"]["sha256"] or sha(PROMPT) != protocol["prompt_sha256"]
            or sha(parser) != protocol["parser_sha256"] or verify(sft) != protocol["sft_hashes"]):
        raise ValueError("Frozen evaluation input changed")
    rows = json.loads(data.read_text())
    anchor = ROOT / "runs/v2_validation_s43/sft_anchor.jsonl"
    predictions, baseline = evaluate_records([json.loads(line) for line in anchor.read_text().splitlines()])
    by_id = {r["sample_id"]: r for r in predictions}
    if set(by_id) != {r["sample_id"] for r in rows}:
        raise ValueError("Historical SFT evaluation does not cover validation exactly")
    for row in rows:
        old = by_id[row["sample_id"]]
        prompt = render_prompt(row["question"], PROMPT)
        if (old["answer"] != row["answer"] or old["prompt_hash"] != digest(prompt)
                or old["model_revision"] != protocol["revision"] or old["model_dtype"] != protocol["model_dtype"]
                or old["checkpoint_hashes"] != protocol["sft_hashes"]):
            raise ValueError(f"Historical SFT record mismatch: {row['sample_id']}")
    if args.dry_run:
        print(f"Validated {len(rows)} frozen validation rows and historical greedy accuracy {baseline['accuracy']:.2%}.")
        return

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Diagnostics require CUDA with BF16 support")
    budget = Budget(args.session)
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(sft, local_files_only=True)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        protocol["model"], revision=protocol["revision"], local_files_only=True, dtype=torch.float32
    )
    model = PeftModel.from_pretrained(model, sft).merge_and_unload().cuda().eval()
    model.config.use_cache = True
    settings = protocol["diagnostic"]
    seen = Counter()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    with (out / "predictions.jsonl").open("x") as stream:
        for i, row in enumerate(rows):
            budget.check()
            set_seed(settings["seed"] + i)
            prompt = render_prompt(row["question"], PROMPT)
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model.generate(
                    **inputs, do_sample=True, temperature=settings["temperature"], top_p=settings["top_p"],
                    max_new_tokens=protocol["decode"]["max_new_tokens"],
                    num_return_sequences=settings["samples_per_question"],
                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                )
            completions = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            rewards = score_batch(completions, [extract_gold(row["answer"])] * len(completions))
            k = sum(r["accuracy"] for r in rewards)
            greedy_correct = by_id[row["sample_id"]]["correct"]
            seen[f"correct_of_eight_{k}"] += 1
            seen["any_correct"] += int(k > 0)
            seen["greedy_wrong_but_sample_correct"] += int(k > 0 and not greedy_correct)
            stream.write(json.dumps(dict(sample_id=row["sample_id"], completions=completions,
                                         rewards=rewards, greedy_correct=greedy_correct), ensure_ascii=False) + "\n")
            stream.flush()
    (out / "summary.json").write_text(json.dumps(dict(
        rows=len(rows), historical_greedy_accuracy=baseline["accuracy"],
        pass_at_8=seen["any_correct"] / len(rows), counts=dict(seen), elapsed_seconds=time.time()-start,
    ), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--session", type=Path, default=PROJECT / "reports/session.json")
    parser.add_argument("--output", type=Path, default=PROJECT / "runs/sft-pass8")
    main(parser.parse_args())
