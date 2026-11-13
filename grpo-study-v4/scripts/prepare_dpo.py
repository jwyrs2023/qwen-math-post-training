"""Build one deterministic preference pair per eligible training question."""
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from math_posttrain.checkpoints import verify
from math_posttrain.parsing import extract_gold, format_ok, prediction_status
from math_posttrain.prompting import render_prompt
from math_posttrain.protocol import require_clean_audit
from math_posttrain.rewards import score_batch

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
SCORING = ROOT / "grpo-study-v3/reports/scoring"
MANIFEST = ROOT / "data/manifests/splits_clean.json"
PROMPT = ROOT / "prompts/math_v1.txt"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    manifest = require_clean_audit(MANIFEST)
    raw = json.loads((ROOT / "data/raw/train.json").read_text())
    ids = {row["sample_id"] for row in manifest["rows"] if row["split"] == "grpo_train"}
    identity = json.loads((SCORING / "identity.json").read_text())
    source = SCORING / "scores.jsonl"
    archived = json.loads((SCORING.parent / "backups/v3-scoring-manifest.json").read_text())
    for path in (source, SCORING / "identity.json"):
        entry = archived["members"][str(path.relative_to(ROOT))]
        if sha(path) != entry["sha256"]:
            raise ValueError(f"Historical scoring artifact hash mismatch: {path}")
    if (identity["manifest_sha256"] != sha(MANIFEST)
            or identity["prompt_sha256"] != sha(PROMPT)
            or identity["sft_hashes"] != verify(ROOT / "runs/sft_frozen")):
        raise ValueError("Scoring provenance mismatch")
    scored = [json.loads(line) for line in source.read_text().splitlines()]
    if len(scored) != len(ids) or {r["sample_id"] for r in scored} != ids:
        raise ValueError("Scores do not match the audited training pool")
    counts = Counter()
    parse_failures = Counter()
    pairs = []
    for row in sorted(scored, key=lambda r: r["sample_id"]):
        sid = row["sample_id"]
        record = raw[int(sid.split(":")[1])]
        gold = extract_gold(record["answer"])
        completions, lengths = row["completions"], row["generated_lengths"]
        if len(completions) != 4 or len(lengths) != 4:
            raise ValueError(f"Invalid completion group: {sid}")
        rewards = score_batch(completions, [gold] * 4)
        if rewards != row["rewards"] or sum(r["accuracy"] for r in rewards) != row["correct_of_four"]:
            raise ValueError(f"Stored reward disagrees with parser: {sid}")
        k = row["correct_of_four"]
        counts[f"correct_of_four_{k}"] += 1
        correct, incorrect = [], []
        for idx, (answer, length) in enumerate(zip(completions, lengths, strict=True)):
            pred, failure = prediction_status(answer)
            if failure:
                parse_failures[failure] += 1
            if pred is None or not format_ok(answer) or length >= identity["max_new_tokens"]:
                continue
            (correct if pred == gold else incorrect).append(idx)
        if not 0 < k < 4:
            continue
        if not correct or not incorrect:
            counts["mixed_without_eligible_pair"] += 1
            continue
        rng = random.Random(f"v4-pair-seed-43:{sid}")
        chosen, rejected = rng.choice(correct), rng.choice(incorrect)
        pairs.append(dict(sample_id=sid, prompt=render_prompt(record["question"], PROMPT),
                          chosen=completions[chosen], rejected=completions[rejected],
                          source_indices=[chosen, rejected]))
    data_path = PROJECT / "data/dpo-pairs.jsonl"
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in pairs)
    if data_path.exists() and data_path.read_text() != payload:
        raise ValueError("Refusing to replace a different frozen preference dataset")
    if not data_path.exists():
        with data_path.open("x") as out:
            out.write(payload)
    audit = dict(pool=len(ids), answers=4 * len(scored), pairs=len(pairs), counts=dict(counts),
                 parse_failures=dict(parse_failures), scores_sha256=sha(source),
                 pairs_sha256=sha(data_path), scoring_identity=identity,
                 selection="One seeded pair per question; both strictly formatted, parseable and below the generation cap")
    (PROJECT / "reports/dpo-data-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({k: audit[k] for k in ("pool", "answers", "pairs", "counts", "parse_failures")}, indent=2))


if __name__ == "__main__":
    prepare()
