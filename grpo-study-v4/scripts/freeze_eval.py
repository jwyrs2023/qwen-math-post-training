"""Freeze validation and held-out evaluation IDs before candidate training."""
import argparse
import json
import re
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

from prepare_dpo import MANIFEST, PROJECT, PROMPT, ROOT, sha
from math_posttrain.checkpoints import verify
from math_posttrain.parsing import extract_gold
from math_posttrain.protocol import require_clean_audit

SOURCE = "https://github.com/arkilpatel/SVAMP/blob/main/SVAMP.json"


def tokens(text):
    return tuple(re.findall(r"\d+(?:\.\d+)?|[a-z]+", text.lower()))


def template(seq):
    return tuple("#" if token[0].isdigit() else token for token in seq)


def shingles(seq):
    return set(zip(seq, seq[1:], seq[2:]))


def freeze(path, data):
    content = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if path.read_text() != content:
            raise ValueError(f"Refusing to replace frozen data: {path}")
        return
    with path.open("x") as stream:
        stream.write(content)


def main(source_path):
    manifest = require_clean_audit(MANIFEST)
    raw = {name: json.loads((ROOT / f"data/raw/{name}.json").read_text())
           for name in ("train", "test")}
    validation = []
    for row in manifest["rows"]:
        if row["split"] == "validation":
            sid = row["sample_id"]
            split, index = sid.split(":")
            validation.append(dict(sample_id=sid, **raw[split][int(index)]))
    if len(validation) != 500:
        raise ValueError("Unexpected validation size")
    seen = {tokens(r["question"]) for rows in raw.values() for r in rows}
    templates = {template(seq) for seq in seen}
    gram_sizes, inverted = [], defaultdict(list)
    for seq in sorted(templates):
        grams = shingles(seq)
        index = len(gram_sizes)
        gram_sizes.append(len(grams))
        for gram in grams:
            inverted[gram].append(index)
    retained, exclusions = [], {}
    svamp = json.loads(source_path.read_text(), parse_float=Decimal)
    for row in svamp:
        sid = "svamp:" + row["ID"]
        question = row["Body"].strip() + " " + row["Question"].strip()
        seq = tokens(question)
        masked = template(seq)
        grams = shingles(masked)
        hits = Counter(index for gram in grams for index in inverted.get(gram, ()))
        if seq in seen or masked in templates:
            exclusions[sid] = "exact_or_number_masked_overlap"
            continue
        # 0.80 Jaccard of number-masked token trigrams.
        if any(5 * count >= 4 * (len(grams) + gram_sizes[i] - count)
               for i, count in hits.items()):
            exclusions[sid] = "near_overlap"
            continue
        answer = "#### " + format(Decimal(str(row["Answer"])), "f")
        extract_gold(answer)
        retained.append(dict(sample_id=sid, question=question, answer=answer))
        seen.add(seq)
        templates.add(masked)
        index = len(gram_sizes)
        gram_sizes.append(len(grams))
        for gram in grams:
            inverted[gram].append(index)
    if len({r["sample_id"] for r in retained}) != len(retained) or len(retained) < 800:
        raise ValueError("Unexpected holdout size or duplicate IDs")
    val_path = PROJECT / "data/validation.json"
    held_path = PROJECT / "data/svamp-heldout.json"
    freeze(val_path, validation)
    freeze(held_path, retained)
    freeze(PROJECT / "reports/evaluation-protocol.json", dict(
        model="Qwen/Qwen2.5-1.5B", revision="8faed761d45a263340a0528343f099c05c9a4323",
        model_dtype="float32", manifest_sha256=sha(MANIFEST), prompt_sha256=sha(PROMPT),
        parser_sha256=sha(ROOT / "src/math_posttrain/parsing.py"),
        sft_hashes=verify(ROOT / "runs/sft_frozen"),
        validation=dict(rows=len(validation), sha256=sha(val_path)),
        holdout=dict(source=SOURCE, raw_sha256=sha(source_path), raw_rows=len(svamp),
                     retained=len(retained), sha256=sha(held_path), exclusions=exclusions,
                     overlap="GSM8K train/test plus earlier retained SVAMP: exact, number-masked exact or number-masked trigram Jaccard >= 0.80"),
        decode=dict(do_sample=False, max_new_tokens=512),
        diagnostic=dict(samples_per_question=8, temperature=0.8, top_p=0.95, seed=43),
        screening=dict(seed=43, min_delta_pp=2.0, split="validation"),
        confirmation=dict(seeds=[42, 43, 44], target_delta_pp=5.0,
                          split="holdout", question_bootstrap_replicates=50000,
                          seed_aggregation="Mean paired per-question delta across fixed training seeds. Report each seed separately."),
        holdout_gate="Evaluate only after the method, seeds and checkpoint roster are locked."
    ))
    print(f"Frozen validation={len(validation)}, SVAMP retained={len(retained)}/{len(svamp)}, exclusions={len(exclusions)}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("svamp_source", type=Path)
    main(parser.parse_args().svamp_source)
