"""Shared experiment-four data, model provenance and budget guard."""
import json
import math
import os
import time
from pathlib import Path

from prepare_dpo import MANIFEST, PROJECT, PROMPT, ROOT, sha
from math_posttrain.checkpoints import seal, verify
from math_posttrain.prompting import render_prompt
from math_posttrain.protocol import require_clean_audit

SFT = ROOT / "runs/sft_frozen"


def protocol():
    p = json.loads((PROJECT / "reports/evaluation-protocol.json").read_text())
    if (sha(MANIFEST) != p["manifest_sha256"] or sha(PROMPT) != p["prompt_sha256"]
            or sha(ROOT / "src/math_posttrain/parsing.py") != p["parser_sha256"]
            or verify(SFT) != p["sft_hashes"]):
        raise ValueError("Frozen evaluation or model inputs changed")
    return p


def training_rows():
    manifest = require_clean_audit(MANIFEST)
    raw = json.loads((ROOT / "data/raw/train.json").read_text())
    return [
        dict(sample_id=row["sample_id"], prompt=render_prompt(
            raw[int(row["sample_id"].split(":")[1])]["question"], PROMPT),
            gold_number=raw[int(row["sample_id"].split(":")[1])]["answer"].split("####")[1].strip())
        for row in manifest["rows"] if row["split"] == "grpo_train"
    ]


class TimeLimit(RuntimeError):
    pass


class Budget:
    def __init__(self, session_path, minutes=None):
        s = json.loads(Path(session_path).read_text())
        start, cap, reserve, price = (
            s[k] for k in ("started_at_unix", "budget_yuan", "reserve_yuan", "hourly_yuan")
        )
        prior = s.get("prior_spend_yuan", 0)
        if (not all(math.isfinite(x) for x in (start, cap, reserve, price, prior))
                or not 0 < price or not 5 <= reserve < cap <= 90
                or not 0 <= prior < cap - reserve or start > time.time()):
            raise ValueError("Invalid session budget or clock")
        self.started = time.time()
        self.deadline = start + (cap - reserve - prior) * 3600 / price
        if minutes is not None:
            if not 0 < minutes <= 135:
                raise ValueError("Route time cap must be within 135 minutes")
            self.deadline = min(self.deadline, self.started + minutes * 60)
        self.check()

    def check(self, headroom=60):
        if time.time() + headroom >= self.deadline:
            raise TimeLimit("Time limit reached with save/shutdown reserve")


def load_policy(p, pad=None):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Requires a CUDA device with BF16 support")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Only a single process is supported")
    tokenizer = AutoTokenizer.from_pretrained(SFT, local_files_only=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = pad or tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        p["model"], revision=p["revision"], local_files_only=True, dtype=torch.float32
    )
    model = PeftModel.from_pretrained(model, SFT).merge_and_unload()
    return model, tokenizer


def policy_lora():
    from peft import LoraConfig
    return LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")


def seal_policy(path, p, kind, **extra):
    (path / "lineage.json").write_text(json.dumps(dict(
        kind=kind, model=p["model"], revision=p["revision"],
        model_dtype=p["model_dtype"], sft_hashes=p["sft_hashes"], **extra,
    ), indent=2))
    return seal(path)


def append(path, record):
    with path.open("a") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def trainable_snapshot(model):
    return {name: tensor.detach().cpu().clone() for name, tensor in model.named_parameters()
            if tensor.requires_grad}


def changed_trainables(model, before):
    import torch
    current = {name: p for name, p in model.named_parameters() if name in before}
    if set(current) != set(before) or not before:
        raise ValueError("Trainable parameters changed identity during pilot")
    if not all(torch.isfinite(p.detach()).all() for p in current.values()):
        raise ValueError("Pilot produced non-finite trainable weights")
    return [name for name, p in current.items() if not torch.equal(p.detach().cpu(), before[name])]
