"""Verify numeric terminal reward through the real TRL reward call on CPU."""
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer
from trl.trainer.utils import get_reward

from math_posttrain.parsing import extract_gold
from math_posttrain.prompting import render_prompt
from math_posttrain.protocol import require_clean_audit
from ppo_reward import AnswerReward

ROOT = Path(__file__).resolve().parents[2]
tokenizer = AutoTokenizer.from_pretrained(ROOT / "runs/sft_frozen", local_files_only=True)
if tokenizer.pad_token_id in (None, tokenizer.eos_token_id):
    tokenizer.pad_token = "<|fim_pad|>"
encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
manifest = require_clean_audit(ROOT / "data/manifests/splits_clean.json")
raw = json.loads((ROOT / "data/raw/train.json").read_text())
gold_by_prompt = {}
for row in manifest["rows"]:
    if row["split"] != "grpo_train":
        continue
    record = raw[int(row["sample_id"].split(":")[1])]
    prompt = tuple(encode(render_prompt(record["question"], ROOT / "prompts/math_v1.txt")))
    if prompt in gold_by_prompt or len(prompt) > 256:
        raise ValueError("Invalid prompt mapping")
    gold_by_prompt[prompt] = extract_gold(record["answer"])
reward = AnswerReward(tokenizer, gold_by_prompt)
prompts = list(gold_by_prompt)[:3]
context = max(map(len, prompts))
pad = tokenizer.pad_token_id
responses = [
    encode(f"Final Answer: {gold_by_prompt[prompts[0]]}") + [tokenizer.eos_token_id],
    encode(f"Final Answer: {gold_by_prompt[prompts[1]] + 1}"),
    encode("No final answer was given.") + [tokenizer.eos_token_id],
]
width = max(map(len, responses))
batch = torch.tensor([
    [pad] * (context - len(p)) + list(p) + a + [pad] * (width - len(a))
    for p, a in zip(prompts, responses, strict=True)
])
_, scores, ends = get_reward(reward, batch, pad, context)
assert scores.tolist() == [1., 0., 0.]
assert ends.tolist() == [context + len(a) - 1 for a in responses]
# A token after the first response padding must not affect the terminal score.
tail = torch.tensor([batch[0].tolist() + [pad] + encode("garbage")])
_, tail_score, _ = get_reward(reward, tail, pad, context)
assert tail_score.tolist() == [1.]
bad = batch.clone()
bad[0, context - 1] = tokenizer.eos_token_id
try:
    get_reward(reward, bad, pad, context)
except ValueError:
    pass
else:
    raise AssertionError("Unrecognized prompt was accepted")
print(f"{len(gold_by_prompt)} audited prompts. TRL rewards correct/wrong/missing=[1,0,0]; terminal positions and unknown-prompt rejection passed.")
