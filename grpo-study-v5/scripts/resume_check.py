"""Compare a continuous run with stop, reload and continuation through real TRL."""
import json
from pathlib import Path
import random
import tempfile

from common import PROTOCOL, REPORTS, atomic_json, load_policy, sha, training_rows, verify
from train import CheckpointMonitor, make_trainer, resume_state


def fixture():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    vocab = {token: i for i, token in enumerate(
        ["[PAD]", "[EOS]", "[UNK]", "Question", ":", "a", "b", "c", "d", "6", "8"]
    )}
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", eos_token="[EOS]",
                                       unk_token="[UNK]", padding_side="left")
    tokenizer.model_input_names = ["input_ids", "attention_mask"]
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(vocab), hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        eos_token_id=1, pad_token_id=0, max_position_embeddings=128, use_cache=False))
    return model, tokenizer


def compare(p, v4, gpu, root):
    import torch
    from safetensors.torch import load_file
    from transformers import set_seed
    from math_posttrain.train_grpo import make_reward_callback
    from common import Budget

    if not gpu:
        torch.set_num_threads(1)
    deadline = Budget().deadline if gpu else float("inf")
    if gpu:
        pool = training_rows()
        planned = random.Random(43).choices(pool, k=16)
    else:
        planned = [dict(sample_id=f"fixture-{i}", prompt=f"Question : {label}", gold_number="6")
                   for i, label in enumerate("abcd"*4)]

    def train_segment(directory, target, start=None):
        directory.mkdir(exist_ok=True)
        set_seed(43)
        model, tokenizer = load_policy(v4) if gpu else fixture()
        model.config.use_cache = False
        previous_path, previous = resume_state(directory, start) if start else (None, {})
        log = directory / f"reward-{target}.jsonl"
        options = p["training"] | dict(output_dir=str(directory), max_steps=4, seed=43, data_seed=43)
        if gpu:
            reward = make_reward_callback("accuracy", log)
        else:
            options.update(use_cpu=True, bf16=False, max_completion_length=8, disable_tqdm=True)
            def reward(completions, sample_id, completion_ids, **_):
                values = [float(i % 4 in (0, 2)) for i in range(len(completions))]
                with log.open("a") as stream:
                    stream.write(json.dumps(dict(ids=sample_id, tokens=completion_ids,
                                                 rewards=values)) + "\n")
                return values
        monitor = CheckpointMonitor(target, set(), deadline, {"pilot": True}, log,
                                    previous.get("segments", []), previous.get("training_seconds", 0))
        trainer = make_trainer(model, tokenizer, planned, options, reward, monitor)
        trainer.train(resume_from_checkpoint=str(previous_path) if previous_path else None)
        assert trainer.state.global_step == target
        final = directory / f"checkpoint-{target}"
        verify(final)
        return final

    full = train_segment(root / "full", 4)
    first = train_segment(root / "resumed", 2)
    resumed = train_segment(root / "resumed", 4, (2, first))

    def rollout_trace(path):
        trace = []
        for segment in json.loads((path / "v5-state.json").read_text())["segments"]:
            data = (path.parent / segment["path"]).read_bytes()[:segment["bytes"]]
            for line in data.splitlines():
                row = json.loads(line)
                if gpu:
                    trace.append(([(r["sample_id"], r["completion"], r["total"]) for r in row["rows"]],
                                  row["generated_token_lengths"]))
                else:
                    trace.append(row)
        return trace

    full_trace = rollout_trace(full)
    assert len(full_trace) == 4 and full_trace == rollout_trace(resumed), "Resume rollout mismatch"
    actual_ids = [item[0][i][0] if gpu else item["ids"][i]
                  for item in full_trace for i in range(0, 16, 4)]
    assert actual_ids == [row["sample_id"] for row in planned], "Reward rows do not follow the plan"
    left = load_file(full / "adapter_model.safetensors")
    right = load_file(resumed / "adapter_model.safetensors")
    initial = load_file(first / "adapter_model.safetensors")
    assert left.keys() == right.keys() == initial.keys()
    assert any(not torch.equal(left[key], initial[key]) for key in left), "Pilot policy did not update"
    max_diff = max(float((left[key]-right[key]).abs().max()) for key in left)
    assert all(torch.allclose(left[k], right[k], rtol=1e-5 if gpu else 0,
                              atol=1e-7 if gpu else 0) for k in left), max_diff
    return dict(device="cuda" if gpu else "cpu", steps=4, resumed_at=2, rollout_matches=True,
                prompt_plan_matches=True, adapter_max_abs_difference=max_diff,
                protocol_sha256=sha(PROTOCOL),
                gpu_name=torch.cuda.get_device_name() if gpu else None,
                torch_version=torch.__version__, cuda=torch.version.cuda if gpu else None)


def run(p, v4, gpu):
    with tempfile.TemporaryDirectory(prefix="v5-resume-", dir=REPORTS if gpu else None) as tmp:
        result = compare(p, v4, gpu, Path(tmp))
    if gpu:
        atomic_json(REPORTS / "gpu-resume-pilot.json", result)
    print(json.dumps(result, indent=2))
