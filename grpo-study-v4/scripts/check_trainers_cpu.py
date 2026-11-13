"""Run actual TRL optimization on a tiny fixture without network or CUDA."""
import argparse
import copy
import math
import json
import random
from pathlib import Path
from types import SimpleNamespace
import tempfile

import torch
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM, set_seed

from ppo_reward import AnswerReward
from runtime import policy_lora
from train_ppo import connect_checkpointing, make_critic


def fixture():
    vocab = {token: i for i, token in enumerate(
        ["[PAD]", "[EOS]", "[UNK]", "Question", "Answer", "Final", ":", "6", "8", "a", "b"]
    )}
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]",
                                       eos_token="[EOS]", unk_token="[UNK]", padding_side="left")
    tokenizer.model_input_names = ["input_ids", "attention_mask"]
    model = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, eos_token_id=1, pad_token_id=0,
        max_position_embeddings=128,
    ))
    return model, tokenizer


def check_dpo(model, tokenizer, output):
    from trl import DPOConfig, DPOTrainer

    dataset = Dataset.from_list([dict(prompt="Question : a", chosen="Final Answer : 6",
                                      rejected="Final Answer : 8")])
    trainer = DPOTrainer(
        model=model, processing_class=tokenizer, train_dataset=dataset, peft_config=policy_lora(),
        args=DPOConfig(output_dir=output, use_cpu=True, bf16=False, max_steps=1,
                       max_length=32, per_device_train_batch_size=1, learning_rate=1e-3,
                       gradient_checkpointing=False, save_strategy="no", report_to="none",
                       disable_tqdm=True),
    )
    before = {n: p.detach().clone() for n, p in trainer.model.named_parameters() if p.requires_grad}
    result = trainer.train()
    assert result.global_step == 1 and math.isfinite(result.training_loss)
    assert trainer.ref_model is None
    assert any(not torch.equal(p, before[n]) for n, p in trainer.model.named_parameters() if n in before)
    print("DPO CPU: one policy update, finite loss, adapter-disabled reference.")


def check_ppo(model, tokenizer, output):
    from fractions import Fraction
    from trl.experimental.ppo import PPOConfig, PPOTrainer

    prompts = [tokenizer.encode(f"Question : {name}") for name in ("a", "b")]
    dataset = Dataset.from_list([dict(input_ids=ids) for ids in prompts])
    verifier = AnswerReward(tokenizer, {tuple(ids): Fraction(6) for ids in prompts})
    critic = make_critic(copy.deepcopy(model.model))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    critic.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    config = PPOConfig(
        output_dir=output, use_cpu=True, bf16=False, per_device_train_batch_size=2,
        gradient_accumulation_steps=1, total_episodes=2, local_rollout_forward_batch_size=2,
        response_length=4, temperature=0.8, stop_token="eos", num_ppo_epochs=1,
        num_mini_batches=1, num_sample_generations=0, learning_rate=1e-3,
        save_strategy="no", report_to="none", disable_tqdm=True,
    )
    trainer = PPOTrainer(args=config, model=model, ref_model=None, reward_model=verifier,
                         value_model=critic, processing_class=tokenizer, train_dataset=dataset,
                         eval_dataset=dataset.select([]), peft_config=policy_lora())
    connect_checkpointing(trainer)
    before = {n: p.detach().clone() for n, p in trainer.model.named_parameters() if p.requires_grad}
    trainer.train()
    assert trainer.state.global_step == 1
    assert abs(trainer.state.log_history[0]["objective/kl"]) < 1e-6
    changed = [n for n, p in trainer.model.named_parameters() if n in before and not torch.equal(p, before[n])]
    assert any(n.startswith("policy.") for n in changed), changed
    assert any(n.startswith("value_model.") for n in changed), changed
    print("PPO CPU: one rollout and update, both policy and critic parameters changed.")


def check_dynamic(model, tokenizer, output):
    from trl import GRPOConfig
    from dynamic_grpo import DynamicGRPOTrainer

    pool = [dict(prompt=f"Question : {name}", sample_id=name) for name in ("a", "b")]
    calls = []

    def reward(completions, **kwargs):
        calls.append(len(completions))
        if len(calls) == 1:
            return [0., 0., 0., 0., 1., 0., 1., 0.]
        return [1., 0., 1., 0.]

    telemetry = Path(output) / "refill.jsonl"
    trainer = DynamicGRPOTrainer(
        model=model, processing_class=tokenizer, train_dataset=Dataset.from_list(pool),
        reward_funcs=reward, peft_config=policy_lora(),
        args=GRPOConfig(output_dir=output, use_cpu=True, bf16=False, max_steps=1,
                        per_device_train_batch_size=8, gradient_accumulation_steps=1,
                        num_generations=4, steps_per_generation=1, max_completion_length=4,
                        loss_type="grpo", scale_rewards="group",
                        learning_rate=1e-3, beta=0.02, save_strategy="no",
                        report_to="none", disable_tqdm=True),
        refill_rows=pool, refill_rng=random.Random(43),
        budget=SimpleNamespace(check=lambda: None), telemetry=telemetry,
    )
    result = trainer.train()
    logs = [json.loads(line) for line in telemetry.read_text().splitlines()]
    assert result.global_step == 1 and math.isfinite(result.training_loss)
    assert calls == [8, 4] and [row["retained_groups"] for row in logs] == [[1], [0]]
    assert all(row["generated_tokens"] > 0 for row in logs)
    print("Dynamic GRPO CPU: generated 3 groups, discarded 1, completed 1 update with 2 retained groups.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=["dpo", "ppo", "dynamic"])
    args = parser.parse_args()
    torch.set_num_threads(1)
    set_seed(43)
    model, tokenizer = fixture()
    with tempfile.TemporaryDirectory(prefix="v4-trainer-cpu-") as output:
        {"dpo": check_dpo, "ppo": check_ppo, "dynamic": check_dynamic}[args.method](model, tokenizer, output)
