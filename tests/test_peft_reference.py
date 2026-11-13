"""Tiny random CPU model tests adapter semantics, not experimental performance."""
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from peft import LoraConfig, get_peft_model


def test_merged_sft_reference_and_reload(tmp_path):
    torch.manual_seed(42)
    config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2)
    model = Qwen2ForCausalLM(config)
    lora = LoraConfig(r=2, lora_alpha=4, lora_dropout=0, target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM')
    sft = get_peft_model(model, lora)
    # Nonzero synthetic adapter ensures merge equivalence test is meaningful.
    with torch.no_grad():
        for name, value in sft.named_parameters():
            if 'lora_B' in name:
                value.normal_(0, 0.05)
    ids = torch.tensor([[1, 2, 3]])
    sft.eval()
    before = sft(ids).logits.detach()
    merged = sft.merge_and_unload().eval()
    assert torch.allclose(before, merged(ids).logits, atol=1e-5, rtol=1e-4)
    merged.save_pretrained(tmp_path)
    reloaded = Qwen2ForCausalLM.from_pretrained(tmp_path).eval()
    assert torch.allclose(before, reloaded(ids).logits, atol=1e-5, rtol=1e-4)
    policy = get_peft_model(reloaded, lora).eval()
    with policy.disable_adapter():
        assert torch.allclose(before, policy(ids).logits, atol=1e-5, rtol=1e-4)
