"""Two real CPU optimizer updates on a synthetic tiny model, no research claims."""
import math
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig


def test_two_sft_updates(tmp_path):
    backend = Tokenizer(WordLevel({'<pad>': 0, '<eos>': 1, '<unk>': 2, 'Question': 3, 'Solution': 4, '2': 5, 'Final': 6, 'Answer': 7, ':': 8}, unk_token='<unk>'))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='<pad>', eos_token='<eos>', unk_token='<unk>')
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=9, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, eos_token_id=1, pad_token_id=0))
    args = SFTConfig(output_dir=str(tmp_path), use_cpu=True, bf16=False, fp16=False,
        max_steps=2, per_device_train_batch_size=1, gradient_accumulation_steps=1,
        learning_rate=1e-3, max_length=32, packing=False, completion_only_loss=True,
        report_to='none', save_strategy='steps', save_steps=1, eval_strategy='steps', eval_steps=1, load_best_model_at_end=True, metric_for_best_model='eval_loss', greater_is_better=False, logging_steps=1, optim='adamw_torch',
        gradient_checkpointing=False, dataloader_pin_memory=False)
    trainer = SFTTrainer(model=model, args=args, processing_class=tokenizer,
        train_dataset=Dataset.from_list([dict(prompt='Question: 2 Solution:', completion='Final Answer: 2<eos>')] * 4),
        eval_dataset=Dataset.from_list([dict(prompt='Question: 2 Solution:', completion='Final Answer: 2<eos>')]),
        peft_config=LoraConfig(r=2, lora_alpha=4, target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM'))
    result = trainer.train()
    assert result.global_step == 2
    assert math.isfinite(result.training_loss)
    trainer.save_model(str(tmp_path / 'adapter'))
    assert (tmp_path / 'adapter/adapter_model.safetensors').is_file()

    checkpoint = tmp_path / 'checkpoint-1'
    for filename in ('optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'trainer_state.json'):
        assert (checkpoint / filename).is_file()
    expected = {name: value.detach().clone() for name, value in trainer.model.named_parameters() if value.requires_grad}
    # Restart from step 1 with the same total schedule; compare the final adapter.
    resumed_model = Qwen2ForCausalLM(model.config)
    resumed = SFTTrainer(model=resumed_model, args=args, processing_class=tokenizer,
        train_dataset=Dataset.from_list([dict(prompt='Question: 2 Solution:', completion='Final Answer: 2<eos>')] * 4),
        eval_dataset=Dataset.from_list([dict(prompt='Question: 2 Solution:', completion='Final Answer: 2<eos>')]),
        peft_config=LoraConfig(r=2, lora_alpha=4, target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM'))
    # Base weights are frozen but must also match the original synthetic initialization.
    with __import__('torch').no_grad():
        original = dict(trainer.model.named_parameters())
        for name, value in resumed.model.named_parameters():
            if not value.requires_grad:
                value.copy_(original[name])
    result = resumed.train(resume_from_checkpoint=str(checkpoint))
    assert result.global_step == 2
    import torch
    for name, value in resumed.model.named_parameters():
        if value.requires_grad:
            assert torch.allclose(value, expected[name], atol=1e-7, rtol=1e-5)
