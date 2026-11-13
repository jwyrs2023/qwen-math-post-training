"""Real synthetic GRPO updates validate CPU callback/PEFT plumbing only."""
import math
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig
from math_posttrain.train_grpo import make_reward_callback


def test_two_grpo_updates(tmp_path):
    backend = Tokenizer(WordLevel({'<pad>': 0, '<eos>': 1, '<unk>': 2, 'Question': 3, 'Solution': 4, '2': 5, 'Final': 6, 'Answer': 7, ':': 8}, unk_token='<unk>'))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='<pad>', eos_token='<eos>', unk_token='<unk>')
    tokenizer.model_input_names = ['input_ids', 'attention_mask']
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=9, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, eos_token_id=1, pad_token_id=0))
    args = GRPOConfig(output_dir=str(tmp_path), use_cpu=True, bf16=False, fp16=False,
        max_steps=2, per_device_train_batch_size=4, gradient_accumulation_steps=1,
        num_generations=4, steps_per_generation=1, max_completion_length=4,
        beta=0.02, temperature=0.8, top_p=0.95, use_vllm=False,
        report_to='none', save_strategy='no', logging_steps=1, optim='adamw_torch',
        gradient_checkpointing=False, dataloader_pin_memory=False)
    trainer = GRPOTrainer(model=model, args=args, processing_class=tokenizer,
        train_dataset=Dataset.from_list([dict(prompt='Question: 2 Solution:', gold_number='2')] * 4),
        reward_funcs=make_reward_callback('accuracy'),
        peft_config=LoraConfig(r=2, lora_alpha=4, target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM'))
    result = trainer.train()
    assert result.global_step == 2
    assert math.isfinite(result.training_loss)
