"""Offline inspection of installed trainer signatures and masking."""
import inspect
import json


def verify():
    import torch
    from trl import SFTTrainer, SFTConfig, GRPOTrainer, GRPOConfig
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
    checks = {}
    for cls in (SFTTrainer, SFTConfig, GRPOTrainer, GRPOConfig):
        checks[cls.__name__] = str(inspect.signature(cls))
    collator = DataCollatorForLanguageModeling(pad_token_id=0, completion_only_loss=True)
    batch = collator([dict(input_ids=[1, 2, 3], completion_mask=[0, 0, 1]),
                      dict(input_ids=[1, 4], completion_mask=[0, 1])])
    assert batch['labels'].tolist() == [[-100, -100, 3], [-100, 4, -100]]
    args = GRPOConfig(output_dir='/tmp/math-posttrain-api-check', use_cpu=True,
                      bf16=False, fp16=False, report_to='none',
                      per_device_train_batch_size=4, gradient_accumulation_steps=4,
                      num_generations=4, beta=0.02, use_vllm=False,
                      max_completion_length=512, steps_per_generation=4)
    checks['masking_verified'] = True
    checks['generation_batch_size'] = args.generation_batch_size
    checks['steps_per_generation'] = args.steps_per_generation
    checks['torch'] = torch.__version__
    checks['gpu_smoke_verified'] = False
    return checks


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
