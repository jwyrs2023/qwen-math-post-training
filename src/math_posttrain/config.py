"""Reject unknown fields and unsafe experiment settings."""
from pathlib import Path
import math
import re
import yaml

DEFAULTS = dict(sft_split='sft_train', model_dtype="bfloat16", eval_steps=None, budget=None, revision=None, save_steps=20, model='Qwen/Qwen2.5-1.5B', seed=42, max_length=1024,
                max_prompt_length=256, max_completion_length=512, group_size=4,
                batch_size=4, gradient_accumulation_steps=4, learning_rate=5e-6,
                beta=0.02, temperature=0.8, top_p=0.95, max_steps=20,
                reward='accuracy', lora_r=16, lora_alpha=32, lora_dropout=0.0,
                manifest='data/manifests/splits.json', prompt='prompts/math_v1.txt')


def load_config(path):
    values = yaml.safe_load(Path(path).read_text())
    if not isinstance(values, dict) or set(values) - set(DEFAULTS):
        raise ValueError('config must be a mapping with known fields only')
    config = DEFAULTS | values
    if config['sft_split'] not in ('sft_train', 'grpo_train'):
        raise ValueError('unsupported SFT training split')
    if config['model_dtype'] not in ('bfloat16', 'float32'):
        raise ValueError('unsupported model_dtype')
    if config['model'] != 'Qwen/Qwen2.5-1.5B':
        raise ValueError('core protocol requires Qwen2.5-1.5B Base')
    for key in ('max_length', 'max_prompt_length', 'max_completion_length', 'group_size',
                'batch_size', 'gradient_accumulation_steps', 'max_steps', 'lora_r', 'lora_alpha', 'save_steps'):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if type(config['seed']) is not int or config['seed'] < 0:
        raise ValueError('seed must be a nonnegative integer')
    if config['revision'] is not None and not re.fullmatch('[0-9a-f]{40}', str(config['revision'])):
        raise ValueError('revision must be a 40-character commit hash')
    for key in ('learning_rate', 'beta', 'temperature', 'top_p', 'lora_dropout'):
        value = config[key]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f'{key} must be finite numeric')
    if config['learning_rate'] <= 0 or config['temperature'] <= 0 or config['beta'] < 0:
        raise ValueError('invalid learning rate, temperature or beta')
    if not 0 < config['top_p'] <= 1 or not 0 <= config['lora_dropout'] < 1:
        raise ValueError('invalid top_p or dropout')
    if config['group_size'] < 2:
        raise ValueError('group_size must be at least two')
    if config['budget'] is not None:
        from .monitoring import RunMonitor
        RunMonitor(Path('/tmp'), config['budget'])
    if config['batch_size'] != 1 and config['batch_size'] % config['group_size']:
        raise ValueError('batch_size must be divisible by group_size')
    if config['eval_steps'] is not None and (type(config['eval_steps']) is not int or config['eval_steps'] <= 0):
        raise ValueError('eval_steps must be positive')
    if config['reward'] not in ('accuracy', 'accuracy_format'):
        raise ValueError('unknown reward')
    for key in ('manifest', 'prompt'):
        if not Path(config[key]).is_file():
            raise ValueError(f'missing {key}: {config[key]}')
    return config
