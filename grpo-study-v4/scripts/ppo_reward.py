"""Expose the existing numeric verifier through TRL 0.26's reward interface."""
from types import SimpleNamespace

import torch
from torch import nn

from math_posttrain.parsing import extract_prediction, is_correct


class AnswerReward(nn.Module):
    """Reward indexed by exact audited prompt token IDs, with no learned weights."""

    base_model_prefix = "verifier"

    def __init__(self, tokenizer, gold_by_prompt):
        super().__init__()
        self.verifier = AnswerVerifier(tokenizer, gold_by_prompt)
        self.score = nn.Identity()


class AnswerVerifier(nn.Module):
    def __init__(self, tokenizer, gold_by_prompt):
        super().__init__()
        if tokenizer.eos_token_id == tokenizer.pad_token_id:
            raise ValueError("PPO verifier requires distinct EOS and padding IDs")
        self.tokenizer = tokenizer
        self.gold_by_prompt = gold_by_prompt
        self.prompt_lengths = sorted({len(ids) for ids in gold_by_prompt})

    def forward(self, input_ids, attention_mask, **kwargs):
        scores = []
        for ids, mask in zip(input_ids.tolist(), attention_mask.tolist(), strict=True):
            start = mask.index(True)
            end = mask.index(False, start) if False in mask[start:] else len(mask)
            tokens = tuple(ids[start:end])
            prompts = [
                tokens[:n] for n in self.prompt_lengths
                if n <= len(tokens) and tokens[:n] in self.gold_by_prompt
            ]
            if len(prompts) != 1:
                raise ValueError("PPO rollout does not identify exactly one audited prompt")
            prompt = prompts[0]
            answer = self.tokenizer.decode(tokens[len(prompt):], skip_special_tokens=True)
            scores.append(float(is_correct(extract_prediction(answer), self.gold_by_prompt[prompt])))
        # TRL selects the last response token after masking padding.
        logits = torch.tensor(scores, device=input_ids.device, dtype=torch.float32)
        return SimpleNamespace(hidden_states=(logits[:, None, None].expand(-1, input_ids.shape[1], 1),))
