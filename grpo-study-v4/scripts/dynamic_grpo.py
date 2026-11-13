"""Single-device GRPO refill using whole mixed-reward groups."""
import torch
from torch.nn.functional import pad
from trl import GRPOTrainer

from runtime import append


def mixed_group_indices(advantages, group_size):
    if advantages.ndim != 1 or len(advantages) % group_size or not torch.isfinite(advantages).all():
        raise ValueError("Malformed grouped advantages")
    groups = advantages.reshape(-1, group_size)
    return torch.where((groups > 0).any(dim=1) & (groups < 0).any(dim=1))[0].tolist()


def combine_batches(batches, pad_token_id):
    keys = set(batches[0]) - {"num_items_in_batch"}
    supported = {"prompt_ids", "prompt_mask", "completion_ids", "completion_mask",
                 "advantages", "old_per_token_logps", "ref_per_token_logps"}
    if not keys <= supported or any(set(b) - {"num_items_in_batch"} != keys for b in batches):
        raise ValueError("Unexpected TRL rollout fields")
    merged = {}
    for key in sorted(keys):
        values = [b[key] for b in batches]
        if key == "advantages":
            merged[key] = torch.cat(values)
            continue
        width = max(v.shape[1] for v in values)
        fill = pad_token_id if key in ("prompt_ids", "completion_ids") else 0
        merged[key] = torch.cat([
            pad(v, (width - v.shape[1], 0) if key.startswith("prompt_") else (0, width - v.shape[1]), value=fill)
            for v in values
        ])
    merged["num_items_in_batch"] = merged["completion_mask"].sum()
    return merged


class DynamicGRPOTrainer(GRPOTrainer):
    """Keep each retained group intact and fill shortages with fresh prompts."""

    def __init__(self, *args, refill_rows, refill_rng, budget, telemetry, **kwargs):
        super().__init__(*args, **kwargs)
        if self.accelerator.num_processes != 1 or self.use_vllm or self.loss_type != "grpo":
            raise ValueError("Dynamic refill is restricted to single-device vanilla GRPO")
        self.refill_rows = refill_rows
        self.refill_rng = refill_rng
        self.budget = budget
        self.telemetry = telemetry

    def _generate_and_score_completions(self, inputs):
        group_size = self.num_generations
        if not inputs or len(inputs) % group_size:
            raise ValueError("Rollout batch must contain whole groups")
        target = len(inputs) // group_size
        kept = []
        remaining = target
        for attempt in range(12):
            self.budget.check()
            generated = super()._generate_and_score_completions(inputs)
            good = mixed_group_indices(generated["advantages"], group_size)[:remaining]
            append(self.telemetry, dict(
                step=self.state.global_step, attempt=attempt,
                candidate_ids=[r["sample_id"] for r in inputs[::group_size]],
                retained_groups=good, generated_tokens=int(generated["num_items_in_batch"]),
            ))
            if good:
                index = torch.tensor(
                    [i * group_size + j for i in good for j in range(group_size)],
                    device=generated["advantages"].device,
                )
                kept.append({k: v.index_select(0, index)
                             for k, v in generated.items() if k != "num_items_in_batch"})
                remaining -= len(good)
            if not remaining:
                return combine_batches(kept, self.pad_token_id)
            inputs = [dict(row) for row in self.refill_rng.choices(self.refill_rows, k=remaining)
                      for _ in range(group_size)]
        raise RuntimeError(f"Dynamic refill could not fill {target} mixed groups in 12 attempts")
