"""Strict trainer callback bridge; reference must be a merged SFT base."""
from .parsing import parse_number
from .rewards import score_batch


def make_reward_callback(mode, telemetry_path=None, group_size=4):
    from pathlib import Path
    import json
    from .signals import SignalWindows
    signal_path = Path(telemetry_path).with_name('signal-state.json') if telemetry_path else None
    windows = SignalWindows(json.loads(signal_path.read_text()) if signal_path and signal_path.exists() else None)
    def reward(completions, gold_number, **kwargs):
        if not all(isinstance(x, str) for x in completions):
            raise ValueError('plain text completion callback required')
        golds = [parse_number(x) for x in gold_number]
        if any(x is None for x in golds):
            raise ValueError('invalid gold_number in dataset')
        rows = score_batch(completions, golds, mode)
        totals = [row['total'] for row in rows]
        if telemetry_path is not None:
            import json
            import time
            from pathlib import Path
            if len(rows) % group_size:
                raise ValueError('reward batch is not divisible by group_size')
            groups = []
            for start in range(0, len(rows), group_size):
                group = rows[start:start + group_size]
                values = totals[start:start + group_size]
                mean = sum(values) / group_size
                std = (sum((x-mean)**2 for x in values) / max(1, group_size-1)) ** 0.5
                groups.append(dict(mean=mean, std=std, zero_variance=std == 0,
                    all_correct=all(r['accuracy'] for r in group),
                    all_wrong=not any(r['accuracy'] for r in group),
                    format_only_count=sum(r['format'] and not r['accuracy'] for r in group),
                    advantages=[(x-mean)/(std+1e-4) for x in values]))
            counts = kwargs.get('completion_ids')
            if counts is not None and len(counts) != len(rows):
                raise ValueError('completion token IDs batch mismatch')
            token_lengths = [len(x) for x in counts] if counts is not None else None
            sample_ids = kwargs.get('sample_id')
            prompts = kwargs.get('prompts')
            if sample_ids is not None and len(sample_ids) != len(rows):
                raise ValueError('sample IDs batch mismatch')
            if prompts is not None and len(prompts) != len(rows):
                raise ValueError('prompts batch mismatch')
            for i, row in enumerate(rows):
                row.update(completion=completions[i], gold_number=gold_number[i],
                           sample_id=sample_ids[i] if sample_ids is not None else None,
                           prompt=prompts[i] if prompts is not None else None)
            if sample_ids is not None:
                for start in range(0, len(rows), group_size):
                    if len(set(sample_ids[start:start+group_size])) != 1:
                        raise ValueError('reward group mixes different sample IDs')
            signal_path.write_text(json.dumps(windows.add(groups), indent=2))
            with Path(telemetry_path).open('a') as stream:
                stream.write(json.dumps(dict(time=time.time(), rows=rows, groups=groups, generated_token_lengths=token_lengths, generated_tokens=sum(token_lengths) if token_lengths is not None else None, token_count_definition='completion IDs including generated EOS; no prompt or padding')) + '\n')
        return totals
    reward.__name__ = 'reward_' + mode
    return reward
