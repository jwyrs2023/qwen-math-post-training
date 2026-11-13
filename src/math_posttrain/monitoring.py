"""Training callbacks with truthful stop status and append-only telemetry."""
import json
import math
import time
from pathlib import Path
from transformers import TrainerCallback


class RunMonitor(TrainerCallback):
    def __init__(self, root, budget=None, clock=time.time):
        self.root = Path(root)
        self.budget = budget
        self.clock = clock
        self.started = clock()
        self.reason = None
        self.prior_seconds = 0
        cost_file = self.root / 'cost.jsonl'
        if cost_file.exists():
            lines = cost_file.read_text().splitlines()
            if lines:
                self.prior_seconds = json.loads(lines[-1]).get('cumulative_elapsed_seconds', 0)
        if budget is not None:
            required = {'limit', 'spent', 'non_gpu', 'reserve', 'hourly_price'}
            if set(budget) != required:
                raise ValueError('budget fields must be limit/spent/non_gpu/reserve/hourly_price')
            if not all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in budget.values()):
                raise ValueError('invalid budget values')
            if budget['hourly_price'] <= 0 or budget['limit'] > 600:
                raise ValueError('positive price and limit <=600 required')

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        if any(isinstance(v, (int, float)) and not math.isfinite(v) for v in logs.values()):
            self.reason = 'nonfinite_metric'
            control.should_training_stop = True
            control.should_save = True
        with (self.root / 'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(step=state.global_step, time=self.clock(), **logs)) + '\n')

    def on_step_end(self, args, state, control, **kwargs):
        signal = self.root / 'signal-state.json'
        if signal.exists() and json.loads(signal.read_text()).get('stop'):
            self.reason = self.reason or 'zero_variance_three_windows'
            control.should_training_stop = True
            control.should_save = True
        elapsed = self.clock() - self.started
        if self.budget is not None:
            b = self.budget
            used = b['spent'] + b['non_gpu'] + (elapsed + self.prior_seconds) / 3600 * b['hourly_price']
            # Reserve the next measured update and checkpoint headroom.
            next_cost = elapsed / max(1, state.global_step - getattr(self, 'initial_step', 0)) / 3600 * b['hourly_price']
            if used + b['reserve'] + next_cost >= b['limit']:
                self.reason = 'budget_limit'
                control.should_training_stop = True
                control.should_save = True
            with (self.root / 'cost.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(step=state.global_step, elapsed_seconds=elapsed,
                    cumulative_elapsed_seconds=elapsed+self.prior_seconds, estimated_total_cost=used, hourly_price=b['hourly_price'],
                    billing='elapsed-time estimate; reconcile with provider invoice', stop_reason=self.reason)) + '\n')
        (self.root / 'status.json').write_text(json.dumps(dict(step=state.global_step,
            max_steps=state.max_steps, elapsed_seconds=elapsed, stop_reason=self.reason,
            status='stopping' if self.reason else 'running'), indent=2))

    def on_train_begin(self, args, state, control, **kwargs):
        self.initial_step = state.global_step

    def on_save(self, args, state, control, **kwargs):
        signal = self.root / 'signal-state.json'
        if signal.exists():
            checkpoint = self.root / f'checkpoint-{state.global_step}'
            (checkpoint / 'signal-state.json').write_text(signal.read_text())
