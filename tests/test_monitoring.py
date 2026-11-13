import json
from types import SimpleNamespace
import pytest
from math_posttrain.monitoring import RunMonitor


def test_budget_stop_and_save(tmp_path):
    ticks = [0]
    monitor = RunMonitor(tmp_path, dict(limit=10, spent=9, non_gpu=0, reserve=0.5, hourly_price=10), clock=lambda:ticks[0])
    ticks[0] = 360
    state = SimpleNamespace(global_step=1, max_steps=100)
    control = SimpleNamespace(should_training_stop=False, should_save=False)
    monitor.on_step_end(None, state, control)
    assert control.should_training_stop and control.should_save
    assert monitor.reason == 'budget_limit'
    assert json.loads((tmp_path / 'status.json').read_text())['status'] == 'stopping'


def test_nan_stops(tmp_path):
    monitor = RunMonitor(tmp_path)
    control = SimpleNamespace(should_training_stop=False, should_save=False)
    monitor.on_log(None, SimpleNamespace(global_step=1), control, {'loss':float('nan')})
    assert control.should_training_stop and control.should_save


def test_bad_budget(tmp_path):
    with pytest.raises(ValueError):
        RunMonitor(tmp_path, {'limit':600})


def test_group_telemetry_zero_variance(tmp_path):
    from math_posttrain.train_grpo import make_reward_callback
    path = tmp_path / 'groups.jsonl'
    callback = make_reward_callback('accuracy_format', path, 4)
    assert callback(['Final Answer: 1'] * 4, ['2'] * 4) == [0.1] * 4
    group = json.loads(path.read_text())['groups'][0]
    assert group['zero_variance']
    assert group['advantages'] == [0] * 4
    assert group['format_only_count'] == 4
