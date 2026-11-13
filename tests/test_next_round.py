import json
from types import SimpleNamespace
import pytest
from math_posttrain.signals import SignalWindows
from math_posttrain.train_grpo import make_reward_callback
from math_posttrain.monitoring import RunMonitor
from math_posttrain.frozen_protocol import freeze_protocol,verify_protocol


def test_strict_threshold_and_persistent_windows():
    w=SignalWindows(window_groups=5)
    w.add([{'zero_variance':True}]*4+[{'zero_variance':False}])
    assert w.state['streak']==0  # exactly 80% is not >80%
    w.add([{'zero_variance':True}]*7)
    restored=SignalWindows(json.loads(json.dumps(w.state)),window_groups=5)
    assert restored.add([{'zero_variance':True}]*8)['stop']


def test_token_counts_and_stop_callback(tmp_path):
    callback=make_reward_callback('accuracy',tmp_path/'reward-groups.jsonl',4)
    for _ in range(60):
        callback(['Final Answer: 1']*4,['1']*4,completion_ids=[[1,2,3]]*4)
    row=json.loads((tmp_path/'reward-groups.jsonl').read_text().splitlines()[0])
    assert row['generated_tokens']==12
    assert json.loads((tmp_path/'signal-state.json').read_text())['stop']
    monitor=RunMonitor(tmp_path)
    control=SimpleNamespace(should_training_stop=False,should_save=False)
    monitor.on_step_end(None,SimpleNamespace(global_step=60,max_steps=100),control)
    assert control.should_training_stop and control.should_save
    checkpoint=tmp_path/'checkpoint-60';checkpoint.mkdir()
    monitor.on_save(None,SimpleNamespace(global_step=60),control)
    assert json.loads((checkpoint/'signal-state.json').read_text())['stop']


def test_frozen_protocol_detects_mutation_and_refuses_overwrite(tmp_path,monkeypatch):
    import math_posttrain.frozen_protocol as fp
    config={'seed':42}
    current={'seed':42,'artifact':'abc'}
    monkeypatch.setattr(fp,'snapshot',lambda c,m:dict(current))
    path=tmp_path/'protocol.json'
    freeze_protocol(config,'models',path)
    verify_protocol(config,'models',path)
    with pytest.raises(FileExistsError): freeze_protocol(config,'models',path)
    current['artifact']='tampered'
    with pytest.raises(ValueError,match='mismatch'): verify_protocol(config,'models',path)
