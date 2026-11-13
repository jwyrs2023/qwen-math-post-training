import copy
import json
from pathlib import Path
import pytest
from math_posttrain.data import build_manifest, audit_manifest


def test_manifest_reproducible_and_tamper_detection(tmp_path):
    train = [dict(question=f'question number {i}', answer='work\n#### 1') for i in range(7473)]
    test = [dict(question=f'test number {i}', answer='#### 2') for i in range(1319)]
    revision = 'a' * 40
    a = build_manifest(train, test, revision)
    assert a == build_manifest(train, test, revision)
    from collections import Counter
    assert Counter(row['split'] for row in a['rows']) == dict(sft_train=3000, grpo_train=2000, validation=500, unused=1973, official_test=1319)
    assert len({row['sample_id'] for row in a['rows']}) == 8792
    (tmp_path / 'raw').mkdir()
    (tmp_path / 'manifests').mkdir()
    (tmp_path / 'raw/train.json').write_text(json.dumps(train))
    (tmp_path / 'raw/test.json').write_text(json.dumps(test))
    a['rows'][0]['content_hash'] = 'tampered'
    path = tmp_path / 'manifests/splits.json'
    path.write_text(json.dumps(a))
    with pytest.raises(ValueError, match='mismatch'):
        audit_manifest(path)
    with pytest.raises(ValueError):
        build_manifest(train, test, 'main')


def test_quarantine_preserves_original_and_test():
    from math_posttrain.data import apply_quarantine
    original = dict(rows=[dict(sample_id='train:0', split='sft_train'),
                          dict(sample_id='test:0', split='official_test')])
    clean = apply_quarantine(original, ['train:0'])
    assert original['rows'][0]['split'] == 'sft_train'
    assert clean['rows'][0]['split'] == 'quarantine'
    assert clean['rows'][0]['original_split'] == 'sft_train'
    assert clean['rows'][1] == original['rows'][1]
    with pytest.raises(ValueError):
        apply_quarantine(original, ['test:0'])
    with pytest.raises(ValueError):
        apply_quarantine(original, ['train:99'])


def test_checkpoint_tampering(tmp_path):
    from math_posttrain.checkpoints import freeze, verify
    source = tmp_path / 'adapter'
    source.mkdir()
    (source / 'adapter_config.json').write_text('{}')
    (source / 'adapter_model.safetensors').write_bytes(b'synthetic test fixture')
    frozen = tmp_path / 'frozen'
    freeze(source, frozen)
    verify(frozen)
    (frozen / 'adapter_model.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        verify(frozen)
