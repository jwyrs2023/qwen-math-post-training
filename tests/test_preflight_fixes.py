import json
import pytest
from math_posttrain.checkpoints import seal, freeze, verify
from math_posttrain.config import load_config


def test_freeze_already_sealed(tmp_path):
    src = tmp_path / 'src'
    src.mkdir()
    (src / 'adapter_config.json').write_text('{}')
    (src / 'adapter_model.safetensors').write_bytes(b'fixture')
    seal(src)
    freeze(src, tmp_path / 'dst')
    verify(tmp_path / 'dst')


@pytest.mark.parametrize('field,value', [('learning_rate', '.nan'), ('beta', '-1'), ('top_p','2'), ('revision','main'), ('group_size','1')])
def test_numeric_validation(tmp_path, field, value):
    p = tmp_path / 'config.yaml'
    p.write_text(f'{field}: {value}\n')
    with pytest.raises(ValueError):
        load_config(p)


def test_raw_tampering_rejected(tmp_path):
    from math_posttrain.data import digest
    from math_posttrain.protocol import require_clean_audit
    (tmp_path / 'raw').mkdir()
    (tmp_path / 'manifests').mkdir()
    manifest = {'source_hashes': {'train': digest([]), 'test':digest([])}}
    p = tmp_path / 'manifests/clean.json'
    p.write_text(json.dumps(manifest))
    (p.parent / 'clean.audit.json').write_text(json.dumps(dict(passed=True, manifest_hash=digest(manifest))))
    (tmp_path / 'raw/train.json').write_text('[{"tampered":true}]')
    with pytest.raises(ValueError, match='raw dataset hash'):
        require_clean_audit(p)
