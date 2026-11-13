import json
from pathlib import Path
import pytest
from math_posttrain.config import load_config
from math_posttrain.data import audit_path, digest
from math_posttrain.training import execute
from math_posttrain.evaluate import evaluate


@pytest.fixture
def audited_manifest(tmp_path):
    raw = tmp_path / 'raw'
    raw.mkdir()
    manifest_path = tmp_path / 'manifests' / 'splits_clean.json'
    manifest_path.parent.mkdir()
    records = [dict(question='What is one plus one?', answer='#### 2')]
    for split in ('train', 'test'):
        (raw / f'{split}.json').write_text(json.dumps(records))
    manifest = dict(source_hashes={split: digest(records) for split in ('train', 'test')})
    manifest_path.write_text(json.dumps(manifest))
    audit_path(manifest_path).write_text(json.dumps(dict(manifest_hash=digest(manifest), passed=True)))
    return manifest_path


def load_config_with_manifest(name, manifest, tmp_path):
    config_path = tmp_path / name
    template = Path('configs', name).read_text()
    config_path.write_text(template.replace('data/manifests/splits_clean.json', str(manifest)))
    return load_config(config_path)


def test_dry_run_no_output_or_download(audited_manifest, tmp_path):
    config = load_config_with_manifest('sft.yaml', audited_manifest, tmp_path)
    result = execute(config, 'dry-run-test', 'sft', dry_run=True)
    assert result['gpu_verified'] is False
    assert not Path('runs/dry-run-test').exists()


def test_grpo_requires_verified_sft(audited_manifest, tmp_path):
    config = load_config_with_manifest('grpo_a.yaml', audited_manifest, tmp_path)
    with pytest.raises(ValueError, match='requires frozen SFT'):
        execute(config, 'unused', 'grpo', dry_run=True)


def test_testset_gate():
    with pytest.raises(ValueError, match='frozen evaluation protocol'):
        evaluate({}, 'unused', 'official_test', 'unused')
