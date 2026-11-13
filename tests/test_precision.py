import pytest
from math_posttrain.precision import validate_eval_precision


def test_grpo_evaluation_must_match_weight_precision():
    with pytest.raises(ValueError, match='model_dtype mismatch'):
        validate_eval_precision({'model_dtype':'bfloat16'}, {'kind':'grpo','model_dtype':'float32'})
    assert validate_eval_precision({'model_dtype':'float32'}, {'kind':'grpo','model_dtype':'float32'}) == 'float32'


def test_baseline_can_be_explicitly_reevaluated_in_common_precision():
    assert validate_eval_precision({'model_dtype':'float32'}, {'kind':'sft'}) == 'float32'
