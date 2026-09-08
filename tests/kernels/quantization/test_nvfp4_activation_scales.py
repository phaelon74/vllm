# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.model_executor.layers.quantization.utils.nvfp4_activation_scales import (
    UNCALIBRATED_FILL_ATTR,
    UNCALIBRATED_FILL_KIND,
    fill_uncalibrated_nvfp4_activation_scale,
    fill_uncalibrated_nvfp4_activation_scales,
    unwritten_nvfp4_activation_scale,
)
from vllm.v1.sample.kld import (
    _activation_scale_substitution,
    _summarize_substitutions,
)


def test_unwritten_sentinel_is_nan_not_zero():
    scale = unwritten_nvfp4_activation_scale(4, 2)
    assert scale.shape == (4, 2)
    assert torch.isnan(scale).all()


def test_complete_scale_is_left_alone():
    scale = torch.tensor([1.0e-4, 2.0e-4, 3.0e-4])
    assert fill_uncalibrated_nvfp4_activation_scale(scale, name="w2") is None
    torch.testing.assert_close(scale, torch.tensor([1.0e-4, 2.0e-4, 3.0e-4]))


def test_partial_fill_uses_the_maximum_not_the_minimum():
    scale = torch.tensor([1.0e-4, 2.0e-3, float("nan")])
    record = fill_uncalibrated_nvfp4_activation_scale(
        scale, name="w2_input_global_scale"
    )
    assert record is not None
    assert record["kind"] == UNCALIBRATED_FILL_KIND
    assert record["unusable"] == 1
    assert record["fill_value"] == pytest.approx(2.0e-3)
    assert record["max_spread"] == pytest.approx(20.0)
    torch.testing.assert_close(scale, torch.tensor([1.0e-4, 2.0e-3, 2.0e-3]))


def test_zero_is_a_loaded_value_and_is_not_filled():
    scale = torch.tensor([0.0, 1.0e-4, float("nan")])
    record = fill_uncalibrated_nvfp4_activation_scale(scale, name="w2")
    assert record["unusable"] == 1
    torch.testing.assert_close(scale, torch.tensor([0.0, 1.0e-4, 1.0e-4]))


def test_exported_zero_survives_disclosure_alongside_a_fill():
    """A repaired gap must not hide a scale the checkpoint exported as zero."""
    layer = torch.nn.Module()
    layer.w2_input_global_scale = torch.nn.Parameter(
        torch.tensor([0.0, 1.0e-4, float("nan"), 2.0e-3])
    )
    layer.w2_input_scale = torch.nn.Parameter(torch.ones(4))
    fill_uncalibrated_nvfp4_activation_scales(layer)
    record = _activation_scale_substitution(layer)["w2_input_global_scale"]
    assert record["filled"]["unusable"] == 1
    assert record["unusable"] == 1


def test_fully_unwritten_scale_is_refused():
    scale = unwritten_nvfp4_activation_scale(8)
    with pytest.raises(ValueError, match="no finite positive slot"):
        fill_uncalibrated_nvfp4_activation_scale(scale, name="w2")


def test_layer_records_fill_evidence_before_values_look_finite():
    layer = torch.nn.Module()
    layer.w2_input_global_scale = torch.nn.Parameter(
        torch.tensor([1.0e-4, float("nan"), 5.0e-4])
    )
    records = fill_uncalibrated_nvfp4_activation_scales(layer)
    assert len(records) == 1
    stored = getattr(layer, UNCALIBRATED_FILL_ATTR)
    assert stored[0]["unusable"] == 1
    assert torch.isfinite(layer.w2_input_global_scale).all()


def test_inspector_reads_fill_attr_not_the_now_finite_tensor():
    layer = torch.nn.Module()
    layer.w13_input_global_scale = torch.nn.Parameter(torch.ones(4, 2))
    layer.w13_input_scale = torch.nn.Parameter(torch.ones(4))
    layer.w2_input_global_scale = torch.nn.Parameter(
        torch.tensor([1.0e-4, float("nan"), 2.0e-3, 3.0e-4])
    )
    layer.w2_input_scale = torch.nn.Parameter(torch.ones(4))
    fill_uncalibrated_nvfp4_activation_scales(layer)
    found = _activation_scale_substitution(layer)
    assert found["w2_input_global_scale"]["filled"]["unusable"] == 1
    assert found["w2_input_global_scale"]["filled"]["kind"] == UNCALIBRATED_FILL_KIND
    assert found["w13_input_global_scale"].get("filled") is None
    summary = _summarize_substitutions(
        [{"activation_scales": found}]
    )
    kinds = {item["kind"]: item for item in summary}
    assert "uncalibrated_experts_filled_from_layer_max" in kinds
    assert kinds["uncalibrated_experts_filled_from_layer_max"]["layers"] == 1
    assert kinds["uncalibrated_experts_filled_from_layer_max"]["unusable_slots"] == 1
