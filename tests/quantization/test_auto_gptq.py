# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests that the auto_gptq quantization method works correctly.

Run `pytest tests/quantization/test_auto_gptq.py -v -s`.
"""

from types import SimpleNamespace

import pytest
import torch

import vllm.envs as envs
import vllm.model_executor.layers.quantization.auto_gptq as auto_gptq
from tests.quantization.utils import is_quant_method_supported
from vllm.model_executor.kernels.linear import (
    MPLinearLayerConfig,
    choose_mp_linear_kernel,
)
from vllm.model_executor.kernels.linear.mixed_precision.emulation import (
    EmulationWNA16LinearKernel,
)
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.quantization.auto_gptq import (
    AutoGPTQConfig,
    AutoGPTQLinearMethod,
    AutoGPTQMoEMethod,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_packed_zero,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    gptq_pack,
    gptq_quantize_weights,
)
from vllm.scalar_type import scalar_types

PROMPT = "On the surface of Mars, we found"

MODELS = [
    "TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ",
]


@pytest.mark.skipif(
    not is_quant_method_supported("auto_gptq"),
    reason="auto_gptq is not supported on this GPU type.",
)
@pytest.mark.parametrize("model_id", MODELS)
def test_auto_gptq_quantization_method(vllm_runner, model_id: str, monkeypatch):
    """Test that quantization='auto_gptq' loads and runs correctly."""
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    with vllm_runner(
        model_id,
        dtype=torch.float16,
        quantization="auto_gptq",
        max_model_len=2048,
        enforce_eager=True,
    ) as llm:

        def check_model(model):
            for name, submodule in model.named_modules():
                if name == "model.layers.0.self_attn.qkv_proj":
                    assert isinstance(submodule.quant_method, AutoGPTQLinearMethod)
                    break

        llm.apply_model(check_model)

        outputs = llm.generate_greedy([PROMPT], max_tokens=8)
        assert outputs
        assert len(outputs[0][1]) > 0


def test_auto_gptq_config_get_name():
    """Test that AutoGPTQConfig.get_name() returns 'auto_gptq'."""
    assert AutoGPTQConfig.get_name() == "auto_gptq"


def test_auto_gptq_config_allows_partial_expert_group(monkeypatch):
    captured = {}
    sentinel = SimpleNamespace(input_dtype=None)

    def supports(layer, group_size, **kwargs):
        captured.update(kwargs)
        return True

    monkeypatch.setattr(auto_gptq, "check_moe_marlin_supports_layer", supports)
    monkeypatch.setattr(
        auto_gptq,
        "get_moe_quant_method",
        lambda *args, **kwargs: sentinel,
    )
    monkeypatch.setattr(auto_gptq, "get_marlin_input_dtype", lambda prefix: None)
    config = AutoGPTQConfig(4, 128, False, True, False, {}, {})
    layer = object.__new__(RoutedExperts)

    assert config.get_quant_method(layer, "model.layers.0.moe") is sentinel
    assert captured == {
        "allow_tile_padding": True,
        "allow_group_padding": True,
    }


def test_auto_gptq_moe_creates_zero_initialized_expert_biases():
    method = object.__new__(AutoGPTQMoEMethod)
    method.quant_config = AutoGPTQConfig(4, 128, False, True, False, {}, {})
    method.input_dtype = None
    method.experts_cls = None
    method.moe = SimpleNamespace(
        w13_num_shards=2,
        intermediate_size_per_partition_unpadded=704,
    )
    layer = torch.nn.Module()

    method.create_weights(
        layer=layer,
        num_experts=2,
        hidden_size=8,
        intermediate_size_per_partition=4,
        params_dtype=torch.float16,
        intermediate_size_full=4,
        weight_loader=lambda *args, **kwargs: None,
    )

    assert layer.w13_bias.shape == (2, 8)
    assert layer.w2_bias.shape == (2, 8)
    assert torch.count_nonzero(layer.w13_bias) == 0
    assert torch.count_nonzero(layer.w2_bias) == 0


def test_auto_gptq_moe_allocates_exporter_padded_reduction():
    method = object.__new__(AutoGPTQMoEMethod)
    method.quant_config = AutoGPTQConfig(4, 128, False, True, False, {}, {})
    method.input_dtype = None
    method.experts_cls = None
    method.moe = SimpleNamespace(w13_num_shards=2)
    layer = torch.nn.Module()

    method.create_weights(
        layer=layer,
        num_experts=2,
        hidden_size=256,
        intermediate_size_per_partition=704,
        params_dtype=torch.float16,
        intermediate_size_full=704,
        weight_loader=lambda *args, **kwargs: None,
    )

    assert layer.w2_qweight.shape == (2, 96, 256)
    assert layer.w2_scales.shape == (2, 6, 256)
    assert layer.w2_qzeros.shape == (2, 6, 32)
    assert layer.w2_g_idx.shape == (2, 768)
    assert layer.w2_g_idx_sort_indices.shape == (2, 768)
    assert layer.w2_qweight.expected_shard_sizes == (88, 96)
    assert layer.w2_scales.expected_shard_sizes == (6,)
    packed_zero = marlin_packed_zero(method.quant_config.quant_type)
    assert torch.all(layer.w2_qweight == packed_zero)

    loaded = torch.randint(
        torch.iinfo(torch.int32).min,
        torch.iinfo(torch.int32).max,
        (88, 256),
        dtype=torch.int32,
    )
    destination = RoutedExperts._narrow_expert_data_for_padding(
        layer.w2_qweight.data[0],
        loaded,
        hidden_dim=1,
        shard_dim=0,
    )
    destination.copy_(loaded)
    method._fill_w2_qweight_padding(layer)
    assert torch.equal(layer.w2_qweight[0, :88], loaded)
    assert torch.all(layer.w2_qweight[:, 88:] == packed_zero)


def test_auto_gptq_linear_keeps_packed_rows_and_partial_scale_group(monkeypatch):
    class DummyKernel:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.auto_gptq."
        "choose_mp_linear_kernel",
        lambda config: DummyKernel,
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    method = object.__new__(AutoGPTQLinearMethod)
    method.quant_config = AutoGPTQConfig(4, 128, False, True, False, {}, {})
    method.input_dtype = None
    method.quant_type = method.quant_config.quant_type
    layer = torch.nn.Module()

    method.create_weights(
        layer=layer,
        input_size_per_partition=2112,
        output_partition_sizes=[256],
        input_size=2112,
        output_size=256,
        params_dtype=torch.float16,
        weight_loader=lambda *args, **kwargs: None,
    )

    assert layer.qweight.shape == (264, 256)
    assert layer.scales.shape == (17, 256)
    assert layer.qzeros.shape == (17, 32)
    assert layer.g_idx.shape == (2112,)


def test_batch_invariant_gptq_linear_selects_emulation(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    config = MPLinearLayerConfig(
        full_weight_shape=(2112, 256),
        partition_weight_shape=(2112, 256),
        weight_type=scalar_types.uint4b8,
        act_type=torch.bfloat16,
        group_size=128,
        zero_points=False,
        has_g_idx=False,
    )

    assert (
        choose_mp_linear_kernel(config, compute_capability=0)
        is EmulationWNA16LinearKernel
    )


def test_gptq_linear_emulation_dequantizes_partial_final_group():
    size_k, size_n, group_size = 24, 16, 16
    config = MPLinearLayerConfig(
        full_weight_shape=(size_k, size_n),
        partition_weight_shape=(size_k, size_n),
        weight_type=scalar_types.uint4b8,
        act_type=torch.bfloat16,
        group_size=group_size,
        zero_points=False,
        has_g_idx=False,
    )
    kernel = EmulationWNA16LinearKernel(
        config,
        w_q_param_name="qweight",
        w_s_param_name="scales",
    )
    padded_weight = torch.zeros(32, size_n, dtype=torch.bfloat16)
    padded_weight[:size_k] = torch.randn(size_k, size_n, dtype=torch.bfloat16)
    reference, quantized, scales, _, _ = gptq_quantize_weights(
        padded_weight,
        scalar_types.uint4b8,
        group_size,
        act_order=False,
    )
    layer = torch.nn.Module()
    layer.register_parameter(
        "qweight",
        torch.nn.Parameter(
            gptq_pack(quantized[:size_k], 4, size_k, size_n),
            requires_grad=False,
        ),
    )
    layer.register_parameter(
        "scales",
        torch.nn.Parameter(scales, requires_grad=False),
    )

    kernel.process_weights_after_loading(layer)
    inputs = torch.randn(3, size_k, dtype=torch.bfloat16)
    output = kernel.apply_weights(layer, inputs)

    assert layer.qweight.shape == (size_k, size_n)
    torch.testing.assert_close(output, inputs @ reference[:size_k])


def test_routed_experts_loads_per_expert_biases():
    class Loader:
        quant_config = None
        quant_method = object()
        moe_config = SimpleNamespace(
            is_act_and_mul=True,
            tp_rank=0,
            moe_parallel_config=SimpleNamespace(tp_size=1),
        )
        _get_hidden_dim = staticmethod(RoutedExperts._get_hidden_dim)
        _narrow_expert_data_for_padding = staticmethod(
            RoutedExperts._narrow_expert_data_for_padding
        )
        _load_w13 = RoutedExperts._load_w13
        _loaded_expert_biases = set()

        @staticmethod
        def _map_global_expert_id_to_local_expert_id(expert_id):
            return expert_id

    loader = Loader()
    w13_bias = torch.nn.Parameter(torch.zeros(1, 8), requires_grad=False)
    w2_bias = torch.nn.Parameter(torch.zeros(1, 4), requires_grad=False)

    for shard_id, loaded in (
        ("w1", torch.tensor([1.0, 2.0, 3.0, 4.0])),
        ("w3", torch.tensor([5.0, 6.0, 7.0, 8.0])),
    ):
        assert RoutedExperts.weight_loader(
            loader,
            w13_bias,
            loaded,
            weight_name="model.layers.0.mlp.experts.w13_bias",
            shard_id=shard_id,
            expert_id=0,
            return_success=True,
        )

    assert RoutedExperts.weight_loader(
        loader,
        w2_bias,
        torch.tensor([9.0, 10.0, 11.0, 12.0]),
        weight_name="model.layers.0.mlp.experts.w2_bias",
        shard_id="w2",
        expert_id=0,
        return_success=True,
    )
    assert torch.equal(w13_bias, torch.arange(1, 9, dtype=torch.float32).reshape(1, 8))
    assert torch.equal(w2_bias, torch.arange(9, 13, dtype=torch.float32).reshape(1, 4))
    assert loader._loaded_expert_biases == {"w13_bias", "w2_bias"}
