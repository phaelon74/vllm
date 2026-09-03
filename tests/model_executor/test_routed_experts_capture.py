# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import types
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.config import ModelConfig, VllmConfig
from vllm.config.compilation import CompilationMode
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.forward_context import override_forward_context
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.fused_moe.experts.trtllm_fp8_moe import (
    TrtLlmFp8ExpertsMonolithic,
)
from vllm.model_executor.layers.fused_moe.forced_routing import (
    prepare_flashinfer_forced_topk_weights,
)
from vllm.model_executor.layers.fused_moe.forced_routing import (
    FORCED_ROUTING_KEY,
    ForcedRouting,
)
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsCapturer,
    RoutedExpertsManager,
    bind_routed_experts_capturer,
    get_routed_experts_attn_gid,
)
from vllm.model_executor.layers.fused_moe.router.base_router import (
    BaseRouter,
    compute_forced_routing_weights,
)
from vllm.model_executor.layers.fused_moe.router.custom_routing_router import (
    CustomRoutingRouter,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
    FusedTopKRouter,
)
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
    GroupedTopKRouter,
)
from vllm.model_executor.layers.fused_moe.router.router_factory import (
    create_fused_moe_router,
)
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.transformers_utils.model_arch_config_convertor import (
    ModelArchConfigConvertorBase,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)

pytestmark = pytest.mark.cpu_test

_REC_MODULE = "vllm.model_executor.layers.fused_moe.routed_experts_capturer"


def _capturer_with_buffer(
    *,
    max_tokens: int = 8,
    num_layers: int = 4,
    num_experts_per_tok: int = 2,
    dp_rank: int = 0,
    tp_size: int = 1,
) -> RoutedExpertsCapturer:
    # Bypass __init__ so the test can use a CPU buffer and skip the
    # VllmConfig dependency. The CUDA device-tensor allocation in the
    # real constructor is not what we are exercising here.
    c = RoutedExpertsCapturer.__new__(RoutedExpertsCapturer)
    c.dp_rank = dp_rank
    c.tp_size = tp_size
    c.device_buffer = torch.full(
        (max_tokens, num_layers, num_experts_per_tok),
        -1,
        dtype=torch.int32,
    )
    return c


class DummyRouter(BaseRouter):
    @property
    def routing_method_type(self) -> RoutingMethodType:
        return RoutingMethodType.FUSED_TOPK

    def _compute_routing(
        self, hidden_states, router_logits, indices_type, *, input_ids=None
    ):
        topk_ids = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
        topk_weights = torch.ones_like(topk_ids, dtype=torch.float32)
        return topk_weights, topk_ids

    def _apply_eplb_mapping(self, topk_ids: torch.Tensor) -> torch.Tensor:
        # Make mapping observable without requiring CUDA EPLB path.
        return topk_ids + 10

    def _compute_forced_weights(
        self, hidden_states, router_logits, forced_topk_ids, *, input_ids=None
    ):
        return torch.ones_like(forced_topk_ids, dtype=torch.float32)


def _make_router(eplb_state: EplbLayerState | None = None) -> DummyRouter:
    return DummyRouter(
        top_k=2,
        global_num_experts=16,
        eplb_state=eplb_state,
    )


def _make_modular_routed_experts():
    return types.SimpleNamespace(
        quant_method=types.SimpleNamespace(is_monolithic=False),
    )


def _make_model_config(hf_config):
    num_experts_per_token = ModelArchConfigConvertorBase(
        hf_config, hf_config
    ).get_num_experts_per_token()
    model_config = SimpleNamespace(
        hf_text_config=hf_config,
        model_arch_config=SimpleNamespace(
            num_experts_per_token=num_experts_per_token,
        ),
    )
    model_config.get_num_experts = lambda: hf_config.num_experts
    model_config.get_num_experts_per_tok = lambda: (
        ModelConfig.get_num_experts_per_tok(model_config)
    )
    model_config.get_total_num_hidden_layers = lambda: hf_config.num_hidden_layers
    return model_config


def test_routed_experts_manager_uses_gemma4_top_k_experts():
    hf_config = SimpleNamespace(
        num_experts=8,
        top_k_experts=2,
        num_hidden_layers=3,
    )
    vllm_config = SimpleNamespace(model_config=_make_model_config(hf_config))
    kv_cache_spec = FullAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layer"], kv_cache_spec)],
    )

    manager = RoutedExpertsManager(vllm_config, kv_cache_config)

    assert manager.routed_experts_by_slot.shape == (8, 3, 2)


def test_routed_experts_manager_uses_kimi_k3_experts_per_token():
    hf_config = SimpleNamespace(
        num_experts=8,
        num_experts_per_token=2,
        num_hidden_layers=3,
    )
    vllm_config = SimpleNamespace(model_config=_make_model_config(hf_config))
    kv_cache_spec = FullAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layer"], kv_cache_spec)],
    )

    manager = RoutedExpertsManager(vllm_config, kv_cache_config)

    assert manager.routed_experts_by_slot.shape == (8, 3, 2)


def test_base_router_capture_pre_eplb_mapping():
    router = _make_router()
    captured = []

    def capture_fn(ids):
        captured.append(ids.clone())

    router.set_capture_fn(capture_fn)
    topk_weights, topk_ids = router.select_experts(
        hidden_states=torch.empty(1),
        router_logits=torch.empty(1),
    )

    assert topk_weights.shape == topk_ids.shape
    assert len(captured) == 1
    assert torch.equal(captured[0], torch.tensor([[1, 2], [3, 4]]))
    assert torch.equal(topk_ids, torch.tensor([[11, 12], [13, 14]]))


def test_base_router_capture_with_eplb_enabled():
    eplb_state = EplbLayerState()
    eplb_state.expert_load_view = torch.zeros(32, dtype=torch.int64)
    eplb_state.logical_to_physical_map = torch.arange(32).view(32, 1)
    eplb_state.logical_replica_count = torch.ones(32, dtype=torch.int64)
    eplb_state.should_record_tensor = torch.ones((), dtype=torch.bool)
    eplb_state.num_unpadded_tokens_tensors = [torch.tensor(0, dtype=torch.int32)]
    router = _make_router(eplb_state=eplb_state)

    captured = []

    def capture_fn(ids):
        captured.append(ids.clone())

    router.set_capture_fn(capture_fn)
    _, topk_ids = router.select_experts(
        hidden_states=torch.empty(1),
        router_logits=torch.empty(1),
    )

    assert len(captured) == 1
    # Capture should see logical ids pre-EPLB mapping.
    assert torch.equal(captured[0], torch.tensor([[1, 2], [3, 4]]))
    # Our DummyRouter mapping adds +10.
    assert torch.equal(topk_ids, torch.tensor([[11, 12], [13, 14]]))


@pytest.mark.parametrize("scoring_func", ["softmax", "sigmoid"])
@pytest.mark.parametrize("renormalize", [False, True])
def test_forced_topk_uses_student_weights(scoring_func, renormalize):
    router = FusedTopKRouter(
        top_k=2,
        global_num_experts=4,
        scoring_func=scoring_func,
        renormalize=renormalize,
    )
    logits = torch.tensor([[3.0, 1.0, -1.0, 0.0]])
    forced_ids = torch.tensor([[2, 0]], dtype=torch.int32)

    weights, selected_ids = router.select_forced_experts(
        hidden_states=torch.empty(1, 1),
        router_logits=logits,
        forced_topk_ids=forced_ids,
    )

    scores = (
        torch.softmax(logits, dim=-1)
        if scoring_func == "softmax"
        else torch.sigmoid(logits)
    )
    expected = scores.gather(1, forced_ids.to(torch.int64))
    if renormalize:
        expected = expected / expected.sum(dim=-1, keepdim=True)
    assert torch.equal(selected_ids, forced_ids)
    torch.testing.assert_close(weights, expected)


def test_forced_ids_are_captured_before_eplb_mapping():
    router = _make_router()
    captured = []
    router.set_capture_fn(lambda ids: captured.append(ids.clone()))
    forced_ids = torch.tensor([[5, 2]], dtype=torch.int32)

    weights, mapped_ids = router.select_forced_experts(
        hidden_states=torch.empty(1, 1),
        router_logits=torch.empty(1, 16),
        forced_topk_ids=forced_ids,
    )

    assert weights.tolist() == [[1.0, 1.0]]
    assert len(captured) == 1
    assert torch.equal(captured[0], forced_ids)
    assert torch.equal(mapped_ids, forced_ids + 10)


def test_forced_routing_refuses_eplb_remapping():
    router = FusedTopKRouter(
        top_k=1,
        global_num_experts=2,
        eplb_state=Mock(),
    )
    with pytest.raises(ValueError, match="requires EPLB to be disabled"):
        router.select_forced_experts(
            hidden_states=torch.empty(1, 1),
            router_logits=torch.tensor([[1.0, 0.0]]),
            forced_topk_ids=torch.tensor([[0]], dtype=torch.int32),
        )


def test_grouped_forced_weights_ignore_selection_bias_and_apply_scale():
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0]])
    forced_ids = torch.tensor([[0]], dtype=torch.int32)
    common = dict(
        top_k=1,
        global_num_experts=4,
        num_expert_group=2,
        topk_group=2,
        scoring_func="sigmoid",
        renormalize=False,
        routed_scaling_factor=3.0,
    )
    unbiased_router = GroupedTopKRouter(**common)
    biased_router = GroupedTopKRouter(
        **common,
        e_score_correction_bias=torch.tensor([0.0, 0.0, 0.0, 5.0]),
    )

    unbiased_weights, _ = unbiased_router.select_forced_experts(
        hidden_states=torch.empty(1, 1),
        router_logits=logits,
        forced_topk_ids=forced_ids,
    )
    biased_weights, _ = biased_router.select_forced_experts(
        hidden_states=torch.empty(1, 1),
        router_logits=logits,
        forced_topk_ids=forced_ids,
    )

    expected = torch.sigmoid(logits[:, :1]) * 3.0
    torch.testing.assert_close(unbiased_weights, expected)
    torch.testing.assert_close(biased_weights, expected)


def test_custom_forced_callback_applies_gemma_per_expert_scale():
    per_expert_scale = torch.tensor([2.0, 3.0, 0.5, 4.0])

    def natural_routing(hidden_states, gating_output, topk, renormalize):
        weights, ids = torch.topk(gating_output, topk, dim=-1)
        return weights, ids

    def gemma_forced_routing(
        hidden_states, gating_output, forced_topk_ids, renormalize
    ):
        return compute_forced_routing_weights(
            gating_output,
            forced_topk_ids,
            scoring_func="softmax",
            renormalize=True,
            per_expert_scale=per_expert_scale,
        )

    router = create_fused_moe_router(
        top_k=2,
        global_num_experts=4,
        custom_routing_function=natural_routing,
        custom_forced_routing_function=gemma_forced_routing,
    )
    assert isinstance(router, CustomRoutingRouter)
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0]])
    forced_ids = torch.tensor([[1, 3]], dtype=torch.int32)

    weights, selected_ids = router.select_forced_experts(
        hidden_states=torch.empty(1, 1),
        router_logits=logits,
        forced_topk_ids=forced_ids,
    )

    selected_scores = torch.softmax(logits, dim=-1).gather(
        1, forced_ids.to(torch.int64)
    )
    selected_scores /= selected_scores.sum(dim=-1, keepdim=True)
    expected = selected_scores * per_expert_scale[forced_ids]
    assert torch.equal(selected_ids, forced_ids)
    torch.testing.assert_close(weights, expected)


def test_custom_router_without_forced_callback_refuses():
    def natural_routing(hidden_states, gating_output, topk, renormalize):
        return torch.topk(gating_output, topk, dim=-1)

    router = CustomRoutingRouter(
        top_k=1,
        global_num_experts=2,
        custom_routing_function=natural_routing,
    )

    with pytest.raises(ValueError, match="custom_forced_routing_function"):
        router.select_forced_experts(
            hidden_states=torch.empty(1, 1),
            router_logits=torch.tensor([[1.0, 0.0]]),
            forced_topk_ids=torch.tensor([[1]], dtype=torch.int32),
        )


def _runner_for_forced_routing(router, *, monolithic=False):
    runner = MoERunner.__new__(MoERunner)
    runner._shared_experts = None
    runner.layer_name = "model.layers.0.mlp.experts"
    runner.moe_config = SimpleNamespace(
        experts_per_token=2,
        num_logical_experts=4,
    )
    quant_method = SimpleNamespace(
        is_monolithic=monolithic,
        topk_indices_dtype=torch.int64,
        apply_monolithic_routed=Mock(return_value=torch.empty(1, 1)),
    )
    runner.router = router
    runner.routed_experts = SimpleNamespace(
        quant_method=quant_method,
        forward_modular=Mock(return_value=torch.empty(1, 1)),
        forward_monolithic=Mock(return_value=torch.empty(1, 1)),
    )
    return runner


def test_modular_runner_dispatches_forced_ids_with_student_weights():
    router = FusedTopKRouter(
        top_k=2,
        global_num_experts=4,
        scoring_func="softmax",
        renormalize=True,
    )
    runner = _runner_for_forced_routing(router)
    logits = torch.tensor([[3.0, 1.0, -1.0, 0.0]])
    forced_ids = torch.tensor([[[2, 0]]], dtype=torch.int32)
    ctx = SimpleNamespace(
        additional_kwargs={FORCED_ROUTING_KEY: ForcedRouting(expert_ids=forced_ids)}
    )

    with override_forward_context(ctx):
        runner._apply_quant_method(torch.empty(1, 1), logits, None)

    call = runner.routed_experts.forward_modular.call_args.kwargs
    assert torch.equal(call["topk_ids"], forced_ids[:, 0].to(torch.int64))
    selected_scores = torch.softmax(logits, dim=-1).gather(
        1, forced_ids[:, 0].to(torch.int64)
    )
    expected = selected_scores / selected_scores.sum(dim=-1, keepdim=True)
    torch.testing.assert_close(call["topk_weights"], expected)


def test_modular_runner_uses_normal_router_without_override():
    router = Mock()
    natural_weights = torch.tensor([[0.75, 0.25]])
    natural_ids = torch.tensor([[0, 1]], dtype=torch.int64)
    router.select_experts.return_value = natural_weights, natural_ids
    runner = _runner_for_forced_routing(router)
    ctx = SimpleNamespace(additional_kwargs={})

    with override_forward_context(ctx):
        runner._apply_quant_method(
            torch.empty(1, 1),
            torch.empty(1, 4),
            None,
        )

    router.select_experts.assert_called_once()
    router.select_forced_experts.assert_not_called()
    call = runner.routed_experts.forward_modular.call_args.kwargs
    assert call["topk_weights"] is natural_weights
    assert call["topk_ids"] is natural_ids


def test_monolithic_runner_dispatches_exact_forced_routing():
    router = FusedTopKRouter(
        top_k=2,
        global_num_experts=4,
        scoring_func="softmax",
        renormalize=True,
    )
    runner = _runner_for_forced_routing(router, monolithic=True)
    logits = torch.tensor([[3.0, 1.0, -1.0, 0.0]])
    forced_ids = torch.tensor([[[2, 0]]], dtype=torch.int64)
    ctx = SimpleNamespace(
        additional_kwargs={FORCED_ROUTING_KEY: ForcedRouting(expert_ids=forced_ids)}
    )

    with override_forward_context(ctx):
        runner._apply_quant_method(
            torch.empty(1, 1),
            logits,
            None,
        )

    call = runner.routed_experts.quant_method.apply_monolithic_routed.call_args.kwargs
    assert call["layer"] is runner.routed_experts
    assert call["topk_ids"].dtype == torch.int32
    assert call["topk_ids"].is_contiguous()
    assert torch.equal(call["topk_ids"], forced_ids[:, 0].to(torch.int32))
    expected = torch.softmax(logits, dim=-1).gather(1, forced_ids[:, 0])
    expected /= expected.sum(dim=-1, keepdim=True)
    assert call["topk_weights"].dtype == torch.float32
    assert call["topk_weights"].is_contiguous()
    torch.testing.assert_close(call["topk_weights"], expected)
    runner.routed_experts.forward_monolithic.assert_not_called()


def test_monolithic_runner_preserves_natural_path():
    runner = _runner_for_forced_routing(Mock(), monolithic=True)
    ctx = SimpleNamespace(additional_kwargs={})

    with override_forward_context(ctx):
        runner._apply_quant_method(
            torch.empty(1, 1),
            torch.empty(1, 4),
            None,
        )

    runner.routed_experts.forward_monolithic.assert_called_once()
    runner.routed_experts.quant_method.apply_monolithic_routed.assert_not_called()


def test_monolithic_kernel_forwards_exact_routing_tensors():
    parallel_config = SimpleNamespace(
        dp_size=1,
        ep_size=1,
        pcp_size=1,
        is_sequence_parallel=False,
    )
    moe_config = SimpleNamespace(
        moe_parallel_config=parallel_config,
        experts_per_token=2,
        should_defer_moe_finalize=lambda num_tokens: False,
    )
    prepare_finalize = SimpleNamespace(
        prepare=Mock(
            side_effect=lambda hidden_states, **kwargs: (
                hidden_states,
                None,
                kwargs["router_logits"],
            )
        ),
        finalize=Mock(side_effect=lambda output: output),
    )
    routed_output = torch.empty(2, 4)
    experts = SimpleNamespace(
        moe_config=moe_config,
        quant_config=SimpleNamespace(),
        expects_unquantized_inputs=True,
        apply_routed=Mock(return_value=routed_output),
    )
    impl = mk.FusedMoEKernelMonolithicImpl(prepare_finalize, experts)
    hidden_states = torch.empty(2, 4)
    topk_ids = torch.tensor([[2, 0], [1, 3]], dtype=torch.int32)
    topk_weights = torch.tensor([[0.75, 0.25], [0.125, 0.875]], dtype=torch.float32)

    output = impl.apply_routed(
        hidden_states=hidden_states,
        w1=torch.empty(1),
        w2=torch.empty(1),
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=Mock(),
        global_num_experts=4,
        expert_map=None,
        apply_router_weight_on_input=False,
    )

    call = experts.apply_routed.call_args.kwargs
    assert call["topk_ids"] is topk_ids
    assert call["topk_weights"] is topk_weights
    assert output is routed_output


def test_trtllm_fp8_forced_route_uses_public_packed_api():
    import flashinfer

    experts = object.__new__(TrtLlmFp8ExpertsMonolithic)
    experts.quant_config = SimpleNamespace(
        block_shape=[1, 32],
        w1_scale=torch.empty(1),
        w2_scale=torch.empty(1),
    )
    experts.gemm1_alpha = None
    experts.gemm1_beta = None
    experts.gemm1_clamp_limit = None
    experts.intermediate_size_per_partition = 4
    experts.ep_rank = 0
    experts.local_num_experts = 4
    experts.routing_method_type = RoutingMethodType.RenormalizeNaive
    experts.moe_config = SimpleNamespace()

    topk_ids = torch.tensor([[2, 0]], dtype=torch.int32)
    topk_weights = torch.tensor([[0.75, 0.25]], dtype=torch.float32)
    packed_routes = torch.tensor([[1, 2]], dtype=torch.int32)
    expected = torch.empty(1, 4)
    target = "trtllm_fp8_block_scale_routed_moe"

    with (
        patch.object(
            flashinfer.fused_moe,
            target,
            return_value=expected,
            create=True,
        ) as routed_moe,
        patch(
            "vllm.model_executor.layers.fused_moe.experts."
            "trtllm_fp8_moe.fi_moe_largest_bucket",
            return_value=1,
        ),
        patch(
            "vllm.model_executor.layers.fused_moe.experts."
            "trtllm_fp8_moe.trtllm_moe_pack_topk_ids_weights",
            return_value=packed_routes,
        ) as pack_routes,
    ):
        output = experts._apply_block_scale_routed(
            hidden_states=torch.empty(1, 4),
            w1=torch.empty(1),
            w2=torch.empty(1),
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=MoEActivation.SILU,
            global_num_experts=4,
            a1q_scale=torch.empty(1),
            apply_router_weight_on_input=False,
        )

    routed = routed_moe.call_args.kwargs
    pack_routes.assert_called_once_with(topk_ids, topk_weights)
    torch.testing.assert_close(routed["topk_ids"], packed_routes)
    assert "routing_logits" not in routed
    assert routed["routing_method_type"] == RoutingMethodType.RenormalizeNaive
    assert output is expected


def test_flashinfer_forced_weights_use_selected_logits_softmax():
    router_logits = torch.tensor([[8.0, -3.0, 5.0, 1.0]], dtype=torch.bfloat16)
    topk_ids = torch.tensor([[2, 0]], dtype=torch.int32)
    fallback = torch.zeros(1, 2)

    actual = prepare_flashinfer_forced_topk_weights(
        router_logits=router_logits,
        topk_ids=topk_ids,
        topk_weights=fallback,
        routing_method=RoutingMethodType.RenormalizeNaive,
    )

    expected = torch.softmax(torch.tensor([[5.0, 8.0]]), dim=-1)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("dp_size", 2, "does not support DP"),
        ("ep_size", 2, "does not support DP"),
        ("pcp_size", 2, "does not support DP"),
        ("is_sequence_parallel", True, "does not support DP"),
    ],
)
def test_monolithic_kernel_rejects_distributed_forced_routing(field, value, match):
    parallel_config = SimpleNamespace(
        dp_size=1,
        ep_size=1,
        pcp_size=1,
        is_sequence_parallel=False,
    )
    setattr(parallel_config, field, value)
    experts = SimpleNamespace(
        moe_config=SimpleNamespace(
            moe_parallel_config=parallel_config,
            experts_per_token=2,
            should_defer_moe_finalize=lambda num_tokens: False,
        )
    )
    impl = mk.FusedMoEKernelMonolithicImpl(SimpleNamespace(), experts)

    with pytest.raises(NotImplementedError, match=match):
        impl.apply_routed(
            hidden_states=torch.empty(1, 4),
            w1=torch.empty(1),
            w2=torch.empty(1),
            topk_weights=torch.empty(1, 2),
            topk_ids=torch.empty(1, 2, dtype=torch.int32),
            activation=Mock(),
            global_num_experts=4,
            expert_map=None,
            apply_router_weight_on_input=False,
        )


def test_monolithic_kernel_rejects_deferred_finalize():
    experts = SimpleNamespace(
        moe_config=SimpleNamespace(
            moe_parallel_config=SimpleNamespace(
                dp_size=1,
                ep_size=1,
                pcp_size=1,
                is_sequence_parallel=False,
            ),
            should_defer_moe_finalize=lambda num_tokens: True,
        )
    )
    impl = mk.FusedMoEKernelMonolithicImpl(SimpleNamespace(), experts)

    with pytest.raises(NotImplementedError, match="deferred finalize"):
        impl.apply_routed(
            hidden_states=torch.empty(1, 4),
            w1=torch.empty(1),
            w2=torch.empty(1),
            topk_weights=torch.empty(1, 2),
            topk_ids=torch.empty(1, 2, dtype=torch.int32),
            activation=Mock(),
            global_num_experts=4,
            expert_map=None,
            apply_router_weight_on_input=False,
        )


def test_public_binding_only_visits_target_model(monkeypatch):
    class DummyFusedMoE:
        def __init__(self, layer_id):
            self.layer_id = layer_id
            self.router = _make_router()
            self._quant_method = _make_modular_routed_experts().quant_method

    target_module = DummyFusedMoE(layer_id=7)
    draft_module = DummyFusedMoE(layer_id=0)

    import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer

    monkeypatch.setattr(fused_moe_layer, "MoERunner", DummyFusedMoE)
    calls = []
    capturer = types.SimpleNamespace(capture=lambda *args: calls.append(args))

    bind_routed_experts_capturer(
        types.SimpleNamespace(modules=lambda: [target_module]), capturer
    )

    assert target_module.router.capture_fn is not None
    assert draft_module.router.capture_fn is None
    topk_ids = torch.tensor([[5, 6]])
    target_module.router.capture_fn(topk_ids)
    assert calls == [(7, topk_ids)]


def test_public_binding_rejects_monolithic_without_replay_support(monkeypatch):
    class DummyFusedMoE:
        def __init__(self):
            self.layer_id = 3
            self.router = _make_router()
            # Use a concrete monolithic expert and override its capability
            # instead of instantiating the abstract base class directly.
            from vllm.model_executor.layers.fused_moe.experts.cpu_moe import (
                CPUExpertsFp8,
            )

            fused_experts = CPUExpertsFp8.__new__(CPUExpertsFp8)
            self.routed_experts = types.SimpleNamespace(
                quant_method=types.SimpleNamespace(
                    is_monolithic=True,
                    moe_kernel=types.SimpleNamespace(
                        impl=types.SimpleNamespace(fused_experts=fused_experts)
                    ),
                )
            )
            self._quant_method = self.routed_experts.quant_method
            self._quant_method.moe_kernel.impl.fused_experts = fused_experts
            fused_experts.supports_routing_replay_capture = lambda: False

    class DummyCapturer:
        def capture(self, layer_id, topk_ids):
            pass

    dummy_module = DummyFusedMoE()
    import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer

    monkeypatch.setattr(fused_moe_layer, "MoERunner", DummyFusedMoE)

    with pytest.raises(ValueError, match="monolithic MoE kernel"):
        bind_routed_experts_capturer(
            types.SimpleNamespace(modules=lambda: [dummy_module]), DummyCapturer()
        )


def test_routed_experts_capturer_single_dp_no_metadata():
    """dp_metadata is None: capture writes the full topk_ids rows."""
    capturer = _capturer_with_buffer(dp_rank=0)
    topk = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int32)
    ctx = SimpleNamespace(dp_metadata=None)
    with patch(f"{_REC_MODULE}.get_forward_context", return_value=ctx):
        capturer.capture(layer_id=0, topk_ids=topk)
    assert torch.equal(capturer.device_buffer[:3, 0, :], topk)
    assert capturer.device_buffer[3, 0, 0].item() == -1


def test_routed_experts_capturer_dp_naive_concatenated_all_ranks():
    """n == sum(num_tokens_dp): slice this rank's segment from concatenated topk."""
    capturer = _capturer_with_buffer(dp_rank=1)
    num_tokens_dp = torch.tensor([2, 3], dtype=torch.int32)
    ctx = SimpleNamespace(
        dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=num_tokens_dp)
    )
    # Concatenated order: rank0 rows then rank1 rows.
    topk = torch.tensor(
        [[0, 1], [2, 3], [10, 11], [12, 13], [14, 15]], dtype=torch.int32
    )
    with patch(f"{_REC_MODULE}.get_forward_context", return_value=ctx):
        capturer.capture(layer_id=0, topk_ids=topk)
    want = topk[2:5]
    assert torch.equal(capturer.device_buffer[:3, 0, :], want)


def test_routed_experts_capturer_dp_modular_local_tokens():
    """n == token_num_per_dp: topk is already local to this DP rank."""
    capturer = _capturer_with_buffer(dp_rank=1)
    num_tokens_dp = torch.tensor([2, 3], dtype=torch.int32)
    ctx = SimpleNamespace(
        dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=num_tokens_dp)
    )
    topk = torch.tensor([[10, 11], [12, 13], [14, 15]], dtype=torch.int32)
    with patch(f"{_REC_MODULE}.get_forward_context", return_value=ctx):
        capturer.capture(layer_id=0, topk_ids=topk)
    assert torch.equal(capturer.device_buffer[:3, 0, :], topk)


def test_routed_experts_capturer_dp_unexpected_batch_raises():
    """Mismatch between topk batch dim and DP layout: fail fast."""
    capturer = _capturer_with_buffer(dp_rank=0)
    num_tokens_dp = torch.tensor([2, 3], dtype=torch.int32)
    ctx = SimpleNamespace(
        dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=num_tokens_dp)
    )
    # total=5, local=2: n=1 matches neither naive (5) nor modular (2).
    topk = torch.tensor([[1, 2]], dtype=torch.int32)
    with (
        patch(f"{_REC_MODULE}.get_forward_context", return_value=ctx),
        pytest.raises(AssertionError, match="unexpected topk_ids batch dim"),
    ):
        capturer.capture(layer_id=0, topk_ids=topk)
    assert capturer.device_buffer[0, 0, 0].item() == -1


def test_routed_experts_attention_group_is_shared_and_fail_closed():
    """Both sides key routing data by this gid, so it must skip non-full-attention
    groups rather than defaulting to 0, and fail closed when none exists."""
    common = dict(num_kv_heads=1, head_size=1, dtype=torch.float32)
    config = SimpleNamespace(
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["swa_layer"],
                SlidingWindowMLASpec(block_size=4, sliding_window=8, **common),
            ),
            KVCacheGroupSpec(["full_layer"], FullAttentionSpec(block_size=4, **common)),
        ]
    )
    assert get_routed_experts_attn_gid(config) == 1

    with pytest.raises(ValueError, match="requires a full-attention KV cache group"):
        get_routed_experts_attn_gid(SimpleNamespace(kv_cache_groups=[]))


def test_routed_experts_attention_group_unwraps_uniform_type_specs():
    """DeepSeek-V4-shaped groups wrap their specs in ``UniformTypeKVCacheSpecs``.

    The wrapper is not a ``FullAttentionSpec``, so a bare isinstance check finds
    no group and fails closed on every worker. Unwrap semantics themselves are
    covered by ``test_is_full_attention_spec_*`` in tests/v1/core.
    """
    common = dict(num_kv_heads=1, head_size=1, dtype=torch.float32)
    swa_spec = SlidingWindowMLASpec(block_size=4, sliding_window=8, **common)
    mla_spec = MLAAttentionSpec(block_size=4, **common)
    config = SimpleNamespace(
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["swa_layer"],
                UniformTypeKVCacheSpecs(
                    block_size=4, kv_cache_specs={"swa_layer": swa_spec}
                ),
            ),
            KVCacheGroupSpec(
                ["mla_layer"],
                UniformTypeKVCacheSpecs(
                    block_size=4, kv_cache_specs={"mla_layer": mla_spec}
                ),
            ),
        ]
    )

    assert get_routed_experts_attn_gid(config) == 1

    swa_only = SimpleNamespace(kv_cache_groups=[config.kv_cache_groups[0]])
    with pytest.raises(ValueError, match="requires a full-attention KV cache group"):
        get_routed_experts_attn_gid(swa_only)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mrv2_async_output_returns_existing_routed_experts_field():
    from vllm.v1.outputs import ModelRunnerOutput, RoutedExpertsTensors
    from vllm.v1.worker.gpu.async_utils import AsyncOutput
    from vllm.v1.worker.gpu.sample.output import SamplerOutput

    routed_experts = RoutedExpertsTensors(
        routing_data=torch.arange(6, dtype=torch.int32, device="cuda").reshape(3, 1, 2),
        slot_mapping=torch.tensor([11, 12, 13], device="cuda"),
    )
    num_sampled = torch.tensor([1], dtype=torch.int32, device="cuda")
    sampler_output = SamplerOutput(
        sampled_token_ids=torch.tensor([[1]], device="cuda"),
        logprobs_tensors=None,
        num_nans=None,
        num_sampled=num_sampled,
        num_rejected=torch.tensor([0], dtype=torch.int32, device="cuda"),
    )
    output = AsyncOutput(
        model_runner_output=ModelRunnerOutput(req_ids=["req"], req_id_to_index={}),
        sampler_output=sampler_output,
        num_sampled_tokens=num_sampled,
        main_stream=torch.cuda.current_stream(),
        copy_stream=torch.cuda.Stream(),
        routed_experts=routed_experts,
    ).get_output()

    assert output.routed_experts is not None
    assert output.routed_experts.routing_data[:, 0, 0].tolist() == [0, 2, 4]
    assert output.routed_experts.slot_mapping.tolist() == [11, 12, 13]


@pytest.mark.parametrize("rank", [0, 1])
def test_all_tp_ranks_initialize_capture(monkeypatch, rank):
    pytest.importorskip("vllm.vllm_flash_attn", exc_type=ImportError)
    import vllm.v1.worker.gpu.model_runner as model_runner

    capturer = Mock()
    constructor = Mock(return_value=capturer)
    bind = Mock()
    monkeypatch.setattr(model_runner, "RoutedExpertsCapturer", constructor)
    monkeypatch.setattr(model_runner, "bind_routed_experts_capturer", bind)

    runner = model_runner.GPUModelRunner.__new__(model_runner.GPUModelRunner)
    runner.max_num_tokens = 32
    runner.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(rank=rank))
    runner.kv_cache_config = SimpleNamespace()
    runner.model = Mock()

    runner.init_routed_experts_capturer()

    constructor.assert_called_once_with(
        max_num_batched_tokens=32,
        vllm_config=runner.vllm_config,
        kv_cache_config=runner.kv_cache_config,
    )
    bind.assert_called_once_with(runner.model, capturer)
    assert runner.routed_experts_capturer is capturer


def test_v2_model_runner_accepts_routed_experts(monkeypatch):
    monkeypatch.setattr("importlib.metadata.entry_points", lambda **_: ())
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            enable_return_routed_experts=True,
            use_mla=False,
            logits_processors=None,
            enable_prompt_embeds=False,
        ),
        speculative_config=None,
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1,
            tensor_parallel_size=1,
            distributed_executor_backend=None,
            pipeline_parallel_size=1,
            enable_dbo=False,
            enable_elastic_ep=False,
        ),
        compilation_config=SimpleNamespace(
            mode=CompilationMode.NONE,
            pass_config=SimpleNamespace(enable_sp=False),
        ),
        cache_config=SimpleNamespace(kv_sharing_fast_prefill=False),
        ec_transfer_config=None,
    )

    unsupported = VllmConfig._get_v2_model_runner_unsupported_features(config)

    assert "routed experts capture" not in unsupported
