# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    gptq_pack,
    pack_cols,
)
from vllm.model_executor.models.gemma4 import (
    _dequantize_router_proj,
    gemma4_forced_routing_weights,
    gemma4_fused_routing_kernel_triton,
    gemma4_routing_function_torch,
)


def sort_by_id(w, ids):
    order = ids.argsort(dim=-1)
    return w.gather(1, order), ids.gather(1, order)


def test_dequantize_router_proj_matches_packed_values_transposed():
    """A quantized router must reload as GateLinear's dense [E, H] weight.

    AutoRound quantizes the router projection, but GateLinear has no quantized
    parameter to receive it, so the load path dequantizes instead. Getting the
    grouping or the orientation wrong yields finite garbage rather than an
    error, which is why this asserts the exact reconstruction.
    """
    torch.manual_seed(0)
    size_k, size_n, group_size = 64, 16, 32
    num_groups = size_k // group_size

    q = torch.randint(0, 16, (size_k, size_n), dtype=torch.int32)
    zp = torch.full((num_groups, size_n), 8, dtype=torch.int32)
    scales = torch.rand(num_groups, size_n, dtype=torch.bfloat16) + 0.5

    shards = {
        "qweight": gptq_pack(q, 4, size_k, size_n),
        "qzeros": pack_cols(zp, 4, num_groups, size_n),
        "scales": scales,
    }
    weight = _dequantize_router_proj(shards, torch.bfloat16)

    expected = (q - zp.repeat_interleave(group_size, dim=0)).to(torch.bfloat16)
    expected = expected * scales.repeat_interleave(group_size, dim=0)

    assert weight.shape == (size_n, size_k)
    assert weight.dtype == torch.bfloat16
    torch.testing.assert_close(weight, expected.t().contiguous())


def test_dequantize_router_proj_refuses_indivisible_group_count():
    size_k, size_n = 64, 16
    q = torch.randint(0, 16, (size_k, size_n), dtype=torch.int32)
    shards = {
        "qweight": gptq_pack(q, 4, size_k, size_n),
        "qzeros": pack_cols(
            torch.full((3, size_n), 8, dtype=torch.int32), 4, 3, size_n
        ),
        "scales": torch.ones(3, size_n, dtype=torch.bfloat16),
    }
    with pytest.raises(ValueError, match="do not divide evenly"):
        _dequantize_router_proj(shards, torch.bfloat16)


def test_gemma4_forced_natural_ids_preserve_triton_weights():
    torch.manual_seed(0)
    gating = torch.randn(32, 128, dtype=torch.bfloat16, device="cuda")
    scales = torch.rand(128, dtype=torch.float32, device="cuda")
    natural_weights, natural_ids = gemma4_fused_routing_kernel_triton(gating, 8, scales)

    forced_weights = gemma4_forced_routing_weights(
        gating,
        natural_ids,
        scales,
        natural_weights,
        natural_ids,
        renormalize=True,
    )

    assert torch.equal(forced_weights, natural_weights)


# Gemma4 MoE Model has context length of 250K
# the minus 1 is to ensure that edge cases are tested
@pytest.mark.parametrize("num_tokens", [1, 2, 2048, 250000])
@pytest.mark.parametrize("num_experts", [128])  # gemma4 moe experts
@pytest.mark.parametrize("topk", [8])  # gemma4 topk
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.half, torch.float32])
def test_gemma4_routing_kernel_triton(
    num_tokens: int,
    num_experts: int,
    topk: int,
    dtype: torch.dtype,
):
    torch.manual_seed(0)

    gating = torch.randn(num_tokens, num_experts, dtype=dtype, device="cuda")
    scales = torch.rand(num_experts, dtype=torch.float32, device="cuda")

    ref_w, ref_ids = gemma4_routing_function_torch(gating, topk, scales)
    tri_w, tri_ids = gemma4_fused_routing_kernel_triton(gating, topk, scales)

    # Sort by expert id — to remove tie-breaking differences
    ref_ws, ref_is = sort_by_id(ref_w, ref_ids)
    tri_ws, tri_is = sort_by_id(tri_w, tri_ids)

    ids_match = (ref_is == tri_is).all().item()
    weights_match = torch.allclose(ref_ws, tri_ws, atol=1e-2, rtol=1e-2)
    all_match = ids_match and weights_match
    max_err = (ref_ws - tri_ws).abs().max().item()
    print(
        f"T={num_tokens:5d} E={num_experts:4d} K={topk} "
        f"{str(dtype).split('.')[-1]:7s} ids={ids_match} max_Δweight={max_err:.2e}"
    )
    if not all_match:
        bad = (ref_is != tri_is).any(dim=-1).nonzero(as_tuple=True)[0]
        if len(bad):
            r = bad[0].item()
            print(
                f"  first bad row {r}: ref_ids={ref_ids[r].tolist()} "
                f"tri_ids={tri_ids[r].tolist()}"
            )
        assert all_match
