# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Detect and fill unwritten NVFP4 activation scales.

NVFP4 MoE and linear layers historically allocated activation scales with
``torch.empty``. A checkpoint that omitted a per-expert (or per-shard) key
then ran on uninitialized memory, which CUTLASS consumes as a per-expert
``a1_gscale`` / ``a2_gscale`` and can take to Inf/NaN. Zeros are a legal
calibrated value in some schemes; NaN is not, so an unwritten slot is
distinguishable from a loaded one.

A missing slot is filled from the maximum of the present positive scales on
that tensor. The direction is not arbitrary. Both exporters store the
reciprocal, ``s = amax / (FP8_MAX * FP4_MAX)``, so ``s`` rises with the
activation range; the kernel consumes ``1 / s`` and a block's e4m3 scale is
``(1 / s) * block_amax / FP4_MAX``, which saturates at 448 once ``s`` is
smaller than that block really needed. Filling from the largest present ``s``
therefore assumes the widest range in the layer and cannot saturate; the
minimum would do the opposite. It is not free, though: an over-large ``s``
pushes a quiet block's e4m3 scale down, and past a spread of roughly a hundred
that scale reaches the e4m3 subnormals and starts losing mantissa. This is why
the spread is disclosed rather than the fill simply being called safe.

The fill is recorded on the layer at fill time. After it, the tensor looks
finite and no later inspector can tell a gap was there.
"""

from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

UNCALIBRATED_FILL_ATTR = "_nvfp4_uncalibrated_fill"
UNCALIBRATED_FILL_KIND = "uncalibrated_experts_filled_from_layer_max"
# Set on every layer the scan visited, filled or not. A disclosure needs a
# denominator, and counting NVFP4 layers after load is guesswork: the dense
# paths delete or overwrite the scale parameter they were named for.
UNCALIBRATED_SCAN_ATTR = "_nvfp4_activation_scales_scanned"

_NVFP4_ACTIVATION_SCALE_ATTRS = (
    "w13_input_global_scale",
    "w2_input_global_scale",
    "w13_input_scale",
    "w2_input_scale",
    "input_global_scale",
    "input_scale",
)


def unwritten_nvfp4_activation_scale(
    *size: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Allocate activation scales so an unwritten slot cannot look loaded."""
    return torch.full(size, float("nan"), dtype=dtype, device=device)


def fill_uncalibrated_nvfp4_activation_scale(
    scale: torch.Tensor,
    *,
    name: str,
) -> dict[str, Any] | None:
    """Fill non-finite slots from the maximum of the present positive scales.

    Returns a disclosure record if anything was filled, else None. Raises
    if no finite positive slot exists to fill from.
    """
    if not isinstance(scale, torch.Tensor) or not scale.is_floating_point():
        return None
    values = scale.detach().clone()
    missing = ~torch.isfinite(values)
    if not bool(missing.any()):
        return None
    present = torch.isfinite(values) & (values > 0)
    if not bool(present.any()):
        raise ValueError(
            f"NVFP4 activation scale {name!r} has no finite positive slot "
            f"({int(values.numel())} unwritten); refusing to invent a scale"
        )
    fill_value = values[present].max().to(dtype=values.dtype)
    with torch.no_grad():
        scale.masked_fill_(missing, fill_value)
    present_min = float(values[present].min())
    present_max = float(fill_value)
    return {
        "parameter": name,
        "kind": UNCALIBRATED_FILL_KIND,
        "slots": int(values.numel()),
        "unusable": int(missing.sum().item()),
        "fill_value": present_max,
        "max_spread": (present_max / present_min) if present_min else None,
    }


def fill_uncalibrated_nvfp4_activation_scales(
    layer: torch.nn.Module,
) -> list[dict[str, Any]]:
    """Fill every NVFP4 activation-scale parameter on ``layer`` that has gaps.

    Must run before a kernel fuses the scales into weight alphas. Records the
    substitution on ``layer._nvfp4_uncalibrated_fill``.
    """
    records: list[dict[str, Any]] = []
    scanned: list[str] = []
    for name in _NVFP4_ACTIVATION_SCALE_ATTRS:
        tensor = getattr(layer, name, None)
        if not isinstance(tensor, torch.Tensor):
            continue
        scanned.append(name)
        record = fill_uncalibrated_nvfp4_activation_scale(tensor, name=name)
        if record is not None:
            records.append(record)
    setattr(layer, UNCALIBRATED_SCAN_ATTR, scanned)
    if records:
        setattr(layer, UNCALIBRATED_FILL_ATTR, records)
        logger.warning_once(
            "NVFP4 checkpoint omitted per-expert activation scales; "
            "unwritten slots were filled from the layer maximum. Too large "
            "wastes range; too small overflows e4m3. See layer.%s.",
            UNCALIBRATED_FILL_ATTR,
        )
    return records
