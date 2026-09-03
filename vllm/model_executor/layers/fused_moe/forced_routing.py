# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import torch

from vllm.forward_context import (
    get_forward_context,
    is_forward_context_available,
)

FORCED_ROUTING_KEY = "forced_moe_routing"

_INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}


@dataclass(frozen=True)
class ForcedRouting:
    """Ordered logical expert IDs to force for one model forward."""

    expert_ids: torch.Tensor

    def for_layer(
        self,
        layer_id: int,
        *,
        num_tokens: int,
        top_k: int,
        num_experts: int,
        device: torch.device,
    ) -> torch.Tensor:
        ids = self.expert_ids
        if ids.ndim != 3:
            raise ValueError(
                "Forced MoE expert IDs must have shape [num_tokens, num_layers, top_k]."
            )
        if ids.shape[0] != num_tokens:
            raise ValueError(
                "Forced MoE expert IDs are not aligned with the current token "
                f"batch: expected {num_tokens}, got {ids.shape[0]}."
            )
        if not 0 <= layer_id < ids.shape[1]:
            raise ValueError(
                f"Forced MoE routing has no data for layer {layer_id}; "
                f"the trace contains {ids.shape[1]} layers."
            )
        if ids.shape[2] != top_k:
            raise ValueError(
                "Forced MoE expert IDs have the wrong top-k width: "
                f"expected {top_k}, got {ids.shape[2]}."
            )
        if ids.dtype not in _INTEGER_DTYPES:
            raise TypeError(f"Forced MoE expert IDs must be integers, got {ids.dtype}.")
        if ids.device != device:
            raise ValueError(
                "Forced MoE expert IDs must be on the router logits device: "
                f"expected {device}, got {ids.device}."
            )

        layer_ids = ids[:, layer_id, :]
        if layer_ids.numel() > 0:
            invalid = (layer_ids < 0) | (layer_ids >= num_experts)
            if torch.any(invalid).item():
                raise ValueError(
                    f"Forced MoE expert IDs must be in [0, {num_experts})."
                )
            sorted_ids = torch.sort(layer_ids.to(torch.int64), dim=-1).values
            if torch.any(sorted_ids[:, 1:] == sorted_ids[:, :-1]).item():
                raise ValueError(
                    "Forced MoE expert IDs must be unique within each token."
                )
        return layer_ids


def get_forced_expert_ids(
    layer_id: int,
    *,
    num_tokens: int,
    top_k: int,
    num_experts: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Return this layer's forced logical IDs from the forward context."""
    if not is_forward_context_available():
        return None

    forced_routing = get_forward_context().additional_kwargs.get(FORCED_ROUTING_KEY)
    if forced_routing is None:
        return None
    if not isinstance(forced_routing, ForcedRouting):
        raise TypeError(
            f"ForwardContext.additional_kwargs[{FORCED_ROUTING_KEY!r}] "
            "must be a ForcedRouting instance."
        )
    return forced_routing.for_layer(
        layer_id,
        num_tokens=num_tokens,
        top_k=top_k,
        num_experts=num_experts,
        device=device,
    )
