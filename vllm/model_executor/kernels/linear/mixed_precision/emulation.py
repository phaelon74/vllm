# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_packed_zero,
    marlin_pad_qweight,
)
from vllm.scalar_type import scalar_types
from vllm.utils.math_utils import round_up

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig


class EmulationWNA16LinearKernel(MPLinearKernel):
    """Batch-invariant WNA16 correctness path using BF16 matmul."""

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if c.weight_type not in (scalar_types.uint4b8, scalar_types.uint8b128):
            return False, "only symmetric GPTQ int4 and int8 are supported"
        if c.act_type not in (torch.float16, torch.bfloat16):
            return False, "only FP16 and BF16 activations are supported"
        if c.zero_points or c.has_g_idx:
            return False, "zero points and activation ordering are unsupported"
        if c.group_size != -1 and c.group_size <= 0:
            return False, f"unsupported group size {c.group_size}"
        pack_factor = 32 // c.weight_type.size_bits
        if c.partition_weight_shape[0] % pack_factor:
            return False, "input size must be divisible by the packing factor"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        qweight = getattr(layer, self.w_q_name)
        scales = getattr(layer, self.w_s_name)
        size_k, size_n = self.config.partition_weight_shape
        group_size = (
            size_k if self.config.group_size == -1 else self.config.group_size
        )
        padded_k = round_up(size_k, group_size)
        expected_groups = padded_k // group_size
        if scales.shape[0] != expected_groups:
            raise ValueError(
                f"scale rows {scales.shape[0]} do not match expected "
                f"group count {expected_groups}"
            )
        expected_qweight_shape = (
            size_k // (32 // self.config.weight_type.size_bits),
            size_n,
        )
        if qweight.shape != expected_qweight_shape:
            raise ValueError(
                f"qweight shape {tuple(qweight.shape)} does not match expected "
                f"{expected_qweight_shape}"
            )
        qweight = marlin_pad_qweight(
            qweight,
            size_n,
            size_k,
            size_n,
            padded_k,
            padding_value=marlin_packed_zero(self.config.weight_type),
        )

        bits = self.config.weight_type.size_bits
        pack_factor = 32 // bits
        shifts = torch.arange(
            pack_factor,
            device=qweight.device,
            dtype=torch.int32,
        ) * bits
        values = (qweight.unsqueeze(-1) >> shifts) & ((1 << bits) - 1)
        values = values.permute(0, 2, 1).reshape(padded_k, size_n)
        values = values.to(scales.dtype) - self.config.weight_type.bias
        expanded_scales = scales.repeat_interleave(group_size, dim=0)
        weight = (values * expanded_scales)[:size_k].contiguous()
        replace_parameter(
            layer,
            self.w_q_name,
            torch.nn.Parameter(weight, requires_grad=False),
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output_shape = (*x.shape[:-1], self.config.partition_weight_shape[1])
        x_2d = x.reshape(-1, x.shape[-1])
        output = torch.matmul(x_2d, getattr(layer, self.w_q_name))
        if bias is not None:
            output = output + bias
        return output.reshape(output_shape)
