# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
)
from vllm.scalar_type import scalar_types

logger = init_logger(__name__)

__all__ = ["CompressedTensorsW6A16"]
W6A16_SUPPORTED_BITS = [6]


class CompressedTensorsW6A16(CompressedTensorsScheme):
    """
    CompressedTensors scheme for W6A16 quantization using adapted FlexQ kernels.
    
    Weights are quantized to 6 bits and packed using FlexQ bit packing format.
    Activations are FP16 (half precision).
    Uses adapted FlexQ CUDA kernels that work with FP16 activations instead of
    quantized activations.
    """
    
    def __init__(
        self,
        strategy: str,
        num_bits: int,
        group_size: int = 128,
    ):
        """
        Initialize W6A16 scheme.
        
        Args:
            strategy: Quantization strategy (should be "group" for FlexQ)
            num_bits: Number of bits for weight quantization (should be 6)
            group_size: Group size for quantization (default 128 for FlexQ)
        """
        self.strategy = strategy
        self.group_size = group_size
        
        if num_bits not in W6A16_SUPPORTED_BITS:
            raise ValueError(
                f"Unsupported num_bits = {num_bits}."
                f"Supported num_bits = {W6A16_SUPPORTED_BITS}"
            )
        self.num_bits = num_bits
        
    @classmethod
    def get_min_capability(cls) -> int:
        """FlexQ kernels require NVIDIA GPUs with compute capability >= 8.0"""
        return 80
    
    def create_weights(
        self,
        layer: torch.nn.Module,
        output_size: int,
        input_size: int,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        """
        Create weights for W6A16 quantization scheme.
        
        Weights are stored in packed format:
        - weight: int32 tensor containing bit-packed 6-bit weights
        - weight_scale: FP16 scales for weight quantization (per-group)
        """
        output_size_per_partition = sum(output_partition_sizes)
        row_parallel = input_size != input_size_per_partition
        
        # FlexQ uses group_size=128
        effective_group_size = self.group_size
        
        # Ensure group_size divides input_size_per_partition
        assert input_size_per_partition % effective_group_size == 0, (
            f"input_size_per_partition {input_size_per_partition}"
            f" not divisible by group_size {effective_group_size}"
        )
        
        # Calculate scale sizes
        # Weight scales: [output_size_per_partition, input_size_per_partition // group_size]
        # Each scale is FP16 (half), stored as half2 pairs
        scales_and_zp_size = input_size_per_partition // effective_group_size
        
        # Weight tensor: packed 6-bit weights
        # Format: [output_size_per_partition, input_size_per_partition * 6 / 32] (int32)
        # FlexQ packs 4 INT6 values into 3 bytes, but we store as int32 for alignment
        weight_packed_size = (input_size_per_partition * 6 + 31) // 32
        
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition, weight_packed_size, dtype=torch.int32
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)
        layer.register_parameter("weight_packed", weight)
        
        # Weight scales: FP16, stored as half2 pairs
        weight_scale = GroupQuantScaleParameter(
            output_dim=0,
            input_dim=1,
            weight_loader=weight_loader,
            data=torch.empty(
                output_size_per_partition, scales_and_zp_size, dtype=torch.float16
            ),
        )
        layer.register_parameter("weight_scale", weight_scale)
        
        # Store metadata for kernel execution
        layer.flexq_group_size = effective_group_size
        layer.flexq_num_bits = self.num_bits
    
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """
        Process weights after loading.
        
        For FlexQ, weights should already be in the correct packed format.
        This method can be used for any post-loading transformations if needed.
        """
        # FlexQ weights are expected to be pre-packed, so no processing needed
        pass
    
    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None
    ) -> torch.Tensor:
        """
        Apply weights using adapted FlexQ kernels for FP16 activations.
        
        Args:
            layer: Layer with registered weights
            x: Input tensor (FP16)
            bias: Optional bias tensor
            
        Returns:
            Output tensor after matrix multiplication (FP16)
        """
        # Import FlexQ ops here to avoid circular imports
        try:
            from vllm._custom_ops import flexq_bmma_w6a16
        except ImportError:
            raise RuntimeError(
                "FlexQ kernels not available. Please ensure FlexQ CUDA extensions are compiled."
            )
        
        weight_packed = layer.weight_packed
        weight_scale = layer.weight_scale
        
        # Call adapted FlexQ kernel for W6A16
        # Adapted kernel signature: (X, W, W_SCALE, M, N, K, D, group_size, bias)
        # Where:
        # - X: FP16 activations (not quantized)
        # - W: quantized weights (int32, bit-packed) 
        # - W_SCALE: FP16 scales
        # - M, N, K: matrix dimensions
        # - D: output tensor (FP16)
        # - group_size: quantization group size (128)
        # - bias: optional bias
        
        M = x.shape[0]  # batch size (or sequence length)
        K = x.shape[-1]  # input dimension
        N = weight_packed.shape[0]  # output dimension
        
        # For now, we'll need to implement the kernel wrapper
        # This is a placeholder - actual implementation will be in the C++ wrapper
        raise NotImplementedError(
            "FlexQ W6A16 kernel integration not yet complete. "
            "Need to implement adapted C++ wrapper and pybind11 bindings."
        )

