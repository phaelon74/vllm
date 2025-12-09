# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)
from vllm.model_executor.parameter import (
    BasevLLMParameter,
    ChannelQuantScaleParameter,
    GroupQuantScaleParameter,
    ModelWeightParameter,
    _ColumnvLLMParameter,
)
from vllm.scalar_type import scalar_types

logger = init_logger(__name__)

__all__ = ["CompressedTensorsW6A8"]
W6A8_SUPPORTED_BITS = [6]


class FlexQScaleParameter(_ColumnvLLMParameter):
    """
    Custom parameter class for FlexQ weight scales.
    
    Handles loading scales from checkpoint format which may have different shapes
    than expected. The checkpoint format stores scales with shape [output_size, 3]
    but we need to reshape/broadcast to [output_size_per_partition, scales_per_group].
    
    Note: This only inherits from _ColumnvLLMParameter (not RowvLLMParameter) because
    we handle row parallelism manually in load_row_parallel_weight.
    """
    
    def load_row_parallel_weight(self, loaded_weight: torch.Tensor):
        """
        Load weight scales for row parallel layers.
        
        The checkpoint format has scales with shape [output_size, 3] (or similar).
        For row parallel layers, we only shard along output_dim (dimension 0), not input_dim.
        The checkpoint scales are already in the correct format - we just need to shard the output dimension.
        """
        # Shard along output dimension (dimension 0) only
        # Don't shard along input dimension (dimension 1) because checkpoint format is [output_size, 3]
        if loaded_weight.shape[0] != self.data.shape[0]:
            # Need to shard along output dimension
            shard_size = self.data.shape[0]
            start_idx = self.tp_rank * shard_size
            if start_idx + shard_size > loaded_weight.shape[0]:
                # Handle edge case where shard doesn't fit
                actual_size = min(shard_size, loaded_weight.shape[0] - start_idx)
                loaded_weight = loaded_weight.narrow(0, start_idx, actual_size)
                # Copy what we can
                self.data[:actual_size].copy_(loaded_weight)
            else:
                loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
                # After sharding output dim, shapes should match
                if loaded_weight.shape == self.data.shape:
                    self.data.copy_(loaded_weight)
                elif loaded_weight.shape[1] == self.data.shape[1]:
                    # Same input dim, copy output dim
                    self.data.copy_(loaded_weight)
                else:
                    # Input dim mismatch - copy what we can
                    min_in = min(loaded_weight.shape[1], self.data.shape[1])
                    self.data[:, :min_in].copy_(loaded_weight[:, :min_in])
        else:
            # Output dimensions match, just copy (handling input dim mismatch if needed)
            if loaded_weight.shape == self.data.shape:
                self.data.copy_(loaded_weight)
            else:
                # Input dimension mismatch - copy what we can
                min_in = min(loaded_weight.shape[1], self.data.shape[1])
                self.data[:, :min_in].copy_(loaded_weight[:, :min_in])


class CompressedTensorsW6A8(CompressedTensorsScheme):
    """
    CompressedTensors scheme for W6A8 quantization using FlexQ kernels.
    
    Weights are quantized to 6 bits and packed using FlexQ bit packing format.
    Activations are quantized to 8 bits.
    Uses FlexQ CUDA kernels for efficient matrix multiplication.
    """
    
    def __init__(
        self,
        strategy: str,
        num_bits: int,
        group_size: int = 128,
        is_static_input_scheme: bool = False,
        input_symmetric: bool = True,
    ):
        """
        Initialize W6A8 scheme.
        
        Args:
            strategy: Quantization strategy (should be "group" for FlexQ)
            num_bits: Number of bits for weight quantization (should be 6)
            group_size: Group size for quantization (default 128 for FlexQ)
            is_static_input_scheme: Whether input activations are statically quantized
            input_symmetric: Whether input quantization is symmetric
        """
        self.strategy = strategy
        self.group_size = group_size
        self.is_static_input_scheme = is_static_input_scheme
        self.input_symmetric = input_symmetric
        
        if num_bits not in W6A8_SUPPORTED_BITS:
            raise ValueError(
                f"Unsupported num_bits = {num_bits}."
                f"Supported num_bits = {W6A8_SUPPORTED_BITS}"
            )
        self.num_bits = num_bits
        
        # FlexQ kernels require GPU compute capability >= 8.0 (Ampere+)
        # This will be checked in get_min_capability
        
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
        Create weights for W6A8 quantization scheme.
        
        Weights are stored in packed format:
        - weight: int32 tensor containing bit-packed 6-bit weights
        - weight_scale: FP16 scales for weight quantization (per-group)
        - input_scale: FP16 scales for input activation quantization (if static)
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
        # Weight scales: The checkpoint format may differ from what we expect
        # The checkpoint might have scales with shape [output_size, 3] or similar
        # We need to match the checkpoint format, not calculate based on group_size
        # For now, let's use a small default size that matches common checkpoint formats
        # The actual scale shape will be determined from the checkpoint during loading
        # FlexQ kernels handle the scale format internally, so we just need to load what's there
        scales_and_zp_size_partition = 3  # Default to match checkpoint format [output_size, 3]
        # TODO: Determine actual scale shape from checkpoint metadata or config
        
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
        
        # Determine scale partitioning
        # The checkpoint format has scales with shape [output_size, 3] (or similar small number)
        # FlexQ kernels handle quantization internally, so we just need to match the checkpoint format
        # Use FlexQScaleParameter for all layers to handle the checkpoint format correctly
        weight_scale = FlexQScaleParameter(
            output_dim=0,
            weight_loader=weight_loader,
            data=torch.empty(
                output_size_per_partition, scales_and_zp_size_partition, dtype=torch.float16
            ),
        )
        layer.register_parameter("weight_scale", weight_scale)
        
        # Input activation scales (if static quantization)
        if self.is_static_input_scheme:
            # Input scales: per-token or per-tensor depending on strategy
            # For now, assume per-tensor scaling
            # Use BasevLLMParameter for scalar scales (no dimension requirements)
            input_scale = BasevLLMParameter(
                data=torch.empty(1, dtype=torch.float16),
                weight_loader=weight_loader,
            )
            layer.register_parameter("input_scale", input_scale)
        
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
        # However, we may need to verify the format or convert if needed
        # The weight scales might need reshaping if the checkpoint format differs
        if hasattr(layer, "weight_scale"):
            weight_scale = layer.weight_scale
            # Check if the scale shape matches what we expect
            # If not, we might need to reshape or broadcast
            pass
    
    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None
    ) -> torch.Tensor:
        """
        Apply weights using FlexQ kernels.
        
        Args:
            layer: Layer with registered weights
            x: Input tensor (will be quantized to 8-bit if needed)
            bias: Optional bias tensor
            
        Returns:
            Output tensor after matrix multiplication
        """
        # Import FlexQ ops here to avoid circular imports
        try:
            from vllm._custom_ops import flexq_bmma_w6a8
        except ImportError:
            raise RuntimeError(
                "FlexQ kernels not available. Please ensure FlexQ CUDA extensions are compiled."
            )
        
        weight_packed = layer.weight_packed
        weight_scale = layer.weight_scale
        
        # Get input scale if static quantization
        input_scale = None
        if self.is_static_input_scheme and hasattr(layer, "input_scale"):
            input_scale = layer.input_scale
        
        # Quantize input activations to 8-bit if needed
        # For now, assume activations are already quantized or will be quantized elsewhere
        # TODO: Implement activation quantization if needed
        
        # Call FlexQ kernel
        # FlexQ kernel signature: (X, W, X_SCALE, W_SCALE, M, N, K, D, group_size, bias)
        # Where:
        # - X: quantized activations (int32, bit-packed)
        # - W: quantized weights (int32, bit-packed) 
        # - X_SCALE: FP16 scales
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
            "FlexQ kernel integration not yet complete. "
            "Need to implement C++ wrapper and pybind11 bindings."
        )

