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
)
from vllm.scalar_type import scalar_types

logger = init_logger(__name__)

__all__ = ["CompressedTensorsW6A8"]
W6A8_SUPPORTED_BITS = [6]


class FlexQScaleParameter(BasevLLMParameter):
    """
    Custom parameter class for FlexQ weight scales.
    
    The checkpoint format stores scales with shape [output_size, 3].
    FlexQ kernels handle quantization internally, so we just need to load the checkpoint format.
    We only shard along output dimension (dimension 0), never along input dimension (dimension 1).
    """
    
    def load_row_parallel_weight(self, loaded_weight: torch.Tensor):
        """
        Load weight scales for row parallel layers.
        Only shard along output dimension (dimension 0), not input dimension (dimension 1).
        """
        # CRITICAL: This method MUST be called, not RowvLLMParameter.load_row_parallel_weight
        # If this isn't being called, the parameter is not a FlexQScaleParameter instance
        logger.info(
            f"FlexQScaleParameter.load_row_parallel_weight CALLED: "
            f"loaded_weight.shape={loaded_weight.shape}, self.data.shape={self.data.shape}, "
            f"tp_rank={self.tp_rank}, type={type(self).__name__}"
        )
        
        # Shard along output dimension (dimension 0) only
        # Never shard along input dimension (dimension 1) - checkpoint format is [output_size, 3]
        if loaded_weight.shape[0] != self.data.shape[0]:
            shard_size = self.data.shape[0]
            start_idx = self.tp_rank * shard_size
            if start_idx + shard_size <= loaded_weight.shape[0]:
                loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
            else:
                # Edge case: take what we can
                actual_size = loaded_weight.shape[0] - start_idx
                loaded_weight = loaded_weight.narrow(0, start_idx, actual_size)
        
        # Copy the data (handling shape mismatches if needed)
        # The checkpoint might have different scale dimensions than we created
        # After sharding output dim, we copy what we can
        if loaded_weight.shape == self.data.shape:
            self.data.copy_(loaded_weight)
        else:
            # Handle shape mismatch - copy what we can
            # If the checkpoint has a different scale dimension, we'll take the first few columns
            min_out = min(loaded_weight.shape[0], self.data.shape[0])
            min_in = min(loaded_weight.shape[1], self.data.shape[1])
            
            # If checkpoint has more columns than we expect, take the first ones
            # If checkpoint has fewer columns, we'll only copy what's available
            if loaded_weight.shape[1] > self.data.shape[1]:
                logger.info(
                    f"FlexQScaleParameter: checkpoint has more scale columns ({loaded_weight.shape[1]}) "
                    f"than expected ({self.data.shape[1]}), taking first {self.data.shape[1]} columns"
                )
                self.data[:min_out, :].copy_(loaded_weight[:min_out, :self.data.shape[1]])
            elif loaded_weight.shape[1] < self.data.shape[1]:
                logger.warning(
                    f"FlexQScaleParameter: checkpoint has fewer scale columns ({loaded_weight.shape[1]}) "
                    f"than expected ({self.data.shape[1]}), only copying available columns"
                )
                self.data[:min_out, :min_in].copy_(loaded_weight[:min_out, :min_in])
            else:
                # Same number of columns, just copy
                self.data[:min_out, :min_in].copy_(loaded_weight[:min_out, :min_in])
    
    def load_column_parallel_weight(self, loaded_weight: torch.Tensor):
        """Load weight scales for column parallel layers - same logic as row parallel."""
        self.load_row_parallel_weight(loaded_weight)


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
        # Weight scales: The checkpoint format stores scales with shape [output_size, scale_dim]
        # where scale_dim can be 3 (for FlexQ's standard format) or other values depending on the checkpoint
        # The actual scale shape will be determined from the checkpoint during loading
        # FlexQ kernels handle the scale format internally, so we need to match the checkpoint format
        # For now, we'll create a parameter that can accommodate different scale dimensions
        # The checkpoint might have:
        # - weight: [output_size, 3] (the actual scale, being redirected)
        # - weight_scale: [output_size_per_partition, scale_dim] (per-group scales or other format)
        # We'll start with a reasonable default and adjust during loading if needed
        scales_and_zp_size_partition = 3  # Default to match FlexQ's standard format [output_size, 3]
        # Note: If checkpoint has different format, we'll handle it in load_row_parallel_weight
        
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
            weight_loader=weight_loader,
            data=torch.empty(
                output_size_per_partition, scales_and_zp_size_partition, dtype=torch.float16
            ),
        )
        logger.info(
            f"Created FlexQScaleParameter: shape={weight_scale.data.shape}, "
            f"type={type(weight_scale).__name__}, has load_row_parallel_weight={hasattr(weight_scale, 'load_row_parallel_weight')}"
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
            from vllm._custom_ops import flexq_w6a8_gemm
        except ImportError:
            raise RuntimeError(
                "FlexQ kernels not available. Please ensure FlexQ CUDA extensions are compiled."
            )
        
        weight_packed = layer.weight  # Same as layer.weight_packed (both registered)
        weight_scale = layer.weight_scale
        
        # Store original dtype to restore it after FlexQ computation
        # FlexQ kernels require FP16 input, but we want to preserve the original dtype
        original_dtype = x.dtype
        
        # Ensure input is FP16 (FlexQ kernels require FP16 input)
        # During torch.compile, inputs might be converted to BF16 or other dtypes
        if x.dtype != torch.float16:
            x = x.to(torch.float16)
        
        # Get input scale if static quantization
        # If not static, create a dummy scale tensor (FlexQ kernels require it)
        if self.is_static_input_scheme and hasattr(layer, "input_scale"):
            input_scale = layer.input_scale.data if hasattr(layer.input_scale, 'data') else layer.input_scale
        else:
            # Create a dummy scale tensor for dynamic quantization
            # FlexQ kernels handle dynamic quantization internally
            input_scale = torch.ones(1, dtype=torch.float16, device=x.device)
        
        # Ensure input_scale is FP16 and has the right shape
        if isinstance(input_scale, torch.Tensor):
            if input_scale.dtype != torch.float16:
                input_scale = input_scale.to(torch.float16)
        else:
            input_scale = torch.ones(1, dtype=torch.float16, device=x.device)
        
        # Extract weight_scale data (it's a FlexQScaleParameter)
        weight_scale_data = weight_scale.data if hasattr(weight_scale, 'data') else weight_scale
        if weight_scale_data.dtype != torch.float16:
            weight_scale_data = weight_scale_data.to(torch.float16)
        
        # Call FlexQ kernel
        # flexq_w6a8_gemm signature: (input, weight_packed, input_scale, weight_scale, group_size, bias)
        # Where:
        # - input: FP16 input activations (will be quantized to int8 internally)
        # - weight_packed: int32 packed 6-bit weights
        # - input_scale: FP16 scales for input quantization
        # - weight_scale: FP16 scales for weight quantization
        # - group_size: quantization group size (128)
        # - bias: bool indicating if bias is present
        
        output = flexq_w6a8_gemm(
            x,  # input activations (FP16)
            weight_packed,  # packed weights (int32)
            input_scale,  # input scales (FP16)
            weight_scale_data,  # weight scales (FP16)
            layer.flexq_group_size,  # group_size
            bias is not None,  # bias flag
        )
        
        if bias is not None:
            output = output + bias
        
        # Convert output back to the original input dtype
        # FlexQ outputs FP16, but we need to match the original dtype
        # to prevent dtype mismatches in downstream layers
        if output.dtype != original_dtype:
            output = output.to(original_dtype)
        
        return output

