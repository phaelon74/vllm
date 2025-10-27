# NVFP4 Kernel Compilation Issue (UNRESOLVED)

## Status: ⚠️ UNRESOLVED - Awaiting vLLM Nightly with SM10.0/SM11.0 Symbol Fixes

## Problem

When trying to run NVFP4 (4-bit floating point) quantized models on Blackwell GPUs (compute capability 12.0), vLLM fails with:

```
NotImplementedError: No compiled nvfp4 quantization kernel
RuntimeError: Error Internal
```

## Root Cause Analysis

The FP4 kernels are **not being compiled** into vLLM. Investigation revealed a complex interaction between:

1. **PyTorch Architecture List**: PyTorch 2.8.0+cu129 includes `sm_100` and `sm_120` in its architecture list
2. **CMake Architecture Intersection**: vLLM's CMakeLists.txt uses `cuda_archs_loose_intersection()` to match available architectures
3. **Missing SM10.0**: We previously removed SM10.0 and SM11.0 from `CUDA_SUPPORTED_ARCHS` due to incomplete kernel implementations and symbol errors
4. **Broken Intersection Logic**: Without SM10.0 in the supported list, the architecture intersection fails when PyTorch reports `sm_100`

## Hardware Details

- **GPUs**: 2x NVIDIA RTX PRO 6000 Blackwell Workstation Edition
- **Compute Capability**: 12.0 (SM_120)
- **CUDA Version**: 12.9
- **PyTorch Version**: 2.8.0+cu129
- **PyTorch Architectures**: `['sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120', 'compute_120']`

## Why We Can't Fix It Now

The straightforward fix would be to add `10.0` and `10.0a` back to `CUDA_SUPPORTED_ARCHS`:

```cmake
set(CUDA_SUPPORTED_ARCHS "7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0;10.0;10.0a;12.0;12.0a")
```

**However**, this brings back **symbol errors** in SM10.0/SM11.0 kernels that we removed them to fix.

The issue is that:
- SM10.0 kernels have incomplete implementations with missing symbols
- But PyTorch's `sm_100` in the architecture list breaks CMake's architecture intersection
- We can't have FP4 kernels without SM10.0 in the supported architectures

## Temporary Solution

**Wait for vLLM nightly** that fixes the SM10.0/SM11.0 symbol issues, then add them back to `CUDA_SUPPORTED_ARCHS`.

## Current CMakeLists.txt Configuration

```cmake
# lines 88-100 in CMakeLists.txt
if(DEFINED CMAKE_CUDA_COMPILER_VERSION AND
   CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
  # Exclude SM10.0 and SM11.0 due to incomplete kernel implementations (keep SM12.0 for GB20x)
  # Use 12.0f suffix for proper FP4 kernel compilation with CUDA 13+
  set(CUDA_SUPPORTED_ARCHS "7.5;8.0;8.6;8.7;8.9;9.0;12.0;12.0f")
elseif(DEFINED CMAKE_CUDA_COMPILER_VERSION AND
   CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
  # Exclude SM10.0 and SM11.0 due to incomplete kernel implementations (keep SM12.0 for GB20x)
  # Use 12.0a suffix for proper FP4 kernel compilation
  set(CUDA_SUPPORTED_ARCHS "7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0;12.0;12.0a")
else()
  set(CUDA_SUPPORTED_ARCHS "7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0")
endif()
```

## What Needs to Happen

1. **vLLM nightly fixes SM10.0/SM11.0 symbols** (waiting on upstream)
2. **Update CMakeLists.txt** to include SM10.0:
   ```cmake
   set(CUDA_SUPPORTED_ARCHS "7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0;10.0;10.0a;12.0;12.0a")
   ```
3. **Rebuild vLLM** to compile FP4 kernels
4. **Verify** FP4 functions exist:
   ```bash
   python -c "import vllm._C as _C; print([x for x in dir(_C) if 'fp4' in x.lower()])"
   ```

## Testing Once Fixed

After the fix, test with:

```bash
vllm serve /path/to/NVFP4/model \
  --trust-remote-code \
  --quantization compressed-tensors \
  --tensor-parallel-size 2 \
  --max-model-len 65535 \
  --gpu-memory-utilization 0.80 \
  --disable-custom-all-reduce
```

Expected FP4 functions that should exist:
- `cutlass_scaled_fp4_mm`
- `scaled_fp4_quant`
- `cutlass_scaled_mm_supports_fp4`

## Technical Details

### FP4 Kernel Build Logic

From `CMakeLists.txt` lines 574-625:

```cmake
# FP4 Archs and flags
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
  cuda_archs_loose_intersection(FP4_ARCHS "12.0f" "${CUDA_ARCHS}")
else()
  cuda_archs_loose_intersection(FP4_ARCHS "12.0a;12.1a" "${CUDA_ARCHS}")
endif()

if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8 AND FP4_ARCHS)
  set(SRCS
    "csrc/quantization/fp4/nvfp4_quant_kernels.cu"
    "csrc/quantization/fp4/activation_nvfp4_quant_fusion_kernels.cu"
    "csrc/quantization/fp4/nvfp4_scaled_mm_kernels.cu"
    # ... more files
  )
  list(APPEND VLLM_EXT_SRC "${SRCS}")
  list(APPEND VLLM_GPU_FLAGS "-DENABLE_NVFP4_SM100=1")
else()
  message(STATUS "Not building NVFP4 as no compatible archs were found.")
endif()
```

### Entry Point Stubs

The entry points always compile (from `csrc/quantization/fp4/nvfp4_quant_entry.cu`):

```cpp
void scaled_fp4_quant(torch::Tensor& output, torch::Tensor const& input,
                      torch::Tensor& output_sf, torch::Tensor const& input_sf) {
#if (defined(ENABLE_NVFP4_SM100) && ENABLE_NVFP4_SM100) || \
    (defined(ENABLE_NVFP4_SM120) && ENABLE_NVFP4_SM120)
  return scaled_fp4_quant_sm1xxa(output, input, output_sf, input_sf);
#endif
  TORCH_CHECK_NOT_IMPLEMENTED(false, "No compiled nvfp4 quantization kernel");
}
```

Without the `ENABLE_NVFP4_*` flags, the entry points compile but only contain error stubs.

## Architecture Intersection Explanation

The `cuda_archs_loose_intersection()` function (from `cmake/utils.cmake:296-370`) matches:
- **Source architectures**: What the kernel supports (e.g., `12.0a`)
- **Target architectures**: What CUDA_ARCHS contains (from PyTorch, e.g., `10.0`, `12.0`)

When PyTorch reports `sm_100` → `10.0`, but `CUDA_SUPPORTED_ARCHS` doesn't include `10.0`, the intersection calculation gets confused and fails to properly match `12.0a`.

## References

- `CMakeLists.txt` lines 88-100: CUDA_SUPPORTED_ARCHS definition
- `CMakeLists.txt` lines 574-625: FP4 kernel build logic
- `cmake/utils.cmake` lines 296-370: cuda_archs_loose_intersection() implementation
- `csrc/quantization/fp4/`: FP4 kernel implementations
- `csrc/quantization/fp4/nvfp4_quant_entry.cu`: Entry point stubs

## Related Issues

This is related to the broader SM10.0/SM11.0 incomplete kernel implementation issue in vLLM for Blackwell GB10x GPUs. The GB20x (SM12.0) kernels are complete, but the build system gets confused by PyTorch's architecture list.

