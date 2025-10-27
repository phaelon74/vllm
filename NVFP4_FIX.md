# NVFP4 Kernel Compilation Fix

## Problem Summary

Your vLLM build was failing to run NVFP4 models with the error:
```
RuntimeError: Error Internal
```

The root cause: **NVFP4 kernels were never compiled** into your vLLM installation.

## Root Cause

When you removed SM10.0 and SM11.0 architectures from `CUDA_SUPPORTED_ARCHS` to fix symbol problems, you kept only `12.0` without the proper architecture suffix. The CMake build system requires:

- **For CUDA 12.8-12.x**: `12.0a` suffix
- **For CUDA 13.0+**: `12.0f` suffix

Without the suffix, the architecture intersection logic fails and the FP4 kernels don't get built.

### Evidence
```bash
$ python -c "import vllm._C as _C; funcs = [x for x in dir(_C) if 'fp4' in x.lower()]; print(funcs)"
# Result: [] (empty - no FP4 functions compiled!)
```

## Solution

Updated `CMakeLists.txt` to include both the base architecture and the suffixed version:

### For CUDA 12.8+:
```cmake
set(CUDA_SUPPORTED_ARCHS "7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0;10.0;10.0a;12.0;12.0a")
```

### For CUDA 13.0+:
```cmake
set(CUDA_SUPPORTED_ARCHS "7.5;8.0;8.6;8.7;8.9;9.0;10.0;10.0f;12.0;12.0f")
```

**Important:** We include `10.0` and `10.0a/10.0f` even though we don't want to build most kernels for SM10.0. This is necessary because PyTorch 2.8.0+cu129 includes `sm_100` in its architecture list, and if we don't include it in CUDA_SUPPORTED_ARCHS, the architecture intersection logic breaks.

The suffixes enable the architecture intersection logic to correctly identify that FP4 kernels should be built for your Blackwell GPUs.

## What to Do Next

### 1. Clean Previous Build
```bash
cd /home/phaedawg/vllm/vllm
pip uninstall vllm -y
rm -rf build/
rm -rf vllm.egg-info/
find . -name "*.so" -delete
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
```

### 2. Rebuild vLLM
```bash
# Make sure you're in the venv
source /home/phaedawg/vllm/venv/bin/activate

# Rebuild with verbose output to verify FP4 compilation
VERBOSE=1 pip install -e . 2>&1 | tee build.log
```

### 3. Verify FP4 Kernels Were Built

Look for these messages in the build output:
```
-- Building NVFP4 for archs: 12.0a
```

Then verify the functions exist:
```bash
python -c "import vllm._C as _C; funcs = [x for x in dir(_C) if 'fp4' in x.lower()]; print('FP4 functions:', funcs)"
```

Expected output:
```
FP4 functions: ['cutlass_scaled_fp4_mm', 'cutlass_scaled_mm_supports_fp4', 'scaled_fp4_quant']
```

### 4. Test Your NVFP4 Model

Run your original command:
```bash
vllm serve /media/fmodels/TheHouseOfTheDude/Behemoth-R1-123B-v2_Compressed-Tensors/NVFP4 \
  --trust-remote-code \
  --quantization compressed-tensors \
  --tensor-parallel-size 2 \
  --max-model-len 65535 \
  --gpu-memory-utilization 0.80 \
  --disable-custom-all-reduce \
  --api-key REDACTED
```

## Technical Details

### Architecture Suffix Explanation

- **12.0**: Base architecture number (SM_120)
- **12.0a**: Variant suffix for data center GPUs (like your RTX PRO 6000 Blackwell)
- **12.0f**: Variant suffix for future/other variants

The CMake `cuda_archs_loose_intersection()` function uses these suffixes to match architectures properly. When it sees `12.0a` in the source list and `12.0` in the target list, it correctly maps them and enables the kernels.

### Why This Happened

**Two problems combined:**

1. **Missing architecture suffixes**: When you removed SM10.0 and SM11.0, you kept only `12.0` without the `12.0a` suffix that the FP4 build logic requires.

2. **PyTorch architecture mismatch**: Your PyTorch 2.8.0+cu129 was compiled with `sm_100` and `sm_120`, which become `10.0` and `12.0` in CMake's architecture list. When `10.0` wasn't in `CUDA_SUPPORTED_ARCHS`, the architecture intersection logic failed, preventing `12.0a` from being properly matched.

The stable vLLM build likely either:
- Was built with an older PyTorch that didn't include `sm_100`
- Or had the complete architecture list before SM10/SM11 removal

## Validation Checklist

- [ ] CMakeLists.txt updated with architecture suffixes
- [ ] Old build cleaned
- [ ] vLLM rebuilt successfully
- [ ] Build log shows "Building NVFP4 for archs: 12.0a"
- [ ] Python verification shows FP4 functions present
- [ ] Model loads and runs without "Error Internal"

## References

- CMakeLists.txt lines 88-99: CUDA_SUPPORTED_ARCHS definition
- CMakeLists.txt lines 574-623: FP4 kernel build logic
- cmake/utils.cmake lines 296-370: cuda_archs_loose_intersection() implementation
- csrc/quantization/fp4/: FP4 kernel implementations

