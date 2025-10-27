# ScoreMode Documentation Index

## Overview

This folder contains complete documentation for the **Score Mode** feature in vLLM, which enables accurate perplexity calculation on quantized models without decompressing weights.

---

## 📚 Documentation Files

### 1. [README.md](README.md) - **START HERE**
Quick overview and getting started guide.

**Contains:**
- What is Score Mode?
- Quick start examples
- Key features summary
- Architecture diagram

**Read this first** to understand what Score Mode does and how to use it.

---

### 2. [VLLM_ScoreMode.md](VLLM_ScoreMode.md) - **COMPLETE IMPLEMENTATION GUIDE**
Comprehensive documentation of ALL changes made to vLLM source code.

**Contains:**
- Every file modified (sampling_params.py, inputs.py, sampler.py)
- Before/after code comparisons
- Detailed explanation of each change
- Usage examples (basic and optimized)
- Testing instructions
- Performance considerations

**Read this** to understand exactly what was changed and why.

---

### 3. [SCORE_MODE_IMPLEMENTATION.md](SCORE_MODE_IMPLEMENTATION.md) - **HIGH-LEVEL OVERVIEW**
High-level overview of the feature design.

**Contains:**
- Problem statement
- Solution approach
- Why it works
- Benefits
- Comparison with alternatives
- Future enhancements

**Read this** for the big picture and design philosophy.

---

### 4. [NVFP4_ISSUE.md](NVFP4_ISSUE.md) - **KNOWN ISSUE**
Documentation of unresolved NVFP4 kernel compilation issue.

**Contains:**
- Problem description
- Root cause analysis
- Why we can't fix it now
- Temporary solution (wait for nightly)
- What needs to happen
- Technical details

**Read this** if you're having issues with NVFP4 models on Blackwell GPUs.

---

## 🐍 Python Scripts

### 5. [test_score_mode.py](test_score_mode.py) - **SIMPLE TEST**
Basic test script using facebook/opt-125m.

**Purpose:**
- Validate score_mode works
- Quick sanity check
- Simple example

**Usage:**
```bash
python ScoreMode/test_score_mode.py
```

---

### 6. [score_mode_perplexity.py](score_mode_perplexity.py) - **FULL EVALUATION**
Complete perplexity evaluation with EXL3-compatible methodology.

**Features:**
- Command-line interface
- WikiText-2 dataset support
- Sliding window evaluation
- Multi-GPU support
- Quantized model support (compressed-tensors, NVFP4, W4A16)
- EXL3-compatible scoring

**Usage:**
```bash
# WikiText-2
python ScoreMode/score_mode_perplexity.py \
    --model facebook/opt-125m \
    --dataset wikitext \
    --dataset-config wikitext-2-raw-v1

# Quantized model
python ScoreMode/score_mode_perplexity.py \
    --model neuralmagic/Meta-Llama-3.1-8B-Instruct-W4A16 \
    --quantization compressed-tensors \
    --text "Your text here"
```

---

## 📖 Reading Order

### For New Users:
1. **README.md** - Understand what Score Mode is
2. **test_score_mode.py** - Run simple test
3. **score_mode_perplexity.py** - Try full evaluation

### For Developers:
1. **VLLM_ScoreMode.md** - See all code changes
2. **SCORE_MODE_IMPLEMENTATION.md** - Understand design
3. Source files in `vllm/` - Review actual implementation

### For Troubleshooting:
1. **README.md** - Check quick start
2. **VLLM_ScoreMode.md** - Verify changes are applied
3. **NVFP4_ISSUE.md** - If using NVFP4 models

---

## ✅ Modified vLLM Files

The following source files in the main vLLM codebase were modified:

| File | Changes |
|------|---------|
| `vllm/sampling_params.py` | Added `score_mode` parameter, modified validation |
| `vllm/inputs.py` | Added `target_token_ids` to `TokensPrompt` |
| `vllm/v1/sample/sampler.py` | Added fast path for target tokens |

See [VLLM_ScoreMode.md](VLLM_ScoreMode.md) for complete details.

---

## 🎯 Key Features

✅ **Accurate perplexity on quantized models** - No decompression  
✅ **Multi-GPU support** - Tensor parallelism for large models  
✅ **Standard formats** - Works with compressed-tensors  
✅ **Exact log probabilities** - No approximations  
✅ **EXL3-compatible** - Comparable to EXL2/EXL3 benchmarks  
✅ **Memory efficient** - Optional target_token_ids optimization  

---

## 🐛 Known Issues

### NVFP4 Kernels Not Compiling (Blackwell SM12.0)

**Status**: ⚠️ UNRESOLVED

**Issue**: FP4 kernels don't compile due to PyTorch architecture mismatch with removed SM10.0/SM11.0 architectures.

**Solution**: Wait for vLLM nightly with SM10.0/SM11.0 symbol fixes.

**Details**: See [NVFP4_ISSUE.md](NVFP4_ISSUE.md)

---

## 🔗 Quick Links

- **Test Script**: [test_score_mode.py](test_score_mode.py)
- **Evaluation Script**: [score_mode_perplexity.py](score_mode_perplexity.py)
- **Implementation Details**: [VLLM_ScoreMode.md](VLLM_ScoreMode.md)
- **Design Overview**: [SCORE_MODE_IMPLEMENTATION.md](SCORE_MODE_IMPLEMENTATION.md)
- **NVFP4 Issue**: [NVFP4_ISSUE.md](NVFP4_ISSUE.md)

---

## 📊 Comparison Table

| Tool | Tests Quantized | Exact Logprobs | Compressed-Tensors |
|------|----------------|----------------|-------------------|
| lm-eval + HF | ❌ | ✅ | ⚠️ |
| lm-eval + vLLM | ✅ | ❌ | ✅ |
| vLLM API | ✅ | ❌ | ✅ |
| ExLlamaV3 | ✅ | ✅ | ❌ |
| **Score Mode** | **✅** | **✅** | **✅** |

---

## 🤝 Contributing

If you make improvements to Score Mode:

1. Update [VLLM_ScoreMode.md](VLLM_ScoreMode.md) with code changes
2. Add examples to [score_mode_perplexity.py](score_mode_perplexity.py)
3. Update this index

---

## 📝 Version History

- **v1.0** - Initial implementation
  - Added `score_mode` parameter to `SamplingParams`
  - Added `target_token_ids` optimization to `TokensPrompt`
  - Created evaluation scripts and documentation

