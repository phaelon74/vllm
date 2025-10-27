# vLLM Score Mode Documentation

This folder contains all documentation and examples related to the **Score Mode** feature added to vLLM for accurate perplexity calculation on quantized models.

## Contents

### Core Documentation

1. **[VLLM_ScoreMode.md](VLLM_ScoreMode.md)** - **START HERE**
   - Complete documentation of all changes made to vLLM
   - Code modifications to `sampling_params.py`, `inputs.py`, and `sampler.py`
   - Usage examples and testing instructions
   - Performance considerations

2. **[SCORE_MODE_IMPLEMENTATION.md](SCORE_MODE_IMPLEMENTATION.md)**
   - High-level overview of the feature
   - Problem statement and solution approach
   - Benefits and comparisons with alternatives

### Example Scripts

3. **[test_score_mode.py](test_score_mode.py)**
   - Simple test script using facebook/opt-125m
   - Validates basic score_mode functionality
   - Quick sanity check

4. **[score_mode_perplexity.py](score_mode_perplexity.py)**
   - Complete perplexity evaluation script
   - EXL3-compatible sliding window methodology
   - WikiText-2 dataset support
   - Command-line interface
   - Support for quantized models (compressed-tensors, NVFP4, etc.)

## Quick Start

### Test Score Mode
```bash
cd /path/to/vllm/repo
python ScoreMode/test_score_mode.py
```

### Run Perplexity Evaluation

**On WikiText-2:**
```bash
python ScoreMode/score_mode_perplexity.py \
    --model facebook/opt-125m \
    --dataset wikitext \
    --dataset-config wikitext-2-raw-v1 \
    --context-length 2048 \
    --stride 512
```

**On Quantized Model:**
```bash
python ScoreMode/score_mode_perplexity.py \
    --model neuralmagic/Meta-Llama-3.1-8B-Instruct-W4A16 \
    --quantization compressed-tensors \
    --text "The quick brown fox jumps over the lazy dog"
```

**On Large Model with Multi-GPU:**
```bash
python ScoreMode/score_mode_perplexity.py \
    --model /path/to/large/model \
    --quantization compressed-tensors \
    --tensor-parallel-size 2 \
    --dataset wikitext \
    --dataset-config wikitext-2-raw-v1 \
    --context-length 2048 \
    --stride 512 \
    --gpu-memory-utilization 0.80 \
    --disable-custom-all-reduce
```

## What is Score Mode?

Score mode is a special mode in vLLM that:

1. **Disables token generation** (`max_tokens=0`)
2. **Enables full vocabulary logprobs** (`prompt_logprobs=-1`)
3. **Returns exact log probabilities** for every token in the prompt
4. **Maintains quantization** - keeps compressed-tensors weights compressed

This allows you to:
- Calculate accurate perplexity on quantized models
- Test the actual quantized weights (no FP16 decompression)
- Use multi-GPU for large models
- Work with standard formats (compressed-tensors, NVFP4, W4A16)

## Key Features

✅ **Accurate perplexity on quantized models** - Test quantized weights without decompression  
✅ **Multi-GPU support** - Works with tensor parallelism for 70B+ models  
✅ **Standard formats** - Works with compressed-tensors (NVFP4/W4A16)  
✅ **Exact log probabilities** - No approximations or top-K limitations  
✅ **EXL3-compatible** - Produces comparable results to EXL2/EXL3 benchmarks  

## Files Modified in vLLM Core

The following vLLM source files were modified:

1. `vllm/sampling_params.py` - Added `score_mode` parameter
2. `vllm/inputs.py` - Added `target_token_ids` optimization
3. `vllm/v1/sample/sampler.py` - Added fast path for target tokens

See [VLLM_ScoreMode.md](VLLM_ScoreMode.md) for complete details of all changes.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  User Code                                                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  llm = LLM(model="W4A16", quantization="ct")        │   │
│  │  sampling_params = SamplingParams(score_mode=True)  │   │
│  │  outputs = llm.generate(prompts, sampling_params)   │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  vLLM Core (Modified)                                       │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  SamplingParams.__post_init__():                    │   │
│  │    if score_mode:                                   │   │
│  │      max_tokens = 0  # No generation                │   │
│  │      prompt_logprobs = -1  # All vocab             │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Sampler.gather_logprobs():                         │   │
│  │    if target_token_ids:                             │   │
│  │      return logprobs.gather(target_token_ids)       │   │
│  │    else:                                            │   │
│  │      return logprobs  # Full vocab                  │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Result: Exact logprobs for all prompt tokens              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  output.prompt_logprobs[i][token_id].logprob        │   │
│  │  → Exact log probability for each token             │   │
│  │  → Computed from quantized weights (no decompress)  │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Comparison with Alternatives

| Tool | Tests Quantized | Exact Logprobs | Compressed-Tensors |
|------|----------------|----------------|-------------------|
| lm-eval + HF | ❌ (decompresses) | ✅ | ⚠️ (decompresses) |
| lm-eval + vLLM | ✅ | ❌ (top-K) | ✅ |
| vLLM API | ✅ | ❌ (top-K) | ✅ |
| ExLlamaV3 | ✅ | ✅ | ❌ (EXL2/3 only) |
| **Score Mode** | **✅** | **✅** | **✅** |

## Support

For issues or questions:
1. Check [VLLM_ScoreMode.md](VLLM_ScoreMode.md) for detailed documentation
2. Review example scripts for usage patterns
3. Ensure you're using the modified vLLM version with score_mode support

## Credits

This implementation was designed to provide a clean, minimal patch to vLLM that leverages existing infrastructure to enable exact log probability calculation on quantized models.

