# Score Mode Implementation for vLLM

## Overview

This implementation adds a "score mode" capability to vLLM that enables accurate perplexity calculation on quantized models (e.g., compressed-tensors format) without decompressing weights to FP16/BF16.

## Problem Statement

Standard tools for perplexity evaluation have a fundamental limitation when working with quantized models:

- **Transformers/lm-eval**: Automatically decompress quantized weights to FP16 during inference, defeating the purpose of testing quantization
- **vLLM's existing API**: Only returns top-K logprobs (optimized for generation), not exact probabilities for arbitrary tokens needed for perplexity calculation

This creates a gap where it's impossible to accurately measure perplexity of compressed-tensors quantized models while keeping weights compressed.

## Solution

The implementation adds a `score_mode` flag to `SamplingParams` that:

1. **Disables generation**: Automatically sets `max_tokens=0`
2. **Enables full vocabulary logprobs**: Sets `prompt_logprobs=-1` to return exact log probabilities for ALL tokens
3. **Computes exact probabilities**: For each prompt token position, computes:
   ```
   logprob = logits[actual_token] - logsumexp(logits)
   ```
4. **Maintains quantization**: Keeps weights compressed throughout inference

## Implementation Details

### Changes Made

#### 1. `vllm/sampling_params.py`

**Added field:**
```python
score_mode: bool = False
```

**Modified `__post_init__`:**
```python
def __post_init__(self) -> None:
    # Handle score_mode: set max_tokens to 0 and enable full prompt logprobs
    if self.score_mode:
        self.max_tokens = 0
        if self.prompt_logprobs is None:
            self.prompt_logprobs = -1  # Return all vocab logprobs
    # ... rest of existing code
```

**Modified `_verify_args`:**
- Allow `max_tokens=0` when `score_mode=True`
- Updated validation to check: `max_tokens >= 0` (was `>= 1`)
- Added check: `max_tokens=0` only allowed in score_mode

**Updated `from_optional`:**
- Added `score_mode` parameter with default `False`

### Why This Works

The existing vLLM infrastructure already supports:

1. **`prompt_logprobs=-1`**: The sampler's `gather_logprobs` method already handles `-1` by returning all vocabulary tokens (see `vllm/v1/sample/sampler.py:190-196`)

2. **Exact token logprobs**: The `gather_logprobs` method uses `logprobs.gather(-1, token_ids)` to get exact log probabilities for specific tokens, not just top-K

3. **Prompt processing without generation**: Setting `max_tokens=0` prevents token generation while still processing the prompt

By combining these existing capabilities, score_mode provides a clean interface for perplexity evaluation.

## Usage

### Basic Example

```python
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

# Load quantized model
llm = LLM(
    model="path/to/W4A16/model",
    quantization="compressed-tensors"
)

# Tokenize text
tokenizer = llm.get_tokenizer()
token_ids = tokenizer.encode("Your text here")

# Enable score mode
sampling_params = SamplingParams(
    score_mode=True,
    temperature=0.0
)

# Get exact logprobs for all prompt tokens
outputs = llm.generate(
    prompts=[TokensPrompt(prompt_token_ids=token_ids)],
    sampling_params=sampling_params
)

# Calculate perplexity
import math
output = outputs[0]
total_nll = sum(
    -output.prompt_logprobs[i][token_ids[i]].logprob
    for i in range(1, len(token_ids))
    if token_ids[i] in output.prompt_logprobs[i]
)
perplexity = math.exp(total_nll / (len(token_ids) - 1))
print(f"Perplexity: {perplexity:.4f}")
```

### Complete Examples

1. **`test_score_mode.py`**: Simple test with facebook/opt-125m model
2. **`examples/score_mode_perplexity.py`**: Full perplexity calculation with:
   - Command-line interface
   - WikiText-2 dataset support
   - Sliding window for long sequences
   - Support for quantized models

## Testing

Run the test script:
```bash
python test_score_mode.py
```

Run perplexity evaluation on WikiText-2:
```bash
python examples/score_mode_perplexity.py \
    --model facebook/opt-125m \
    --dataset wikitext-2-raw-v1 \
    --context-length 2048 \
    --stride 512
```

Test with a quantized model:
```bash
python examples/score_mode_perplexity.py \
    --model neuralmagic/Meta-Llama-3.1-8B-Instruct-W4A16 \
    --quantization compressed-tensors \
    --text "The quick brown fox jumps over the lazy dog"
```

## Benefits

1. **Accurate perplexity on quantized models**: Test the actual quantized weights without decompression
2. **Multi-GPU support**: Works with tensor parallelism for large models (70B+)
3. **Standard format**: Works with compressed-tensors (NVFP4/W4A16) stored in Hugging Face format
4. **Exact log probabilities**: No approximations or top-K limitations
5. **Comparable results**: Produces results comparable to standard benchmarks (EXL2/EXL3)

## Performance Considerations

- Computing full vocabulary logprobs (`prompt_logprobs=-1`) is computationally expensive
- For large vocabularies (>100K tokens), consider the memory overhead
- Use sliding windows for sequences longer than the model's context length
- The implementation benefits from vLLM's optimizations (CUDA graphs, kernel fusion, etc.)

## Comparison with Alternatives

| Method | Tests Quantized Weights | Exact Logprobs | Supports Compressed-Tensors |
|--------|------------------------|----------------|----------------------------|
| lm-eval + hf | ❌ (decompresses) | ✅ | ⚠️ (but decompresses) |
| lm-eval + vLLM | ✅ | ❌ (top-K only) | ✅ |
| vLLM API | ✅ | ❌ (top-K only) | ✅ |
| ExLlamaV3 | ✅ | ✅ | ❌ (EXL2/EXL3 only) |
| **vLLM Score Mode** | **✅** | **✅** | **✅** |

## Future Enhancements

Potential improvements:

1. **Batch processing**: Process multiple sequences in parallel for faster evaluation
2. **Memory optimization**: Option to compute only ground-truth token logprobs (not full vocab) for memory savings
3. **Integration with lm-eval**: Native integration with lm-evaluation-harness
4. **Caching**: Cache prompt logprobs for repeated evaluations

## Documentation

See `docs/source/features/score_mode.md` for complete user-facing documentation.

## Credits

This implementation follows the recommendation:
> "Plan A (recommended): a tiny vLLM patch = exact log-probs with Compressed Tensors, no decompression"

The key insight is that vLLM already has all the necessary infrastructure - we just needed to expose it through a clean API (`score_mode=True`) that automatically configures the right parameters.

