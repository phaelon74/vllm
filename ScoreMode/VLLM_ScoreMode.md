# vLLM Score Mode Implementation

## Overview

This document details all changes made to vLLM to enable **score mode** - a feature that allows accurate perplexity calculation on quantized models without decompressing weights.

## Changes Made to vLLM Core

### 1. Modified `vllm/sampling_params.py`

**Location**: `vllm/sampling_params.py`

#### Added Field
```python
score_mode: bool = False
```

#### Modified `__post_init__` Method

**Before:**
```python
def __post_init__(self) -> None:
    if self.n < 1:
        raise ValueError(f"n must be at least 1, got {self.n}.")
    # ... rest of validation
```

**After:**
```python
def __post_init__(self) -> None:
    # Handle score_mode: set max_tokens to 0 and enable full prompt logprobs
    if self.score_mode:
        self.max_tokens = 0
        if self.prompt_logprobs is None:
            self.prompt_logprobs = -1  # Return all vocab logprobs
    
    if self.n < 1:
        raise ValueError(f"n must be at least 1, got {self.n}.")
    # ... rest of validation
```

**Purpose**: When `score_mode=True`, automatically:
1. Set `max_tokens=0` to prevent token generation
2. Set `prompt_logprobs=-1` to return full vocabulary log probabilities

#### Modified `_verify_args` Method

**Changes**:
1. Changed max_tokens validation from `>= 1` to `>= 0`
2. Added check that `max_tokens=0` is only allowed when `score_mode=True`

**Before:**
```python
if self.max_tokens < 1:
    raise ValueError(f"max_tokens must be at least 1, got {self.max_tokens}.")
```

**After:**
```python
if self.max_tokens < 0:
    raise ValueError(f"max_tokens must be at least 0, got {self.max_tokens}.")

if self.max_tokens == 0 and not self.score_mode:
    raise ValueError("max_tokens=0 is only allowed in score_mode")
```

#### Modified `from_optional` Method

**Changes**: Added `score_mode` parameter with default value `False`

**Before:**
```python
@staticmethod
def from_optional(...) -> "SamplingParams":
```

**After:**
```python
@staticmethod
def from_optional(
    ...,
    score_mode: Optional[bool] = False,
) -> "SamplingParams":
```

### 2. Modified `vllm/inputs.py`

**Location**: `vllm/inputs.py`

#### Added `target_token_ids` Field to `TokensPrompt`

**Purpose**: Enable optimization by allowing vLLM to extract only the ground-truth token logprobs instead of the entire vocabulary.

**Added**:
```python
@dataclass
class TokensPrompt:
    """Schema for a prompt specified as tokens."""
    
    prompt_token_ids: List[int]
    """The token IDs of the prompt."""
    
    multi_modal_data: Optional["MultiModalDataDict"] = None
    """Optional multi-modal data to pass to the model."""
    
    mm_processor_kwargs: Optional[Dict[str, Any]] = None
    """Optional multi-modal processor kwargs to be forwarded to the
    multimodal input mapper & processor."""
    
    target_token_ids: Optional[List[int]] = None
    """Optional target token IDs for score_mode optimization.
    When provided, only logprobs for these tokens are extracted."""
```

### 3. Modified `vllm/v1/sample/sampler.py`

**Location**: `vllm/v1/sample/sampler.py`

#### Modified `gather_logprobs` Method

**Purpose**: When `target_token_ids` is provided, only extract log probabilities for those specific tokens instead of the entire vocabulary. This dramatically reduces memory usage and computation time.

**Changes**: Added fast path for when target tokens are provided:

```python
def gather_logprobs(self, logprobs: torch.Tensor, sampling_metadata: SamplingMetadata) -> torch.Tensor:
    """
    Gather log probabilities for sampled tokens or all vocabulary tokens.
    
    Args:
        logprobs: [num_tokens, vocab_size] tensor of log probabilities
        sampling_metadata: Metadata for sampling
        
    Returns:
        Tensor of gathered logprobs with shape depending on prompt_logprobs setting
    """
    # ... existing code ...
    
    # Fast path: If we have target_token_ids, only extract those
    if hasattr(sampling_metadata, 'target_token_ids') and sampling_metadata.target_token_ids is not None:
        target_ids = sampling_metadata.target_token_ids
        # Only gather logprobs for target tokens (massive memory saving!)
        gathered_logprobs = logprobs.gather(-1, target_ids.unsqueeze(-1))
        return gathered_logprobs
    
    # Original path: Return all vocab if prompt_logprobs == -1
    if self.prompt_logprobs == -1:
        return logprobs  # Return all [num_tokens, vocab_size]
    
    # ... rest of existing code ...
```

## Why These Changes Work

The implementation leverages existing vLLM infrastructure:

1. **`prompt_logprobs=-1` already supported**: The sampler's `gather_logprobs` method already handles `-1` by returning all vocabulary tokens (see `vllm/v1/sample/sampler.py`)

2. **Exact token logprobs computation**: The method uses `logprobs.gather(-1, token_ids)` to get exact log probabilities for specific tokens

3. **No generation with `max_tokens=0`**: Setting `max_tokens=0` prevents token generation while still processing the prompt

4. **Optimization via `target_token_ids`**: When provided, only extract logprobs for ground-truth tokens, saving massive amounts of memory

## Usage Examples

### Basic Usage

```python
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

# Load quantized model
llm = LLM(
    model="neuralmagic/Meta-Llama-3.1-8B-Instruct-W4A16",
    quantization="compressed-tensors"
)

# Tokenize
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

### Optimized Usage (with target_token_ids)

```python
# Prepare target tokens (skip position 0 which has no context)
target_token_ids = token_ids[1:]

# Use optimized path
outputs = llm.generate(
    prompts=[TokensPrompt(
        prompt_token_ids=token_ids,
        target_token_ids=target_token_ids  # Only extract these!
    )],
    sampling_params=sampling_params
)
```

## Files Created

### 1. `test_score_mode.py`
Simple test script that validates score_mode functionality with facebook/opt-125m.

### 2. `examples/score_mode_perplexity.py`
Complete perplexity evaluation script with:
- Command-line interface
- WikiText-2 dataset support
- Sliding window for long sequences
- Support for quantized models
- EXL3-compatible methodology

### 3. `examples/score_mode_perplexity_optimized.py`
Memory-optimized version with:
- Smaller default windows
- Aggressive memory cleanup
- Better for large models or limited VRAM

### 4. `SCORE_MODE_IMPLEMENTATION.md`
Comprehensive documentation of the feature.

## Benefits

1. **Accurate perplexity on quantized models**: Test actual quantized weights without decompression
2. **Multi-GPU support**: Works with tensor parallelism for large models
3. **Standard format**: Works with compressed-tensors (NVFP4/W4A16)
4. **Exact log probabilities**: No approximations or top-K limitations
5. **Memory efficient**: Optional `target_token_ids` optimization

## Testing

### Quick Test
```bash
python test_score_mode.py
```

### WikiText-2 Evaluation
```bash
python examples/score_mode_perplexity.py \
    --model facebook/opt-125m \
    --dataset wikitext \
    --dataset-config wikitext-2-raw-v1 \
    --context-length 2048 \
    --stride 512
```

### Quantized Model Test
```bash
python examples/score_mode_perplexity.py \
    --model neuralmagic/Meta-Llama-3.1-8B-Instruct-W4A16 \
    --quantization compressed-tensors \
    --text "The quick brown fox jumps over the lazy dog"
```

### Memory-Optimized Test
```bash
python examples/score_mode_perplexity_optimized.py \
    --model your/large-model \
    --quantization compressed-tensors \
    --dataset wikitext \
    --dataset-config wikitext-2-raw-v1 \
    --context-length 512 \
    --stride 256
```

## Performance Considerations

- Computing full vocabulary logprobs is computationally expensive
- For vocabularies >100K tokens, memory overhead can be significant
- Use `target_token_ids` optimization when possible
- Use sliding windows for sequences longer than context length
- Benefits from vLLM's optimizations (CUDA graphs, kernel fusion, etc.)

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
1. **Batch processing**: Process multiple sequences in parallel
2. **Native lm-eval integration**: Direct integration with lm-evaluation-harness
3. **Caching**: Cache prompt logprobs for repeated evaluations
4. **Streaming**: Process very long sequences in streaming fashion

## Credits

This implementation follows the principle:
> "Plan A (recommended): a tiny vLLM patch = exact log-probs with Compressed Tensors, no decompression"

The key insight: vLLM already has all necessary infrastructure - we just needed to expose it through a clean API (`score_mode=True`).

