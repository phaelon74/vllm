# Score Mode for Perplexity Evaluation

## Overview

Score mode is a feature in vLLM that enables accurate perplexity calculation on quantized models without decompressing weights. When enabled, vLLM computes exact log probabilities for all prompt tokens without generating new tokens.

## Motivation

Standard inference engines have a gap when it comes to perplexity evaluation of quantized models:

- **Scoring tools** (lm-eval, transformers) decompress quantized weights to FP16 during inference, defeating the purpose of quantization testing
- **Quantized inference engines** (vLLM) are optimized for generation and typically only return top-K logprobs, not exact probabilities for arbitrary tokens
- **Perplexity calculation** requires the exact log probability of the ground-truth token at each position

Score mode bridges this gap by providing exact log probabilities while keeping weights compressed.

## How It Works

When `score_mode=True` is set in `SamplingParams`:

1. **No Generation**: `max_tokens` is automatically set to 0
2. **Full Vocabulary Logprobs**: `prompt_logprobs` is set to -1 (return all vocab tokens)
3. **Exact Probabilities**: For each prompt token position, computes:
   ```
   logprob = logits[actual_token] - logsumexp(logits)
   ```
4. **No Decompression**: Quantized weights stay compressed throughout inference

## Usage

### Basic Example

```python
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

# Load a quantized model
llm = LLM(
    model="path/to/quantized/model",
    quantization="compressed-tensors",
)

# Tokenize your text
tokenizer = llm.get_tokenizer()
token_ids = tokenizer.encode("Your text here")

# Create sampling params with score_mode
sampling_params = SamplingParams(
    score_mode=True,
    temperature=0.0,
)

# Get logprobs for all prompt tokens
outputs = llm.generate(
    prompts=[TokensPrompt(prompt_token_ids=token_ids)],
    sampling_params=sampling_params,
)

# Access prompt logprobs
output = outputs[0]
for i, logprobs_dict in enumerate(output.prompt_logprobs):
    if logprobs_dict and i < len(token_ids):
        actual_token = token_ids[i]
        if actual_token in logprobs_dict:
            logprob = logprobs_dict[actual_token].logprob
            print(f"Token {i}: logprob={logprob:.4f}")
```

### Perplexity Calculation

```python
import math

def calculate_perplexity(llm, token_ids):
    sampling_params = SamplingParams(score_mode=True, temperature=0.0)
    
    outputs = llm.generate(
        prompts=[TokensPrompt(prompt_token_ids=token_ids)],
        sampling_params=sampling_params,
    )
    
    output = outputs[0]
    total_nll = 0.0
    num_tokens = 0
    
    # Skip first token (no context)
    for i in range(1, len(output.prompt_logprobs)):
        if output.prompt_logprobs[i]:
            actual_token = token_ids[i]
            if actual_token in output.prompt_logprobs[i]:
                logprob = output.prompt_logprobs[i][actual_token].logprob
                total_nll += -logprob
                num_tokens += 1
    
    avg_nll = total_nll / num_tokens
    perplexity = math.exp(avg_nll)
    
    return perplexity
```

### Sliding Window for Long Sequences

For sequences longer than the model's context length, use a sliding window:

```python
def calculate_perplexity_sliding_window(
    llm, token_ids, context_length=2048, stride=512
):
    sampling_params = SamplingParams(score_mode=True, temperature=0.0)
    
    total_nll = 0.0
    total_tokens = 0
    
    num_windows = (len(token_ids) - context_length) // stride + 1
    
    for i in range(num_windows):
        start_idx = i * stride
        end_idx = min(start_idx + context_length, len(token_ids))
        window_tokens = token_ids[start_idx:end_idx]
        
        outputs = llm.generate(
            prompts=[TokensPrompt(prompt_token_ids=window_tokens)],
            sampling_params=sampling_params,
        )
        
        output = outputs[0]
        
        # Determine which tokens to evaluate in this window
        # (avoid double-counting in overlapping regions)
        start_eval = 1 if i == 0 else stride
        end_eval = len(window_tokens)
        
        for j in range(start_eval, end_eval):
            if j < len(output.prompt_logprobs) and output.prompt_logprobs[j]:
                actual_token = window_tokens[j]
                if actual_token in output.prompt_logprobs[j]:
                    logprob = output.prompt_logprobs[j][actual_token].logprob
                    total_nll += -logprob
                    total_tokens += 1
    
    perplexity = math.exp(total_nll / total_tokens)
    return perplexity
```

## Examples

See `examples/score_mode_perplexity.py` for a complete example that includes:
- Loading quantized models
- WikiText-2 dataset evaluation
- Sliding window implementation
- Command-line interface

Run it with:
```bash
python examples/score_mode_perplexity.py \
    --model neuralmagic/Meta-Llama-3.1-8B-Instruct-W4A16 \
    --quantization compressed-tensors \
    --dataset wikitext-2-raw-v1 \
    --context-length 2048 \
    --stride 512
```

## API Reference

### SamplingParams

**`score_mode`** (bool, default: False)

If True, enables scoring mode for perplexity evaluation:
- Automatically sets `max_tokens=0` (no token generation)
- Automatically sets `prompt_logprobs=-1` (return all vocab logprobs)
- Computes exact log probabilities for all prompt tokens

### Output Format

When `score_mode=True`, the output will have:

- `prompt_logprobs`: List of dictionaries, one per prompt token
  - Each dict maps token_id → Logprob object
  - Logprob object contains:
    - `logprob` (float): Log probability of the token
    - `rank` (int): Rank of the token (1 = most likely)
    - `decoded_token` (str): Decoded string representation
- `outputs[0].token_ids`: Empty list (no tokens generated)
- `outputs[0].text`: Empty string (no text generated)

## Limitations

- Score mode requires `prompt_logprobs` to be enabled, which adds computational overhead
- For very large vocabularies (>100K tokens), returning all logprobs may be memory-intensive
- The first token in a sequence has no context, so its logprob is typically omitted from perplexity calculations

## Comparison with Alternatives

### vs. lm-evaluation-harness with `hf` backend
- ✅ Score mode: Tests quantized weights
- ❌ lm-eval + hf: Decompresses to FP16

### vs. vLLM OpenAI API
- ✅ Score mode: Exact logprobs for all tokens
- ❌ API: Only top-K logprobs, not suitable for perplexity

### vs. ExLlamaV3
- ✅ Score mode: Supports compressed-tensors format
- ❌ ExLlamaV3: Only supports EXL2/EXL3 formats

## See Also

- [Perplexity Evaluation Guide](perplexity_evaluation.md)
- [Quantization Guide](quantization.md)
- [Compressed Tensors Format](compressed_tensors.md)

