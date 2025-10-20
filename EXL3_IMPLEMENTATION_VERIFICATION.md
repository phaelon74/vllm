# EXL3 Implementation Verification

## Developer's Exact Description

From the EXL3 developer:

```python
# 1. Tokenize the entire test set
tokenized_set = tokenize(test_data)

# 2. Create overlapping windows
input_ids[0] = tokenized_set[0:2048]
input_ids[1] = tokenized_set[512:2560]
input_ids[2] = tokenized_set[1024:3072]
# etc.

# 3. Process each window independently
for each input_ids:
    # Treat each window as a NEW sequence
    logits = model.forward(input_ids[i])
    logprobs = log_softmax(logits)
    
    # Gather target token logprobs
    gather targets from logprobs
    
    # Sum up gathered logprobs
    sum up gathered logprobs

# 4. Final perplexity
ppl = exp(-mean of all logprobs)
```

## Our Implementation (Verified)

### Step 1: Tokenize
```python
# In main()
token_ids = tokenizer.encode(text)  # Entire test set tokenized
```
✅ **Matches**: Tokenizes entire test set upfront

### Step 2: Create Overlapping Windows
```python
# In calculate_perplexity()
num_windows = (len(token_ids) - context_length) // stride + 1
for i in range(num_windows):
    start_idx = i * stride              # 0, 512, 1024, ...
    end_idx = start_idx + context_length  # 2048, 2560, 3072, ...
    windows.append((start_idx, end_idx))

# Produces:
# windows[0] = (0, 2048)      → token_ids[0:2048]
# windows[1] = (512, 2560)    → token_ids[512:2560]
# windows[2] = (1024, 3072)   → token_ids[1024:3072]
```
✅ **Matches**: Creates overlapping windows with stride=512

### Step 3: Process Each Window
```python
for i in range(num_windows):
    start_idx, end_idx = windows[i]
    window_tokens = token_ids[start_idx:end_idx]  # Extract window
    
    # Treat as NEW sequence - forward pass for THIS window only
    outputs = llm.generate(
        prompts=[TokensPrompt(prompt_token_ids=window_tokens)],
        sampling_params=sampling_params,
    )
    
    # logprobs already computed via log_softmax in vLLM's Sampler
    output = outputs[0]
    
    # Gather target logprobs (skip position 0 with no context)
    for j in range(1, len(window_tokens)):
        actual_token = window_tokens[j]
        logprob = output.prompt_logprobs[j][actual_token].logprob
        
        # Sum up gathered logprobs
        total_nll += -logprob  # Accumulate negative log prob
        total_tokens += 1       # Count tokens
```
✅ **Matches**: Each window is treated as independent sequence, logprobs gathered and summed

### Step 4: Final Perplexity
```python
# ppl = exp(-mean of all logprobs)
avg_nll = total_nll / total_tokens  # mean of -logprob
perplexity = math.exp(avg_nll)       # exp(-mean(logprobs))
```
✅ **Matches**: `exp(-mean(all logprobs))`

## Mathematical Verification

Given the EXL3 formula: `ppl = exp(-mean(all logprobs))`

Our implementation:
```
total_nll = Σ(-logprob_i)           # Sum of negative log probs
avg_nll = total_nll / N              # Mean of negative log probs
ppl = exp(avg_nll)                   # exp(-mean(logprobs))
```

**Proof of equivalence**:
```
avg_nll = (1/N) × Σ(-logprob_i)
        = -(1/N) × Σ(logprob_i)
        = -mean(logprobs)

ppl = exp(avg_nll)
    = exp(-mean(logprobs))  ✅ Matches EXL3 formula
```

## Example Trace (500 samples)

### Input
- Dataset: WikiText-2, 500 samples
- Total tokens: ~39,217
- Context: 2048
- Stride: 512

### Window Creation
```
num_windows = (39217 - 2048) // 512 + 1 = 73

Window  0: tokens[    0: 2048]
Window  1: tokens[  512: 2560]
Window  2: tokens[ 1024: 3072]
...
Window 72: tokens[36864:38912] (or shorter if near end)
```

### Processing Each Window
```
Window 0:
  - Forward pass: token_ids[0:2048] (2048 tokens)
  - Evaluate positions: 1-2047 (2047 tokens)
  - Accumulate: total_nll += sum of 2047 negative logprobs

Window 1:
  - Forward pass: token_ids[512:2560] (2048 tokens)
  - Evaluate positions: 1-2047 (2047 tokens)
  - Note: Tokens 512-2047 are EVALUATED AGAIN (different context)
  - Accumulate: total_nll += sum of 2047 negative logprobs

Window 2:
  - Forward pass: token_ids[1024:3072] (2048 tokens)
  - Evaluate positions: 1-2047 (2047 tokens)
  - Note: Tokens 1024-2559 are EVALUATED AGAIN (3rd time for some)
  - Accumulate: total_nll += sum of 2047 negative logprobs

... (repeat for all 73 windows)
```

### Token Evaluation Counts

| Token Position | Evaluated In Windows | Times Evaluated |
|----------------|---------------------|-----------------|
| 0 | - | 0 (skip, no context) |
| 1-511 | Window 0 only | 1x |
| 512-1023 | Windows 0, 1 | 2x |
| 1024-1535 | Windows 0, 1, 2 | 3x |
| 1536-2047 | Windows 0, 1, 2, 3 | 4x |
| 2048-2559 | Windows 1, 2, 3, 4 | 4x |
| ... | ... | 4x (steady state) |
| ~37000-38000 | Last 4 windows | 4x → 3x → 2x → 1x |

**Total evaluations**: ~149,431 token evaluations (vs ~39,216 unique tokens)
**Average evaluations per token**: ~3.8x

### Final Perplexity
```
total_nll = sum of all 149,431 negative log probs
total_tokens = 149,431
avg_nll = total_nll / 149431
ppl = exp(avg_nll)
```

## Key Insights

### 1. Each Window is Independent
✅ No KV cache reuse between windows  
✅ Each window starts fresh (position 0 = start of sequence)  
✅ This is what "treat each window as a new sequence" means  

### 2. Overlapping Tokens Evaluated Multiple Times
✅ Token at position 1500 evaluated in Windows 0, 1, 2, 3  
✅ Each evaluation has different context (0:1499 vs 512:1499 vs 1024:1499 vs 1536:1499)  
✅ All 4 evaluations contribute to final average  

### 3. Why This Lowers Perplexity
- Early in window: "cold" context (few prior tokens)
- Late in window: "warm" context (many prior tokens)
- Most tokens are evaluated in their "warm" state multiple times
- This biases the average toward better (lower) perplexity

### 4. Implementation Status
✅ **vLLM core changes**: Correct (enable full vocab logprobs)  
✅ **Script logic**: Correct (matches EXL3 exactly)  
✅ **Comments**: Updated to reflect EXL3 methodology explicitly  

## Conclusion

**The implementation is CORRECT and matches the EXL3 developer's description exactly.**

No code changes were needed - only documentation/comments were updated to make the methodology explicit.

### Verification Checklist
- ✅ Windows created with stride (0:2048, 512:2560, 1024:3072, ...)
- ✅ Each window processed independently (new forward pass)
- ✅ ALL tokens in each window evaluated (positions 1 to end)
- ✅ Overlapping tokens evaluated multiple times
- ✅ All logprobs summed across all windows
- ✅ Final perplexity: exp(-mean(all logprobs))
- ✅ Comments clearly explain EXL3 methodology

### Ready to Use
The script is now fully documented and ready for EXL3-comparable perplexity evaluation.

**Command to run**:
```bash
python examples/score_mode_perplexity.py \
    --model /media/fmodels/TheHouseOfTheDude/Llama-3.1-8B-Instruct/W4A16/ \
    --quantization compressed-tensors \
    --dataset wikitext \
    --dataset-config wikitext-2-raw-v1 \
    --num-samples 500 \
    --context-length 2048 \
    --stride 512 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.95 \
    --trust-remote-code
```

**Expected results**:
- 73 windows processed
- ~149,431 total token evaluations
- ~34 min per window
- ~41 hours total runtime
- Perplexity score directly comparable to EXL3

