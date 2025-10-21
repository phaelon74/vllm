# Final Fix: Correct Logprob Extraction

## The Critical Bug

I was treating `logprobs_tensor[pos, token_id]` as if `token_id` was a direct index, but:

**Reality**: When `gather_logprobs` is called with `num_logprobs=-1` (full vocab), it does:
```python
topk_logprobs, topk_indices = torch.topk(logprobs, vocab_size, dim=-1)
```

This returns tokens **sorted by logprob** (highest first), NOT by token ID!

So:
- `token_ids_tensor[pos, 0]` = highest probability token at position `pos`
- `token_ids_tensor[pos, 1]` = 2nd highest probability token
- ...
- `logprobs_tensor[pos, i]` = logprob of the `i`-th highest probability token

**My bug**: I was doing `logprobs_tensor[pos, target_token_id]`, treating the second dimension as token ID.

**Correct**: I need to **search** for where `target_token_id` appears in `token_ids_tensor[pos]`, then use that index.

## The Fix

```python
# Create a comparison mask to find where target token appears
target_expanded = target_token_ids_tensor.unsqueeze(1)  # [num_pos, 1]
matches = (token_ids_tensor == target_expanded)  # [num_pos, vocab_size]

# Find the index (in sorted list) where target token appears
indices = matches.long().argmax(dim=1)  # [num_pos]

# Extract logprobs at those indices
target_logprobs = logprobs_tensor[position_indices, indices]

# Rank = position in sorted list (index + 1 for 1-based ranking)
target_ranks = indices + 1
```

## Why This Fixes Perplexity

**Before**: Extracting logprobs at position=target_token_id → completely wrong tokens → nonsense perplexity (2270782)

**After**: Search for target_token_id in sorted list → extract correct logprob → correct perplexity (~7-15)

## Rebuild & Test

```bash
cd /path/to/vllm
rm -rf build/ dist/ *.egg-info
find . -name "*.pyc" -delete
find . -name "__pycache__" -delete
pip uninstall vllm -y
pip install -e . --no-build-isolation
```

Then:
```bash
python examples/score_mode_perplexity.py \
    --model /media/fmodels/TheHouseOfTheDude/Llama-3.1-8B-Instruct/W4A16/ \
    --quantization compressed-tensors \
    --dataset wikitext \
    --dataset-config wikitext-2-raw-v1 \
    --num-samples 10 \
    --context-length 2048 \
    --stride 512 \
    --gpu-memory-utilization 0.95 \
    --trust-remote-code
```

**Expected**:
- ✅ ~30-40 seconds (optimization working!)
- ✅ **Perplexity: ~7-15** (CORRECT VALUE!)
- ✅ Clean exit

This should finally give correct results!

