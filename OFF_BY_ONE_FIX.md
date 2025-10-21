# Off-By-One Bug Fix

## The Problem

Your W4A16 model was getting perplexity **7.607**, which is BETTER than FP32's **7.635**. This is impossible - quantization should make it worse!

## Root Cause: TWO Bugs

### Bug 1: Misaligned Indexing in Script (FIXED)

**Location**: `examples/score_mode_perplexity.py`

```python
# WRONG (old code):
for j in range(start_eval, end_eval):  # j = 1, 2, 3, ...
    actual_token = window_tokens[j]
    if actual_token in output.prompt_logprobs[j]:  # BUG!
```

- `window_tokens[1]` = token at position 1
- `prompt_logprobs[1]` = logprobs for position **2** (position 0 is excluded!)

**We were comparing the wrong tokens!**

**Fix**:
```python
# CORRECT (new code):
for i in range(len(output.prompt_logprobs)):  # i = 0, 1, 2, ...
    actual_token = window_tokens[i + 1]  # +1 to account for position 0 being excluded
    if actual_token in output.prompt_logprobs[i]:  # Now aligned!
```

### Bug 2: Wrong Extraction Logic in Fast Path (FIXED)

**Location**: `vllm/v1/engine/logprobs.py` - `_update_prompt_logprobs_fast_path()`

The fast path was treating the Sampler output as if it contained the FULL vocabulary, when it actually already extracted ONLY the target tokens!

**What the Sampler returns** (when using `target_token_ids`):
```python
# gather_target_logprobs() returns:
LogprobsTensors(
    indices=target_token_ids_2d.to(torch.int32),  # shape: [num_positions, 1]
    logprobs=target_logprobs,                      # shape: [num_positions, 1]
    ranks=target_ranks                             # shape: [num_positions]
)
```

**Old (WRONG) fast path**:
```python
# Tried to search for target tokens in full vocab (which doesn't exist!)
matches = (token_ids_tensor == target_expanded)  # BUG!
indices = matches.long().argmax(dim=1)  # Returns wrong indices
```

**New (CORRECT) fast path**:
```python
# Data is already extracted by Sampler - just squeeze and transfer!
target_logprobs_cpu = logprobs_tensor.squeeze(-1).cpu().tolist()
target_ranks_cpu = ranks_tensor.cpu().tolist()
target_token_ids_cpu = token_ids_tensor.squeeze(-1).cpu().tolist()
```

## How These Bugs Combined

1. **Fast path extracted wrong tokens** (e.g., token 271 instead of 284)
2. **Script indexing was off-by-one**, so it compared wrong positions
3. **Sometimes they accidentally aligned** and found a match
4. **When they didn't align**, tokens were silently skipped
5. **Skipping difficult tokens** → artificially low (better) perplexity

## Rebuild and Test

```bash
cd ~/vllm/vllm
pip uninstall vllm -y
rm -rf build/ dist/ *.egg-info
pip install -e . --force-reinstall
```

Then test:
```bash
python examples/score_mode_perplexity.py \
    --model /media/fmodels/TheHouseOfTheDude/Llama-3.1-8B-Instruct/W4A16/ \
    --quantization compressed-tensors \
    --dataset wikitext \
    --dataset-config wikitext-2-raw-v1 \
    --num-samples 2612 \
    --context-length 2048 \
    --stride 512 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.30 \
    --trust-remote-code
```

**Expected**: Perplexity should now be **>7.635** (worse than FP32, as expected for quantized models!)

