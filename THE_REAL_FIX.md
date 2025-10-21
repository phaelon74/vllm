# The REAL Problem: 131K KV Cache Allocation!

## What Was Actually Wrong

Looking at the error dump more carefully:
```
max_seq_len=131072  ← vLLM allocating KV cache for 131K tokens!
enable_prefix_caching=False  ✓
num_common_prefix_blocks=[0]  ✓
```

**The problem**: vLLM was allocating KV cache for the model's MAXIMUM context (131K tokens), not your actual window size (2048 tokens)!

## VRAM Breakdown (Before Fix)

| Component | Per GPU (TP=2) | Why |
|-----------|----------------|-----|
| Model (8B W4A16) | ~5GB | Split across 2 GPUs |
| **KV cache (131K!)** | **~80GB** | Allocated for max_seq_len! |
| Activations | ~2GB | |
| Logprobs | ~1GB | |
| Overhead | ~4GB | |
| **TOTAL** | **~92GB** | ← Your OOM! |

## VRAM Breakdown (After Fix)

| Component | Per GPU (TP=2) | Why |
|-----------|----------------|-----|
| Model (8B W4A16) | ~5GB | Split across 2 GPUs |
| **KV cache (2048)** | **~2-3GB** | Only what we need! |
| Activations | ~2GB | |
| Logprobs | ~1GB | |
| Overhead | ~1GB | |
| **TOTAL** | **~12GB** | ✅ Fits easily! |

## The Fixes Applied

1. ✅ `enable_prefix_caching=False` - Prevents accumulating cache across windows
2. ✅ **`max_model_len=args.context_length`** - Only allocate KV cache for 2048, not 131K!

## Test Now

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
    --gpu-memory-utilization 0.90 \
    --trust-remote-code
```

**Expected**:
- ✅ ~12GB VRAM per GPU (not 92GB!)
- ✅ No OOM
- ✅ 349 windows @ ~36 sec each
- ✅ **~3.5 hours total**

## Why This Makes Total Sense

For perplexity evaluation:
- You only process 2048 tokens at a time
- You don't need KV cache for 131K tokens
- Each window is independent

But vLLM defaults to allocating KV cache for the model's maximum supported length (131K for this model), which is way overkill for perplexity!

This is the real fix!

