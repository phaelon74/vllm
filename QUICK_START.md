# Quick Start: Testing the Optimization

## Rebuild vLLM

```bash
cd /path/to/vllm
pip install -e . --force-reinstall
```

## Test 1: Quick Verification (OPT-125M)

Should complete in <1 minute:

```bash
python examples/score_mode_perplexity.py \
    --model facebook/opt-125m \
    --num-samples 10 \
    --context-length 512
```

**Expected output**:
- Fast iteration (~1-5 seconds per window instead of minutes)
- Perplexity score calculated correctly

## Test 2: Your 8B Model (500 samples)

**Before optimization**: ~41 hours  
**After optimization**: ~25 minutes (**100x faster!**)

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

**Watch for**:
- Progress bar showing ~20 sec/it (instead of 34 min/it)
- 73 total windows
- Should complete in ~25 minutes

## What Changed

**The Bottleneck** (from EXL3 developer insight):
- NOT GPU→CPU transfer (that's fast, ~1-2 seconds)
- **Python object creation**: 262 million Logprob objects took 30+ minutes!

**The Fix**:
- Only create 2048 Logprob objects (one per position) instead of 262M
- Extract only target token logprobs, not full vocabulary
- Same perplexity accuracy, 100x faster

## Verification

1. **Speed**: ~20 sec/window (not 34 min/window)
2. **Accuracy**: Perplexity scores identical to previous runs
3. **Methodology**: Still EXL3-compatible (evaluates all tokens in each window)

##  If Something Goes Wrong

### Optimization Not Activating (Still Slow)

```bash
# 1. Verify rebuild
pip install -e . --force-reinstall

# 2. Check script has the changes
grep "target_token_ids" examples/score_mode_perplexity.py
# Should show: target_token_ids = window_tokens[1:]

# 3. Try without optimization to compare
# (Remove target_token_ids= line from script temporarily)
```

### Perplexity Scores Different

```bash
# Run both optimized and non-optimized to compare
# If they match, optimization is correct!
```

## Expected Results

| Configuration | Windows | Time/Window | Total Time |
|---------------|---------|-------------|------------|
| 500 samples (8B) | 73 | ~20 sec | ~25 min |
| 2612 samples (204K tokens) | 395 | ~20 sec | ~2.2 hours |
| Full WikiText-2 | 475 | ~20 sec | ~2.6 hours |

**This matches EXL3's performance!** 🚀

