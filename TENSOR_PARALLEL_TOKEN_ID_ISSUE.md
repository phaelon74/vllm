# Tensor Parallel Token ID Issue

## Problem
All token IDs are 0 when using `tensor_parallel_size=2` with Ministral3ForCausalLM model.

## Root Cause
Token IDs are only generated on the last rank (where `lm_head` and sampling run), but they're not being gathered/broadcast to other ranks before being put into `EngineCoreOutput`. Non-last ranks produce zeros.

## Evidence
- Logs show: `WARNING: All token IDs are 0! new_token_ids=[0]...`
- Usage shows 20 completion tokens were generated (so generation is working)
- But all token IDs are zeros

## Workaround
Try running with `tensor_parallel_size=1` to see if the issue is specific to tensor parallelism:

```bash
vllm serve /media/fmodels/TheHouseOfTheDude/Devstral-2-123B-Instruct-2512_Compressed-Tensors/W4A16 \
  --quantization compressed-tensors \
  --tensor-parallel-size 1 \
  --max-model-len 65535 \
  --gpu-memory-utilization 0.90 \
  --api-key 26c2027f7cfe2c127b55ab02918ad3de454c50f9d21699806b34bf0621cdfa73
```

## Next Steps
1. Test with `tensor_parallel_size=1` to confirm this is a TP issue
2. If confirmed, investigate how token IDs are gathered in tensor-parallel setups
3. Check if there's a bug in vLLM's engine core for TP token ID gathering
4. May need to fix the engine core code to properly gather token IDs from the last rank

