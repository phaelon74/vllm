# The Nondeterminism Floor in Self-KLD

This note explains why KL-divergence measurements against reference logits are
currently unusable for GLM-5.3-Flash, how large the error is, and where we
believe it originates. It is a working document for the score-mode KLD
evaluation path, not a description of shipped behavior.

## Why zero is the requirement, not a nice-to-have

Score-mode KLD is a two-phase measurement. Phase 1 runs a full-precision model
over a corpus and saves the prompt logits as a reference. Phase 2 runs a second
model over the same tokens and reports the KL divergence of its distribution
from the reference. The number that comes out is attributed entirely to the
difference between the two models, which is the whole point: it is how we decide
whether a quantization recipe is acceptable.

That attribution only holds if the engine is a deterministic function of its
inputs. Run the *same* model in both phases and the divergence must be exactly
0.0, because both phases evaluate the identical function on identical tokens.
Any non-zero result is measurement error, and it is indistinguishable from the
model difference we are trying to measure. Self-KLD is therefore not one test
among many; it is the calibration that tells us whether any other KLD number
means anything.

## What we measure instead

On 4xB200, TP=4, BF16, eager, `moe_backend=triton`, GLM-5.3-Flash scored against
reference logits captured from itself:

**Mean self-KLD: 0.007206 nats.** Required: 0.0.

The floor is not uniform noise. Comparing two captures of the same prompt in the
same process, the logits are bit-identical up to a sharp boundary and then
degrade with length:

| Prompt length | Rows differing | max abs delta-logit | argmax flips |
|---|---|---|---|
| 129 | 0 | 0 | 0 |
| 130 | 1 (row 128) | 0.2 | 0 |
| 192 | 63 | — | — |
| 512 | 383 | — | — |
| 2048 | 1919 | ~8 | ~40 |

Two facts in that table do most of the damage. The divergence is *not* a
rounding-scale wobble; by 2048 tokens a single logit moves by up to 8, which is
far outside anything BF16 accumulation order can explain on its own. And ~40
positions per 2048 change their argmax, meaning the model's own top-1 prediction
disagrees with itself on roughly 2% of positions while nothing about the model
has changed.

## Why 0.007 nats is disqualifying rather than merely untidy

A quantization evaluation is a comparison of small numbers. The decision we want
to support is "recipe A degrades the distribution less than recipe B," and the
differences that separate a good 4-bit recipe from a bad one are themselves
small divergences. A fixed 0.007-nat pedestal under every measurement destroys
that comparison in three ways.

It sets a resolution limit. Nothing below the floor is observable, so any recipe
whose true divergence is at or under the noise reads as "indistinguishable from
full precision" regardless of whether it actually is.

It is not a constant that can be subtracted out. The error grows with position
within a window, so the pedestal depends on context length, stride, and window
count. Two runs with different windowing produce different floors, which means
KLD numbers are not comparable across configurations, and a change in the floor
can masquerade as a change in the model.

It is not conservative. The argmax flips mean the reference logits themselves
encode a top-1 prediction that the same model would not reproduce. A quantized
model that happens to agree with the reference's flipped positions scores
*better* than one that agrees with what the reference model actually predicts
most of the time. The metric can rank a worse model higher.

We have not yet calibrated the floor against a known-good quantization on this
model, so we cannot state the exact signal-to-noise ratio. That measurement is
worth doing, but it does not change the conclusion: a metric whose defined
minimum is 0.0 and which reports 0.007 for its own calibration case is not
reporting model difference.

## Where it happens, within a process

`scripts/glm53_layer_bisect.py` installs forward hooks inside the TP workers,
runs the 130-token reproducer twice, and compares every hooked module's prefill
output. Hooking all 45 layers plus their key operations in a single pair of runs
localizes the origin to one module:

```text
[result] first divergent module (execution order):
         language_model.model.layers.3.self_attn.mla_attn.mla_attn
```

Everything upstream of it is bit-identical: the layer's fused norm, the fused
QKV-A projection, the query projection, and all four sparse-indexer submodules
including `indexer.indexer_op`. The attention kernel receives identical queries
and identical keys and values — and returns a different result.

The magnitude points at accumulation order. The attention output differs by
9.765625e-04, which is exactly one BF16 ULP at that tensor's scale, as are the
deltas in the modules immediately downstream (2⁻⁸, 2⁻⁷, 2⁻⁵). A single rounding
step means the same values were summed in a different order.

## The cause: unstable order in the sparse index rows

Fingerprinting every tensor argument the kernel receives, on entry and across
two runs, isolates the difference to `block_tables` at the very first MLA layer,
before any upstream divergence exists. Hashing those indices both as passed and
after sorting separates the two possible causes, and the answer is unambiguous:
the sorted hash matches while the raw hash does not. The kernel is handed the
same set of KV positions in a different order.

The order is unstable because at these lengths the indexer is not selecting.
With `index_topk = 2048` and a 130-token prompt, every token is a candidate, so
the top-k rows enumerate all valid positions; only their order varies between
runs. Attention does not depend on that order, but the kernel accumulates in it,
so a permutation costs one ULP per layer. Over 41 MLA layers that grows roughly
250x into a 0.25-to-0.78 logit delta, ~40 argmax flips per 2048 positions, and
the 0.007-nat KLD floor.

Within a single process there is a second, compounding effect: the KV block
allocator hands the second request different physical blocks, so the same
logical content arrives at different addresses (token index 1152 in one run,
6912 in the next). Repeating a capture inside one process therefore measures
allocator rotation as well as index order, which makes it a poor determinism
test. Two fresh processes replaying the same prompts allocate identically, and
that is what the two-phase KLD workflow actually does.

The fix is to sort each index row before it reaches the kernel, keeping the -1
padding at the end. This preserves the exact set of selected tokens, so the
attention result is mathematically unchanged, while fixing the accumulation
order. It is gated behind `VLLM_SPARSE_MLA_SORT_TOPK=1` in
`vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py`. With it enabled, two
independent processes capturing the same 130-token prompt produce bit-identical
logits: `identical=True max|d|=0.0`.

## What the kernel is not

The kernel itself was cleared before the argument fingerprinting was written.
`scripts/glm53_sparse_mla_kernel_probe.py` calls
`trtllm_batch_decode_with_kv_cache_mla` directly with this model's shapes and
reproduces itself bit-for-bit at every row count tested (1 through 512, spanning
the point where the cubin switches from the `MultiCtasKv` variant to
`Persistent`), and also when the same logical KV content is placed at permuted
physical slots. An earlier draft of this note attributed the floor to a
multi-CTA reduction inside FlashInfer; that hypothesis is retired. The kernel is
order-sensitive, which is normal for a fused reduction, and the engine was
handing it an unstable order.

The path to that kernel is still worth spelling out, because the obvious fix
sites do not apply. This model's MLA dimensions are
`(qk_nope=256, qk_rope=0, v=256)`, which no MLA prefill backend accepts:

```text
WARNING [mla_attention.py:587] No MLA prefill backend supports this model;
sparse MLA will use the top-k MQA path only (no dense-MHA prefill).
```

With no dense or masked prefill available, every token — prefill included — goes
through `trtllm_batch_decode_with_kv_cache_mla` in
`vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py`, one row per token with
its own per-token valid KV count. That is also why the batch-invariance flags do
not help: they guard the dense MLA backends, which this model never reaches.

Layers 0 through 2 are clean because GLM-5.3-Flash is hybrid; layer 3 is the
first MLA layer. Only MLA layers originate divergence.

## A prerequisite that was silently broken

Independently of the numerics, `return_prompt_logits` did nothing on this model.
The flag was implemented in `vllm/v1/worker/gpu_model_runner.py`, but GLM-5.3
runs on Model Runner V2 (`vllm/v1/worker/gpu/`), selected by config and
overridable with `VLLM_USE_V2_MODEL_RUNNER`. V2's prompt path computed only
top-k logprobs, so requests asking for raw logits received `None` and requests
asking for KLD were never scored. Any KLD number produced before the V2 port
should be treated as unverified.

## A note on method

This cost us a wrong answer twice, both times for the same reason: measuring a
nondeterministic system with a method that assumes determinism.

The fault is stochastic per invocation, so searching layers in one pair of runs
and then submodules in a second pair compares two different experiments: the
reported culprit layer moved between runs, and a layer's own input appeared
dirty while its predecessor read clean. Any bisection of a nondeterministic
fault has to measure every candidate in the same pair of runs, which is what
`--pattern` in the bisect script is for.

The same trap appears in the measurement itself. Comparing two captures inside
one process also compares two different KV block allocations, so it reports a
floor that the two-phase workflow — separate processes, same prompt order — does
not actually suffer. Determinism claims here must state whether they are
within-process or across-process, because the answers differ.

## What we ruled out

Uninitialized memory. Running under `torch.use_deterministic_algorithms` with
`torch.utils.deterministic.fill_uninitialized_memory` produced no NaNs, so the
divergence is not a read of memory that was never written.

FlashInfer autotuning. Disabling it changed nothing, which also matches the log:
the autotuner saves 0 configs on this model, so it has nothing to vary.

Prefill chunking. The boundary sits at the same position with chunked prefill at
512 tokens and with the whole window in a single chunk, so it is not an artifact
of how the prompt is split into batches.

## The obvious fix is not available

vLLM has a batch-invariance mode (`VLLM_BATCH_INVARIANT=1`) intended for exactly
this class of problem. It cannot be used here. GLM-5.3-Flash requires
`use_sparse=True`, and no attention backend satisfies both constraints: every
backend that reports `supports_batch_invariance()` is dense, and every sparse
backend declines batch invariance. The engine fails at startup with no valid
backend. `TRITON_MLA` is the near miss, objecting only to sparse; forcing it
would run dense attention on a sparse architecture, which produces different
logits and so cannot serve as a reference capture even though it might be
deterministic.

The dense-MLA backends do carry the relevant guard — `TRITON_MLA` and the FA4
MLA prefill path both pin their split count to 1 under `VLLM_BATCH_INVARIANT` so
the reduction is order-stable — but neither is reachable for this model, so
patching them has no effect on it.

This is worth stating plainly because it means the floor cannot be waved away
with the existing environment variable. It had to be fixed on the path this
model actually takes.

## Reproducing and gating

`scripts/glm53_determinism_probe.py` measures the floor: it sweeps prompt
lengths to locate the boundary, repeats captures to compare them pairwise, and
runs a self-KLD round trip. Its `--gate` mode turns those checks into hard
pass/fail and exits non-zero, and the capture launcher runs it before Phase 1 so
a reference capture cannot silently inherit a floor. `--gate-max-self-kld`
defaults to `0.0`; raising it records a known floor deliberately rather than by
accident.

## Status

With `VLLM_SPARSE_MLA_SORT_TOPK=1`, two independent processes capturing the same
130-token prompt produce bit-identical prompt logits. The cross-process floor,
which is the one the two-phase KLD workflow is exposed to, is gone.

Still open: whether within-process repeats are also bit-identical once the
allocator hands out different physical blocks, and a quality check confirming
that sorting the index rows leaves model outputs unchanged. Sorting cannot
change the mathematics — the same tokens are attended with the same weights —
but that should be demonstrated on an eval rather than argued.

Without the sort, treat any GLM-5.3-Flash KLD number above roughly 0.007 nats as
containing an unknown contribution from measurement error, and any number at or
below it as unmeasured.

## Windowing (correction)

The earlier score-mode example used `--stride 512` with `--context-length 2048`
and described that as EXL3-compatible. That was wrong. Turbo's
`exllamav3/eval/model_diff.py` calls `get_test_tokens(..., eval_len, eval_len)`,
so stride equals length and rows never overlap. Shallow-context removal is a
prefix discard (`first = n_ctx // 2`) on the llama.cpp-parity PPL path, not a
sliding window.

The KLD script now defaults to non-overlapping rows (`--rows 100`,
`--context-length 2048`, stride = length) and `--score-from 0`. Depth buckets
are always printed. `--stride` remains only to regenerate historical overlapping
numbers.

## Hidden-state replay (measurement pending)

Teacher hidden states replayed through the teacher LM head are a candidate
replacement for full-logit storage. The capture path can write both tensors and
run the replay probe through a reconstructed TP-aware `ParallelLMHead` and the
loaded model's `compute_logits` / logits-processor path. It requires **bitwise**
equality with live teacher logits.

That probe has not yet been measured on this stack. Their published replay error
of ~1.23e-6 fails the zero doctrine if reproduced here. Until the probe prints
`identical: True`, keep `--storage logits` and treat hidden-state scoring as
disabled. Do not enable `--storage hidden` or `auto` on a failed probe.
