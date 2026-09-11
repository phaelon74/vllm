# QxQ and BxQ — Routed-Model Distribution Fidelity

**Applies to:** laws version 14, Laws 14 and 17.
**Read first:** [`LAWS.md`](LAWS.md) for the laws, [`README.md`](README.md) for the
harness.

This document explains what the QxQ and BxQ cells measure, why measuring them at
all requires bitwise-exact repeat from every kernel on the path, which MoE
backends have earned the right to produce a published number, and how the
pipeline refuses the ones that have not.

## 1. Why one number is not enough for a routed model

For a dense model, quantization error is weight rounding. One KLD against the
BF16 reference describes it, and the only question left is how it distributes
across text.

A Mixture-of-Experts model breaks that. Two unrelated things happen at once, and
a single mean hides both:

1. The experts round their weights, exactly as a dense layer would.
2. The router sends tokens to *different experts* than the reference would,
   because the activations arriving at the router are themselves quantized.

These have opposite consequences. Rounding is a fidelity tax that shrinks
predictably with bit width. Routing divergence means the model is evaluating a
different function — a different subnetwork per token — and it does not shrink
predictably with anything. A candidate whose loss is mostly rounding and a
candidate whose loss is mostly rerouting can post identical means and behave
differently in deployment. Ranking them on that shared mean is the error Law 14
exists to prevent.

## 2. The two cells

Both cells score the same candidate against the same BF16 teacher on the same
frozen token suite, and differ only in who chooses the experts.

**QxQ** is the candidate exactly as deployed: quantized weights, and its own
router choosing its own experts. This is the number that describes what a user
would actually run, and it is what routed candidates are ranked on.

**BxQ** replays the same tokens with the teacher's expert IDs forced into the
student, while the student keeps its own gating *weights*. Forcing the IDs
removes the routing disagreement and leaves the rounding, so BxQ is the fidelity
the candidate would have if its router agreed with the reference.

The keeping of student gating weights is deliberate and easy to get wrong. If
BxQ also took the teacher's gate values it would stop being a measurement of the
student's experts and become a partial reconstruction of the teacher. Only the
selection is intervened on; the weighting stays the candidate's own.

**QxQ − BxQ** is therefore the routing contribution, reported as the paired
intervention delta. The **natural route flip rate** reports how often the two
disagree at all, measured over `(token, layer)` choices on the natural QxQ run —
not on the intervened one, where by construction there is nothing to count.

A worked example from the gemma-4-26B-A4B-it family: `Intel/gemma-4-26B-A4B-it-int4-AutoRound`
scores QxQ 1.17816288 and BxQ 0.90308200, a delta of +0.27508088, and flips
93.07% of natural routing choices. It sits mid-pack on deployed fidelity, but the
decomposition says something no mean could: nearly all of its loss is the router,
not the rounding. Its neighbours in the same group carry deltas near +0.15 at
comparable flip rates.

## 3. Why the subtraction demands bitwise-exact repeat

QxQ − BxQ is a difference between two separate forward passes. If scoring the
same tokens twice does not reproduce the same logits bit for bit, then a delta of
+0.15 is indistinguishable from kernel noise, and the decomposition is
storytelling.

So the pipeline does not assume reproducibility, it measures it. Every routed
candidate is scored, replayed, and the two results compared; the published
`repeat_delta` is that difference and it is required to be exactly `0.0`. A
candidate whose backend cannot deliver that is marked uncertified for exact
repeat and cannot be published as a Law 14 measurement.

This is the entire reason the effort behind this document was determinism work
rather than metric work. The metric is a subtraction; the difficulty is earning
the right to subtract.

## 4. How MoE kernels break exact repeat

Batch invariance is the property that a token's output does not depend on which
other tokens share its batch, nor on their order. Most MoE kernels violate it,
and not by accident: grouping tokens by expert and reducing partial sums in
whatever order the grouping produced is the fast way to do it. That is a correct
optimization for serving, where nobody compares two runs, and it is fatal here.

Four fixes were required before routed models could be scored at all.

**Marlin token ordering and reduction.** The Marlin MoE path was ported to a
canonical token order with a full-K reduction, which is what makes
`MarlinExperts` and `BatchedMarlinExperts` batch invariant. This carries the bulk
of the field: every int4 AWQ, GPTQ, and W4A16 routed candidate scores here.

**Qwen GDN attention.** Certified only on its NVIDIA CUDA, non-speculative,
per-sequence path, with FlashInfer GDN context parallelism disabled. Without this
the Qwen3.6 family could not run under `VLLM_BATCH_INVARIANT` at all.

**Dense NVFP4 linear layers.** Weight-only W4A16 NVFP4 linear layers use
deterministic emulation, because dense Marlin is not batch invariant.

**Expert parallelism is never certified.** An EP path reduces across ranks in
completion order. No allowlist entry overrides this.

## 5. Certification fails closed

A kernel's `_supports_batch_invariance()` is a claim about itself. It is not
evidence. `CutlassExpertsFp4` declared True and still produced NaN on W4A4
checkpoints; the NaN was uninitialized per-expert activation scales, not the
kernel, but the lesson stands: a self-declaration is not a probe.

`inspect_model_moe_backends` in `vllm/v1/sample/kld.py` therefore grants
`certified_for_exact_repeat` only when all three of these hold:

1. The expert class is named in `_EXACT_REPEAT_CERTIFIED_EXPERTS`, an allowlist
   earned by passing an exact-repeat probe on real suite content. The report
   carries this separately as `exact_repeat_probed`.
2. The kernel also self-declares batch invariance.
3. The run is not expert-parallel.

The allowlist currently holds `MarlinExperts`, `BatchedMarlinExperts`,
`TritonExperts`, `Nvfp4QuantizationEmulationTritonExperts`, and
`CutlassExpertsFp4`. CUTLASS earned its place after the SM120 bitwise
permutation test (24/24, `atol=0`) and after the W4A4 NaN was shown to be a
loader gap, not a kernel defect. The run's own zero-tolerance exact-repeat
control remains the binding gate.

The default is refusal. An unprobed backend — including one that arrives with a
future vLLM bump — is uncertified until somebody probes it, and an uncertified
backend is reported as uncertified rather than published as a number. This
matters more than it sounds: the failure mode it replaces is a kernel quietly
producing garbage that the harness dutifully writes down as a fidelity result.

DeepGEMM, FlashInfer MoE, AITER, XPU, CPU, and every expert-parallel path remain
uncertified.

**Certification and backend selection now meet.** With the MoE backend pin removed
(§7) the choice belongs to vLLM's oracle, and the oracle ranks by expected
performance, not by whether a kernel is certified here. `AVAILABLE_BACKENDS` in
`vllm/model_executor/layers/fused_moe/oracle/nvfp4.py` puts `FLASHINFER_TRTLLM`
first and nothing in that path consults `VLLM_BATCH_INVARIANT`, so nothing
*guarantees* a certified kernel. The refusal is upstream of the cost, at least:
scoring reads certification off the loaded model and raises before any forward pass,
so an uncertified choice costs a model load rather than a scored pass.

Measured, on ten gemma4-26b-a4b candidates at one row each: the oracle chose
`VLLM_CUTLASS` for all four NVFP4 MoE exports, `TRITON` for the FP8 export, and
`MARLIN` for the four INT4 ones. Every one is in the allowlist and every control
repeated at exactly 0.000e+00. FlashInfer was first in the list each time and its
`is_supported_config` declined each time, for reasons logged only at debug level.
So the outcome is good and it is a reading rather than a promise: run the `smoke`
stage over a routed family before committing a campaign to it, and read the backend
each candidate names.

**A kernel is verified on the next run, not pinned.** The oracle chooses per layer
from the checkpoint, the device, the installed FlashInfer and its own source, and
§9's comparability key binds all four — the last of them because the oracle is a
`.py` file under `vllm/` and therefore inside the numerics digest. Under one binding
the choice is a function, so a second run of it owes the same answer. Scoring holds
it to that: `_prior_kernel_identity` reads the previous report for the candidate,
and when its binding matches the live one, the expert class and kernel read back
after the student load must match what that report recorded. A mismatch is refused
before any forward pass, because a published number that a rerun would not
reproduce is worse than no number. The comparison is narrowed to `quant_method`,
`kernel` and `experts`: tensor and expert parallelism are recorded on the report but
a rescore is entitled to move them. Where the binding differs the expectation is
released, since new numerics or newly built kernels are allowed a new choice — that
is a new number, not a broken one. This is the safe half of pinning. A pin had to
predict the kernel before the load and could refuse a checkpoint outright (§7); a
comparison happens after the load and can only refuse a contradiction.

## 6. Case study: uninitialized scales, not a broken kernel

`CutlassExpertsFp4` self-declares batch invariance. Under
`VLLM_BATCH_INVARIANT=1`, on two W4A4 NVFP4 checkpoints, it took finite inputs
to NaN. The scorer refused a non-finite KLD and the engine died. The first
reading was that the kernel's claim was false. That was wrong.

The kernel is bitwise batch-invariant on this hardware. The CUDA grouped GEMM
asserts at compile time that it uses `PersistentTileSchedulerSm100Group` "for
batch invariance", and `tests/v1/determinism/test_cutlass_batch_invariance.py`
passed 24/24 on SM120 at `atol=0, rtol=0` across both activations, both expert
counts (40, 64), both top-k values, and all three shape cases.

The NaN was uninitialized memory. NVFP4 allocated per-expert activation scales
with `torch.empty`. Checkpoints that omitted `input_scale` keys for some experts
in some layers — `bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4` (14 layers short)
and `Neural-ICE/Gemma-4-26B-A4B-it-NVFP4` (16 layers short) — left those slots
holding whatever was on the GPU. CUTLASS consumes one scale per expert
(`a1_gscale` / `a2_gscale` of length `e`); a NaN slot damages that expert. The
unsloth export, with every slot written, never went NaN.

Loading does not catch this. Strict all-parameters-loaded tracking is off by
default for quantized models, and any module with `process_weights_after_loading`
has every parameter force-marked as loaded, so a partially filled tensor is
invisible to it by construction.

Localizing it took two purpose-built tools.

`scripts/scan_checkpoint_nonfinite.py` reads a checkpoint's tensors and counts
non-finite values, zero scales, and per-row magnitude outliers in scale tensors.
`torch.isfinite` has no CPU kernel for several float8 dtypes, so `nonfinite_mask`
tests FP8 bit patterns directly — `(bits & 0x7F) == 0x7F` for `e4m3fn`, and
`>= 0x7C` for `e5m2`. The scan came back clean, which established that the NaN
was generated at runtime rather than baked into the weights, and moved the
investigation off the checkpoint.

`scripts/nan_first_module_probe.py` installs forward hooks and reports the first
module whose output goes non-finite from finite inputs, together with the first
row index at which it happens. Two capabilities were decisive. `--context-file`
replays a real suite context instead of a synthetic prompt, because synthetic
prompts are repetitive and never route diversely enough to trigger the bug.
`--prompt-logprobs` computes logits at every prompt position, which is what the
scoring harness does and what a generation-only probe does not.

With both, the failure is exactly reproducible: `moe.experts` at layer 0, row 814
of `context-0002`, and layer 14, row 170 of `context-0004`.

The `--moe-backend` override then isolated the failure to the native path on
the two incomplete checkpoints, not to batch invariance itself:

| `VLLM_BATCH_INVARIANT` | Backend | Result |
| --- | --- | --- |
| 0 | `FLASHINFER_CUTLASS` (auto) | pass |
| 1 | `VLLM_CUTLASS` (auto) | **NaN** on incomplete exports |
| 1 | `marlin` (forced) | pass (drops activation scales) |
| 1 | `emulation` (forced) | pass (collapses per-expert scales) |

The fail-closed allowlist was the right reaction to a NaN that looked like a
kernel defect. Once the scales were the cause, keeping CUTLASS off the list
was the thing standing between native BxQ and a published number.

## 7. W4A4 NVFP4 and the kernel the loader actually builds

A W4A16 export and a W4A4 export of the same model differ only in whether
activations are also quantized, and that difference decides which kernels can
score the checkpoint honestly. A repository name says nothing reliable about it,
so `_declared_expert_activation_quant` in
`examples/offline_inference/score_mode_kld.py` reads the checkpoint: for every
`quantization_config.config_groups` entry that targets experts, it reports what
that group declares for `input_activations` — `4-bit float`, `8-bit float`, or
`unquantized`. Scoped to expert groups, because a group covering only attention
says nothing about the kernels the MoE will run, and scanning every group is what
made a checkpoint with W4A4 attention and FP8 experts look like a W4A4 MoE.

It reports a list, not a verdict, because a checkpoint can declare more than one
width inside its own MoE. And it is a reading of the checkpoint, not a prediction
about the run: what the loader builds is a separate fact, read back after load.

**The scorer no longer pins a MoE backend, and the removal was not a
simplification.** It used to pin `moe_backend="cutlass"` for anything it judged
W4A4, and two real checkpoints showed that a pre-load pin is being asked two
questions it cannot answer. `nvidia/gemma-4-26B-A4B-it-NVFP4` declares
`num_bits: 4, type: float` weights *and* activations for `mlp.experts`; the
modelopt `MIXED_PRECISION` loader built those experts weight-only, and the pinned
W4A4 CUTLASS kernel refused a configuration ending `(symmetric)xNone` — the pin
failed the candidate over a kernel the checkpoint was never going to get.
`unsloth/gemma-4-26B-A4B-it-NVFP4` declares FP8 experts in its last eight layers
and W4A4 in the rest, which no single global pin can serve at all. Both failures
were ours, not the checkpoints'.

So the choice is left to the oracle, which answers per layer and knows what it
built, and `inspect_model_moe_backends` reads the answer back off the loaded
model. That is the measurement this project prefers to a declaration, and it is
the same discipline as `assert_unquantized_kv_cache` in §9.

The kernels themselves still differ, and which one ran still decides what a number
means. vLLM CUTLASS is the only native W4A4 MoE path that keeps a per-expert
activation-scale vector. FlashInfer collapses every expert to one scalar via
`amax_for_moe_activation_quant(...).repeat(num_experts)` — the same defect as
emulation. Marlin drops activation scales and scores W4A4 as W4A16.

Unwritten slots are now a NaN sentinel, filled from the maximum of the present
per-expert scales before CUTLASS fuses them into the weight alphas. Too large
wastes quantization range; too small overflows e4m3. A layer with no finite
positive slot is refused rather than invented. The fill is recorded on the
layer at fill time as `uncalibrated_experts_filled_from_layer_max` and Law 17
discloses it. A complete export scored on CUTLASS records an empty substitution
list, because the kernel used the checkpoint's own scales. A complete export is
not automatically such a run: with the pin gone, unsloth's mixed-precision MoE
gets whichever kernel the loader builds for it, and if that kernel collapses the
scales the substitution is disclosed and then priced.

Both halves of the model are walked, because a dense NVFP4 projection can omit a
shard's scale exactly as an expert can. `inspect_model_moe_backends` covers the
routed experts and `inspect_model_nvfp4_dense_scales` covers everything else,
reported separately so the denominators stay meaningful — a fill on 3 of 200
dense layers is a different claim than 3 of 30 experts. The fill records a scan
marker on every layer it visits, filled or not, because counting NVFP4 layers
after load is guesswork: the dense paths delete or overwrite the very parameter
they were named for. Without the dense walk a dense W4A4 candidate reported no
substitution, and Law 17 read that silence as a clean bill of health rather than
as an absence of evidence, which is why it now returns `NOT_APPLICABLE` when
nothing was inspected at all.

| Path | Per-expert activation scales | Exact repeat |
| --- | --- | --- |
| vLLM CUTLASS FP4 | honoured (per-expert vector) | certified on SM120 |
| FlashInfer FP4 | collapsed to one scalar for the layer | uncertified |
| Marlin | dropped entirely (scores W4A16) | certified |
| Emulation | collapsed to a layer maximum | certified |

**A collapse is priced, not just disclosed.** Law 17 states that a run used one
layer-wide scalar where the checkpoint exported a scale per expert, and that left
a reader to guess whether the substitution was worth a decimal place or the
ranking. When a report discloses `per_expert_collapsed_to_layer_scalar`, the
campaign scores that candidate once more with `--moe-backend cutlass`, which keeps
the per-expert scales, and records the difference as `per_expert_collapse_cost`.
The one-pager prints both numbers and the delta.

Conditional by construction: a candidate that collapsed nothing pays nothing,
which is every dense candidate and every routed one whose experts kept their own
scales. A candidate that did collapse pays one student load and one scoring pass,
because the teacher capture does not depend on which kernel the experts run and is
shared with the run that just finished. If the non-collapsing kernel will not load
the checkpoint on this hardware, the cost is recorded as unpriced with the reason;
withdrawing a measured, compliant, deployed number because a second run vLLM never
has to perform did not work would be the wrong trade.

The deployed number does not change. Which kernel the oracle builds for a
checkpoint on this hardware is a fact about deploying it, and this index reports
what deployment does. The counterfactual is priced beside it, never in place of it,
and the two are not comparable to each other's leaderboard groups because the
substitution is part of the comparability key.

**A named backend belongs on the student only.** `--moe-backend` reaches
`student_kwargs`, not the shared `llm_kwargs`. Applied to the latter it propagates
to the unquantized BF16 teacher, which has no such scheme, and reference engine
initialization fails.

A weight-only W4A16 NVFP4 result still measures the quantization scheme rather than
a native FP4 kernel's own rounding, because dense Marlin is not batch invariant
and those linear layers use deterministic emulation. That limit is stated on the
published card.

### History: the emulation collapse we published through

Before the loader gap was understood, W4A4 scoring was pinned to emulation.
The emulation branch of `convert_to_nvfp4_moe_kernel_format` collapses the
per-expert activation scales to one scalar per layer —
`a13_scale = 1.0 / a13_scale.max()` — and vLLM's own comment says taking the
largest global scale "likely results in overflowing the FP8 range for other
experts." Since `a2_scale` holds `1 / w2_input_global_scale`, `.max()` applies
the *smallest* per-expert scale. Measured on
`unsloth/gemma-4-26B-A4B-it-NVFP4`:

| Parameter | Slots | Distinct values | Worst spread | Collapsed |
| --- | --- | --- | --- | --- |
| `w13_input_global_scale` | 256 | 1 | 1.0x | no |
| `w2_input_global_scale` | 128 | 86–99 | 150.5x | yes, all 30 layers |

Four NVFP4 exports carrying byte-identical QDQ diagnostics (0.66652247 and
1.43642041) scored QxQ between 1.16299506 and 1.82545539 on that path, and that
0.6 nat spread was written up as living "entirely in the activation scheme and
kernel path." That reading was unsupported. The spread was substantially an
artifact of the measurement. It stays here because it is the exact shape of
mistake this document exists to prevent: a real, reproducible, bitwise-exact
number that is nonetheless measuring the harness rather than the checkpoint.

Law 17 still exists for the remaining real substitutions — an uncalibrated
expert filled from the layer maximum is one — so a substituted result ranks
only against candidates measured the same way, and is never withdrawn for
having a disclosed fill.

### How far to discount a filled result

The disclosure states a spread. It does not say what that spread costs, and the
first native-CUTLASS campaign answered the question by accident.

The two filled candidates landed on top of each other:

| Candidate | Layers filled | Worst spread | QxQ | BxQ |
| --- | --- | --- | --- | --- |
| `Neural-ICE/Gemma-4-26B-A4B-it-NVFP4` | 15 of 30 | 230.6x | 1.82281421 | 1.58512109 |
| `bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4` | 12 of 30 | 240.3x | 1.82307292 | 1.59361302 |

0.00026 nats apart on QxQ, from different publishers, with different fill
footprints, and with one quantizing its LM head while the other does not. They
are not the same checkpoint: distinct `student_weights_sha256`, and distinct
per-position KLD digests in `kld_evidence`, so the two runs did not produce
bitwise-identical output either.

Two independent quantizations do not agree to four decimal places on their own.
The reading that fits is that once a tensor's per-expert scales are replaced by
one layer maximum on half the layers, the fill sets the number and the
checkpoint's own choices stop being visible in it. On the same suite the clean
NVFP4 exports separate normally — 1.13846019 for unsloth against 1.77968754 for
RedHatAI — so the collapse is not the suite failing to discriminate.

This is n=2 and therefore a strong hint rather than a proof; a fill-direction
sweep on one checkpoint would settle it, and has not been run. Treat a filled
QxQ as an upper bound on that family of exports rather than a measurement of the
particular one. The leaderboard says so on the row: a filled candidate is ranked
beside the clean ones, because it was measured on the same suite, geometry, and
runtime, and carries a `†` naming what was substituted. It is not exiled into a
section of its own, which would hide the comparison a reader came for while the
substitution stays bound in its comparability key either way.
It is also the argument against ever promoting the fill out of the harness: see
§13.

## 8. Checkpoint defects the pipeline had to fix, not tolerate

Determinism gets you a repeatable number. It does not get you a *correct* one if
the checkpoint is being loaded wrong, and two classes of loading bug were found
by scoring rather than by tests.

**AutoRound int4 geometry.** AutoRound exports use group sizes that do not divide
the input dimension, producing a partial final group. The packed-row and
scale-group arithmetic — `scales_size`, `num_groups`, `size_k`, and the encoded
symmetric zero used to pad — was wrong at exactly those boundaries, in both the
Marlin and AutoGPTQ paths. A read-only `safetensors` header probe established the
real geometry before any code changed; assumptions about row counts had already
cost several wrong fixes. Regression tests cover the padded reduction allocation,
the encoded-zero padding, single-rank partial groups, and the dense path's packed
rows with a partial scale group.

**A quantized router with nowhere to land.** The AutoRound Gemma-4 export
quantizes the router projection. vLLM's `Gemma4Router` hardcodes its `GateLinear`
as unquantized BF16, so the checkpoint's `qweight`, `qzeros`, and `scales` had no
destination, no loader complained, and `router.proj.weight` was left as
uninitialized memory. The router emitted NaN from finite inputs and KLD went
non-finite.

Note how this presents: identical symptom to §6, entirely different cause. The
module probe is what separated them, naming `layers.0.router.proj` rather than
`moe.experts`. The fix dequantizes those tensors into BF16 at load time in
`Gemma4Model.load_weights`, with tests in `tests/kernels/moe/test_gemma4router.py`
covering the transposed packed values and the refusal of an indivisible group
count. That checkpoint now scores 1.17816288 and passes all seventeen laws.

## 9. The runtime the numbers are bound to

Scoring pins `VLLM_BATCH_INVARIANT=1`, disables DeepGEMM
(`VLLM_MOE_USE_DEEP_GEMM=0`) and FlashInfer autotune, sets `NCCL_DETERMINISTIC=1`
and `CUBLAS_WORKSPACE_CONFIG=:4096:8`, enforces eager execution, disables prefix
caching, holds `max_num_seqs=1`, and refuses a quantized KV cache.

None of that is optional and none of it is a performance setting. Each one closes
a path by which two runs of the same tokens could diverge.

The KV cache closes the widest such path found so far, and it was open for
the whole campaign. Left at `auto`, vLLM resolves the KV cache dtype from a
scheme declared in the candidate's own config, so
`unsloth/gemma-4-26B-A4B-it-NVFP4` — which declares an 8-bit float KV cache —
had its attention read and written through a quantized cache while every
candidate it was ranked against used an unquantized one. That is a difference in
how the measurement was taken, not a property of the checkpoint being measured.
The cache is never quantized: scoring holds 4096 tokens, so there is no memory
pressure that quantizing it could relieve, and nothing to weigh against the loss
of comparability.

Unquantized is not one dtype. It was a literal `bfloat16` at first, which refused
three AWQ checkpoints published as float16: FlashAttention will not take a float16
query against a bfloat16 key, so the run died in `mha_varlen_fwd` rather than
scoring. `unquantized_kv_cache_dtype` in `vllm/v1/sample/kld.py` reads each
checkpoint's own `torch_dtype` and caches in that, so a float16 model caches in
float16 and a bfloat16 model in bfloat16. Both are unquantized, which is the
property the policy is about; the resolved dtype is recorded on the report and
carried in the comparability key, so a reader is told which one a number was taken
under and two candidates cached differently are not silently ranked together.
`assert_unquantized_kv_cache` refuses anything outside
`UNQUANTIZED_KV_CACHE_DTYPES`, and still refuses `auto`.

The rescore that followed measured what it had cost, and doubled as the cleanest
determinism evidence in this document. unsloth is the only one of the nine
candidates whose inspection reported a declared KV cache scheme, and it is the
only one whose number moved: 1.16570700 to 1.13846019, so its declared FP8 cache
had been costing it 0.02724681 nats, or 2.3% of its KLD, and enough to place it
above `gemma-4-26B-A4B-it-W4A16` at 1.15926068 when it belongs below. Five of the
remaining eight are quoted at their pre-pin values elsewhere in this document —
0.69415039, 1.17816288 with its BxQ and delta, 1.77968754, 1.82281421, and
1.82307292 — and every one came back identical to eight decimal places on a
different `vllm_commit`. So the defect was worth a rescore and was not worth a
panic, the blast radius was exactly the set the mechanism predicted, and the
harness reproduced everything outside it. `assert_unquantized_kv_cache` reads the dtype the engine actually
resolved and refuses both a quantized value and `auto`, because the failure being
prevented is precisely a value nobody checked.

**A number that moves on a rescore is not a number that was wrong.** The Qwen
families were first scored before the NVFP4 uncalibrated-scale fill, before the
Marlin determinism work, and before MoE batch invariance; rescored under all three,
some of those values moved by as much as 6%. That is the correct amount of movement
for a runtime that gained a fill where it had been consuming uninitialized scales
and a batch-invariant expert path where it had not had one. Each value was a
faithful measurement of the runtime that produced it, which is why the runtime is
in the comparability key and why a legacy number is never quietly ranked beside a
current one. Read a movement of this size as the runtime changing, and look for the
change; the alarming case is a number that moves when nothing that computes it did,
which is what the digest below is for.

Because that runtime *is* part of the result, its identity is bound into the
comparability key: `numerics_digest`,
`compiled_extensions_sha256`, `torch`, `driver`, `gpu_names`, and
`kv_cache_dtype`, alongside the suite and geometry. Two candidates are ranked
against each other only when all of it matches. `kv_cache_dtype` is in that list
because of the unsloth case above: the key's one job is to bound a ranking to
runs that ran alike, and it had nothing to say about a candidate whose attention
ran at a different precision than its neighbours'.

The commit used to sit in that list, and the consequence was that **any commit
invalidated every published number.** A documentation paragraph, a campaign
config, a new script: the next scoring run read a different `vllm_commit`,
declared 45 compliant reports stale, and spent GPU-days reproducing numbers that
were already right — and, being a rescore rather than a refusal, it did so
silently. That was over-refusal dressed as rigour. An index of quantization
fidelity is under continuous development by construction, so a currency test that
cannot tell a docs edit from a kernel change makes the index unmaintainable.

What bounds a result is whether the code that computed it would compute it again.
`numerics_digest` in `vllm/v1/sample/kld.py` hashes every `.py` under `vllm/`
together with the scorer, and `compiled_extensions_sha256` covers the built
kernels, so between them they answer that question directly. Kernel sources are
not hashed, because a source edit cannot move a number until it is rebuilt and the
rebuild changes the extension digest. The digest is deliberately coarse — a
comment in `vllm/` moves it — because deciding which edits inside the runtime are
numerically inert is exactly the judgement a currency test must not be trusted
with, and the cost of that coarseness is a rescore rather than a wrong number.

The commit is still recorded on every report and printed on every one-pager. It
says *when* a number was taken, which is provenance under Law 6, and provenance is
not a currency test. A result reads: at commit `abc123`, under numerics digest
`def456`, this KLD was measured. The commit may have moved a hundred times since;
the number stands until the digest moves. Harness fixes are still best batched,
now because a rescore wave costs GPU-hours rather than because the alternative is
a stale library.

## 10. Gates that stop a correct measurement from being published wrong

Each of the following was found by a law refusing to publish, at the end of a
multi-hour campaign, rather than by a test. They are recorded here because the
failure mode is characteristic: the measurement is fine, and the metadata binding
it to a suite, a geometry, or a reference capture is not.

**Suite-driven geometry.** `_expected_geometry` derives the expected row count and
context length from the suite manifest and the active partition, not from the
dataset-driven `rows` and `context_length` on the config. Those config fields
describe a dataset-driven run only; reading them for a suite-driven one compared
768 real rows against a config default of 1024 and failed every candidate.

**A field nobody wrote.** `context_length` was absent from every report, so a gate
comparing it always failed. The writer in `score_mode_kld.py` now records it.
Existing reports were backfilled from their bound capture manifests after
verifying the recorded `capture_manifest_sha256`, refusing any mismatch.

**Captures that moved underneath a report.** Assembly publishes the reference
capture beside the report that cites it. If the capture is rebuilt after a report
scores, the pair cites a manifest nothing hashes to, and Laws 5 and 14 refuse it —
after the whole campaign. `_score_report_is_current` now treats a changed capture
manifest as staleness, where a rescore is cheap.

**Currency must not depend on processing order.** The gate above compares against
whatever capture is on disk at that instant, and silently passes when the file is
absent. That is order-dependent: early candidates in a run match the capture their
predecessor left behind and are skipped, then a later rescore replaces it and
strands them. `_candidate_complete` therefore delegates to the same
`_score_report_is_current` the scorer uses, which compares each report's recorded
commit against the live runtime and cannot be defeated by ordering.

The last one has a worked example. In the gemma-4-26B-A4B-it family, one candidate
was skipped early in a run on a stale `ec11b8…` capture, while a later rescore
rebuilt the capture as `d1f03dea…` on a new commit. Nine candidates published on
the new commit; the tenth kept a report from `fdc0b57e` bound to a capture the
family no longer published. It failed Law 12 alone and ranked in a comparability
group of one — every other law passed, because they read the candidate's own
manifest. It was withdrawn via `excluded_candidates`, which records the repo,
revision, and reason in `excluded-candidates.json` beside the family, rather than
being dropped from the config silently.

## 11. Reproducing and extending

Certify a new MoE backend before trusting it. In order:

```bash
# 1. Is the NaN in the weights, or generated at runtime?
python scripts/scan_checkpoint_nonfinite.py --model <checkpoint> --stats

# 2. Which module goes non-finite first, on real content, at which row?
VLLM_BATCH_INVARIANT=1 python scripts/nan_first_module_probe.py \
    --model <checkpoint> --context-file <suite>/contexts/context-0002.json \
    --prompt-logprobs

# 3. Is it the kernel? Re-run pinning each backend in turn.
VLLM_BATCH_INVARIANT=1 python scripts/nan_first_module_probe.py \
    --model <checkpoint> --context-file <suite>/contexts/context-0002.json \
    --prompt-logprobs --moe-backend marlin

# 4. One candidate end to end: finite KLD, and repeat exactly 0.
python fidelity/campaign.py smoke --config <campaign>.json \
    --only-candidate <name>

# 5. The family.
python fidelity/campaign.py all --config <campaign>.json
```

A backend that passes the probe on real content, at full context, with prompt
logprobs on, may be added to `_EXACT_REPEAT_CERTIFIED_EXPERTS`. Nothing else
qualifies it, and self-declaration never does.

Note that `--only-candidate` applies to `smoke` only. To rescore a single
candidate of an assembled family, delete its report from
`<work>/reports/<tag>.json`; the completeness gate then rescores exactly that one
and skips the rest. A rescore the campaign decides on for itself moves the old
report to `<work>/prior/<tag>.json` and holds the rebuild to the kernel that report
read back. Deleting the report by hand skips that, which is the way to accept a
kernel change deliberately rather than argue with the check.

## 12. What these numbers do not say

**They are not comparable outside their group.** Not against numbers from another
suite, geometry, runtime, or laws version, and not against any published
elsewhere. The comparability key is printed with every leaderboard group for
exactly this reason.

**A W4A4 NVFP4 MoE result is not guaranteed to be a native CUTLASS number.** It
used to be, by a pin, and the pin failed two checkpoints it should have scored
(§7). A result is now a number from whichever expert kernel the loader built for
that checkpoint on this hardware, which is the kernel deploying it would get. Read
it off the one-pager's declared-against-built table rather than inferring it from
the scheme label: a checkpoint declaring 4-bit activations for its experts may have
been built weight-only, and one built on a kernel that collapses per-expert
activation scales carries a disclosed substitution and a priced cost for it. A
W4A16 dense NVFP4 result still measures the scheme rather than a native FP4 kernel,
because dense Marlin is not batch invariant.

**A declared quantization is not a performed one.** The checkpoint's
`quantization_config` is a statement by whoever exported it, and the kernels vLLM
builds are a separate fact. `nvidia/gemma-4-26B-A4B-it-NVFP4` declares W4A4
experts and was built weight-only; its number is honest about what ran and says
nothing about what a W4A4 kernel would have scored.

**The QDQ ladder is diagnostic, never a candidate.** Those cells round weights on
synthetic BF16 checkpoints and route naturally. They are not QxQ or BxQ, they are
not rankable against deployed candidates, and the published tables separate them.

**A scheme label names the narrowest group, not the whole model.** A checkpoint
may quantize attention at one width and its experts at another, and may declare a
KV cache scheme. `unsloth/gemma-4-26B-A4B-it-NVFP4` is `format:
"mixed-precision"`: FP8 W8A8 attention, NVFP4 W4A4 experts and dense MLP, and a
declared FP8 KV cache. Its QxQ of 1.13846019 against 1.77968754 for a complete
all-`Linear` NVFP4 export is therefore mostly the 8-bit attention, not a better
NVFP4 export, and it lands between the all-FP8 candidate at 0.69415039 and the
all-NVFP4 ones exactly where a hybrid should. The declared KV cache is a separate
matter and is no longer in that number: §9 holds an unquantized cache, which is
worth 0.02724681 of the 0.64 separating it from the complete export, so the
attention width still carries the result. This is not a comparability failure — the key deliberately excludes
the candidate's scheme, because ranking schemes against one reference is the
point — but it was a labelling one until `scheme_mix` and `kv_cache_scheme` were
added to the inspection. Component coverage cannot substitute: it counts how many
weights are quantized, not at what width, so a hybrid and a uniform export both
read `all`.

**A flip rate is not an error rate.** Two experts disagreeing on a token is not
by itself a wrong answer; the delta is what quantifies the cost. AutoRound's
93.07% flip rate with a +0.275 delta is the point — high disagreement, bounded
consequence.

**Certification is about repeatability, not accuracy.** `exact_repeat: certified`
says two runs agree bit for bit. It says nothing about whether the kernel computes
the right thing, which is what the zero baseline (Law 1) and the reference binding
(Laws 12 and 16) are for.

## 13. The fill is a harness policy, not a vLLM fix

Half of the loader change here — allocating consumed NVFP4 scales as a NaN
sentinel instead of `torch.empty`, so an unwritten slot is detectable rather
than arbitrary — is not ours to contribute. Three open upstream PRs already do
it on the same files by the same mechanism: #54444 on the ModelOpt linear
methods and fused experts, #45320 on the ModelOpt per-expert scales, #52501 on
the linear per-block `weight_scale`. A fourth, #55073, is actively reworking the
same compressed-tensors and ModelOpt scale code.

Where we differ is the policy after detection, and the difference is deliberate
on both sides. All three upstream PRs **reject**: they raise at load time naming
the parameter and the affected experts. #45320 states the position outright —
"this remains fail-fast only, it does not guess missing calibration statistics
or add an imputation policy." Filling from the layer maximum is exactly the
imputation policy they declined.

They are right for a serving engine, and the evidence for that is in §7's
convergence table rather than in any argument from principle. A user served a
filled checkpoint gets a model whose numerics are substantially set by an
invented scale, with nothing on the surface to say so. Refusing to load is the
better failure.

The harness can do what the engine should not, because it discloses. Law 17
records the fill on the report, the substituted parameters enter the
comparability key, and a filled candidate ranks only against others measured the
same way. That is the whole justification, and it does not transfer to a serving
path that has no comparability key to put anything in.

Two consequences follow.

**Those two candidates will stop loading on stock vLLM.** When any of the
rejecting PRs lands, `Neural-ICE/Gemma-4-26B-A4B-it-NVFP4` and
`bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4` will refuse at load. Their
published numbers stay valid for what they are and become unreproducible without
this fill, so the fill has to be maintained as a standing, disclosed divergence
rather than treated as a fix awaiting merge.

**The contribution worth making is evidence, not code.** #45320, #54444, and
#55073 all report that no end-to-end evaluation was run: no Blackwell hardware,
or no affected checkpoint, or both. This harness has SM120, two affected
checkpoints, and measured numbers for what the missing scales cost. That is the
gap in those PRs, and it is not a competing patch.
