# QxQ and BxQ — Routed-Model Distribution Fidelity

**Applies to:** laws version 12, Law 14.
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
evidence, and at least one kernel's claim is false in a way that silently
destroys the measurement (§6).

`inspect_model_moe_backends` in `vllm/v1/sample/kld.py` therefore grants
`certified_for_exact_repeat` only when all three of these hold:

1. The expert class is named in `_EXACT_REPEAT_CERTIFIED_EXPERTS`, an allowlist
   earned by passing an exact-repeat probe on real suite content. The report
   carries this separately as `exact_repeat_probed`.
2. The kernel also self-declares batch invariance.
3. The run is not expert-parallel.

The allowlist currently holds `MarlinExperts`, `BatchedMarlinExperts`,
`TritonExperts`, and `Nvfp4QuantizationEmulationTritonExperts`.

The default is refusal. An unprobed backend — including one that arrives with a
future vLLM bump — is uncertified until somebody probes it, and an uncertified
backend is reported as uncertified rather than published as a number. This
matters more than it sounds: the failure mode it replaces is a kernel quietly
producing garbage that the harness dutifully writes down as a fidelity result.

DeepGEMM, FlashInfer MoE, AITER, XPU, CPU, and every expert-parallel path remain
uncertified.

## 6. Case study: a kernel that declares batch invariance and returns NaN

`CutlassExpertsFp4` self-declares batch invariance. Under
`VLLM_BATCH_INVARIANT=1`, on W4A4 NVFP4 checkpoints, it takes finite inputs to
NaN. The scorer then refuses a non-finite KLD and the engine dies, which is how
this surfaced: two NVFP4 candidates failing with `KLD is not finite`, no report
produced, and nothing to indicate the kernel rather than the checkpoint.

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

The `--moe-backend` override then isolated the kernel:

| `VLLM_BATCH_INVARIANT` | Backend | Result |
| --- | --- | --- |
| 0 | `FLASHINFER_CUTLASS` (auto) | pass |
| 1 | `VLLM_CUTLASS` (auto) | **NaN** |
| 1 | `marlin` (forced) | pass |
| 1 | `emulation` (forced) | pass |

The bug is specific to the CUTLASS FP4 experts under batch invariance, not to
NVFP4, not to the checkpoints, and not to batch invariance itself. It is why §5
fails closed: this kernel's self-declaration was the only thing standing between
a broken forward pass and a published fidelity number.

## 7. W4A4 NVFP4 scores on faithful emulation

A W4A16 export and a W4A4 export of the same model differ only in whether
activations are also quantized, and that difference decides which kernels can
score the checkpoint honestly. A repository name says nothing reliable about it,
so `_quantizes_activations_to_fp4` in `examples/offline_inference/score_mode_kld.py`
reads the checkpoint: any entry in `quantization_config.config_groups` whose
`input_activations` declares `num_bits: 4` and `type: "float"` makes it W4A4.

For those checkpoints the scorer pins `moe_backend="emulation"`, which
quantize-dequantizes both activation stages and is batch invariant by
construction.

**This path is not yet faithful for MoE, and no W4A4 result may be published on
it.** The emulation branch of `convert_to_nvfp4_moe_kernel_format` collapses the
per-expert activation scales to one scalar per layer —
`a13_scale = 1.0 / a13_scale.max()` — and warns when the per-expert values differ.
vLLM's own comment on that line records the consequence: taking the largest global
scale "likely results in overflowing the FP8 range for other experts." A
checkpoint whose per-expert scales span an order of magnitude is therefore scored
against an activation scale it never uses, and the resulting mean is biased by an
amount nothing in the run discloses.

So W4A4 NVFP4 currently has **no faithful batch-invariant MoE path at all**:

| Path | Per-expert activation scales | Exact repeat |
| --- | --- | --- |
| CUTLASS / FlashInfer FP4 | honoured | uncertified, and NaN on calibration gaps |
| Marlin | dropped entirely (scores W4A16) | certified |
| Emulation | collapsed to a layer maximum | certified |

A candidate is **not withdrawn** for this. Withdrawal is for a result that cannot
be interpreted at all — one bound to a capture nothing publishes, say. A
substituted activation scale produces a number that means something precise; it is
the *label* that would be wrong, not the measurement. The remedy is disclosure:
the substitution is recorded on the report, it enters the comparability key so
substituted results rank against each other and not against faithful ones, and the
leaderboard marks the affected cells. Four NVFP4 candidates in the
gemma-4-26B-A4B-it family carry this disclosure.

Two details matter and both were bugs first:

**The pin belongs on the student only.** It is applied to `student_kwargs`, not
the shared `llm_kwargs`. Applied to the latter it propagates to the unquantized
BF16 teacher, which has no such scheme, and reference engine initialization fails.

**Marlin is refused here, not merely not preferred.** Marlin is batch invariant
and would run, but its MoE path drops activation scales — it would score a W4A4
checkpoint as though it were W4A16 and report a fidelity the checkpoint never
delivers. A wrong number that passes every law is worse than a refusal.

A weight-only W4A16 NVFP4 result measures the quantization scheme rather than a
native FP4 kernel's own rounding, and that limit is stated on the published card.

A cautionary note on reading results from this path. Before the collapse above was
understood, four NVFP4 exports carrying byte-identical QDQ diagnostics (0.66652247
and 1.43642041) scored QxQ between 1.16299506 and 1.82545539, and that 0.6 nat
spread was written up here as living "entirely in the activation scheme and kernel
path." That reading was unsupported. Those checkpoints differ in how widely their
per-expert activation scales spread, so they differ in how much the layer-maximum
substitution costs them, and the spread was substantially an artifact of the
measurement path. It is recorded here because it is the exact shape of mistake this
document exists to prevent: a real, reproducible, bitwise-exact number that is
nonetheless measuring the harness rather than the checkpoint.

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
count. That checkpoint now scores 1.17816288 and passes all sixteen laws.

## 9. The runtime the numbers are bound to

Scoring pins `VLLM_BATCH_INVARIANT=1`, disables DeepGEMM
(`VLLM_MOE_USE_DEEP_GEMM=0`) and FlashInfer autotune, sets `NCCL_DETERMINISTIC=1`
and `CUBLAS_WORKSPACE_CONFIG=:4096:8`, enforces eager execution, disables prefix
caching, and holds `max_num_seqs=1`.

None of that is optional and none of it is a performance setting. Each one closes
a path by which two runs of the same tokens could diverge.

Because that runtime *is* part of the result, its identity is bound into the
comparability key: `vllm_commit`, `vllm_dirty_digest`,
`compiled_extensions_sha256`, `torch`, `driver`, and `gpu_names`, alongside the
suite and geometry. Two candidates are ranked against each other only when all of
it matches. A consequence worth internalizing before committing to this
repository: **any commit changes `vllm_commit`, and a dirty tree changes
`vllm_dirty_digest`, so the next scoring run treats every prior report as stale
and rescores the family.** That is correct behaviour, not a bug, and it is why
harness fixes are best batched.

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
and skips the rest.

## 12. What these numbers do not say

**They are not comparable outside their group.** Not against numbers from another
suite, geometry, runtime, or laws version, and not against any published
elsewhere. The comparability key is printed with every leaderboard group for
exactly this reason.

**An NVFP4 result measures the scheme, not a native FP4 kernel.** See §7. The
faithful-emulation path is a deliberate substitution, disclosed rather than hidden.

**The QDQ ladder is diagnostic, never a candidate.** Those cells round weights on
synthetic BF16 checkpoints and route naturally. They are not QxQ or BxQ, they are
not rankable against deployed candidates, and the published tables separate them.

**A flip rate is not an error rate.** Two experts disagreeing on a token is not
by itself a wrong answer; the delta is what quantifies the cost. AutoRound's
93.07% flip rate with a +0.275 delta is the point — high disagreement, bounded
consequence.

**Certification is about repeatability, not accuracy.** `exact_repeat: certified`
says two runs agree bit for bit. It says nothing about whether the kernel computes
the right thing, which is what the zero baseline (Law 1) and the reference binding
(Laws 12 and 16) are for.
