# Addendum for vllm-project/vllm#54444

End-to-end evidence for *[Bugfix][Quantization] Reject NVFP4 checkpoints with
missing global scales (linear + MoE)*.

**This is not a competing patch.** It is the validation the PR states it does not
have: "No unit test is added... I exercised the validation predicates directly on
CPU." The same gap is stated on #45320 ("No local full-model result is claimed")
and #55073 ("Model evaluation: NOT RUN locally... the available RTX 4060 Laptop is
SM89 and cannot execute the affected Blackwell NVFP4 MoE path").

We have the hardware, two affected published checkpoints, and measured output.
Everything below is reproducible from the pinned revisions given.

---

## 1. Two affected checkpoints, both in this PR's scope

Both are `quant_method: "modelopt"`, so they load through
`ModelOptNvFp4FusedMoE` — the class this PR patches.

| Checkpoint | Revision | ModelOpt producer |
| --- | --- | --- |
| `Neural-ICE/Gemma-4-26B-A4B-it-NVFP4` | `659c96699c565f371094dd04f911858ca8e84789` | `0.42.0` |
| `bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4` | `a15dd6f161881b62db952303a5bfb7be118ed15e` | `0.43.0rc2.dev57+g87ea8babe` |

Both declare `quant_algo: "NVFP4"`, `targets: ["Linear"]`, and — the detail that
makes the missing scale unambiguous — `input_activations.dynamic: false`. The
input scale is a static calibration value. There is no runtime path that could
legitimately supply it, so an absent one is a gap rather than an intentional
omission.

Two independent ModelOpt versions produce the same defect, which suggests this
is a property of the export path rather than one bad upload.

A third checkpoint, `RedHatAI/gemma-4-26B-A4B-it-NVFP4` at
`5557756b8dce33ac72f2bd702b11729fdba3b839` (compressed-tensors, all-`Linear`
W4A4), has every scale present. It is the control below.

### Reproducing the gap without downloading anything

Safetensors headers are readable over HTTP range requests, so a reviewer without
Blackwell hardware can still confirm which experts are short:

```python
import json, requests
from huggingface_hub import HfApi

repo = "Neural-ICE/Gemma-4-26B-A4B-it-NVFP4"
rev  = "659c96699c565f371094dd04f911858ca8e84789"
shards = [
    e.path
    for e in HfApi().list_repo_tree(repo, revision=rev, recursive=True)
    if e.path.endswith(".safetensors")
]
for name in shards:
    url = f"https://huggingface.co/{repo}/resolve/{rev}/{name}"
    n = int.from_bytes(
        requests.get(url, headers={"Range": "bytes=0-7"}).content, "little"
    )
    keys = json.loads(
        requests.get(url, headers={"Range": f"bytes=8-{8+n-1}"}).content
    )
    # group by module prefix; compare experts carrying weight_scale_2
    # against experts carrying input_scale
```

Key on `weight_scale_2` (ModelOpt) or `weight_global_scale`
(compressed-tensors), not on `weight_scale`. FP8 layers also have a
`weight_scale`, and keying on it reports every FP8 layer of a mixed-precision
checkpoint as a missing NVFP4 scale. See §6.

---

## 2. What the missing scale does on Blackwell

Hardware: SM120 (consumer/SoC Blackwell). Scored under `VLLM_BATCH_INVARIANT=1`,
`VLLM_MOE_USE_DEEP_GEMM=0`, FlashInfer autotune off, `NCCL_DETERMINISTIC=1`,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, eager execution, prefix caching off,
`max_num_seqs=1`.

With `torch.empty` allocation on `main`, both checkpoints took **finite inputs to
NaN** inside `moe.experts`, and the failure was exactly reproducible:

- layer 0, prompt row 814
- layer 14, prompt row 170

Two details mattered for reproduction, and both are reasons a reviewer may have
failed to see this:

1. **Synthetic prompts do not trigger it.** They are too repetitive to route
   across enough experts to hit an uncalibrated one. Real content at full
   context does.
2. **Generation-only testing does not trigger it.** Logits must be computed at
   every prompt position (`prompt_logprobs`), which is what a scoring or
   evaluation harness does.

The backend matters, and confirms this PR's own claims about which paths consume
the scales:

| `VLLM_BATCH_INVARIANT` | MoE backend | Result |
| --- | --- | --- |
| 0 | `FLASHINFER_CUTLASS` (auto) | pass — collapses per-expert scales to one scalar, so a NaN slot is averaged away |
| 1 | `VLLM_CUTLASS` (auto) | **NaN** — consumes a genuine per-expert vector |
| 1 | `marlin` (forced) | pass — drops activation scales entirely |
| 1 | `emulation` (forced) | pass — collapses to a layer maximum |

Only the path that honours per-expert scales exposes the bug. That is worth
stating plainly because it means **the severity of this bug is backend-dependent,
and the safest-looking backends are the ones hiding it.**

### Why vLLM's own loading checks cannot catch this

Independently confirmed while diagnosing it, and it argues for putting the guard
exactly where this PR puts it:

- Strict all-parameters-loaded tracking is off by default for quantized models.
- Any module defining `process_weights_after_loading` has **every** parameter
  force-marked as loaded, so a partially populated tensor is invisible to that
  tracking by construction.

A load-time validation inside `process_weights_after_loading` is therefore the
only place this is detectable.

---

## 3. The failure mode that is worse than the NaN

A scan of both checkpoints' stored tensors came back clean: no non-finite values
on disk. The NaN was generated at runtime, from `torch.empty` contents.

That is the important part. `torch.empty` returns whatever was in that memory,
and **most of the time it is finite.** We saw NaN because NaN is loud. A finite
piece of allocator garbage in the same slot produces a silently wrong per-expert
alpha, a quietly degraded expert, and no error anywhere.

The NaN is the detectable corner of the bug. The undetectable interior is why the
sentinel is the right mechanism rather than an `isfinite` check on
`torch.empty` output.

---

## 4. Measured cost, and why it supports rejecting rather than filling

We needed those two checkpoints to produce a number, so our measurement harness
does what this PR deliberately declines to do: it fills a missing per-expert
scale from the maximum of the present positive scales on that tensor, and
discloses the substitution on the published result.

**Our own data says your choice is the right one for a serving engine.**

Setup: KL divergence of each quantized candidate against the unquantized BF16
`google/gemma-4-26B-A4B-it`, 768 contexts of 2048 tokens, identical suite and
runtime for every row, bitwise-exact repeat certified at `atol=0` for all of
them.

| Checkpoint | Per-expert scales | Experts filled | Worst spread | KLD vs BF16 |
| --- | --- | --- | --- | --- |
| `RedHatAI/gemma-4-26B-A4B-it-NVFP4` | complete | — | — | **1.77968754** |
| `Neural-ICE/Gemma-4-26B-A4B-it-NVFP4` | incomplete | 15 of 30 layers, 26 slots | 230.6x | **1.82281421** |
| `bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4` | incomplete | 12 of 30 layers, 25 slots | 240.3x | **1.82307292** |

Two things to take from this.

**The fill is not free.** Both filled checkpoints score worse than the complete
export. The spread column is why: filling from the layer maximum assumes the
widest activation range in the layer, which cannot saturate e4m3, but at a spread
past roughly a hundred it pushes a quiet block's e4m3 scale into the subnormal
range and starts costing mantissa. At 230–240x that is happening.

**The fill erases the checkpoint.** The two filled results land **0.00026 nats
apart.** They are not the same checkpoint: different publishers, different
ModelOpt versions, different fill footprints (15/26 versus 12/25), distinct
weight digests, and distinct per-position KLD digests, so the two runs did not
even produce bitwise-identical output. Two independent quantizations do not agree
to four decimal places on their own. On the same suite, complete NVFP4 exports
separate normally, so this is not the measurement failing to discriminate.

The reading that fits is that once a tensor's per-expert scales are replaced by
one layer maximum across half the layers, the imputed value sets the behaviour
and the checkpoint's own quantization stops being visible in it.

That is an argument for `ValueError`. A user served a filled checkpoint gets a
model whose numerics are substantially decided by a value the loader invented,
with nothing on the surface saying so. Our harness can do it only because it
discloses the substitution and refuses to rank a filled result against a measured
one. A serving engine has no such mechanism, and #45320's position — "it does not
guess missing calibration statistics or add an imputation policy" — is correct.

Two caveats stated honestly. This is n=2: a strong indication, not a proof, and a
fill-direction sweep would settle it, which we have not run. And the complete
control is a compressed-tensors export while the two incomplete ones are ModelOpt,
so the comparison is not exporter-matched — it is the closest available control
(all-`Linear` W4A4 NVFP4, `lm_head` excluded in all three, identical suite,
reference, and runtime), not a perfect one. The 0.00026 convergence between the
two filled results does not depend on that control at all.

---

## 5. Independent confirmation of the backend-consumption table

This PR restricts validation to the scales the selected path actually consumes.
We had to characterise the same thing to choose a scoring backend, and we agree:

| Path | Per-expert activation scales |
| --- | --- |
| vLLM CUTLASS FP4 | honoured as a per-expert vector (`a1_gscale` / `a2_gscale`, length `e`) |
| FlashInfer FP4 | collapsed to one scalar via `amax_for_moe_activation_quant(...).repeat(num_experts)` |
| Marlin | dropped entirely; effectively scores W4A16 |
| Emulation | collapsed to a layer maximum via `1.0 / a13_scale.max()` |

Two notes that may be useful for the PR text:

- Because `a2_scale` holds `1 / w2_input_global_scale`, the `.max()` in the
  emulation path selects the **smallest** per-expert scale. vLLM's own comment
  there already warns this "likely results in overflowing the FP8 range for other
  experts."
- Excluding the W4A16 `input_scale` from validation is right. It is a discarded
  placeholder on that path and is legitimately absent from weight-only exports.

---

## 6. One review request: do not fire on mixed-precision checkpoints

A real published checkpoint that a validator must not reject:
`unsloth/gemma-4-26B-A4B-it-NVFP4` at
`20df0542b1a86ce19f495ac2eca2c7c12bce82f9`, `quant_method: "compressed-tensors"`,
`format: "mixed-precision"`, llmcompressor `0.17.2.a20260707`:

- `group_0`, targets `re:.*self_attn\.(q|k|v|o)_proj$` — **FP8**, 8-bit weights
  per channel, 8-bit dynamic per-token activations
- `group_1`, targets `re:.*\.experts\.\d+\.(gate|up|down)_proj$` and
  `re:.*language_model.*\.mlp\.(gate|up|down)_proj$` — **NVFP4**, 4-bit,
  `tensor_group`, group size 16
- plus an 8-bit float `kv_cache_scheme`

All 115 of its attention projections carry a `weight_scale` and correctly carry no
NVFP4 `input_global_scale`, because they are not NVFP4 layers at all. (115 rather
than 120 because `attention_k_eq_v: true` shares k and v on the five
`full_attention` layers, so those have three projections instead of four.) A check
that keys on `weight_scale` presence flags every one of them. We hit this exact
false positive and report it because it is the shape of checkpoint most likely to
produce a spurious rejection from an otherwise correct validator.

Since this PR validates inside the NVFP4 quantization methods rather than by
scanning key names, we would expect it to be unaffected — worth confirming, since
a false `ValueError` at load is a worse regression than the bug being fixed.

---

## 7. Summary of what we are offering

1. Two named, pinned, publicly available checkpoints in this PR's exact code
   path, from two ModelOpt versions.
2. A no-download reproduction of the missing keys from safetensors headers.
3. Confirmation on SM120 Blackwell that the current `torch.empty` behaviour
   reaches NaN from finite inputs, with the reproducing layer and row, and the
   two harness conditions required to see it.
4. The observation that NaN is the loud corner of a mostly silent bug, since
   `torch.empty` garbage is usually finite.
5. Measured KL divergence showing what the missing scales cost, and evidence
   that imputing them destroys the distinction between two different
   checkpoints — support for this PR's fail-fast policy over any request to
   guess the values.
6. Independent confirmation of the backend-consumption table, plus a
   mixed-precision checkpoint worth testing against for false rejections.

Happy to run additional configurations on SM120 against either checkpoint if
that would help the review.

---

*Numbers in §4 come from a determinism-controlled KL divergence harness, not from
`lm-eval`. Every candidate was scored on one frozen token suite, one runtime, and
one reference, with bitwise-exact repeat verified at `atol=0` before any value was
recorded. The absolute KLD values are only comparable to each other, not to
figures published elsewhere.*
