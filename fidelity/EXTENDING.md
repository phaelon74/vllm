# Extending the suite to a model it was not built for

The models this suite has scored are dense and routed transformers with
conventional attention. The models arriving next are not only that: linear and
hybrid attention, latent KV caches, multi-token-prediction heads, always-on shared
experts, and quantization schemes that did not exist when the certification set was
written. Something in the list above will break this suite, and this document is
how to find out which thing and what to do about it.

Organised by mechanism rather than by model, deliberately. Model names age out
within months; "this checkpoint has a latent KV cache" stays true.

## The one thing to understand first

**The suite refuses rather than guesses.** Almost every way a new model can be
unsupported ends in a hard error before any number is produced, which means your
job is usually not debugging a wrong number — it is reading a refusal, deciding
whether the gate is right, and either extending it honestly or excluding the
candidate with a stated reason.

The dangerous cases are the two that *don't* refuse, and both are covered below:
a routed model misread as dense, which silently skips BxQ; and a kernel whose
class name is already in the certification set but whose behaviour is not what the
name was certified for.

## Triage, in order

Do these in sequence and stop at the first failure. Each step is cheap relative to
the one after it.

### 0. Does upstream vLLM run this model at all?

```bash
python -c "
from vllm import LLM
llm = LLM(model='<path-or-repo>', max_model_len=4096, enforce_eager=True)
print(llm.generate(['The capital of France is'])[0].outputs[0].text)
"
```

If this fails, the problem is model support and belongs upstream, not here. Do not
start extending the fidelity suite for a model vLLM cannot load. If it fails only
on this fork, check that against upstream at the same commit before assuming the
fork is at fault — and if the fork *is* at fault, that is a bug worth fixing rather
than working around.

### 1. Is it seen as routed?

```bash
python -c "
import sys; sys.path.insert(0, 'fidelity')
from campaign import qdq_routing
print(qdq_routing('<reference-path>'))
"
```

`None` means dense, and dense means no BxQ, no Law 14, no routing analysis. If the
model *is* routed and this prints `None`, you have found a real gap — see
"Routed models whose expert count has a new name" below. This is the highest-value
step in the list because it is the one failure that is silent.

### 2. What did the loader actually build?

```bash
python -c "
from vllm import LLM
from vllm.v1.sample.kld import (inspect_model_moe_backends,
                                inspect_model_recurrent_backends,
                                inspect_model_lm_heads)
import json, os
os.environ['VLLM_BATCH_INVARIANT'] = '1'
llm = LLM(model='<candidate-path>', max_model_len=4096, enforce_eager=True)
for fn in (inspect_model_moe_backends, inspect_model_recurrent_backends,
           inspect_model_lm_heads):
    print(fn.__name__)
    print(json.dumps(llm.apply_model(fn), indent=2)[:4000])
"
```

This is the single most informative command in this document. It reports, per
layer, the quantization method, the kernel class, the expert class, the router, the
recurrent backend, and — crucially — `batch_invariant_supported` and
`certified_for_exact_repeat` for each.

Two different certification mechanisms show up here and it is worth knowing which
you are looking at. MoE experts are certified by an allowlist of class names,
`_EXACT_REPEAT_CERTIFIED_EXPERTS`, and-ed with the backend's own
`batch_invariant_supported` and with expert parallelism being off. Recurrent layers
are certified purely by asking the backend `supports_batch_invariance()`. The
allowlist is the thing a new model is most likely to fall outside of.

If either inspection returns no layers for a model that plainly has those layers,
the inspection walk does not recognise the module type. That is its own kind of
gap: see the mechanism sections.

### 3. Will it repeat?

```bash
python fidelity/campaign.py smoke --config <your-config>.json \
  --only-candidate <name>
```

A pass prints the chosen backend and a natural control of exactly
`max=0.000e+00`. Any other value means the model does not reproduce itself and no
number from it is publishable. `scripts/glm53_determinism_probe.py` and
`scripts/glm53_layer_bisect.py` exist for exactly this hunt — the first sweeps
determinism-relevant environment flags, the second bisects to the layer where two
supposedly identical runs first diverge.

### 4. Is the KV cache dtype right?

```bash
python -c "
import sys; sys.path.insert(0, '.')
from vllm.v1.sample.kld import unquantized_kv_cache_dtype
print(unquantized_kv_cache_dtype('<reference-path>'))
"
```

The rule is that the cache is never quantized, not that it is always bfloat16 —
forcing bfloat16 onto a float16 checkpoint leaves query and key in different dtypes
and FlashAttention refuses the pair. If this prints something your checkpoint did
not declare, see "A dtype or a cache kind the policy does not know" below.

## By mechanism

### Routed models whose expert count has a new name

**What breaks:** `qdq_routing` decides routed-versus-dense by looking for
`num_experts`, `n_routed_experts`, `num_local_experts`, or `moe_num_experts` in
`config.json` or in `text_config`, and requires the value to exceed 1. A model
declaring its expert count under a fifth name is classified as dense.

**Why it matters more than it looks:** nothing refuses. The candidate scores, gets
a `mean_kld`, publishes, and the smoke step prints "dense reference has no BxQ
smoke test" — which reads as a fact about the model rather than as a failure to
recognise it. You get a QxQ-only row for a routed model and no indication that the
routing contribution was never measured.

**Fix:** add the key to the tuple in `qdq_routing` (`campaign.py`). This is a
one-line change in `fidelity/`, so it is outside the numerics digest and costs no
rescore — but candidates already published as dense will need rescoring to gain
their BxQ cell, because that is a real change in what was measured.

**Check it worked:** step 1 above returns a dict, and step 3's smoke now runs a BxQ
probe instead of skipping it.

### A new quantization scheme, or a new expert kernel class

**What breaks:** the expert class name is not in `_EXACT_REPEAT_CERTIFIED_EXPERTS`,
so `certified_for_exact_repeat` is false for every routed layer and scoring refuses
after the model load and before any forward pass.

**What not to do:** add the name to the set because the smoke test passed. The
allowlist is a record of kernels that have been *probed*, not a list of kernels
believed to be fine, and a name in it is trusted permanently and silently — that is
precisely how a build without the fork's Marlin patch ends up certified
([`INSTALL.md`](INSTALL.md) §4).

**The certification procedure**, which is what earns a name its place:

1. Confirm the kernel's own `supports_batch_invariant` is true. If the
   implementation does not claim batch invariance, there is nothing to certify and
   the work is upstream in the kernel.
2. Prove repeat: the same window scored twice in one process must agree bitwise.
   This is the smoke step's natural control and it must be exactly zero, not small.
3. Prove batch invariance, which is the part the natural control does not cover.
   The same prompt scored in differently-composed batches must produce identical
   logits. This is the property the Marlin MoE patch exists to provide, and it is
   the one that quietly fails when a kernel's reduction or thread configuration
   depends on the batch shape.
4. Prove it across a process restart, so you are not certifying a warm cache.
5. Record what you did. A name added to that set without a written probe is a
   liability for everyone downstream.

`scripts/validate_deterministic_qxq_bxq.sh` is the harness closest to steps 2–4.

**If the kernel genuinely is not batch-invariant:** do not certify it. Exclude the
candidate with a reason — see the last section — or fix the kernel. There is no
third option that produces a publishable number.

**Also check the activation-scale path.** A new low-precision scheme probably has
per-tensor or per-expert activation scales, and `_FP4_ACTIVATION_SCALES` plus
`_activation_scale_substitution` in `kld.py` only know the two NVFP4 key pairs
(`w13_input_global_scale`/`w13_input_scale` and the `w2_` equivalents). A scheme
with different key names gets no substitution disclosure at all, which means Law 17
reports nothing and a collapse like the one in [`DECISIONS.md`](DECISIONS.md) §4
would go unrecorded. Extend the tuple and the reader together.

### Shared or always-on experts

**What breaks:** BxQ forces the student to use the reference's expert selections.
A shared expert that runs for every token is not *selected*, so it is not part of
what gets forced — which is correct — but it does mean the QxQ-to-BxQ gap no longer
partitions cleanly into "routing" and "experts." Some of the expert arithmetic was
never subject to routing in the first place.

**What to do:** this is a reporting problem rather than a correctness one. The
number is right; the interpretation in [`QXQ.md`](QXQ.md) §2 is what needs a
caveat. Before publishing a family with shared experts, record the shared-expert
count alongside the routed count so a reader can see how much of the model the
routing analysis does not speak to. Check what the inspection reports for those
modules — if they are not `FusedMoE`-shaped they may not appear in
`inspect_model_moe_backends` at all, and an absence in the layer list should never
be read as an absence in the model.

### Linear and hybrid attention

Applies to Mamba-style state-space layers, gated delta networks, and the mixed
stacks that interleave them with full attention — the mechanism behind most models
marketed as "Flash" or "Next."

**What breaks, in order of likelihood.** First, determinism: recurrent kernels
carry separate prefill and decode paths, and both must be batch-invariant for the
model to repeat. `inspect_model_recurrent_backends` reports
`prefill_backend` and `decode_kernel` separately for exactly this reason, and
certification here is the backend's own `supports_batch_invariance()` rather than
an allowlist. Second, recognition: the walk keys on `MambaBase`, so a new recurrent
layer that does not inherit from it is invisible to inspection — and an empty layer
list is indistinguishable from a model with no recurrent layers.

**Precedent to read before starting:** the fork's largest single-file change is
`vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`, and it is a
determinism fix in a linear-attention layer. Whatever you are about to do has been
done once already; read that diff first.

**Cost estimate:** this is the most expensive mechanism on the list, because
making a recurrent kernel batch-invariant is upstream kernel work, not harness
work. Budget accordingly, and note that the number is unobtainable until it is
done — there is no partial credit.

### MLA and latent KV caches

Applies to DeepSeek-lineage and GLM-lineage attention.

**What breaks:** the cache the KV policy reasons about is a latent projection
rather than per-head keys and values. `unquantized_kv_cache_dtype` still answers a
dtype question correctly, because it reads the checkpoint's declared dtype and does
not care about the cache's shape. What needs checking is whether the latent cache
respects that dtype, and whether any part of the MLA path quantizes internally —
because "the KV cache is never quantized" is a constraint on the arithmetic, not
on the name of a config field.

**Determinism:** sparse and paged MLA backends select kernels by shape. The fork
already touches `vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py`, which is
the shape of problem to expect. Run step 3 before believing anything.

**What to record:** if the model has a latent cache, say so on the artifact. A
reader comparing a latent-cache model against a conventional one is comparing two
different memory regimes, and the comparability key does not capture that because
it is about the runtime rather than the architecture.

### Multi-token-prediction and speculative heads

**What breaks:** this suite scores one distribution per position, produced by one
LM head, against a reference's distribution at the same position. A model with an
MTP head has more than one head and can produce more than one distribution per
position, and nothing in the current code decides which one is *the* distribution.

**What the suite does today:** `inspect_model_lm_heads` reports the heads it finds,
including whether each is quantized, and `detect_lm_head_quantization` reads the
checkpoint. So the presence of extra heads is visible, but no policy exists for
them.

**The decision you have to make, and make explicitly:** score the main head only,
with speculation disabled, and disclose that the MTP head was present and not
measured. That is the honest default and it is what we would do — the main head is
what the comparison against an unquantized reference is meaningful for, and
speculative decoding is a throughput mechanism whose output is supposed to be
distributionally identical to the main head's. If it is *not* identical in a
quantized checkpoint, that is a genuinely interesting result and a separate
measurement, not something to fold into QxQ.

**Do not** let speculation run during scoring and then report the number as if one
head produced it. Verify by checking that the report's head inspection lists the
extra head and that the run's configuration disabled speculation.

### A dtype or a cache kind the policy does not know

**What breaks, concretely:** `UNQUANTIZED_KV_CACHE_DTYPES` includes `float32`, but
`_KV_CACHE_DTYPE_ALIASES` maps only `bfloat16`/`bf16`, `float16`/`fp16`/`half`. A
checkpoint declaring `torch_dtype: "float32"` therefore falls through to the
bfloat16 default. The constant anticipates a case the alias map does not produce.

**Fix:** add the alias. The fallback is deliberately safe rather than correct —
bfloat16 is what vLLM would pick anyway — so this is a silent narrowing rather
than a failure, which is why it needs finding on purpose.

**For a genuinely new kind of cache** (a convolutional state, an SSM state, a
compressed cache), the question to answer is not what dtype it is but whether it is
quantized, and if so whether that quantization is part of the checkpoint under test
or part of the runtime. If it is part of the runtime, turn it off. If it is part of
the checkpoint, it is a property of the candidate and must be disclosed like any
other substitution.

### A new tokenizer or vocabulary

**What breaks:** token IDs are not portable. A suite minted for one tokenizer
scored against a model with a different vocabulary is measuring nonsense, and the
suite's identity is bound into the artifact precisely so this cannot happen
quietly.

**What to do:** mint a new suite, following "Building the token suite" in
[`README.md`](README.md). Then accept the consequence: your family's numbers are
not comparable with any family on a different suite, and the comparability key will
say so. That is the correct outcome, not a limitation to work around.

**Watch for vocabulary padding.** `tokenizer_unpadded_vocab_size` and
`resolve_vocab_size` exist because a padded vocabulary makes the KLD sum over
positions that carry no probability mass. A new tokenizer with unusual padding is
worth checking against those two functions before minting.

### A new field on the report, and the law that reads it

**What breaks:** nothing, if you add the field. Everything, if you add a law that
*requires* it.

A law that reads a field old reports do not carry makes every old report
permanently uncurrent, and no amount of re-rendering will fix it — only rescoring
will. That is not hypothetical; it cost this project a multi-day rescore of 45
candidates ([`DECISIONS.md`](DECISIONS.md) §2).

**The rule:** add fields to the report generously, because they are free. Add laws
that depend on new fields rarely, and when you do, know that you are buying a
rescore of the entire library and say so before starting. If the law can be written
to treat a missing field as "no expectation" rather than as a failure, write it that
way — that is the pattern `_prior_kernel_identity` uses, returning an empty result
when there is nothing to hold a run to.

## Rules for extending

Short, and each one is a reversal we already paid for
([`DECISIONS.md`](DECISIONS.md)).

- **Never pin a kernel to make a model work.** A pin can refuse to load a
  checkpoint. Let the oracle choose and verify afterwards.
- **Never add a name to the certification allowlist without a probe.** The set is
  a record of evidence, and it is trusted silently and permanently.
- **Never quantize the KV cache**, for any model, for any reason.
- **Never enforce a rule in two places.** If a gate must exist at two stages, the
  second stage calls the first stage's implementation.
- **Never drop a candidate quietly.** Use `excluded_candidates`.
- **Prefer a refusal to a guess.** If the suite cannot tell whether a number means
  what it should, the correct behaviour is to produce no number.

## When to stop and exclude the candidate

Sometimes the honest answer is that this model cannot be scored yet. A kernel that
is not batch-invariant, a recurrent backend nobody has made deterministic, a
checkpoint whose scales cannot be recovered — these are real, and pretending
otherwise produces exactly the kind of number
[`DECISIONS.md`](DECISIONS.md) §4 is an apology for.

Record it:

```json
"excluded_candidates": [
  {
    "hf_repo": "org/model-quantized",
    "revision": "<sha>",
    "reason": "SomeNewExperts is not batch-invariant; repeats differ at 1e-3"
  }
]
```

That writes `excluded-candidates.json` into the library, surfaces on the card and
in `kld-vs-size.json`, and refuses if the candidate is also active at the same
revision. An exclusion with a specific, falsifiable reason is a contribution — it
tells the next person what to fix. "Did not work" is not.
