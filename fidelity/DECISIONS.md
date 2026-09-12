# Decisions, and the ones we reversed

[`LAWS.md`](LAWS.md) states the rules a published artifact obeys.
[`QXQ.md`](QXQ.md) explains the measurement and the failure modes it guards
against. Neither says why the design is shaped the way it is, and several of the
laws exist because an earlier, more obvious design was tried and produced a wrong
number that looked right.

This document is that record. It is written for two readers: someone deciding
whether to trust a published figure, and someone about to change this code and
tempted by an idea we already tried. Where a decision is enforced somewhere, the
enforcement is named, because a documented intention with no code behind it is
worth very little.

## How we got here

The suite began as a determinism investigation, not a benchmark. The branch is
still called `feature/glm53-kld-determinism`, and the seventeen probes under
`scripts/` — layer bisection, first-non-finite-module search, a packet capture of
logits, an environment sweep across determinism flags — are the residue of
chasing a model whose outputs would not repeat. That work produced the machinery
before it produced the metric: if you can make a forward pass bitwise repeatable
and can prove you have, then the difference between two models becomes a
measurement rather than an anecdote.

The metric came second. A quantized model's divergence from its own unquantized
reference, measured as KL divergence over a frozen token suite, is a number that
answers the question people actually have about a quantization: not "does it still
score well on a benchmark" but "how far did the distribution move." Making that
number defensible is where the rest of the system comes from — the laws, the
provenance, the digests, and the refusals.

The publication layer came third, and is the reason for the strictness. A number
kept in a notebook needs only to be right. A number published to a leaderboard has
to be right, has to say what it is bound to, and has to refuse to be compared
against numbers it is not comparable with.

## The reversals

### 1. Two cells, not one number

**First:** one KLD per candidate.

**What broke:** for a routed model it conflates two independent failures.
Quantization can move the expert distributions, and it can move the router so a
different expert runs. One number cannot distinguish "the experts got worse" from
"a different expert answered," and those have different fixes.

**Now:** QxQ measures the quantized model against the reference end to end. BxQ
holds routing to the reference's own selections and measures only the expert
arithmetic. The gap between them is the routing contribution. See
[`QXQ.md`](QXQ.md) §1–2 and Law 14.

**Consequence accepted:** BxQ requires forcing expert selection, which needs
hooks throughout the request path, and it exists only for routed models. Dense
candidates publish `mean_kld` alone, and the smoke step says so out loud rather
than emitting a fabricated cell.

### 2. Binding a result to the git commit, replaced by a numerics digest

**First:** every report recorded the commit that produced it, and a result was
current only at that commit.

**What broke:** the binding was far too coarse. Editing this documentation,
adjusting a chart, or fixing a typo in a law moved the commit and invalidated
every number in the library. The cost of a docs change was a multi-day rescore,
which in practice means the docs do not get fixed.

**Now:** `numerics_digest()` in `vllm/v1/sample/kld.py` hashes every `.py` under
`vllm/` plus `examples/offline_inference/score_mode_kld.py` — the code that can
actually reach a number. `_NUMERICS_TREES` and `_NUMERICS_FILES` are the whole
definition. The commit is still recorded, for provenance under Law 6, but it is
not what currency is judged on.

**Consequence, and it is deliberate:** the entire `fidelity/` tree is *outside*
the digest. Changing `campaign.py`, `compliance.py`, `publish.py`, a law, or this
sentence triggers no rescore. Rendering and orchestration cannot change a
measured number, so they are not allowed to invalidate one.

**Where this still bites:** the digest cannot know that a *new law* reads a field
old reports do not carry. Law 10's comparability key needs `numerics_digest` on
the report, and reports written before that field existed can never satisfy it, so
they are permanently uncurrent and must be rescored. That happened, and it cost
days. It is the price of the design rather than a bug in it, and the mitigation is
to add fields to the report generously and laws that depend on them sparingly.

### 3. Pinning the MoE kernel, replaced by choosing freely and verifying afterwards

**First:** name the expert backend explicitly so every run used a known kernel.

**What broke:** a pin can refuse to load a checkpoint. Naming a backend that a
particular quantization cannot use turns a scoreable candidate into a failed one,
and the failure is caused by the measurement apparatus rather than by the model.

**Now:** three separate things, and the ordering matters.

The oracle picks per layer from the checkpoint, the device, and the installed
FlashInfer. Then scoring reads certification off the *loaded* model and refuses
before any forward pass if the chosen kernel is not certified for exact repeat, so
an uncertified choice costs a model load rather than a scored pass. Then, on a
later run of the same binding, the built kernel is compared against the previous
report's and a difference is a hard error — because under one binding the oracle's
choice is a function, and a second call to a function owes the same answer.

**Why a comparison is safe where a pin was not:** a pin can refuse to load a
checkpoint; a comparison cannot. It adds no failure mode of its own, and it only
speaks when it has a prior report of the same binding to speak from.

**Still unsatisfying:** certification is keyed on the kernel's class *name*. A
build missing the fork's Marlin patch reports the same name and is certified
anyway. [`INSTALL.md`](INSTALL.md) §4 is the mitigation, and it is documentation
rather than enforcement.

### 4. Emulation as the W4A4 path, retracted

**First:** W4A4 NVFP4 scoring ran on the emulation branch, because the native path
was not understood yet.

**What broke:** emulation collapses every expert's activation scale to one scalar
per layer. On `unsloth/gemma-4-26B-A4B-it-NVFP4` that meant 86–99 distinct
`w2_input_global_scale` values replaced by one, in all thirty layers, a 150-fold
spread. Four exports with byte-identical QDQ diagnostics scored 0.6 nats apart,
and that spread was written up as living "entirely in the activation scheme and
kernel path."

**Now:** the native CUTLASS path, and the earlier reading is retracted in place
rather than deleted. [`QXQ.md`](QXQ.md) §7 keeps it under "History: the emulation
collapse we published through," because it is the exact shape of mistake the
document exists to prevent: a real, reproducible, bitwise-exact number that
measures the harness instead of the checkpoint.

**The lesson we generalized from it:** bitwise reproducibility is necessary and
nowhere near sufficient. A number can be perfectly repeatable and still be about
the wrong thing, which is why substitution disclosure (Law 17) reports what the
loader did to the checkpoint rather than only what the checkpoint contained.

### 5. Refusing a checkpoint with missing scales, replaced by pricing the damage

**First:** refuse to publish a candidate whose per-expert activation scales the
loader had to fill in.

**What broke:** refusal hides a real and measurable effect, and it silently
removes exactly the checkpoints a reader most needs warning about.

**Now:** publish, disclose the fill under Law 17, and say how far to discount it.
The discount is not a guess: two independently produced NVFP4 exports, from
different publishers, with different fill footprints, one quantizing its LM head
and the other not, landed 0.00026 nats apart. Two independent quantizations do not
agree to four decimal places on their own. Once half the layers have their
per-expert scales replaced by one layer maximum, the fill sets the number and the
checkpoint stops being visible in it. On the same suite, clean exports separate
normally. [`QXQ.md`](QXQ.md) §7, "How far to discount a filled result."

**Related decision:** the fill is a harness policy, not a claim about the
checkpoint ([`QXQ.md`](QXQ.md) §13), and the argument that it should be fixed
upstream instead is [`PR54444-Adendum.md`](PR54444-Adendum.md).

### 6. Two copies of the weight-collision rule, reduced to one

**First:** `publish.py` carried its own collision check, and `campaign.py` carried
`weight_collisions()`.

**What broke:** they disagreed, and the copy in `publish.py` was the version
`campaign.py`'s docstring records having already fixed. It treated an equal mean
KLD with unequal weight digests as a contradiction. But `weights_identity()`
hashes shard names, sizes, and file bytes *including headers*, so it guarantees
one direction only: equal digests prove equal bytes, and unequal digests prove
nothing at all. Two upstream repos that had repackaged one quantization into
differently-sized shards produced exactly that pattern, and `publish.py` refused
an entire five-candidate family over it.

**Now:** `publish.py` imports `weight_collisions` and gets its three-way answer.
`impossible` (same digest, different mean) is a hard stop. `duplicates` (same
digest, same mean) and `equivalent` (same mean, different digest) both publish,
with the relationship disclosed on the card.

**The general lesson:** a rule enforced in two places is a rule with two versions.
When a gate has to exist at two stages, the second stage calls the first stage's
implementation.

**A second lesson, about diagnosis:** the first response to that refusal was to
recommend rescoring both candidates, on the assumption that a digest difference
proved a tensor difference. It does not. The correct diagnosis cost no GPU time
at all — comparing shard sizes over the Hub API showed 168 bytes of difference
with 56 MB moved between shards, and a ranged read proved 2,418 identical tensor
names with zero differing bytes across a sample. Reach for the cheap evidence
before the expensive remedy.

### 7. Deleting the superseded report, replaced by moving it aside

**First:** a rescore deleted the old report before starting.

**What broke:** deletion is the right instinct — a rescore that dies halfway must
not leave a stale number where a fresh one belongs — but it also destroys the only
record of which kernel the previous run built, which is what decision 3's
verification compares against.

**Now:** the old report moves to `<work>/prior/<tag>.json`, outside `reports/` so
nothing that enumerates published reports finds it, and the scorer is pointed at
it with `--prior-report`.

**The escape hatch, deliberately left open:** deleting a report by hand skips the
comparison. That is the documented way to accept a kernel change on purpose
rather than argue with a check.

### 8. The laws version as a rescore trigger, removed

**First:** the natural reading — a new laws version means the old results do not
comply, so rescore.

**What broke:** it conflates two different costs. Rescoring re-runs the model.
Relabelling re-renders the artifact. A change to a law's wording, a chart, or a
card costs the second and has no business costing the first.

**Now:** `_score_report_is_current` in `campaign.py` does not consult
`LAWS_VERSION` at all. It checks the capture manifest hash, the KV dtype, routing
currency, the reference weights, and the two runtime fields. `assemble` runs a
completeness gate that rescores only what it independently finds uncurrent, so a
laws-or-rendering change costs an `assemble` — minutes — rather than an `all` —
days.

## Standing constraints

These are decisions rather than defaults, and they are not up for casual
revision.

**The KV cache is never quantized.** Quantizing it introduces a second error
source, uncontrolled and interacting, whose contribution would be silently
attributed to the weights. The KV dtype is part of what a report is bound to, so a
run that changed it would be detected rather than compared.

**The repository is frozen while a run is in progress.** The runtime digest is
read per report at score time, not once per campaign, so a commit or an edit
mid-run splits one campaign across two bindings. Law 11 is the formal version;
`bootstrap.sh` warns on a dirty tree for the same reason.

**Exact repeat is a gate, not a metric.** It is checked and it can refuse; it is
never reported as a quality score. A candidate that cannot reproduce itself does
not get a number at all.

**A candidate is dropped with disclosure or not at all.** `excluded_candidates`
takes a repo, a revision, and a *reason*; it writes `excluded-candidates.json`
into the library, and the exclusion surfaces on the card and in
`kld-vs-size.json`. Quietly removing a row from a config is not an available
option, and the mechanism refuses if a candidate is both active and excluded at
the same revision.

**A number states what it is comparable with.** Every leaderboard group carries a
comparability key (Law 10). Two numbers from different bindings are not compared,
by construction rather than by the reader's diligence.

## Known rough edges

Recorded so they are found on purpose rather than by surprise.

- Certification is by kernel class name, so a build without the fork's Marlin
  patch is certified while not being batch-invariant. See
  [`INSTALL.md`](INSTALL.md) §4.
- `AVAILABLE_BACKENDS` puts FlashInfer first and nothing in that path consults
  `VLLM_BATCH_INVARIANT`, so nothing *guarantees* a certified kernel — the refusal
  happens after the load instead. In practice FlashInfer's own
  `is_supported_config` has declined every time, for reasons logged only at debug
  level. [`QXQ.md`](QXQ.md) §5.
- The selftests are invoked inconsistently: a `--selftest` flag on most modules, a
  positional `selftest` on `suite.py` and `artifact.py`.
- `sweep.py` counts `__bxq_smoke__*` reports as orphans, though they are
  deliberate products of the smoke step.
