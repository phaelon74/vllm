# Local Inference Lab — Distribution Fidelity Laws

**Laws version:** 12
**Status:** draft, pending coordination with `local-inference-lab` on the
publication namespace and suite format.

Version 2 adds Law 14. A receipt at version 1 says nothing about component
attribution either way, so a version-1 result for a routed model is not a
version-2 result and should be rescored rather than relabeled.

Version 3 adds Law 15. A version-2 result reports one mean over a suite built
from a dozen kinds of text and never says which of them the divergence landed
on, so it cannot be rebadged as version 3; the per-context records Law 15 reads
come from the scoring run itself.

Version 4 changes what Law 14 means. Versions 2 and 3 took the router weight cell
as the routing term and its ranking floor. That was wrong: routing changes because
the activations reaching an identical router changed, so a checkpoint shipping a
BF16 router reported a floor of zero while its routing term was not zero. Version
4 requires the routing term to be measured from the run's own expert selections
and makes that measurement the floor. A version-2 or version-3 receipt for a
routed model overstates how comparable two candidates are and must be rescored,
not relabeled.

Version 5 tightens Law 12. Versions 1 through 4 checked that a reference was
published, not that it was the reference the number came from, and an artifact
shipped with reference tensors bound to different tokens and a different row count
than the candidate beside them. Every other law passed, because they read the
candidate's manifest and never the one published with the tensors. Unlike the
earlier version changes this one is a relabeling, not a rescore: the measurement
is unaffected, so a version-4 artifact becomes a version-5 artifact by
reassembling it and passing the new check.

Version 6 changes what Law 14 cells match. Versions 2 through 5 selected
weights by component name pattern and rounded every match, so a mixed-precision
checkpoint that quantized only some attention or expert weights produced a
composite cell heavier than the deployment it claimed to decompose. Version 6
matches per tensor. Because a reference may fuse what a pack names separately —
one `experts.gate_up_proj` stack against a quantized pack's per-expert names —
each quantized name is resolved to the reference tensor that carries it, and a
fused tensor the pack covers only in part is refused rather than rounded whole.
A version-5 receipt for a partly quantized candidate is not a version-6
receipt and must be rescored, not relabeled.

Version 7 separates a saturated routing measurement from an absent one. Versions
2 through 6 failed Law 14 whenever the routing excess was null and advised a
rescore with `--measure-routing`, which was the right advice for a run that
never measured routing and useless for one where every scored position rerouted
or none did. Both of those are measurement outcomes a rescore reproduces
exactly. Version 7 states which case holds and publishes a saturated candidate
with its deployed mean marked unranked. This is a relabeling, not a rescore: a
version-6 receipt becomes a version-7 receipt by reassembling it.

Version 8 adds Law 16. Versions 1 through 7 bound a report to its tokens, its
capture, and its reference, and never to the candidate weights themselves: the
report named a directory, and nothing said what that directory held when it was
read. Two candidates in one campaign then published the same mean from different
checkpoints, which is only possible if one was scored against the other's
weights, and every law passed. Version 8 requires the digest of the scored
tensors on the report and refuses a family in which distinct weights produced an
identical mean. A version-7 report whose weights were released cannot be bound
after the fact; it publishes under a recorded Law 13 deviation saying so, or it
is rescored.

Version 9 states what agreement between tensor-parallel workers means, under
Law 8. Versions 1 through 8 required the workers to return identical floats for
the trunk KLD, which no correct multi-worker run can do: each rank multiplies its
own slice of the vocabulary on its own device, and a sum over the vocabulary
amplifies the last-bit difference in the gathered logits. Every tensor-parallel
candidate was therefore refused for a fault it did not have, and the refusal
reported no numbers, so the cause stayed hidden. Version 9 requires agreement to
a stated bound, publishes the observed divergence beside the figure so no reader
takes more precision from the digits than was measured, and refuses a real
disagreement with the field, the position, and both values named. A version-8
receipt from a single-worker run becomes a version-9 receipt by reassembling it;
a multi-worker candidate that version 8 refused must be scored.

Version 10 replaces Law 14's synthetic component-cell interpretation with a
paired routed-model intervention. QxQ is the unchanged quantized candidate under
its natural routing. BxQ is the same candidate with the BF16 teacher's expert IDs
forced while the student computes its own gating weights for those experts.
Earlier `expert_cell`, `router_cell`, and `composite_cell` results remain useful
synthetic QDQ diagnostics, but none measured BxQ or QxQ. Routed candidates must
therefore be rescored for BxQ; an existing deployed report may supply QxQ only
after the paired-run controls and bindings are established.

Version 11 replaces Law 14's fixed bitwise-style natural-control threshold with
a measured repeatability envelope. The paired protocol runs two natural and two
forced-natural samples, retains the fixed numerical floor for repeatable
kernels, and otherwise permits no more than twice the larger observed
within-path span. Natural expert IDs may differ across the two samples when
arithmetic drift propagates into later routers, so their selection flip rate is
recorded. Both forced-natural controls still replay the first QxQ sample's exact
IDs.
The paired routed-score and BxQ protocol versions are now 3. Version-10 paired
reports must be rescored rather than relabeled.

Version 12 restores Law 14's publication requirement to exact repeated QxQ and
BxQ. Version 11's repeatability envelope remains diagnostic evidence of kernel
drift, but it does not authorize a drifting canonical score. Publication
requires two natural QxQ samples, two forced-natural controls, and two BxQ
samples to agree exactly in routes and per-position KLD. Uncertified or
nondeterministic backends fail. Version-11 paired reports must be rescored
rather than relabeled. The paired routed-score and BxQ protocol versions are
now 4.

These laws govern every distribution-fidelity measurement this program
publishes. They are not guidance. The pipeline refuses to produce or upload an
artifact that violates one, and the only way past a refusal is a recorded,
named human approval that is then printed beside every number the campaign
produced (Law 13).

Each law states what is required, why, the automated check that enforces it,
and whether an override is permitted at all.

## What this program measures

Teacher-forced next-token distribution fidelity: how closely a candidate
checkpoint reproduces a reference checkpoint's output distribution over a frozen
set of token IDs, measured as `KL(reference || candidate)` over the tokenizer's
real vocabulary.

## What this program does not measure

Free-running generation quality, benchmark accuracy, instruction following,
long-context behavior beyond the suite's context length, multimodal behavior,
tool use, or throughput. A fidelity number is evidence about one property. It is
not a quality verdict, and Law 10 forbids presenting it as one.

## Versioning

The laws are versioned as a whole. Every artifact records the laws version it
was produced under. Changing a law's meaning increments the version; artifacts
are never retroactively judged against a version they predate.

---

## Law 1 — Zero baseline

**Required.** Every campaign begins by scoring the reference checkpoint against
a capture of itself. The mean KLD must be exactly `0.0`, over every scored
position, on the same runner and tensor-parallel size the campaign will use for
its candidates.

**Why.** Zero is the only self-evidently correct answer, so it is the only check
that distinguishes a working measurement from a plausible-looking one. A
pipeline that cannot reproduce a distribution it just produced cannot be trusted
to attribute a difference to quantization.

**Check.** `mean_kld == 0.0` exactly — not rounded, not below a threshold — in
the self-comparison report, with `num_positions` equal to
`rows * (context_length - 1 - score_from)`.

**Override.** Permitted only under Law 13, and only with all three of:

1. a named approval recorded before any candidate is scored;
2. a repeat-capture study, in which the reference is captured three times over
   the identical tokens and all three pairwise comparisons are scored, which
   establishes the magnitude of the nondeterminism the campaign is carrying;
3. headline disclosure. The non-zero baseline and the widest pairwise repeat
   spread appear in the artifact's opening section, not a footnote, and every
   candidate mean in the campaign is annotated with the floor it sits above.

A candidate difference no larger than the repeat spread is not a result. Under an
override, ranking two candidates requires repeated candidate capture as well.

## Law 2 — Determinism

**Required.** Eager execution enforced. No autotuned kernel selection, no
inference-time JIT kernel selection, no CUDA graphs, prefix caching disabled,
and a fixed `max_num_seqs`. Tensor-parallel size is recorded, and reference and
candidate are scored under identical settings.

**Why.** Timing-based autotuners and JIT kernel choice make the arithmetic a
function of machine load, which silently converts run-to-run noise into apparent
quantization error. Programs that skip this are forced to spend statistical
machinery resolving differences smaller than their own noise floor. Determinism
is what makes Law 1's exact zero attainable.

**Check.** `enforce_eager` true in the capture manifest and the live config;
prefix caching disabled; compiled runs are labeled `authoritative: false` and
are rejected from any published comparison.

**Override.** Not permitted for published numbers. Compiled runs may be recorded
as exploratory and must be labeled as such.

## Law 3 — Frozen input

**Required.** Comparisons consume stored token IDs from a published suite.
Never text retokenized at run time.

**Why.** Tokenizer versions, normalization, and concatenation policy all change
the token stream, and a changed token stream changes the number while looking
identical in the command line. Stored IDs make the input an auditable object.

**Check.** The SHA-256 of the scored token IDs matches the suite manifest and
the capture manifest. Mismatch aborts before any model loads.

**Override.** Not permitted. A different token set is a different artifact.

## Law 4 — Real vocabulary

**Required.** KLD is computed over the tokenizer's actual vocabulary. Output
rows that exist only as alignment padding are never scored. The scored width is
published as `kld_vocab_size`.

**Why.** Padding rows carry no token ID and can never be emitted. Scoring them
measures how two checkpoints treat dead weights, and because padding size is a
per-checkpoint alignment choice, including it makes checkpoints incomparable for
reasons invisible in the output.

**Check.** `kld_vocab_size` equals the tokenizer's unpadded vocabulary size and
is strictly less than or equal to the checkpoint's declared `vocab_size`.

**Override.** Not permitted.

## Law 5 — Manifest binding

**Required.** A reference capture binds itself to the tokenizer identity, token
hash, context length, row count, `score_from`, scored vocabulary size, tensor
parallel size, eager mode, and runtime identity. Scoring against a capture whose
manifest disagrees with the live configuration aborts.

**Why.** Reusing a capture across configurations is the easiest way to publish a
number that compares two different things. The binding must fail closed, because
a coerced comparison is indistinguishable from a valid one in the output.

**Check.** Every bound field matches, and a capture directory lacking a manifest
is refused rather than inferred.

**Override.** Not permitted. Changing a bound field mints a new capture.

## Law 6 — Provenance

**Required.** No number publishes without its environment report: host and
kernel, GPU inventory with driver, VBIOS, ECC mode, persistence mode and clock
limits, CUDA toolchain, torch and vLLM versions with the exact commit, the
resolved interpreter, the full installed package set, and content hashes for
every checkpoint involved.

**Why.** A fidelity number is a property of a checkpoint measured through a
stack. Kernel selection, driver version, and even ECC state can move a bitwise
result, so a number without its stack cannot be reproduced or defended.

**Never a credential.** Provenance records that a variable was set, never a
credential's value. The watched prefixes put `HF_TOKEN` in the same namespace as
`HF_HOME`, so a variable with a whole name-word of `TOKEN`, `KEY`, `SECRET`,
`PASSWORD`, `AUTH`, `COOKIE`, or `SESSION` is recorded with its value replaced.
Redaction applies at capture, again at rendering, and again as a refusal to upload,
because
a published credential cannot be unpublished.

**Not reused across a moving tree.** The environment report is recaptured whenever
the repository's HEAD differs from the commit it recorded, and republished into
every model root on each assembly. A capture carried forward describes a stack that
did not produce the numbers beside it, which is the failure this law exists to
prevent and the one hardest to notice, because every field is populated and every
check passes.

**Check.** The artifact contains an environment report whose GPU, driver, torch,
vLLM, and commit fields are populated, and whose recorded checkpoint hashes match
those referenced by the reports. Publication additionally scans every text file in
the artifact, refusing on a recognizable credential shape or on a field whose name
is exactly a credential's name and still carries a value, and that refusal cannot
be overridden or skipped.

**Override.** Not permitted.

## Law 7 — Storage integrity

**Required.** Reference distributions may be stored as hidden states plus a
language-model head only when a replay probe proves, on that same capture, that
replaying the stored hidden states through the stored head reproduces the live
logits bitwise. Otherwise full logits are stored.

**Why.** Hidden-state storage is what makes a reusable multi-thousand-context
reference affordable, but it is only valid if replay is exact. An approximate
replay silently adds a floor to every candidate comparison built on it.

**Check.** The capture manifest's replay probe records bitwise identity. Hidden
storage without an exact probe is refused at capture and again at scoring.

**Override.** Not permitted. Fall back to logits storage instead.

## Law 8 — Head transparency

**Required.** Every comparison reports both trunk KLD, computed by pushing the
candidate's hidden states through the reference's language-model head, and
deployed KLD, computed through the candidate's own head.

**Why.** Head quantization and trunk error are different defects with different
remedies. Reporting only the deployed number conflates them, and reporting only
the trunk number hides a quantized head entirely.

**Check.** Both values present, along with the non-additive delta between them
and the detected head state for reference and candidate.

**Override.** Permitted under Law 13 when the candidate's head is proven
bit-identical to the reference's, in which case the delta is necessarily zero and
the second scoring pass buys no information.

### Agreement, not identity, across tensor-parallel workers

The trunk figure is computed on every tensor-parallel worker from the same three
files, so the workers cross-check each other: if a rank installed the reference
head wrongly or holds the wrong shard, its answer diverges. But the workers are
not required to produce identical floats, and demanding that they do is wrong.
Each rank multiplies its own slice of the vocabulary on its own device, and the
gathered logits differ in their last bits; a sum over a quarter-million-token
vocabulary amplifies that difference into the KLD.

So the contract is agreement to a stated bound — currently `1e-5` absolute per
position and `1e-7` relative on the mean. Within it, the run proceeds, the trunk
figure is rank 0's, and the one-pager states how far the ranks actually diverged,
so nobody reads more precision into the digits than was measured. Beyond it, the
ranks disagree about the model rather than about rounding, and scoring refuses
naming the field, the position, and both values. A top-1 flip between ranks is
disclosed and counted rather than refused: an argmax over two near-tied logits
can flip on last-bit noise without either rank being wrong.

## Law 9 — Tail and depth disclosure

**Required.** A published mean is always accompanied by median, p90, p99, and
maximum, by the per-position-depth profile, and by top-1 agreement.

**Why.** The mean and the tail can rank candidates in opposite directions. A
checkpoint can be more faithful at the typical position and far worse at rare
ones, and a checkpoint whose error accumulates with context depth looks fine at
short context. Publishing the mean alone hides both failure modes.

**Check.** All of `median_kld`, `p90_kld`, `p99_kld`, `max_kld`, depth buckets,
and `top1_agreement` present in the report and rendered in the one-pager.

**Override.** Not permitted.

## Law 10 — Comparability

**Required.** Numbers compare only against numbers produced from the identical
suite, geometry, and runtime identity. No threshold imported from another model,
corpus, tokenizer, vocabulary, or serving stack carries a verdict. No fidelity
number is presented as a capability, accuracy, or general quality claim.

**Why.** KLD has no absolute scale. Its value depends on the corpus, the context
depth distribution, the vocabulary width, and the runtime. Cross-harness
comparison is the single easiest way to publish a confident falsehood.

**Check.** Each result records its suite ID, geometry, laws version, and runtime
manifest hash. The leaderboard groups strictly by that tuple and refuses to place
rows from differing tuples in one ranking. The suite identity is read from what the
scoring run recorded, never from a suite manifest supplied to the audit, or a run
that tokenized at run time reports a complete key by borrowing the identity of a
suite it never opened.

**Override.** Not permitted.

## Law 11 — Freeze before qualification

**Required.** The suite is partitioned into analysis and qualification contexts.
Quantization parameters, layer assignments, and acceptance thresholds are frozen
on analysis results and recorded before any qualification result is read.

**Why.** Tuning against the set you report on converts a measurement into a
fitting exercise. The freeze receipt is what makes a qualification number an
out-of-sample claim.

**Check.** A freeze receipt exists, is timestamped before the qualification run,
and hashes the frozen parameters. Qualification scoring refuses to run without
it.

**Override.** Permitted under Law 13 for exploratory work, which may never be
published as a qualification result.

## Law 12 — Reusable reference

**Required.** A published artifact includes everything a third party needs to
score a new candidate without loading the reference checkpoint: the token suite,
the reference distributions, the language-model head, every manifest, and the
comparator configuration. File hashes are published for all of it.

The published reference must be bound to the same identity the report was scored
against, on every field Law 5 binds. Shipping the files is not enough; they have
to be the files the number came from.

**Why.** A fidelity claim that only its author can reproduce is an assertion, not
evidence. Publishing the reference also removes the largest cost from anyone
else's comparison, which is what makes independent replication realistic.

The identity clause exists because presence and reusability are different
properties, and an artifact once shipped with reference tensors describing
different tokens and a different row count than the candidate beside them. Every
other law passed, because they all read the candidate's manifest rather than the
one published with the tensors. A reader reusing that reference gets an abort at
best and needs the reference checkpoint after all, which is the one thing this law
exists to prevent.

**Check.** The artifact contains the suite, reference tensors, head, manifests,
and a `checksums.txt` covering every file, and the checksums verify. The manifest
published under `reference/` agrees with the scored capture manifest on every
bound field: token hash, tokenizer, context length, rows, start offset, comparator
vocabulary size, tensor-parallel degree, eager mode, and runtime.

**Override.** Permitted under Law 13 where redistribution of a derivative is
restricted by the reference checkpoint's license, in which case the artifact
records the restriction and publishes hashes so a locally regenerated reference
can be verified against ours.

## Law 13 — Recorded deviation

**Required.** Any override permitted above requires a named human approver, a
written justification, and a timestamp, recorded in the artifact's compliance
receipt. The deviation is then displayed on the one-pager next to the affected
numbers, not in a footnote.

**Why.** Fail-closed enforcement is only meaningful if the escape hatch is
auditable. An override that is invisible in the published artifact is
indistinguishable from a bug.

**Check.** Every compliance receipt entry with status `override` carries
`approver`, `justification`, and `timestamp`, and the one-pager renders them.

**Override.** None. This law has no exceptions.

## Law 14 — Routed-model QxQ/BxQ intervention

**Required.** A reference that routes tokens to experts publishes a paired
intervention over the identical frozen tokens:

1. **QxQ** (`qxq_cell`) is the unchanged quantized candidate running normally,
   with its natural expert IDs and its own gating weights.
2. **BxQ** (`bxq_cell`) is that same unchanged quantized candidate with the BF16
   teacher's ordered logical expert IDs forced at every routed layer. The student
   computes the gating weights for those forced experts from its own router
   logits, using its native scoring, bias, normalization, and scaling rules.

The first axis is the source of the expert IDs: `B` means BF16 teacher IDs and
`Q` means the quantized student's natural IDs. The second `Q` is the candidate,
which is identical in both runs. It does not mean "quantized experts" in a
synthetic checkpoint, and the first axis does not mean router-weight precision.

**Binding.** Both cells carry `mean_kld`, their supporting report path, partition,
token SHA-256, reference-config SHA-256, and `candidate_weights_sha256`. Every
binding equals the deployed report, and QxQ's mean equals the deployed natural
mean. BxQ additionally carries a 64-hex `routing_trace_sha256`,
`routing_mode: teacher_ids_student_weights`, and the supported
`protocol_version`. The trace binds the forced IDs to the same teacher, tokens,
layer order, and routing geometry used by both scores.

**Backend control.** Exact ID replay is a capability, not an assumption.
`backend_evidence` names the active backend or kernel, and `replay_supported` is
true only when that path injects logical teacher IDs before placement mapping and
dispatch. If replay uses a different kernel path, the candidate is first run with
that path and no override. The control protocol runs two deployed-natural, two
forced-natural, and two BxQ samples. Publication requires exact agreement:
identical natural expert IDs across the two QxQ samples, identical per-position
KLD across QxQ repeats, forced-natural controls, and BxQ repeats, all within
the fixed numerical floor. Observed spans and route-flip rates are stored as a
diagnostic addendum and never authorize a drifting canonical score. There is no
backend-name allowlist or blanket tolerance. `natural_control_parity.passed` is
true only when those exact-repeat conditions hold. An uncertified or
nondeterministic backend fails. The candidate weight digest must remain unchanged
and `candidate_weights_unchanged` must be true.
Forced-natural controls and BxQ remain bound to the first QxQ sample's exact
ordered IDs. QxQ always uses the model's own router.

**Natural routing divergence.** The QxQ run also measures the student's natural
expert IDs against the teacher trace. The artifact reports selection flip rate,
position flip rate, conditional KLD where routing held and flipped, per-layer
rates, and the existing routing-excess state. Conditioning after the run is not
BxQ: it observes the subset where routes happened to agree, while BxQ forces the
teacher IDs over the full suite.

**Paired delta.** `routing_intervention_delta = QxQ mean KLD - BxQ mean KLD`.
It may be positive or negative. It is a paired intervention result, not an
additive decomposition and not a claim that routing alone contributed that much.

**Synthetic QDQ diagnostics remain synthetic.** `expert_cell`, `router_cell`,
`composite_cell`, and ladder rungs round selected BF16 weights, run on BF16
kernels, and route naturally. They may still diagnose weight rounding and report
their own route flips, but they are never labeled BxQ or QxQ. In particular,
`expert_cell` is not BxQ because it changes the checkpoint and does not force
teacher IDs.

**Check.** For a manifest declaring experts, compliance requires both paired
cells and all bindings above; exact QxQ/deployed equivalence; a complete trace
digest; protocol, certified backend, and passing exact-repeat control evidence;
the exact paired delta; measured natural routing divergence; and QxQ plus BxQ
domain records. A manifest declaring no experts is
`not_applicable`. Missing replay support, stale or unbound traces, failed exact
repeat, an uncertified backend, and missing BxQ are failures. They are never
converted into an override or a synthetic substitute.

**Override.** Not permitted. Dense models satisfy this law as
`not_applicable`; a routed model without supported replay and exact repeated
QxQ/BxQ does not.

## Law 15 — Domain disclosure

**Required.** When the suite is stratified, a result publishes its mean per
stratum alongside the overall mean. Each stratum's row carries the number of
contexts behind it, the position-weighted mean, the spread of per-context means
within it, and the reference's own mean top-1 probability on it. Every stratum
the run actually scored appears; a stratum is never dropped for being small or
inconvenient.

**Why.** A model does not lose fidelity uniformly across kinds of text. The suite
is built from a dozen strata — encyclopedic reference, worked mathematics, source
code, dialogue, Chinese, structured data and tool calls — precisely because a
single corpus cannot represent a deployment. Averaging them back into one number
throws away the only part of the measurement that answers the question a reader
actually has, which is not "how far apart are these two models" but "how far
apart are they on the work I am going to give them". Two candidates with the same
mean are not interchangeable if one loses its fidelity on prose and the other on
tool calls.

**Confounding is disclosed, not resolved.** A stratum's KLD depends on how
predictable its text is to begin with. The reference's top-1 probability is
published beside every stratum mean so a reader can see the difference between a
candidate that diverges most where the reference was already uncertain, which is
weak evidence, and one that diverges most where the reference was confident,
which is strong. The program does not normalize the means to hide this; it
reports both numbers and says which case each stratum is.

**Spread, because a stratum is tens of contexts.** A stratum holds tens of
documents, not thousands, so one pathological document can carry it. The median
and p90 of the per-context means, and the worst context's identity, are published
with the mean. A stratum whose mean is four or more times its median is reported
as driven by outlier contexts rather than as a weak domain.

**Ladder.** Where Law 14 applies, the per-stratum table is also reported for the
cells, so the artifact says whether a domain's weakness follows the model or the
format. It is a different finding if NVFP4 is worst on code while FP8 is worst on
mathematics than if both are worst on the same domain.

**Check.** The compliance receipt requires per-context records in the report, one
per scored row, each naming its suite context. Every scored context must be in
the suite manifest, the published domain table must cover exactly the strata that
were scored, and its per-stratum context counts must sum to the number of rows.
A suite that declares fewer than two strata is `not_applicable`.

**Override.** Permitted under Law 13 for an unstratified suite the manifest fails
to describe, and for a zero-baseline or bounded-prefix run that is not published
as a candidate measurement. Not permitted for a published candidate on a
stratified suite.

## Law 16 — Candidate weight binding

**Required.** A published result carries a digest of the weights it scored, taken
from the checkpoint at the moment it was read, and that digest must match the
inspection of the candidate the artifact publishes. Within one family, two
candidates whose digests differ must not report the same mean.

**Why.** Every other binding in these laws points at the inputs a candidate was
measured against — the tokens under Law 3, the capture under Law 5, the reference
under Law 12 — and none of them points at the candidate. A report named a path,
and a path is not evidence. A directory that was repopulated, or a scorer handed
the wrong one of two similarly named checkpoints, produces a fully compliant
artifact attributing one vendor's numbers to another's work. That is the worst
error this program can make, because it is invisible: the number is real, the
receipt is honest, and the name on it is wrong.

**Content digest, not a file list.** The bond hashes each shard name and every
byte of every safetensors shard, including its header. A checkpoint with the same
tensor names, dtypes, shapes, and file sizes but different values therefore
cannot preserve the digest. Two directories that share it hold byte-identical
weight shards, whatever their repos are called.

**The same mean from different weights is refused, not explained.** Distinct
quantizations of one reference do not land on an identical mean to eighteen
digits; a suite of this size does not produce that coincidence. When it appears,
one report was scored against the other's checkpoint, and nothing in the
artifacts says which. Both results are withdrawn and rescored. The converse —
two candidates sharing a digest — is not an error in the measurement but a fact
about the upstream repositories: one is a verbatim re-upload of the other. It
publishes once only when both reports agree; conflicting scores from identical
weights are refused. The artifact names an agreeing re-upload as such rather
than presenting it as an independent quantization.

**Check.** The compliance receipt requires `student_weights_sha256` on the report
and an inspection of the published candidate carrying the same digest.
Assembly refuses a family containing an identical mean across differing digests,
and reports a shared digest as a duplicate.

**Override.** Permitted under Law 13 only for a result scored before this law
existed whose weights have since been released, where no digest can be recovered
without a rescore. The deviation is printed beside the number, which then states
that its weights are unbound. Not permitted for a new measurement, and never
permitted for a refused identical mean.

## Numbering

Laws are append-only. A published receipt cites its laws by number, so renumbering
would silently change what an existing artifact claims to have satisfied. A law
that is superseded is marked as such and keeps its number.
