# Local Inference Lab — Distribution Fidelity Laws

**Laws version:** 1
**Status:** draft, pending coordination with `local-inference-lab` on the
publication namespace and suite format.

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
credential's value. Watched variable names are matched broadly, which puts
`HF_TOKEN` in the same namespace as `HF_HOME`, so any name that looks like a
token, key, password, or session is recorded with its value replaced. Redaction
applies at capture, again at rendering, and again as a refusal to upload, because
a published credential cannot be unpublished.

**Check.** The artifact contains an environment report whose GPU, driver, torch,
vLLM, and commit fields are populated, and whose recorded checkpoint hashes match
those referenced by the reports. Publication additionally scans every text file in
the artifact for credential shapes and refuses on any hit; that refusal cannot be
overridden or skipped.

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
rows from differing tuples in one ranking.

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

**Why.** A fidelity claim that only its author can reproduce is an assertion, not
evidence. Publishing the reference also removes the largest cost from anyone
else's comparison, which is what makes independent replication realistic.

**Check.** The artifact contains the suite, reference tensors, head, manifests,
and a `checksums.txt` covering every file, and the checksums verify.

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
