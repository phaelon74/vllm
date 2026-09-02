# Local Inference Lab — Distribution Fidelity

Tooling that turns a KLD scoring run into a publishable, law-compliant,
independently reproducible artifact.

Read [`LAWS.md`](LAWS.md) first. The laws are enforced, not advised: the
compliance checker exits non-zero on any violation, so a pipeline wired to it
cannot publish a non-compliant result.

## Components

| File | Role |
|---|---|
| `LAWS.md` | The immutable laws, their automated checks, and the override procedure |
| `suite.py` | Mints the frozen token suite; verifies one; self-tests offline |
| `suites/recipe-v1.json` | The base recipe: sources, revisions, strata, benchmarks |
| `bootstrap.sh` | Creates the venv, installs this branch, fetches checkpoints |
| `campaign.py` | Orchestrates a campaign: download, score, assemble, release, render |
| `compliance.py` | Fail-closed law evaluation; emits the compliance receipt |
| `artifact.py` | Renders the one-pager, the leaderboard, and `checksums.txt` |
| `publish.py` | Uploads to the Hub; refuses anything non-compliant or tampered |
| `redaction.py` | The one secret policy shared by capture, rendering, and publish |
| `tails.py` | Whether a mean describes the distribution or a few hundred positions |
| `qdq.py` | Builds single-component quantize-dequantize variants of a BF16 checkpoint, and inspects a quantized one for its scheme |
| `curate.py` | Interactive or flag-driven Hub curator: picks candidates, pins revisions, emits a campaign JSON |
| `provenance.py` | Refuses a candidate whose architecture is not the reference's |
| `strata.py` | Attributes a mean to the kinds of text it was measured on (Law 15) |
| `sweep.py` | Reports and reclaims scratch the published library does not use |
| `campaigns/*.json` | Campaign definitions: suite, geometry, models, candidates |

Scoring itself lives in `examples/offline_inference/score_mode_kld.py`, driven by
`scripts/kld_run_matrix.sh`. Environment capture lives in
`scripts/kld_env_report.sh`. The measurement runbook is
[`docs/design/kld_manual_verification.md`](../docs/design/kld_manual_verification.md).

## Building the token suite

The suite is the evaluation input, not a description of it (Law 3). Candidates
consume the stored token IDs directly; retokenizing source text does not
reproduce the evaluation.

```bash
python fidelity/suite.py selftest        # exercise the builder offline first
python fidelity/suite.py probe           # then prove every source is reachable

python fidelity/suite.py build \
  --recipe fidelity/suites/recipe-v1.json \
  --tokenizer /media/fmodels/Qwen/Qwen3.6-27B \
  --out /mnt/kld/suites/qwen3.6-1024x2048-v1

python fidelity/suite.py verify --suite /mnt/kld/suites/qwen3.6-1024x2048-v1
```

`recipe-v1.json` replicates the Kimi K3 distribution-fidelity recipe: the same
sources at the same pinned dataset revisions, the same ten allocation strata and
counts totalling 1,024 contexts, the same exact and near-duplicate dedup, the same
benchmark-overlap scan, and the same 768/256 analysis and qualification split.
Where it cannot follow the reference exactly, `upstream.deviations` in the recipe
records what changed and why, and the reason is carried into the published source
registry rather than left implicit.

`probe` streams a bounded prefix of every source and benchmark, reporting the
columns each one actually has and how many records survive its filters. Harvesting
costs minutes per source, so a renamed column, a revoked revision, or a gated
repository needs to surface in the first minute rather than after the expensive
work. Sources that only exposed a loader script, which current `datasets` refuses
to execute, name their data files directly at the same pinned revision, and a
source may list alternative column names when one lineage renames a field.

Two sources are gated and need their terms accepted once, on the Hub, under the
same account as `HF_TOKEN`: `bigcode/starcoderdata` and `Idavidrein/gpqa`. `probe`
reports either as unreachable until then.

Harvested pools are cached under `<out>-cache`, keyed by everything that can
change a pool: the source definition, the tokenizer identity, the context length,
and the dedup parameters. A build that fails late therefore re-harvests only what
changed, and `--refresh <key>` forces one source to be re-read. `--no-cache`
disables it. The cache sits outside the suite directory and is never published.

`--oversample` sets how many candidates are harvested per allocated slot, giving
dedup and leakage rejections room to reject without leaving a slot empty. A source
that filters for a rare file type reads many records per retained candidate, so it
may declare its own `oversample` in the recipe rather than pay the global factor.

**Token IDs are not portable across tokenizers.** Their suite is Kimi-tokenized
and unusable for Qwen, and ours is equally unusable for a model with a different
vocabulary. The recipe is what transfers, so mint one suite per tokenizer family;
`suite.py` refuses to score a suite against a tokenizer whose vocabulary size
differs from the one it was minted for.

What the builder guarantees, all checked by `selftest`:

- One context per coherent source unit, so a long article cannot dominate a
  stratum. Sources that share an identifier space, such as the three that index
  GitHub repositories, are deduplicated against each other too, so one repository
  cannot enter the suite twice through two sources.
- Exact token-duplicate and MinHash near-duplicate rejection before allocation.
- A benchmark-overlap scan against HumanEval, MMLU, and GPQA-Diamond at pinned
  revisions. For MMLU a match needs the question *and* every answer choice, so a
  short general question occurring naturally in reference prose is not mistaken
  for contamination. Documents with a complete overlap are blocked from the suite.
  The scan is indexed by the interior words of each item fragment, a necessary
  condition for containment that `selftest` checks against the direct comparison
  it replaced; the direct form is quadratic and does not finish against MMLU.
- Deterministic everything: token offsets, allocation order, partition
  assignment, and therefore every hash. Rebuilding from identical inputs
  reproduces the suite bit for bit, which `selftest` asserts by building twice.

Sentinel contexts are assigned deterministically but not captured routinely.
Law 1 sets the noise floor at exactly zero, so repeat captures are only needed if
a Law 1 override is ever approved; the sentinel set is agreed in advance so the
choice cannot be made after seeing results.

Score a partition with `--token-suite <dir> --suite-partition analysis`. The
suite supplies rows, context length, and stride, and every context's hash is
re-derived from the file it was read from before a model loads.

## Running a campaign

```bash
bash fidelity/bootstrap.sh fidelity/campaigns/qwen3.6.json
python fidelity/campaign.py all --config fidelity/campaigns/qwen3.6.json
```

`bootstrap.sh` is idempotent: it creates the venv only if absent, installs vLLM
only if it cannot already be imported, and downloads only checkpoints that are
missing and have an `hf_repo`. A checkpoint that is missing with no repo to fetch
it from is a hard error, listed by path.

`campaign.py` runs the stages in the order the laws require, and each stage skips
work whose output already exists, so an interrupted campaign resumes. Stages can
also be run individually as `download`, `score`, `assemble`, or `release`.

A candidate that fails provenance, download, or scoring is recorded and skipped.
The rest of the sweep continues, assembly publishes whoever produced a report,
and the process exits non-zero so a partial campaign is not mistaken for a clean
one. Law 1 is the exception: a non-zero self-KLD still stops the model.

### Multi-candidate campaigns

`curate.py` writes the campaign JSON. With no `--base` it asks, in order: what
the base model is; whether you have one candidate, several, or want the pipeline
to pull X from the Hub to review; where the weights should live (default
`/media/fmodels2`, laid out as `author/modelname`); whether a local copy of the
reference can be reused; and whether to delete each candidate after scoring.
Flags pre-answer those prompts. `--picks` pins an explicit list; omitting it
searches the Hub and fills `--slots` (default `fp8=1,nvfp4=5,int4=5`) after
dropping GGUF, MLX, bnb, and derivative-weight name markers. Search also
recognizes `int8` / `w8a8` / `w8a16`. Every repo is pinned to a commit sha.

```bash
python fidelity/curate.py
python fidelity/curate.py --base Qwen/Qwen3.8-27B \
  --picks fidelity/campaigns/picks/qwen3.8-27b.json \
  --out fidelity/campaigns/qwen3.8-27b.json \
  --suite-dir /path/to/suite
```

A generated config stores weights under `--models-root` (default
`/media/fmodels2`) as `org/name`, and sets `"fetch": "lease"` unless you pass
`--fetch upfront`. Lease means `score` downloads the BF16 reference once, then
each candidate in turn: fetch, score, delete those weights. Peak disk is the
reference plus one candidate. Only a directory this campaign fetched is
eligible: `score` writes `work/leases/<name>.json` on a successful download, and
will not touch a checkpoint that was already on disk or the reference itself. A
candidate that fails keeps its weights so you can inspect the failure; drop them
later with:

```bash
python fidelity/campaign.py release --config fidelity/campaigns/qwen3.8-27b.json
```

Published artifacts stay reproducible after the weights are gone: each candidate
pins `hf_repo` and `revision`, and `inspect.json` is stored under
`work/inspect/` (copied into the assembled tree) rather than only beside the
checkpoint.

Scoring also binds each report to the weights it read, hashing tensor names,
dtypes, and shapes plus every shard's name and size from the safetensors headers.
Law 16 requires that digest to match the published `inspect.json`, so a directory
repopulated from a different repo cannot be scored under the old name, and two
candidates that differ in weights cannot publish the same mean. A report whose
digest disagrees with the checkpoint now on disk is refused rather than rewritten.

Before a candidate is scored, `provenance.py` compares `architectures`,
`hidden_size`, `num_hidden_layers`, `layer_types`, `vocab_size`,
`intermediate_size`, and `head_dim` against the reference (unwrapping
`text_config` when those fields are nested). A mismatch is a failed candidate,
not a KLD number. The comparison is published as `provenance.json`. Scoring
writes `inspect.json` so the format matrix is known before GPU time is spent.
A checkpoint whose `config.json` will crash vLLM on load (for example a Quark
W4A16 export whose `algo_config` is a list of dicts, which
`WeightsMapper.apply_list` cannot map) is refused from `config.json` alone.
That is a skipped candidate, not an EngineCore abort, and installing
`amd-quark` does not change it.

Every model in a campaign is ingested, scored, and assembled by the identical
code path. That is the point: a comparison between two candidates means nothing
if anything about the procedure differed between them.

### The Law 1 gate

Before any candidate of a model is scored, the campaign scores that model's
reference against itself. If the result is not exactly `0.0`, the campaign stops
and prints the baseline report path. It does not fall back, warn and continue, or
round. Proceeding requires the approval and the repeat-capture study described
under [Overriding Law 1](#overriding-law-1-specifically).

## Mandatory pipeline order

The steps are order-dependent, and getting it wrong produces spurious Law 12
failures because the checker verifies the assembled artifact:

1. Capture the environment (`scripts/kld_env_report.sh`) into `environment/`.
2. Score the zero baseline: reference against itself, on the runner and
   tensor-parallel size the candidates will use. Law 1 stops the campaign here if
   it is not exactly `0.0`.
3. Score each candidate against the same reference capture.
4. Assemble the artifact tree: suite, reference tensors, head, manifests,
   baselines, per-candidate reports.
5. Write `checksums.txt` over the assembled tree.
6. Run `compliance.py` per candidate, writing `results/<candidate>/compliance.json`.
7. Render one-pagers, then the leaderboard.
8. Publish.

Steps 5 and 6 cannot be swapped. Law 12 verifies that `checksums.txt` and the
reusable reference exist, so compliance is evaluated against a finished tree. It
also verifies that the manifest published under `reference/` is bound to the same
identity the report was scored against, which is why step 4 replaces a reference
whose identity has moved rather than keeping whatever landed there first.

## Library layout

One root per reference model; one folder per candidate beneath it. Everything
shared by that model's candidates — the suite, the reusable reference, the zero
baseline, the environment — lives once at the model root.

```text
<library>/
  LAWS.md                       the laws these artifacts were produced under
  leaderboard.md, leaderboard.csv
  Qwen3.6-27B/
    LAWS.md                     shipped inside the artifact, which is published alone
    README.md                   this model's identity and its candidate table
    suite/suite-manifest.json   token hashes, sources, strata, partitions
    suite/tokens/               the frozen token IDs that are the evaluation input
    reference/manifest.json     the capture manifest, bound per Law 5
    reference/hidden_*.safetensors
    reference/lm_head.safetensors
    baselines/                  the exact-zero proofs required by Law 1
    environment/                kld_env_report.sh output, verbatim
    checksums.txt               authoritative for file integrity
    Qwen3.6-27B-FP8/
      report.json, report.md, manifest.json, compliance.json
    Qwen3.6-27B-AWQ/
      report.json, report.md, manifest.json, compliance.json
  Qwen3.6-35B-A3B/
    ...
```

`artifact.py leaderboard` walks this tree at any depth. Rankings are grouped
by comparability key, which includes the reference checkpoint, so candidates of
different models never share a table. Each row splits the campaign path into
**family** (the reference), **author** (the Hub org), and **quant** (the Hub
basename with the family prefix stripped), so `unsloth/Qwen3.8-27B-NVFP4` and
`RadixArk/Qwen3.8-27B-NVFP4` stay distinct in the published index.

One-pagers are also mirrored to a central index so a reader can find a model's
mean KLD without downloading tens of gigabytes of reference tensors.

## Usage

Evaluate the laws and write a receipt:

```bash
python fidelity/compliance.py \
  --report results/Qwen3.6-27B-FP8/report.json \
  --manifest results/Qwen3.6-27B-FP8/manifest.json \
  --self-report baselines/Qwen3.6-27B-self.json \
  --env-dir environment \
  --suite suite/suite-manifest.json \
  --artifact-dir . \
  --out results/Qwen3.6-27B-FP8/compliance.json
```

Render the one-pager, the leaderboard, and the checksums:

```bash
python fidelity/artifact.py onepager \
  --report results/Qwen3.6-27B-FP8/report.json \
  --manifest results/Qwen3.6-27B-FP8/manifest.json \
  --receipt results/Qwen3.6-27B-FP8/compliance.json \
  --self-report baselines/Qwen3.6-27B-self.json \
  --env-dir environment \
  --label Qwen3.6-27B-FP8 \
  --out results/Qwen3.6-27B-FP8/report.md

python fidelity/artifact.py leaderboard \
  --results-root results --out leaderboard.md --csv leaderboard.csv

python fidelity/artifact.py checksums --root .
```

Verify a downloaded artifact before using it:

```bash
sha256sum --check checksums.txt
```

## Publishing

```bash
export LIL_HF_NAMESPACE=your-hf-name    # or the org, once it is agreed
python fidelity/publish.py --library /mnt/kld/library --dry-run
python fidelity/publish.py --library /mnt/kld/library
```

The namespace is never guessed: with neither `--namespace` nor
`$LIL_HF_NAMESPACE`, publishing refuses to run.

Two destinations, because the artifacts are large and the answers are small.
Each model becomes its own dataset repo carrying the suite, the reusable
reference tensors, the head, the environment, the baseline, and every candidate's
receipts. A single small index repo carries every one-pager plus the leaderboard,
so a reader can find a mean KLD without downloading tens of gigabytes.

Publishing is gated. A model is refused if its `checksums.txt` does not verify, if
no candidate carries a receipt, or if any candidate is not law-compliant.
`--skip-noncompliant` publishes the compliant candidates and records the withheld
ones in the index. A refusal is reflected in the exit status even when other
models publish, so automation cannot mistake a partial publish for a clean one.

Every text file in the artifact is also scanned two ways: for recognizable
credential shapes — Hugging Face, GitHub, and OpenAI-style tokens, AWS keys,
private key blocks — and, in JSON, for any field whose name *is* a credential's
name (`token`, `api_key`, `password`, and the like) still carrying a value, which
catches a secret that resembles nothing in particular. Field names match exactly,
so this artifact's own `token_sha256`, `cache_key`, and `eos_token` are not
credentials and are not flagged. Either kind of hit refuses the model and the
upload, and
that refusal is not skippable, because a published credential cannot be
unpublished. Assembly redacts credential values out of `environment/runtime.json`
on every run — in the work directory as well as the library copy — so an artifact
captured before this policy existed becomes safe the next time it is assembled.

## Routed models: what the mean hides

Routing cost saturates. A perturbation to a router logit changes an expert
selection only when it crosses a near-tie, and how many near-ties a model has is a
property of the model rather than of the perturbation, so an eight-bit router and a
four-bit router cost about the same. On one MoE checkpoint:

| Cell | MXFP8 | NVFP4 | Ratio |
|---|---|---|---|
| Experts only | 0.002953 | 0.106539 | 36× |
| Router only | 0.200938 | 0.193147 | 1.0× |
| Deployed | 0.211934 | 0.253426 | 1.2× |

The expert weights differ by a factor of 36. The deployed means differ by 1.2,
because a routing term near 0.20 dominates both. A reader given only the deployed
column would conclude the formats are nearly equivalent, and would be wrong by an
order of magnitude. Two engineering conclusions follow: never quantize a router,
since the weights are a rounding error in the parameter count and the cost barely
improves with precision; and never rank schemes on a deployed mean.

### Routing is not router precision

The row labelled "Router only" above is a quantize-dequantize cell: it rounds the
router's weights and measures what that costs. That is *not* the routing term, and
reading it as one gets the mechanism backwards. Routing changes because the
activations arriving at the router changed — quantized attention and quantized
experts upstream have already moved the residual stream — so a checkpoint that
ships its router in BF16 has a bit-identical router and still selects different
experts. On such a checkpoint the router weight cell is exactly zero while the
routing term is not.

The same argument says no QDQ cell is routing-free. Rounding any weight moves the
residual stream, and every router downstream of that change sees different inputs,
so the experts-only cell reroutes tokens too. The artifact reports each cell's own
selection-change rate rather than claiming any cell holds routing fixed.

So the routing term is measured, not emulated. `--measure-routing` records the
reference's selected experts per token and per layer in a separate pass over the
reference — expert IDs are a few hundred megabytes against tens of gigabytes of
hidden states, so an existing capture stays valid — and the candidate's selections
are compared against them while it is scored. The report carries the share of
(token, layer) selections that changed, the share of positions where any layer
rerouted, the mean divergence where routing held against where it changed, how
divergence grows with the number of rerouted layers, and the per-layer rate, which
rises with depth exactly as accumulated perturbation predicts.

The ranking floor is the excess that rerouting puts into the mean: the flipped
fraction times the gap between the two conditional means. It answers "how much of
this mean would disappear if rerouted tokens diverged no more than the rest", and
two deployed means closer together than that do not rank anything.

Law 14 makes the decomposition mandatory for any reference whose config declares
experts. `campaign.py` detects that from the checkpoint, inspects the deployed
candidate for its scheme and router coverage, builds the cells with `qdq.py`,
scores them against the capture the deployed run already produced, and writes
`attribution.json`. Compliance refuses a routed candidate that has none, one whose
capture manifest predates routing detection, and one scored without a measured
routing term, so a routed model cannot exempt itself by omission. The one-pager
prints the cells, the routing divergence, and the ranking floor, and links every
supporting report and QDQ manifest; the leaderboard carries experts-only and
routing-term columns next to the deployed mean.

Cells are named for the component carrying the error rather than written as `B×Q`
or `Q×B`. That notation leaves the order of the factors to the reader, and it is
routinely read both ways, which turns a shared metric into two different metrics
wearing one label.

### Where each cell sits on the perturbation ladder

Quantization is not the only thing between a reference and a served token. Ordered
by what each rung adds:

| Rung | What it adds | Cell |
|---|---|---|
| 0 | Two implementations of the same high-precision weights | not measured — see below |
| 1 | Expert weight rounding, plus whatever rerouting it causes | `expert_cell` |
| 2 | Rounding every quantized component, plus its rerouting | `composite_cell` |
| 3 | Quantized kernels and activations, batch 1, deterministic, BF16 KV | deployed `report.json` |
| 4 | A quantized KV cache | not measured |
| 5 | Realistic batching and shapes | not measured |

Rerouting is not a rung of its own, because it is not something a rung adds: every
rung from 1 up carries it, and each cell reports its own selection-change rate. The
routing term is measured across the deployed run and reported separately.

Rung 3 minus rung 2 is published as the "beyond weight rounding" term. Where the
checkpoint quantizes activations, that term carries activation quantization as well
as kernel arithmetic, and the one-pager says which from the checkpoint's own
`activation_scheme`. On `Qwen3.6-35B-A3B-FP8` it came to +0.00744 against a
deployed mean of 0.01838 — 40% of the divergence, invisible to every weight-only
cell. Rungs 4 and 5
are out of scope by construction — Law 2 requires eager execution, batch 1, and no
prefix caching, because a measurement that includes serving variance cannot
attribute anything. They belong in a serving-variance study that cites this
artifact as its floor, not in this artifact.

Rung 0 is not the zero baseline. Law 1 compares a reference to itself through one
implementation and demands exactly 0.0, which is a determinism check. Comparing two
*implementations* of the same BF16 weights — vLLM against Transformers, say —
measures something else and would be a genuine floor for cross-stack claims. This
program does not measure it, and no number here should be read as if it did.

The ladder is the same expert cell at each scheme, which is what makes a later
comparison possible without rerunning the campaign:

```json
{"ladder": ["fp8_block", "mxfp8", "nvfp4"], "prune_variants": true}
```

Variant weights are deleted once a variant's report exists, keeping its
`qdq-manifest.json`; a three-scheme ladder is otherwise several full copies of the
reference on disk. Candidates of one model now share a single teacher capture,
since the manifest binds a capture to the tokens and geometry rather than to the
candidate — without that, a ladder would recapture identical reference tensors once
per cell.

## Which domains rated poorest

The suite is stratified over a dozen kinds of text, so the headline mean is an
average across encyclopedic prose, worked mathematics, source code, dialogue,
Chinese, structured data, and the rest. Law 15 requires the artifact to say which
of them the divergence landed on, because the useful question is not how far apart
two models are but how far apart they are on the work a deployment will give them.

The join is cheap and needs no extra model output. Every scoring run over a suite
records one `per_context` entry per row — mean, median, p99, max, reference top-1
probability, top-1 agreement — keyed to the suite `context_id` it came from.
`strata.py` groups those entries by the suite manifest's `stratum` and
`source_key` and writes `strata.md` and `strata.json` beside the report:

```bash
python fidelity/strata.py --suite /mnt/kld/suites/qwen3.6-1024x2048-v1 \
    --cell deployed=library/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-FP8/report.json \
    --cell nvfp4=library/.../attribution/experts-nvfp4.json \
    --out strata.md --json strata.json
```

`campaign.py assemble` does this for every candidate automatically, adding one
column per ladder rung so the table answers whether a domain's weakness follows
the model or the format. The one-pager carries the domain table and the
leaderboard carries each candidate's weakest domain.

Two things are reported with every stratum mean rather than resolved silently.
The reference's own top-1 probability on that stratum, because a domain the
reference finds harder diverges more for that reason alone and the finding is
weaker than one where the reference was confident. And the spread of per-context
means within the stratum, because a stratum holds tens of documents and one
pathological document can carry it; `strata.py` says so explicitly when a
stratum's worst context is four or more times its median.

Reports produced before per-context recording cannot be attributed after the
fact. They carry no `per_context`, `strata.py` refuses to guess, and Law 15 fails
with that reason — the fix is a rescore, which reuses the existing capture.

## Assembly will not downgrade a published result

Assembly overwrites a candidate's files before compliance runs on them, so a
campaign pointed at the wrong config can replace a compliant result with a failing
one. When the previously published receipt was compliant and the new one is not,
the report, manifest, receipt, one-pager, and attribution are restored and the
stage exits non-zero:

```text
REVERTED Qwen3.6-27B-FP8: the published result was law-compliant and this one is
not, so the previous report, receipt, and one-pager were restored.
```

`--force` replaces them anyway, which is the right call when the new failure is
the honest one — a law was added, or the old result was compliant under weaker
rules.

The environment is the one thing the guard cannot protect, because it is refreshed
before the candidate loop and belongs to the model rather than to any candidate.
Assembly therefore leaves it alone when the work directory contributes no report
and no baseline, since in that case the published environment is the only truthful
record of whatever did produce the numbers:

```text
!!! Qwen3.6-27B: no reports or baseline in /work/qwen3.6; leaving the published
environment untouched
```

The published reference is left in place across assemblies rather than rewritten,
because it is tens of gigabytes and a hard link cannot be re-made cheaply. That is
only sound while the file already there came from a capture bound to the same
identity, so assembly compares the two manifests on the fields Law 5 binds and
replaces the reference when they differ:

```text
REPLACING reference in .../Qwen3.6-27B/reference: the published reference is bound
to a different identity than Qwen3.6-27B-ref-v0-tp1-analysis-c2048-s0 (rows)
```

## Reclaiming scratch space

The library is the index. `sweep.py` digests every published file and then asks of
each work tree whether anything in it is byte-identical to something published.
Matching by content rather than by path matters: a report is copied into the
library under a different name than it carries in the work tree, and a run
assembled from the wrong config leaves files whose names look plausible.

```bash
python fidelity/sweep.py --library /path/to/library
```

The dry run is the default and names four categories:

- **Stale trees.** No report in them was ever published. This is the shape a
  campaign run against the wrong config leaves behind, and it is the one category
  that is safe to remove wholesale (`--delete-stale`).
- **Unpruned QDQ variants.** Rebuildable from the reference in minutes, while the
  `qdq-manifest.json` beside them is the published provenance and stays
  (`--prune-variants`).
- **Reference captures.** A cache, not scratch. Deleting one costs a forward pass
  and tens of gigabytes of rewriting the next time a candidate is scored against
  that reference, so it takes an explicit `--delete-captures`.
- **Unpublished reports and logs.** Listed, never deleted. A report is what makes
  a rescore unnecessary, and a log is the audit trail behind a published number;
  both are small enough that reclaiming them is a bad trade in either direction.

The sweep refuses to delete anything while a published candidate lacks a passing
receipt, because until every receipt passes there is no settled latest run to
sweep against.

## Overriding a law

A deviation needs a named approver, a written justification, and a timestamp
(Law 13). Anything less is not an approval and the underlying failure stands.
Only Laws 1, 8, 11, 12, 14, and 16 permit an override at all, and Law 16 permits
one only for a report that carries no weight digest. A digest that contradicts the
published checkpoint is absolute: an approval there would excuse the single error
the law exists to catch, so it is rejected even when the approval is complete.

```json
{
  "12": {
    "approver": "Your Name",
    "justification": "reference redistribution restricted by checkpoint license",
    "timestamp": "2026-08-29T21:00:00Z"
  }
}
```

Pass it with `--approvals`. The override is then rendered on the one-pager beside
the affected numbers, never in a footnote.

### Overriding Law 1 specifically

A non-zero zero-baseline is the most consequential deviation possible, so an
approval alone is not enough. The campaign must also capture the reference three
times over the identical tokens and score all three pairwise comparisons, which
measures the nondeterminism floor it is carrying:

```json
{
  "captures": 3,
  "pairwise": [
    {"pair": "00-01", "mean_kld": 0.0000312, "positions": 204700},
    {"pair": "00-02", "mean_kld": 0.0000298, "positions": 204700},
    {"pair": "01-02", "mean_kld": 0.0000305, "positions": 204700}
  ]
}
```

Pass it with `--repeat-study`. Without it, an approved Law 1 failure stays a
failure. With it, the widest pairwise mean becomes the published floor, and the
one-pager opens with it — a candidate difference no larger than that floor is not
a result.
