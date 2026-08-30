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
| `campaign.py` | Orchestrates a campaign: score, gate, assemble, render |
| `compliance.py` | Fail-closed law evaluation; emits the compliance receipt |
| `artifact.py` | Renders the one-pager, the leaderboard, and `checksums.txt` |
| `publish.py` | Uploads to the Hub; refuses anything non-compliant or tampered |
| `redaction.py` | The one secret policy shared by capture, rendering, and publish |
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
also be run individually as `download`, `score`, or `assemble`.

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
reusable reference exist, so compliance is evaluated against a finished tree.

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

`artifact.py leaderboard` walks this tree at any depth, so a result's label is
its `<model>/<candidate>` path. Rankings are grouped by comparability key, which
includes the reference checkpoint, so candidates of different models never share
a table.

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

Every text file in the artifact is also scanned for credential shapes — Hugging
Face, GitHub, and OpenAI-style tokens, AWS keys, private key blocks — and any hit
refuses the model and the upload. That refusal is not skippable, because a
published credential cannot be unpublished. Assembly redacts credential values
out of `environment/runtime.json` on every run, so an artifact captured before
this policy existed becomes safe the next time it is assembled.

## Overriding a law

A deviation needs a named approver, a written justification, and a timestamp
(Law 13). Anything less is not an approval and the underlying failure stands.
Only Laws 1, 8, 11, and 12 permit an override at all.

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
