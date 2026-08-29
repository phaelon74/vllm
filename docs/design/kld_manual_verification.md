# KLD measurement rework: manual verification

This runbook is for the human who executes tests. The implementation was not
run against GPU or pytest in the authoring session. Each section has the
exact command, expected output, and a pass/fail line.

Use the repo venv. Never use system `python3` or bare `pip`.

```bash
source .venv/bin/activate
```

The venv does not have to be repo-local. Both helper scripts resolve the
interpreter from `KLD_PYTHON`, then `$VIRTUAL_ENV/bin/python`, then
`./.venv/bin/python`, and print which one they picked; the manual commands
below assume the venv is active, so substitute `$VIRTUAL_ENV/bin/python` for
`.venv/bin/python` if yours lives elsewhere.

Execution order: Step 0 and Step 1 below come first and gate everything else.
The `Phase A`..`Phase 9` sections are the per-behavior regression checklist; a
non-zero self-KLD in Step 1 invalidates every number they produce, so do not
start them until Step 1 is clean.

## Step 0 — enshrine the environment

A KLD value is only meaningful next to the stack that produced it. Capture the
host, driver, toolchain, package set, repo commit, and checkpoint fingerprints
into one directory before the first run, and re-capture after any driver,
torch, vLLM, or checkpoint change.

```bash
export KLD_RUN="$HOME/kld-artifacts/$(date +%Y%m%d)-$(hostname -s)"
mkdir -p "$KLD_RUN"

bash scripts/kld_env_report.sh "$KLD_RUN/env" \
  /media/fmodels/Qwen/Qwen3.6-27B \
  /media/fmodels/Qwen/Qwen3.6-27B-FP8 \
  /media/fmodels/Qwen/Qwen3.6-35B-A3B \
  /media/fmodels/Qwen/Qwen3.6-35B-A3B-FP8

cat "$KLD_RUN/env/summary.md"
```

`summary.md` is the front page; the rest of the directory holds the raw
probes. `gpu-smi-query.txt` matters more than it looks: ECC mode, persistence
mode, clock caps, and throttle reasons all sit in there, and a change in any of
them is a legitimate explanation for a bitwise result that stopped reproducing.
`runtime.json` includes the exact `capture_runtime_manifest()` fields the
capture directories bind themselves to, plus the TF32 and deterministic-algorithm
torch flags. Model directories are fingerprinted by listing hash, byte total,
and `config.json` / tokenizer hashes; set `KLD_HASH_WEIGHTS=1` to sha256 every
safetensors shard instead, which is worth doing once per checkpoint but is slow
over a network mount.

**Pass:** `summary.md` names four GPUs, a driver version, a torch version, a
vLLM version, and the `feature/glm53-kld-determinism` commit, and each model
file lists a `config.json` whose `architectures` you recognize. **Fail:**
missing driver or GPU lines (the report ran without GPU visibility), or
`runtime.json` contains `torch_error` / `vllm_error` / a
`capture_runtime_manifest_error`.

## Step 1 — self-KLD must be exactly zero

Score each BF16 checkpoint against a capture of itself. This is the only check
that separates "the KLD pipeline is correct" from "the student is close to the
teacher", because the sole correct answer is zero. Run it per model and per
runner, since a capture manifest refuses to be replayed under a different
runner.

Both Qwen3.6 checkpoints are hybrid architectures, so the default runner
selection lands on V1; V2 requires the explicit env override and is not the
default path for them. Treat V1 as authoritative and V2 as an additional
check — if V2 aborts with a config error, record the message in the artifact
directory and move on rather than forcing it.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

self_kld () {
  model="$1"; label="$2"; runner="$3"
  VLLM_USE_V2_MODEL_RUNNER="$runner" \
  .venv/bin/python examples/offline_inference/score_mode_kld.py \
    --model "$model" \
    --reference-model "$model" \
    --reference-logits "$KLD_RUN/${label}-v${runner}-self" \
    --dataset Salesforce/wikitext \
    --dataset-config wikitext-2-raw-v1 \
    --rows 1 \
    --context-length 2048 \
    --score-from 0 \
    --storage auto \
    --probe-replay \
    --language-model-only \
    --max-num-seqs 1 \
    --report-json "$KLD_RUN/${label}-v${runner}-self.json" \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.90 \
    2>&1 | tee "$KLD_RUN/${label}-v${runner}-self.log"
}

self_kld /media/fmodels/Qwen/Qwen3.6-27B      dense-27b   0
self_kld /media/fmodels/Qwen/Qwen3.6-35B-A3B  moe-35b-a3b 0
self_kld /media/fmodels/Qwen/Qwen3.6-27B      dense-27b   1
self_kld /media/fmodels/Qwen/Qwen3.6-35B-A3B  moe-35b-a3b 1
```

The script pins `max_model_len` to twice `--context-length` (4096 here) and
disables prefix caching, and eager execution is the default, so no extra flags
are needed for determinism. `--max-num-seqs 1` keeps the hybrid state
allocation small and removes batch composition as a variable.

**Expected:** each run prints the requested runner, `Mean KLD (ref ||
student): 0.00000000`, 2047 scored positions, and — for hidden storage —
`'identical': True` from the replay probe.

**Pass:** exact zero on both models under V1, and under V2 for whichever
model V2 accepts. **Fail:** any non-zero mean KLD, or a replay probe that is
not bitwise identical. On a non-zero self-KLD, stop; on a failed replay probe
only, rerun that model with `--storage logits` rather than trusting hidden
storage.

Only after this passes do the FP8 checkpoints become meaningful: they are the
real student/teacher pairs on this host, BF16 as `--reference-model` and FP8 as
`--model`, each pair needing its own capture directory.

## Step 2 — the BF16 vs FP8 matrix

`scripts/kld_run_matrix.sh` runs the Step 1 baselines and the FP8 comparisons
with per-run capture directories, logs, and report JSONs named after the runner,
row count, context length, and `score_from`, and appends one row per run to
`$KLD_RUN/matrix-results.tsv`. It re-asserts the zero invariant from the report
JSON rather than from console text, and aborts on the first failure unless
`KEEP_GOING=1`.

```bash
bash scripts/kld_run_matrix.sh selfkld          # Step 1, gates the rest
bash scripts/kld_run_matrix.sh pairs            # BF16 teacher vs FP8 student
RUNNER=1 bash scripts/kld_run_matrix.sh all     # repeat under V2, if accepted
```

Knobs are environment variables: `MODEL_ROOT`, `RUNNER`, `ROWS`,
`CONTEXT_LENGTH`, `SCORE_FROM`, `TP_SIZE`, `GPU_UTIL`, `STORAGE`,
`MAX_NUM_SEQS`, `SELF_KLD_FP8`, `KEEP_GOING`. `SCORE_FROM=1024` gives the
deep-context-only view and, because `score_from` is manifest-bound, lands in
its own capture directory automatically.

**Pass:** every row in `matrix-results.tsv` reads `OK`, the two `*-self` rows
report `mean_kld` of exactly `0.0`, and both FP8 pairs produce a finite
non-zero mean. **Fail:** any `EXIT_*`, `MISSING*`, or `NONZERO_SELF_KLD` row.

## Phase A — GLM-5.3 work is on the branch

```bash
git log --oneline --stat feature/glm53-kld-determinism
git status
```

**Expected:** four commits covering (1) V2 score-mode port, (2) sparse-MLA
top-k sort, (3) diagnostic scripts, (4) design doc / script loading.
`glm53-diag.tar.gz`, `sparse_mla_sort_topk.patch`, and `v2_prompt_logits.patch`
are gitignored, not committed.

**Pass:** those four commits exist and `git status` is clean except ignored
paths. **Fail:** artifacts are tracked, or the GLM-5.3 runner/sort work is
missing.

## Phase 0 — vocabulary truncation and tokenizer fail-closed

### Padding vocab changes the number

Capture and score a checkpoint whose `config.json` `vocab_size` is larger than
`tokenizer.vocab_size` (padding rows). Compare mean KLD against a pre-fix run
that softmaxed the full padded width.

```bash
.venv/bin/python examples/offline_inference/score_mode_kld.py \
  --model /path/to/STUDENT \
  --reference-model /path/to/TEACHER \
  --dataset wikitext --dataset-config wikitext-2-raw-v1 \
  --rows 4 --context-length 128 \
  --tensor-parallel-size 1 --gpu-memory-utilization 0.85
```

**Expected:** header prints `kld_vocab_size` equal to
`tokenizer.actual_vocab_size` when provided, otherwise `tokenizer.vocab_size`.
Mean KLD differs from the pre-truncation value when
the padding gap is large. V1 and V2 agree to printed precision on a model
that can run both (not GLM-5.3).

**Pass:** truncated KLD != padded KLD on a padded-vocab pair, and both
runners match. **Fail:** numbers identical to the pre-fix padded run, or V1
and V2 disagree.

### Mismatched tokenizer aborts

Capture once, then score with a different `--model` whose tokenizer
`name_or_path` / vocab size differs, without passing `--reference-model`.

```bash
.venv/bin/python examples/offline_inference/score_mode_kld.py \
  --model /path/to/DIFFERENT_TOKENIZER_MODEL \
  --reference-logits /path/to/capture_dir \
  --dataset wikitext --dataset-config wikitext-2-raw-v1 \
  --rows 4 --context-length 128
```

**Expected:** `ValueError` / abort: `Capture manifest does not match this
scoring run` mentioning `tokenizer` and/or `token_sha256`. No mean KLD is
printed.

**Pass:** process exits non-zero before scoring. **Fail:** it scores anyway.

## Phase 1 — non-overlapping rows

```bash
.venv/bin/python examples/offline_inference/score_mode_kld.py \
  --model /path/to/STUDENT \
  --reference-model /path/to/TEACHER \
  --dataset wikitext --dataset-config wikitext-2-raw-v1 \
  --rows 100 --context-length 2048 \
  --tensor-parallel-size 1 --gpu-memory-utilization 0.85
```

**Expected header:**

- `Rows: 100`
- `unique tokens: 204800`
- `Positions:` about `204700` (100 rows times 2047 scored positions)

`--num-samples 10` vs `--num-samples 200` must change unique-token coverage
when the corpus is long enough (the old `99 * stride` cap is gone).

```bash
.venv/bin/python examples/offline_inference/score_mode_kld.py \
  --model /path/to/STUDENT \
  --reference-logits /path/to/capture_dir \
  --dataset wikitext --dataset-config wikitext-2-raw-v1 \
  --rows 100 --context-length 2048 --stride 512
```

**Expected:** a deprecation warning for `--stride`, and overlapping window
counts (not 204800 unique tokens).

**Pass:** default run matches 100 / 204800 / ~204700; `--stride` warns;
`--num-samples` changes coverage. **Fail:** still 100 windows from a
`99 * stride` cap, or no warning.

## Phase 2 — per-position reporting

On an unchanged configuration, confirm the printed **mean** matches the old
scalar to printed precision. Also confirm:

- reverse KL (`student || ref`) is printed
- median / p90 / p99 / max are printed
- confidence buckets use `[0.00, 0.25) ... [0.95, 1.00)`
- depth buckets are printed even with `--score-from 0`
- top-K agreement for K=1..5 is printed

Repeat with `--score-from 1024`. For 100 full 2048-token rows, `Positions:`
must be `102300` (100 × 1023), proving the shallow prefix was removed from
every row rather than only the first.

**Pass:** all of the above appear; mean KLD is concentrated in low-confidence
bins on a typical language-model capture. **Fail:** only a single scalar, or
missing reverse KL / buckets.

## Phase 3 — manifest and GLM-5.3 gate

### Mutated manifest refuses scoring

After a successful capture, edit `manifest.json` (change `token_sha256` or
`kld_vocab_size`) and re-run Phase 2 against that directory.

**Expected:** abort with a mismatch listing the edited field.

**Pass:** non-zero exit, no KLD number. **Fail:** scoring proceeds.

### GLM-5.3 gate

```bash
unset VLLM_SPARSE_MLA_SORT_TOPK
.venv/bin/python examples/offline_inference/score_mode_kld.py \
  --model /path/to/STUDENT \
  --reference-model /path/to/GLM-5.3-Flash \
  --dataset wikitext --dataset-config wikitext-2-raw-v1 \
  --rows 1 --context-length 130 --capture-only
```

**Expected:** architecture detection selects the GLM gate, then the process
exits non-zero because `VLLM_SPARSE_MLA_SORT_TOPK=1` was not explicitly set.

With the sort enabled:

```bash
export VLLM_SPARSE_MLA_SORT_TOPK=1
# same command
```

**Expected:** gate passes (self-KLD max 0.0 and bitwise identity) and capture
writes `manifest.json`.

**Pass:** unset sort blocks capture; set sort allows it. **Fail:** capture
succeeds with the sort unset, or the `GlmMoeDsaForCausalLM` architecture skips
the gate even when the checkpoint directory has been renamed.

Non-GLM models print `Determinism gate: skipped` and continue.

## Phase 4 — LM-head detection

```bash
.venv/bin/python - <<'PY'
from vllm.v1.sample.kld import detect_lm_head_quantization
print(detect_lm_head_quantization("/path/to/unquantized_or_ignored_head"))
print(detect_lm_head_quantization("/path/to/head_quantized_checkpoint"))
PY
```

**Expected:** `state` is `unquantized` for a bf16/fp16 `lm_head.weight` or an
ignored/tied head, and `quantized` when packed keys (`qweight`,
`weight_packed`, `scales`, …) are present. Cross-check against
`model.safetensors.index.json` (`sha256sum` of the index is enough to confirm
you inspected the same files).

The KLD script header must print static and authoritative runtime teacher /
student LM-head states.

**Pass:** printed state matches the index. **Fail:** a packed head reports
`unquantized`, or the header omits the state.

## Phase 5 — replay probe (stop on failure)

```bash
.venv/bin/python examples/offline_inference/score_mode_kld.py \
  --model /path/to/STUDENT \
  --reference-model /path/to/TEACHER \
  --dataset wikitext --dataset-config wikitext-2-raw-v1 \
  --rows 1 --context-length 128 \
  --storage auto --probe-replay --capture-only
```

**Expected:** a line `Replay probe: {... 'identical': True ...}`.

- If `identical: True`: `--storage auto` may select hidden; `--storage hidden`
  is allowed.
- If `identical: False`: the script must keep logits (auto) or refuse
  (explicit hidden). **Stop here.** Full-logit storage stays authoritative.
  Record `max_abs` and `num_differing` from the probe dict.

`--decompose-head` requires a hidden capture. Trunk mean KLD is student
hidden × teacher head vs teacher hidden × teacher head through the same
TP-aware logits-processor path. Deployed mean KLD is in-engine student logits
vs teacher replay. The printed deployed-minus-trunk value is a diagnostic
delta, not an additive KL decomposition.

**Pass:** probe prints bitwise equality, or inequality is reported and hidden
storage stays off. **Fail:** hidden scoring proceeds after a failed probe.

## Full suite (AGENTS.md)

```bash
.venv/bin/python -m pytest tests/v1/sample/test_score_mode.py -v
.venv/bin/python -m pytest tests/renderers/test_completions.py -k ScoreKld -v
pre-commit run --all-files
```

The CUDA-marked parity test captures once, then scores the same logits through
both `VLLM_USE_V2_MODEL_RUNNER=0` and `=1`. It must report exact equality and
zero self-KLD. This is the required end-to-end V1/V2 check; merely invoking the
shared math helper is not sufficient.

If `update-dockerfile-graph` fails for missing `/bin/bash` on Windows, skip
only that hook.

For a model-affecting change, also run one eval from `tests/evals/` (for
example GSM8K on a small model you already use for scoring). Record the
command and result in the PR.

**Pass:** new unit tests green; renderer field copy includes `kld_vocab_size`;
pre-commit clean except a documented Windows hook skip. **Fail:** vocab
truncation / windowing / self-KLD-zero unit tests fail, or eval is skipped
with no justification.

## Artifact checklist

| Path | Role |
|------|------|
| `vllm/v1/sample/kld.py` | Shared math, manifest, head detection, replay |
| `examples/offline_inference/score_mode_kld.py` | Capture / score CLI |
| `examples/offline_inference/score_mode_perplexity.py` | PPL CLI (same windowing) |
| `docs/features/score_mode.md` | User-facing flags and API |
| `docs/design/kld_determinism_floor.md` | Floor + windowing correction |
| `tests/v1/sample/test_score_mode.py` | Unit coverage (not run here) |

## Fresh four-RTX-6000 host

These commands assume Linux, four visible 48 GB RTX 6000 GPUs, and this branch
checked out locally. The Qwen3.6 checkpoints use the built-in Qwen3.5
architecture classes in their `config.json`; no remote model code is needed.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements/lint.txt
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
uv pip install -r requirements/test/cuda.in
uv pip install datasets huggingface_hub
pre-commit install

hf download Qwen/Qwen3.6-27B --local-dir /media/fmodels/Qwen/Qwen3.6-27B
hf download Qwen/Qwen3.6-35B-A3B \
  --local-dir /media/fmodels/Qwen/Qwen3.6-35B-A3B
```

Then run Step 0 and Step 1 above; they are the entry point on a host whose
checkpoints are already in place.

For a 100-row candidate comparison, use a new capture directory and point
`--model` at the quantized student while keeping `--reference-model` on the
BF16 checkpoint. Set `VLLM_USE_V2_MODEL_RUNNER` to whichever runner passed
Step 1 for that model:

```bash
VLLM_USE_V2_MODEL_RUNNER=0 \
.venv/bin/python examples/offline_inference/score_mode_kld.py \
  --model /media/fmodels/Qwen/Qwen3.6-27B-FP8 \
  --reference-model /media/fmodels/Qwen/Qwen3.6-27B \
  --reference-logits "$KLD_RUN/dense-27b-fp8-rows100" \
  --dataset Salesforce/wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --rows 100 \
  --context-length 2048 \
  --score-from 0 \
  --storage auto \
  --probe-replay \
  --decompose-head \
  --language-model-only \
  --report-json "$KLD_RUN/dense-27b-fp8-rows100.json" \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.90
```

Repeat with the MoE paths. To reproduce the deep-context-only view, use
`--score-from 1024`; this requires a separate capture directory because
`score_from` is manifest-bound.

For Qwen3.6's 248,320-wide padded output, full FP32 logits are roughly 2.0 GB
per 2048-token row (about 203 GB for 100 rows). Hidden storage is therefore the
practical 100-row mode, but only after the exact replay probe passes.
