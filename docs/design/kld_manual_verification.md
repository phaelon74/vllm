# KLD measurement rework: manual verification

This runbook is for the human who executes tests. The implementation was not
run against GPU or pytest in the authoring session. Each section has the
exact command, expected output, and a pass/fail line.

Use the repo venv. Never use system `python3` or bare `pip`.

```bash
source .venv/bin/activate
```

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

mkdir -p "$HOME/models" "$HOME/kld-captures"
hf download Qwen/Qwen3.6-27B \
  --local-dir "$HOME/models/Qwen3.6-27B"
hf download Qwen/Qwen3.6-35B-A3B \
  --local-dir "$HOME/models/Qwen3.6-35B-A3B"
```

Start with a one-row self-KLD on each runner. Captures are deliberately
runner-specific because the manifest refuses a V1 capture under V2 and vice
versa.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

run_self_kld () {
  model="$1"
  label="$2"
  runner="$3"
  capture="$HOME/kld-captures/${label}-v${runner}-smoke"
  VLLM_USE_V2_MODEL_RUNNER="$runner" \
  .venv/bin/python examples/offline_inference/score_mode_kld.py \
    --model "$model" \
    --reference-model "$model" \
    --reference-logits "$capture" \
    --dataset Salesforce/wikitext \
    --dataset-config wikitext-2-raw-v1 \
    --rows 1 \
    --context-length 2048 \
    --score-from 0 \
    --storage auto \
    --probe-replay \
    --language-model-only \
    --report-json "$HOME/kld-captures/${label}-v${runner}-smoke.json" \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.90
}

run_self_kld "$HOME/models/Qwen3.6-27B" dense-27b 0
run_self_kld "$HOME/models/Qwen3.6-27B" dense-27b 1
run_self_kld "$HOME/models/Qwen3.6-35B-A3B" moe-35b-a3b 0
run_self_kld "$HOME/models/Qwen3.6-35B-A3B" moe-35b-a3b 1
```

Each run must print the requested runner, `Mean KLD (ref || student):
0.00000000`, and 2047 positions. For hidden storage, the replay probe must
also print `'identical': True`. Stop if self-KLD is non-zero. If replay alone
is not bitwise exact, rerun with `--storage logits`; do not force hidden
storage.

For a 100-row candidate comparison, use a new capture directory and replace
`--model` with the matching quantized student checkpoint while keeping
`--reference-model` on the BF16 checkpoint:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 \
.venv/bin/python examples/offline_inference/score_mode_kld.py \
  --model /path/to/Qwen3.6-27B-STUDENT \
  --reference-model "$HOME/models/Qwen3.6-27B" \
  --reference-logits "$HOME/kld-captures/dense-27b-v2-rows100" \
  --dataset Salesforce/wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --rows 100 \
  --context-length 2048 \
  --score-from 0 \
  --storage auto \
  --probe-replay \
  --decompose-head \
  --language-model-only \
  --report-json "$HOME/kld-captures/dense-27b-v2-rows100.json" \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.90
```

Repeat with the MoE paths. To reproduce the deep-context-only view, use
`--score-from 1024`; this requires a separate capture directory because
`score_from` is manifest-bound.

For Qwen3.6's 248,320-wide padded output, full FP32 logits are roughly 2.0 GB
per 2048-token row (about 203 GB for 100 rows). Hidden storage is therefore the
practical 100-row mode, but only after the exact replay probe passes.
