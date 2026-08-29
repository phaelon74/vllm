#!/usr/bin/env bash
# Drive the KLD matrix on a host whose checkpoints are already local: self-KLD
# baselines first, then every BF16-teacher vs FP8-student pair. Reports, logs,
# and capture directories all land under $KLD_RUN.
#
# Usage:
#   export KLD_RUN="$HOME/kld-artifacts/$(date +%Y%m%d)-$(hostname -s)"
#   bash scripts/kld_run_matrix.sh selfkld   # zero-KLD baselines, gates the rest
#   bash scripts/kld_run_matrix.sh pairs     # BF16 vs FP8 comparisons
#   bash scripts/kld_run_matrix.sh all
#
# Knobs (all optional):
#   MODEL_ROOT      checkpoint parent directory (default /media/fmodels/Qwen)
#   RUNNER          VLLM_USE_V2_MODEL_RUNNER value (default 0; hybrid models
#                   default to V1, so V2 is an explicit opt-in)
#   ROWS            evaluation rows for the pair sweep (default 100)
#   CONTEXT_LENGTH  tokens per row (default 2048)
#   SCORE_FROM      leading positions to skip; manifest-bound (default 0)
#   TP_SIZE         tensor parallel size (default 4)
#   GPU_UTIL        gpu memory utilization (default 0.90)
#   STORAGE         logits | hidden | auto (default auto)
#   SELF_KLD_FP8    also self-KLD the FP8 checkpoints (default 0)
#   KEEP_GOING      continue after a failed run instead of aborting (default 0)

set -uo pipefail

MODE=${1:-all}
MODEL_ROOT=${MODEL_ROOT:-/media/fmodels/Qwen}
RUNNER=${RUNNER:-0}
ROWS=${ROWS:-100}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-2048}
SCORE_FROM=${SCORE_FROM:-0}
TP_SIZE=${TP_SIZE:-auto}
GPU_UTIL=${GPU_UTIL:-auto}
KV_CACHE_GIB=${KV_CACHE_GIB:-8}
WEIGHT_FRACTION=${WEIGHT_FRACTION:-0.60}
HEADROOM_GIB=${HEADROOM_GIB:-12}
STORAGE=${STORAGE:-auto}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}
SELF_KLD_FP8=${SELF_KLD_FP8:-0}
KEEP_GOING=${KEEP_GOING:-0}

if [[ -z ${KLD_RUN:-} ]]; then
  echo "KLD_RUN is not set. See docs/design/kld_manual_verification.md Step 0." >&2
  exit 2
fi
mkdir -p "$KLD_RUN" || exit 1

REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)

# Resolve a virtualenv interpreter: an explicit override, then the activated
# venv (which need not live in the repo), then a repo-local .venv. Never fall
# through to a system interpreter.
resolve_python () {
  local candidate
  for candidate in \
    "${KLD_PYTHON:-}" \
    "${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}" \
    "$REPO_ROOT/.venv/bin/python"; do
    if [[ -n $candidate && -x $candidate ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

if ! PY=$(resolve_python); then
  echo "no virtualenv interpreter found. Activate the venv you installed vLLM" \
    "into, or set KLD_PYTHON=/path/to/venv/bin/python." >&2
  exit 2
fi

if ! "$PY" -c 'import vllm' 2>/dev/null; then
  echo "$PY cannot import vllm; this is not the venv vLLM is installed into" >&2
  exit 2
fi
echo "interpreter: $PY"

DENSE_BF16="$MODEL_ROOT/Qwen3.6-27B"
DENSE_FP8="$MODEL_ROOT/Qwen3.6-27B-FP8"
MOE_BF16="$MODEL_ROOT/Qwen3.6-35B-A3B"
MOE_FP8="$MODEL_ROOT/Qwen3.6-35B-A3B-FP8"

visible_gpu_count () {
  if [[ -n ${CUDA_VISIBLE_DEVICES:-} ]]; then
    awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES"
    return 0
  fi
  nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l
}

gpu_total_gib () {
  local mib
  mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits \
    2>/dev/null | head -n1)
  [[ -n $mib ]] || return 1
  awk -v m="$mib" 'BEGIN { printf "%.2f", m / 1024 }'
}

checkpoint_gib () {
  local path=$1 bytes
  bytes=$(find "$path" -type f \( -name '*.safetensors' -o -name '*.bin' \) \
    -printf '%s\n' 2>/dev/null | awk '{s += $1} END {print s + 0}')
  if [[ ${bytes:-0} -le 0 ]]; then
    bytes=$(du -sb "$path" 2>/dev/null | cut -f1)
  fi
  awk -v b="${bytes:-0}" 'BEGIN { printf "%.2f", b / 1073741824 }'
}

# Pick the smallest tensor-parallel size whose per-GPU weight share leaves room
# for activations, then size utilization to weights + pinned KV + headroom.
# Scoring is memory-hungry in a way vLLM's profiler does not see: every scored
# position needs a full padded-vocabulary logits row, several GiB per request
# on these checkpoints, so the KV cache must not be allowed to claim the card.
plan_gpus () {
  local weights=0 candidate total ngpu
  for candidate in "$@"; do
    weights=$(checkpoint_gib "$candidate" | awk -v a="$weights" \
      '{print ($1 > a) ? $1 : a}')
  done
  if ! total=$(gpu_total_gib); then
    echo "cannot query GPU memory via nvidia-smi; set TP_SIZE and GPU_UTIL" >&2
    return 1
  fi
  ngpu=$(visible_gpu_count)
  [[ ${ngpu:-0} -ge 1 ]] || ngpu=1

  read -r PLAN_TP PLAN_UTIL < <(awk \
    -v w="$weights" -v total="$total" -v ngpu="$ngpu" \
    -v wf="$WEIGHT_FRACTION" -v kv="$KV_CACHE_GIB" -v head="$HEADROOM_GIB" '
    BEGIN {
      budget = total * wf
      n = split("1 2 4 8", cand, " ")
      tp = 0
      for (i = 1; i <= n; i++) {
        if (cand[i] > ngpu) continue
        if (w / cand[i] <= budget) { tp = cand[i]; break }
      }
      if (tp == 0) tp = ngpu
      util = (w / tp + kv + head) / total
      if (util > 0.95) util = 0.95
      if (util < 0.15) util = 0.15
      printf "%d %.2f\n", tp, util
    }')

  PLAN_WEIGHTS=$weights
  PLAN_TOTAL=$total
  echo "plan: weights ${weights} GiB, ${ngpu} x ${total} GiB visible" \
    "-> TP=$PLAN_TP util=$PLAN_UTIL kv=${KV_CACHE_GIB} GiB"
}

RESULTS="$KLD_RUN/matrix-results.tsv"
[[ -f $RESULTS ]] || printf 'label\tstatus\tmean_kld\tpositions\treport\n' >"$RESULTS"

# read_report LABEL REPORT_JSON [require_zero]
# Appends one row to $RESULTS and returns non-zero if the report is missing or,
# when require_zero is 1, if mean KLD is not exactly 0.
read_report () {
  local label=$1 report=$2 require_zero=${3:-0}
  REPORT_LABEL=$label REPORT_PATH=$report REQUIRE_ZERO=$require_zero \
    "$PY" - >>"$RESULTS" <<'PYEOF'
import json
import os
import sys

label = os.environ["REPORT_LABEL"]
path = os.environ["REPORT_PATH"]
require_zero = os.environ["REQUIRE_ZERO"] == "1"

try:
    with open(path) as fh:
        report = json.load(fh)
except OSError as exc:
    print(f"{label}\tMISSING\t\t\t{path}")
    sys.exit(f"{label}: no report at {path} ({exc})")

mean = report.get("mean_kld")
positions = report.get("num_positions")
status = "OK"
if require_zero and mean != 0.0:
    status = "NONZERO_SELF_KLD"
print(f"{label}\t{status}\t{mean!r}\t{positions!r}\t{path}")
if status != "OK":
    sys.exit(f"{label}: self-KLD must be exactly 0.0, got {mean!r}")
PYEOF
}

# run_kld LABEL STUDENT TEACHER ROWS REQUIRE_ZERO [extra args...]
run_kld () {
  local label=$1 student=$2 teacher=$3 rows=$4 require_zero=$5
  shift 5

  for path in "$student" "$teacher"; do
    if [[ ! -d $path ]]; then
      printf '%s\tMISSING_MODEL\t\t\t%s\n' "$label" "$path" >>"$RESULTS"
      echo "skipping $label: no checkpoint at $path" >&2
      return 1
    fi
  done

  local tp=$TP_SIZE util=$GPU_UTIL
  if [[ $tp == auto || $util == auto ]]; then
    plan_gpus "$student" "$teacher" || return 1
    [[ $tp == auto ]] && tp=$PLAN_TP
    [[ $util == auto ]] && util=$PLAN_UTIL
  fi

  # Every knob the capture manifest binds itself to belongs in the directory
  # name, tensor parallelism included: a capture taken at TP=4 is refused by a
  # TP=1 scoring run, and silently reusing the directory turns that into a
  # confusing abort instead of a fresh capture.
  local tag="${label}-v${RUNNER}-tp${tp}-r${rows}-c${CONTEXT_LENGTH}-s${SCORE_FROM}"
  local capture="$KLD_RUN/$tag"
  local report="$KLD_RUN/$tag.json"
  local log="$KLD_RUN/$tag.log"

  echo "=== $tag (TP=$tp util=$util kv=${KV_CACHE_GIB}GiB)"
  VLLM_USE_V2_MODEL_RUNNER="$RUNNER" \
    "$PY" examples/offline_inference/score_mode_kld.py \
    --model "$student" \
    --reference-model "$teacher" \
    --reference-logits "$capture" \
    --dataset Salesforce/wikitext \
    --dataset-config wikitext-2-raw-v1 \
    --rows "$rows" \
    --context-length "$CONTEXT_LENGTH" \
    --score-from "$SCORE_FROM" \
    --storage "$STORAGE" \
    --probe-replay \
    --language-model-only \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --report-json "$report" \
    --tensor-parallel-size "$tp" \
    --gpu-memory-utilization "$util" \
    --kv-cache-memory-gib "$KV_CACHE_GIB" \
    "$@" 2>&1 | tee "$log"

  local rc=${PIPESTATUS[0]}
  if [[ $rc -ne 0 ]]; then
    printf '%s\tEXIT_%d\t\t\t%s\n' "$tag" "$rc" "$log" >>"$RESULTS"
    quarantine_capture "$capture" "$log"
    return 1
  fi
  read_report "$tag" "$report" "$require_zero"
}

# Move a capture directory aside so the next run recaptures instead of tripping
# over it. Two cases need this: an interrupted capture, which has window files
# but no manifest.json and would make Phase 1 skip and then fail closed with a
# misleading error; and a complete capture the scoring run rejected, which is
# manifest-bound to conditions this host no longer reproduces.
quarantine_capture () {
  local capture=$1 log=${2:-}
  [[ -d $capture ]] || return 0

  local reason=""
  if [[ -n $log && -f $log ]] &&
    grep -q "Capture manifest does not match" "$log"; then
    reason="rejected"
  elif [[ -f "$capture/manifest.json" ]]; then
    return 0
  elif [[ -z $(ls -A "$capture" 2>/dev/null) ]]; then
    rmdir "$capture" 2>/dev/null
    return 0
  else
    reason="incomplete"
  fi

  local aside="$capture.$reason-$(date +%H%M%S)"
  mv "$capture" "$aside" && echo "quarantined $reason capture: $aside" >&2
  return 0
}

guard () {
  if ! "$@"; then
    if [[ $KEEP_GOING == 1 ]]; then
      echo "run failed; KEEP_GOING=1, continuing" >&2
      FAILURES=$((FAILURES + 1))
      return 0
    fi
    echo "run failed; aborting. See $RESULTS" >&2
    exit 1
  fi
}

FAILURES=0

case $MODE in
  selfkld | pairs | all) ;;
  *)
    echo "unknown mode '$MODE' (expected selfkld, pairs, or all)" >&2
    exit 2
    ;;
esac

if [[ $MODE == selfkld || $MODE == all ]]; then
  # One row is enough: the only acceptable answer is exact zero.
  guard run_kld dense-27b-self "$DENSE_BF16" "$DENSE_BF16" 1 1
  guard run_kld moe-35b-a3b-self "$MOE_BF16" "$MOE_BF16" 1 1
  if [[ $SELF_KLD_FP8 == 1 ]]; then
    guard run_kld dense-27b-fp8-self "$DENSE_FP8" "$DENSE_FP8" 1 1
    guard run_kld moe-35b-a3b-fp8-self "$MOE_FP8" "$MOE_FP8" 1 1
  fi
fi

if [[ $MODE == pairs || $MODE == all ]]; then
  DECOMPOSE=()
  [[ $STORAGE != logits ]] && DECOMPOSE=(--decompose-head)
  guard run_kld dense-27b-fp8 "$DENSE_FP8" "$DENSE_BF16" "$ROWS" 0 "${DECOMPOSE[@]}"
  guard run_kld moe-35b-a3b-fp8 "$MOE_FP8" "$MOE_BF16" "$ROWS" 0 "${DECOMPOSE[@]}"
fi

echo
echo "=== $RESULTS"
cat "$RESULTS"
[[ $FAILURES -eq 0 ]] || echo "$FAILURES run(s) failed" >&2
exit $((FAILURES > 0))
