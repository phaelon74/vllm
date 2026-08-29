#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# GLM-5.3-Flash prompt-logits determinism sweep.
#
# Established 2026-08-28 on 4x180GB, TP=4, BF16, eager:
#   * identical 2048-token prompts do not produce identical prompt logits
#   * positions 0..127 are always bit-identical, 128..2046 always differ
#   * every pair of runs differs, so there is no leftover-state fixed point
#   * in-process self-KLD is 6.06e-3; the KLD path itself is exact
#
# The 128 boundary implicates the GLM KV-pooling path. This script collects the
# source needed to read that code plus a configuration sweep that separates the
# candidate mechanisms.
#
# Run order matters: environment capture happens first so we keep the artifacts
# even if the GPU work dies.

set -euo pipefail

# Layout detection mirrors packet_capture_glm53_flash_logits.sh exactly, so
# both scripts agree on where the clone, venv, weights and dataset live no
# matter how the next rig is laid out.
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_DETECTED_VLLM=""
_DETECTED_WORK=""
if [[ -f "${PWD}/examples/offline_inference/score_mode_kld.py" && -d "${PWD}/vllm" ]]; then
  _DETECTED_VLLM="$(pwd -P)"
  _DETECTED_WORK="$(cd "${_DETECTED_VLLM}/.." && pwd -P)"
elif [[ -f "${_SCRIPT_DIR}/../examples/offline_inference/score_mode_kld.py" \
     && -d "${_SCRIPT_DIR}/../vllm" ]]; then
  _DETECTED_VLLM="$(cd "${_SCRIPT_DIR}/.." && pwd -P)"
  _DETECTED_WORK="$(cd "${_DETECTED_VLLM}/.." && pwd -P)"
fi

WORK_DIR="${WORK_DIR:-${_DETECTED_WORK:-${HOME}/glm53-flash-kld}}"
mkdir -p "${WORK_DIR}"
WORK_DIR="$(cd "${WORK_DIR}" && pwd -P)"
VLLM_DIR="${VLLM_DIR:-${_DETECTED_VLLM:-${WORK_DIR}/vllm}}"

# Names match the launcher; CTX/TP/MODEL are accepted as shorthand.
DATASET_CONFIG="${DATASET_CONFIG:-wikitext-2-raw-v1}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-${CTX:-2048}}"
STRIDE="${STRIDE:-512}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-${TP:-4}}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
MOE_BACKEND="${MOE_BACKEND:-triton}"
REPEATS="${REPEATS:-4}"
LENGTHS="${LENGTHS:-129,130,192,256,512}"

MODEL_DIR="${MODEL_DIR:-${MODEL:-}}"
DATASET_DIR="${DATASET_DIR:-}"
VENV_DIR="${VENV_DIR:-}"
PYTHON="${PYTHON:-}"

# Window-0 logits for a cross-day baseline. Prefer whatever the launcher
# already captured locally; fall back to the repo the launcher uploads to.
HF_USER="${HF_USER:-phaedawg}"
REFERENCE="${REFERENCE:-}"
REFERENCE_REPO="${REFERENCE_REPO:-${HF_REPO:-${HF_USER}/ref-logits-glm-5.3-flash}}"
REFERENCE_FILE="${REFERENCE_FILE:-logits_0.safetensors}"

OUT_DIR="${OUT_DIR:-${VLLM_DIR}/determinism_probe}"
# Prefer a probe sitting next to this script, so keeping both in $HOME works
# as well as running them from inside the clone's scripts/ directory.
if [[ -z "${PROBE:-}" ]]; then
  if [[ -f "${_SCRIPT_DIR}/glm53_determinism_probe.py" ]]; then
    PROBE="${_SCRIPT_DIR}/glm53_determinism_probe.py"
  else
    PROBE="${VLLM_DIR}/scripts/glm53_determinism_probe.py"
  fi
fi
if [[ -z "${BISECT:-}" ]]; then
  if [[ -f "${_SCRIPT_DIR}/glm53_layer_bisect.py" ]]; then
    BISECT="${_SCRIPT_DIR}/glm53_layer_bisect.py"
  else
    BISECT="${VLLM_DIR}/scripts/glm53_layer_bisect.py"
  fi
fi
DIAG_DIR="${OUT_DIR}/diagnostics"
PTH_FILE=""
LAST_RC=0
GATE_RC=""

die() { echo "ERROR: $*" >&2; exit 1; }
log() { echo "=== $* ==="; }

# --------------------------------------------------------------------------
# Step 0: resolve layout using the launcher's candidate lists, so a rig that
# puts the venv or weights somewhere else still works without hand-editing.
# --------------------------------------------------------------------------
pick_venv() {
  local c
  for c in "${VENV_DIR}" "${VIRTUAL_ENV:-}" "${WORK_DIR}/.venv" \
           "${VLLM_DIR}/.venv" "${HOME}/glm53-flash-kld/.venv"; do
    if [[ -n "${c}" && -x "${c}/bin/python" ]]; then
      VENV_DIR="$(cd "${c}" && pwd -P)"
      return 0
    fi
  done
  return 1
}

has_model() {
  [[ -f "${MODEL_DIR}/config.json" ]] \
    && [[ -f "${MODEL_DIR}/model.safetensors.index.json" ]]
}

pick_model() {
  local c
  for c in "${MODEL_DIR}" "${WORK_DIR}/models/GLM-5.3-Flash-BF16" \
           "${WORK_DIR}/models/zai-org/GLM-5.3-Flash-BF16" \
           "${VLLM_DIR}/../models/GLM-5.3-Flash-BF16"; do
    [[ -n "${c}" && -d "${c}" ]] || continue
    MODEL_DIR="${c}"
    if has_model; then
      MODEL_DIR="$(cd "${c}" && pwd -P)"
      return 0
    fi
  done
  return 1
}

has_dataset() {
  [[ -f "${DATASET_DIR}/${DATASET_CONFIG}/test-00000-of-00001.parquet" ]]
}

pick_dataset() {
  local c
  for c in "${DATASET_DIR}" "${WORK_DIR}/datasets/Salesforce-wikitext" \
           "${WORK_DIR}/datasets/wikitext" \
           "${VLLM_DIR}/../datasets/Salesforce-wikitext"; do
    [[ -n "${c}" && -d "${c}" ]] || continue
    DATASET_DIR="${c}"
    if has_dataset; then
      DATASET_DIR="$(cd "${c}" && pwd -P)"
      return 0
    fi
  done
  return 1
}

# The launcher writes reference logits to this directory; reuse them rather
# than pulling 1.3GB back down from Hugging Face.
pick_local_reference() {
  [[ -z "${REFERENCE}" ]] || return 0
  local dir="${VLLM_DIR}/ref_logits_$(basename "${MODEL_DIR}")"
  dir="${dir}_ctx${CONTEXT_LENGTH}_s${STRIDE}"
  if [[ -f "${dir}/${REFERENCE_FILE}" ]]; then
    REFERENCE="${dir}/${REFERENCE_FILE}"
    log "using local reference ${REFERENCE}"
  fi
}

# GLM-5.3 JIT-compiles TileLang MHC kernels at load time, so the workers need
# nvcc. The capture launcher exports these in its own shell; mirror it here.
resolve_cuda() {
  local c
  if [[ -z "${CUDA_HOME:-}" ]]; then
    for c in ${NVCC_CANDIDATES:-/usr/local/cuda-12.8/bin/nvcc /usr/local/cuda/bin/nvcc}; do
      if [[ -x "${c}" ]]; then
        CUDA_HOME="$(cd "$(dirname "${c}")/.." && pwd -P)"
        break
      fi
    done
  fi
  if [[ -z "${CUDA_HOME:-}" ]]; then
    log "WARNING: no nvcc found; TileLang MHC JIT will likely fail"
    return 0
  fi
  export CUDA_HOME
  export CUDACXX="${CUDACXX:-${CUDA_HOME}/bin/nvcc}"
  case ":${PATH}:" in
    *":${CUDA_HOME}/bin:"*) ;;
    *) export PATH="${CUDA_HOME}/bin:${PATH}" ;;
  esac
  log "CUDA_HOME=${CUDA_HOME} nvcc=$("${CUDACXX}" --version 2>/dev/null \
      | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -1)"
}

resolve_layout() {
  log "resolving layout"
  resolve_cuda
  pick_venv || die "no venv with bin/python found; run the capture launcher first"
  PYTHON="${PYTHON:-${VENV_DIR}/bin/python}"
  # The launcher sources the venv, which puts ninja and other console scripts on
  # PATH. We invoke the venv python directly, so add it ourselves; torch's JIT
  # extension loader shells out to `ninja` for the TileLang MHC kernels.
  case ":${PATH}:" in
    *":${VENV_DIR}/bin:"*) ;;
    *) export PATH="${VENV_DIR}/bin:${PATH}" ;;
  esac
  if ! command -v ninja >/dev/null 2>&1; then
    log "ninja not on PATH; installing into ${VENV_DIR}"
    "${PYTHON}" -m pip install -q ninja \
      || log "WARNING: ninja install failed; TileLang JIT will fail"
  fi
  log "ninja=$(command -v ninja || echo MISSING)"
  pick_model || die "no complete model dir found (tried ${WORK_DIR}/models/...)"
  pick_dataset || die "no ${DATASET_CONFIG} parquet found (tried ${WORK_DIR}/datasets/...)"

  if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((TENSOR_PARALLEL_SIZE - 1)))"
  fi
  GPUS="${GPUS:-${CUDA_VISIBLE_DEVICES}}"

  log "WORK_DIR=${WORK_DIR}"
  log "VLLM_DIR=${VLLM_DIR}"
  log "VENV_DIR=${VENV_DIR}"
  log "MODEL_DIR=${MODEL_DIR}"
  log "DATASET_DIR=${DATASET_DIR}"
  log "TP=${TENSOR_PARALLEL_SIZE} CUDA_VISIBLE_DEVICES=${GPUS} CTX=${CONTEXT_LENGTH}"
}

# --------------------------------------------------------------------------
# Step 1: capture everything needed to read the overlay offline.
# --------------------------------------------------------------------------
collect_environment() {
  log "collecting overlay source and environment"
  mkdir -p "${DIAG_DIR}"
  cd "${VLLM_DIR}"

  # Paths taken from the overlay list in packet_capture_glm53_flash_logits.sh.
  # sparse_attn_indexer_kpool.py is the prime suspect for the 128 boundary.
  cat > "${DIAG_DIR}/overlay-filelist.txt" <<'EOF'
vllm/models/glm5next
vllm/model_executor/layers/sparse_attn_indexer_kpool.py
vllm/third_party/flash_linear_attention/ops/kda.py
vllm/third_party/flash_linear_attention/ops/fused_recurrent.py
vllm/model_executor/layers/mamba
vllm/model_executor/layers/mla.py
vllm/model_executor/layers/mhc.py
vllm/model_executor/layers/attention/mla_attention.py
vllm/model_executor/models/registry.py
vllm/model_executor/warmup/kernel_warmup.py
vllm/transformers_utils/configs/glm5_next.py
vllm/transformers_utils/config.py
vllm/v1/attention/backends
vllm/v1/core/kv_cache_coordinator.py
vllm/v1/core/kv_cache_utils.py
vllm/v1/core/single_type_kv_cache_manager.py
vllm/v1/core/sched/scheduler.py
vllm/v1/kv_cache_interface.py
vllm/v1/worker/gpu_model_runner.py
vllm/v1/worker/gpu/model_runner.py
vllm/v1/worker/utils.py
vllm/config
vllm/inputs/engine.py
vllm/renderers/base.py
vllm/v1/engine/input_processor.py
examples/offline_inference/score_mode_kld.py
EOF
  tar czf "${DIAG_DIR}/overlay-src.tar.gz" --ignore-failed-read \
    -T "${DIAG_DIR}/overlay-filelist.txt" 2>"${DIAG_DIR}/overlay-tar.err" \
    || echo "overlay tar reported missing paths; see overlay-tar.err"

  # Safety net for anything the explicit list misses.
  grep -rIl -e kpool -e KDA -e kda_ -e Glm5 -e glm5 \
    vllm/ --include='*.py' > "${DIAG_DIR}/overlay-grep-hits.txt" 2>/dev/null || true
  echo "grep-matched overlay files: $(wc -l < "${DIAG_DIR}/overlay-grep-hits.txt")"
  if [[ -s "${DIAG_DIR}/overlay-grep-hits.txt" ]]; then
    tar czf "${DIAG_DIR}/overlay-grep-src.tar.gz" --ignore-failed-read \
      -T "${DIAG_DIR}/overlay-grep-hits.txt" 2>/dev/null || true
  fi

  cp "${MODEL_DIR}/config.json" "${DIAG_DIR}/model-config.json" 2>/dev/null || true

  git log --oneline -30      > "${DIAG_DIR}/git-log.txt"      2>&1 || true
  git status --porcelain     > "${DIAG_DIR}/git-status.txt"   2>&1 || true
  git diff                   > "${DIAG_DIR}/git-diff.txt"     2>&1 || true
  git diff --stat HEAD       > "${DIAG_DIR}/git-diffstat.txt" 2>&1 || true

  "${PYTHON}" -m pip freeze  > "${DIAG_DIR}/pip-freeze.txt"   2>&1 || true
  nvidia-smi                 > "${DIAG_DIR}/nvidia-smi.txt"   2>&1 || true
  nvidia-smi topo -m         > "${DIAG_DIR}/nvidia-topo.txt"  2>&1 || true

  # A toolchain change between capture and comparison would confound the
  # cross-day [vs-ref] numbers, so record it explicitly.
  {
    echo "CUDA_HOME=${CUDA_HOME:-unset}"
    echo "CUDACXX=${CUDACXX:-unset}"
    echo "which nvcc: $(command -v nvcc 2>/dev/null || echo none)"
    nvcc --version 2>&1 || echo "nvcc unavailable"
    ls -d /usr/local/cuda-* 2>/dev/null || true
    "${PYTHON}" -c 'import torch; print("torch", torch.__version__,
                                        "cuda", torch.version.cuda)' 2>&1
  } > "${DIAG_DIR}/toolchain.txt" || true
  cp "${PROBE}" "${DIAG_DIR}/" 2>/dev/null || true
  cp "${BISECT}" "${DIAG_DIR}/" 2>/dev/null || true
}

# --------------------------------------------------------------------------
# Step 2: a corrupted worker file cost us a full session; fail fast instead.
# --------------------------------------------------------------------------
verify_tree() {
  log "verifying the vLLM tree compiles"
  cd "${VLLM_DIR}"
  "${PYTHON}" -m compileall -q vllm > "${DIAG_DIR}/compileall.txt" 2>&1 \
    || { cat "${DIAG_DIR}/compileall.txt"; die "vLLM tree has syntax errors"; }
  echo "tree compiles clean"
}

maybe_fetch_reference() {
  pick_local_reference
  [[ -z "${REFERENCE}" ]] || return 0
  [[ -n "${REFERENCE_REPO}" ]] || return 0
  log "fetching ${REFERENCE_FILE} from ${REFERENCE_REPO}"
  local dest
  if dest="$(REPO="${REFERENCE_REPO}" FILE="${REFERENCE_FILE}" \
        DEST="${DIAG_DIR}/reference" "${PYTHON}" -c '
import os
from huggingface_hub import hf_hub_download
print(hf_hub_download(repo_id=os.environ["REPO"],
                      filename=os.environ["FILE"],
                      local_dir=os.environ["DEST"],
                      token=os.environ.get("HF_TOKEN") or None))
' 2>"${DIAG_DIR}/reference-download.err")"; then
    REFERENCE="${dest}"
    log "reference at ${REFERENCE}"
  else
    echo "reference download failed (see reference-download.err); continuing"
  fi
}

# --------------------------------------------------------------------------
# torch.use_deterministic_algorithms() must be set inside the TP workers, and
# spawned workers do not inherit it from the parent. A gated .pth runs at
# interpreter startup in every process, including workers. Side benefit: in
# deterministic mode torch.empty() fills with NaN, so a pooled-cache read of
# uninitialized memory shows up as NaN logits.
# --------------------------------------------------------------------------
install_determinism_pth() {
  local site
  site="$("${PYTHON}" -c 'import site; print(site.getsitepackages()[0])')"
  PTH_FILE="${site}/zzz_glm53_determinism.pth"
  cat > "${PTH_FILE}" <<'EOF'
import os; exec("if os.environ.get('GLM53_TORCH_DETERMINISTIC') == '1':\n    import torch\n    torch.use_deterministic_algorithms(True, warn_only=True)\n    torch.utils.deterministic.fill_uninitialized_memory = True\n")
EOF
  echo "installed ${PTH_FILE}"
}

remove_determinism_pth() {
  [[ -n "${PTH_FILE}" && -f "${PTH_FILE}" ]] || return 0
  rm -f "${PTH_FILE}"
  echo "removed ${PTH_FILE}"
  PTH_FILE=""
}

trap remove_determinism_pth EXIT

run_config() {
  local name="$1" extra_env="$2" extra_args="$3"
  local logf="${OUT_DIR}/${name}.log"
  log "config: ${name}  env[${extra_env}]  args[${extra_args}]"

  local ref_args=()
  [[ -n "${REFERENCE}" ]] && ref_args=(--reference "${REFERENCE}")

  cd "${VLLM_DIR}"
  set +e
  # shellcheck disable=SC2086
  env \
    CUDA_VISIBLE_DEVICES="${GPUS}" \
    CUDA_HOME="${CUDA_HOME:-}" \
    CUDACXX="${CUDACXX:-}" \
    VLLM_USE_V2_MODEL_RUNNER=0 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    VLLM_ALLREDUCE_USE_FLASHINFER=0 \
    VLLM_USE_DEEP_GEMM=0 \
    VLLM_MOE_USE_DEEP_GEMM=0 \
    VLLM_DEEP_GEMM_WARMUP=skip \
    VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}" \
    ${extra_env} \
    "${PYTHON}" "${PROBE}" \
      --config-name "${name}" \
      --model "${MODEL_DIR}" \
      --dataset-dir "${DATASET_DIR}" \
      --dataset-config "${DATASET_CONFIG}" \
      --out-dir "${OUT_DIR}" \
      --ctx "${CONTEXT_LENGTH}" \
      --stride "${STRIDE}" \
      --tp "${TENSOR_PARALLEL_SIZE}" \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
      --moe-backend "${MOE_BACKEND}" \
      --repeats "${REPEATS}" \
      --lengths "${LENGTHS}" \
      ${ref_args[@]+"${ref_args[@]}"} \
      ${extra_args} \
    2>&1 | tee "${logf}"
  local rc=${PIPESTATUS[0]}
  set -e

  {
    echo ""
    echo "----- ${name} (exit ${rc}) -----"
    grep -E '^\[(math|sweep|capture|kld|vs-ref|[0-9]+v[0-9]+)' "${logf}" || true
  } >> "${OUT_DIR}/SUMMARY.txt"

  # Nondeterministic-op warnings name the offending kernel outright.
  grep -h -o 'does not have a deterministic implementation[^"]*' "${logf}" \
    | sort -u >> "${OUT_DIR}/nondeterministic-ops.txt" 2>/dev/null || true

  [[ ${rc} -eq 0 ]] || echo "config ${name} failed (exit ${rc}); continuing"
  LAST_RC=${rc}
}

# Localize the divergence to a single module instead of a single logit row.
run_bisect() {
  local extra_env="${1:-}"
  local logf="${OUT_DIR}/layer_bisect.log"
  [[ -f "${BISECT}" ]] || { echo "missing bisect script: ${BISECT}" >&2; return 0; }
  log "layer bisect  env[${extra_env}]"

  cd "${VLLM_DIR}"
  set +e
  # shellcheck disable=SC2086
  env \
    CUDA_VISIBLE_DEVICES="${GPUS}" \
    CUDA_HOME="${CUDA_HOME:-}" \
    CUDACXX="${CUDACXX:-}" \
    VLLM_USE_V2_MODEL_RUNNER=0 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    VLLM_ALLREDUCE_USE_FLASHINFER=0 \
    VLLM_USE_DEEP_GEMM=0 \
    VLLM_MOE_USE_DEEP_GEMM=0 \
    VLLM_DEEP_GEMM_WARMUP=skip \
    VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}" \
    VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
    ${extra_env} \
    "${PYTHON}" "${BISECT}" \
      --model "${MODEL_DIR}" \
      --dataset-dir "${DATASET_DIR}" \
      --dataset-config "${DATASET_CONFIG}" \
      --out-dir "${OUT_DIR}" \
      --length "${BISECT_LENGTH:-130}" \
      --tp "${TENSOR_PARALLEL_SIZE}" \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
      --moe-backend "${MOE_BACKEND}" \
      ${BISECT_PATTERN:+--pattern "${BISECT_PATTERN}"} \
      ${BISECT_CONTROL:+--control} \
      ${BISECT_FINGERPRINT:+--fingerprint-mqa} \
    2>&1 | tee "${logf}"
  local rc=${PIPESTATUS[0]}
  set -e

  {
    echo ""
    echo "----- layer_bisect (exit ${rc}) -----"
    grep -E '^\[(hook|result|done|warn|control|unchecked)' "${logf}" || true
  } >> "${OUT_DIR}/SUMMARY.txt"

  [[ ${rc} -eq 0 ]] || echo "layer bisect failed (exit ${rc}); continuing"
}

main() {
  [[ -f "${PROBE}" ]] || die "missing probe: ${PROBE}"
  [[ -d "${VLLM_DIR}/vllm" ]] || die "not a vLLM clone: ${VLLM_DIR}"
  mkdir -p "${OUT_DIR}"

  resolve_layout
  collect_environment
  verify_tree
  maybe_fetch_reference

  : > "${OUT_DIR}/SUMMARY.txt"
  : > "${OUT_DIR}/nondeterministic-ops.txt"

  # batch_invariant is deliberately absent: GLM-5.3-Flash needs a sparse MLA
  # backend, and no sparse backend reports supports_batch_invariance(), so the
  # engine cannot start. See the case below.
  local configs="${CONFIGS:-baseline bisect no_fi_autotune torch_deterministic chunked_512 single_chunk}"
  for cfg in ${configs}; do
    case "${cfg}" in
      baseline)
        run_config baseline "" "" ;;
      batch_invariant)
        # Known to fail at startup on GLM-5.3-Flash: the model requires
        # use_sparse=True, and every sparse MLA backend declines batch
        # invariance while every batch-invariant backend declines sparse.
        # TRITON_MLA is the near miss - its only objection is "sparse not
        # supported". Kept selectable so the failure can be re-confirmed.
        echo "note: batch_invariant cannot start on a sparse-MLA model" \
             "(no backend satisfies use_sparse=True + use_batch_invariant=True)"
        run_config batch_invariant "VLLM_BATCH_INVARIANT=1" "" ;;
      torch_deterministic)
        install_determinism_pth
        run_config torch_deterministic \
          "GLM53_TORCH_DETERMINISTIC=1 CUBLAS_WORKSPACE_CONFIG=:4096:8" ""
        remove_determinism_pth ;;
      batch_invariant_deterministic)
        install_determinism_pth
        run_config batch_invariant_deterministic \
          "VLLM_BATCH_INVARIANT=1 GLM53_TORCH_DETERMINISTIC=1 CUBLAS_WORKSPACE_CONFIG=:4096:8" ""
        remove_determinism_pth ;;
      no_fi_autotune)
        run_config no_fi_autotune "" "--disable-flashinfer-autotune" ;;
      chunked_512)
        run_config chunked_512 "" "--max-num-batched-tokens 512" ;;
      single_chunk)
        run_config single_chunk "" \
          "--max-num-batched-tokens ${CONTEXT_LENGTH}" ;;
      gate)
        # Precondition for a capture, not a diagnostic: exit 3 means the
        # reference logits this rig would produce are not reproducible.
        run_config gate "${GATE_ENV:-}" \
          "--gate --gate-max-self-kld ${GATE_MAX_SELF_KLD:-0.0} ${GATE_ARGS:-}"
        GATE_RC=${LAST_RC} ;;
      bisect)
        run_bisect "${BISECT_ENV:-}" ;;
      *)
        echo "unknown config: ${cfg}" >&2 ;;
    esac
  done

  log "packaging results"
  tar czf "${OUT_DIR}/../glm53-determinism-results.tar.gz" \
    -C "$(dirname "${OUT_DIR}")" "$(basename "${OUT_DIR}")"
  echo ""
  cat "${OUT_DIR}/SUMMARY.txt"
  echo ""
  log "done"
  echo "results:  ${OUT_DIR}"
  echo "tarball:  $(cd "${OUT_DIR}/.." && pwd)/glm53-determinism-results.tar.gz"

  if [[ -n "${GATE_RC}" && "${GATE_RC}" != "0" ]]; then
    echo "GATE FAILED (exit ${GATE_RC}): this rig is not fit to capture" \
         "reference logits" >&2
    return 3
  fi
}

main "$@"
