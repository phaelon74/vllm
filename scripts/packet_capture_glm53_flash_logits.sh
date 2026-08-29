#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Capture GLM-5.3-Flash BF16 WikiText reference logits on a Packet.AI GPU node
# and upload them to Hugging Face as a private repo.
#
# Packet.AI notes (research, Aug 2026):
#   Target SKU: **4x NVIDIA B200 Dedicated** (~720 GB HBM, ~720 GB RAM, ~1.4 TB
#   local NVMe, ~$23.60/hr). That is enough VRAM for this capture job (weights
#   ~643 GB + a few GB of KV/logits). Do not use 1x Dynamic/Dedicated B200.
#   8x B200 is nicer disk/RAM headroom but is often sold out.
#   Weights and WikiText are pulled with `hf download --local-dir` so they
#   land only under WORK_DIR (no ~/.cache/huggingface/hub blob copy).
#   Native GLM-5.3-Flash is overlaid from vllm-project/vllm#53906 (not on main
#   yet). gpu_model_runner.py is left untouched so score mode stays intact,
#   then patched to pass KVBlockZeroer(num_blocks=...) from the PR.
#   After overlay, KLD scheduler output plumbing is reapplied, and score/KLD
#   TokensPrompt fields are forwarded through the renderer into EngineInput.
#   TileLang MHC needs CUDA Toolkit 12.8+ (sm_100a). The script checks nvcc,
#   removes a duplicate CUDA apt source if present, and installs the compiler
#   packages (cuda-nvcc + cuda-cudart-dev), not the full toolkit.
#   FlashInfer TRTLLM-GEN BF16 MoE is skipped (VLLM_MOE_BACKEND=triton): the
#   sm100f cubin segfaults in cuModuleGetFunction on this host.
#   Capture uses the original two-phase score_mode_kld.py flow (Phase 1
#   logits, then Phase 2 KLD). Do not pass --capture-only.
#   CUDA/drivers are preinstalled. This script runs on the VM host.
#   Docs: https://packet.ai/cli
#         https://packet.ai/bare-metal-gpu-servers
#         https://packet.ai/features
#
# Do NOT put your Hugging Face token in this file. From the clone:
#   export HF_TOKEN=hf_...
#   bash scripts/packet_capture_glm53_flash_logits.sh
# Or from ~/glm53-flash-kld/vllm (pwd you showed): same command; WORK_DIR
# is detected as the parent so .venv and models/ are reused.
#
# Re-run the same command to resume. Each step is skipped when its output
# already exists (venv, vLLM import, GLM-5.3 native overlay from PR 53906,
# weight shards, WikiText dataset, CAPTURE_COMPLETE, UPLOAD_COMPLETE).
# Optional overrides:
#   FORCE_INSTALL=1  FORCE_GLM53_PR=1  FORCE_NVCC=1  FORCE_DOWNLOAD=1  FORCE_DATASET=1  FORCE_CAPTURE=1  FORCE_UPLOAD=1
#   SKIP_INSTALL=1   SKIP_GLM53_PR=1   SKIP_NVCC=1   SKIP_DOWNLOAD=1   SKIP_DATASET=1   SKIP_GATE=1   SKIP_CAPTURE=1   SKIP_UPLOAD=1
#   GATE_ENFORCE=0 downgrades a failed determinism gate to a warning.

set -euo pipefail

HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
HF_USER="${HF_USER:-phaedawg}"
HF_REPO="${HF_REPO:-${HF_USER}/ref-logits-glm-5.3-flash}"
HF_REPO_TYPE="${HF_REPO_TYPE:-model}"

VLLM_GIT_URL="${VLLM_GIT_URL:-https://github.com/phaelon74/vllm.git}"
VLLM_GIT_BRANCH="${VLLM_GIT_BRANCH:-feature/score-mode-ppl-kld}"
# Native GLM-5.3-Flash is not on public vLLM main yet (open PR 53906).
GLM53_PR_GIT="${GLM53_PR_GIT:-https://github.com/vllm-project/vllm.git}"
GLM53_PR_REF="${GLM53_PR_REF:-pull/53906/head}"
GLM53_PR_NUMBER="${GLM53_PR_NUMBER:-53906}"

HF_MODEL_ID="${HF_MODEL_ID:-zai-org/GLM-5.3-Flash-BF16}"
HF_DATASET_ID="${HF_DATASET_ID:-Salesforce/wikitext}"
DATASET_CONFIG="${DATASET_CONFIG:-wikitext-2-raw-v1}"
# If you launch from ~/glm53-flash-kld/vllm (or via scripts/ in the clone),
# detect WORK_DIR as the parent so an existing venv + weights are reused.
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_DETECTED_VLLM=""
_DETECTED_WORK=""
if [[ -f "${PWD}/examples/offline_inference/score_mode_kld.py" && -d "${PWD}/vllm" ]]; then
  _DETECTED_VLLM="$(pwd -P)"
  _DETECTED_WORK="$(cd "${_DETECTED_VLLM}/.." && pwd -P)"
elif [[ -f "${_SCRIPT_DIR}/../examples/offline_inference/score_mode_kld.py" && -d "${_SCRIPT_DIR}/../vllm" ]]; then
  _DETECTED_VLLM="$(cd "${_SCRIPT_DIR}/.." && pwd -P)"
  _DETECTED_WORK="$(cd "${_DETECTED_VLLM}/.." && pwd -P)"
fi
WORK_DIR="${WORK_DIR:-${_DETECTED_WORK:-${HOME}/glm53-flash-kld}}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-2048}"
STRIDE="${STRIDE:-512}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
# 4x B200 Dedicated has ~1.4 TB NVMe. Weights ~643 GB + logits ~80-130 GB.
MIN_FREE_GB="${MIN_FREE_GB:-850}"
# Compiler only (not cuda-toolkit-*-*). 12.8 is the Blackwell/GLM-5.3 floor.
CUDA_APT_SERIES="${CUDA_APT_SERIES:-12-8}"
# The full toolkit, not just the compiler. FlashInfer JIT-builds its sampling
# and sparse-MLA modules at runtime and includes curand.h, cublasLt.h and
# nvrtc.h; a compiler-only install fails mid-run, after weights are loaded.
# Override with the compiler-only set if the box already has full CUDA headers:
#   CUDA_APT_PACKAGES="cuda-nvcc-12-8 cuda-cudart-dev-12-8"
CUDA_APT_PACKAGES="${CUDA_APT_PACKAGES:-cuda-toolkit-${CUDA_APT_SERIES}}"
# Preference order for a system nvcc. Pin this when a run must reproduce an
# earlier run's numerics, since a toolchain change is a confounder:
#   NVCC_CANDIDATES=/usr/local/cuda-12.8/bin/nvcc NVCC_STRICT=1
NVCC_CANDIDATES="${NVCC_CANDIDATES:-/usr/local/cuda-13.0/bin/nvcc /usr/local/cuda-12.9/bin/nvcc /usr/local/cuda-12.8/bin/nvcc /usr/local/cuda/bin/nvcc}"
# 1 = never fall back to whatever nvcc is on PATH.
NVCC_STRICT="${NVCC_STRICT:-0}"

SKIP_INSTALL="${SKIP_INSTALL:-0}"
SKIP_GLM53_PR="${SKIP_GLM53_PR:-0}"
SKIP_NVCC="${SKIP_NVCC:-0}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
SKIP_DATASET="${SKIP_DATASET:-0}"
SKIP_CAPTURE="${SKIP_CAPTURE:-0}"
SKIP_UPLOAD="${SKIP_UPLOAD:-0}"
# Reference logits are only worth capturing if this rig reproduces itself
# bit-for-bit. The gate proves that before spending hours on a capture.
SKIP_GATE="${SKIP_GATE:-0}"
GATE_ENFORCE="${GATE_ENFORCE:-1}"
GATE_LENGTHS="${GATE_LENGTHS:-129,130}"
GATE_REPEATS="${GATE_REPEATS:-2}"
GATE_MAX_SELF_KLD="${GATE_MAX_SELF_KLD:-0.0}"
FORCE_INSTALL="${FORCE_INSTALL:-0}"
FORCE_GLM53_PR="${FORCE_GLM53_PR:-0}"
FORCE_NVCC="${FORCE_NVCC:-0}"
FORCE_DOWNLOAD="${FORCE_DOWNLOAD:-0}"
FORCE_DATASET="${FORCE_DATASET:-0}"
FORCE_CAPTURE="${FORCE_CAPTURE:-0}"
FORCE_UPLOAD="${FORCE_UPLOAD:-0}"

export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
unset HF_HUB_ENABLE_HF_TRANSFER || true
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# Host nvcc on Packet often cannot compile sm_100a (compute_100a).
# Skip optional JIT compilers; they are not the KLD math.
export VLLM_ALLREDUCE_USE_FLASHINFER="${VLLM_ALLREDUCE_USE_FLASHINFER:-0}"
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
export VLLM_MOE_USE_DEEP_GEMM="${VLLM_MOE_USE_DEEP_GEMM:-0}"
export VLLM_DEEP_GEMM_WARMUP="${VLLM_DEEP_GEMM_WARMUP:-skip}"
# Blackwell auto-selects FlashInfer TRTLLM-GEN BF16 MoE (sm100f cubin).
# That path segfaults in cuModuleGetFunction on this Packet host.
# Triton is the portable unquantized MoE kernel; not KLD math.
export VLLM_MOE_BACKEND="${VLLM_MOE_BACKEND:-triton}"
export HF_TOKEN
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN}}"

log() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
die() { log "ERROR: $*"; exit 1; }

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing command: $1"
}

free_gb() {
  df -BG --output=avail "${1}" | tail -1 | tr -dc '0-9'
}

nvcc_release() {
  local bin="${1:-nvcc}"
  if command -v "${bin}" >/dev/null 2>&1; then
    bin="$(command -v "${bin}")"
  elif [[ ! -x "${bin}" ]]; then
    return 1
  fi
  "${bin}" --version 2>/dev/null \
    | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' \
    | head -1
}

nvcc_ge_12_8() {
  local ver="${1:-}"
  [[ -n "${ver}" ]] || return 1
  local maj="${ver%%.*}"
  local min="${ver#*.}"
  min="${min%%.*}"
  [[ "${maj}" -gt 12 ]] || { [[ "${maj}" -eq 12 ]] && [[ "${min}" -ge 8 ]]; }
}

use_nvcc() {
  local bin="$1"
  local prefix
  [[ -x "${bin}" ]] || return 1
  nvcc_ge_12_8 "$(nvcc_release "${bin}")" || return 1
  prefix="$(cd "$(dirname "${bin}")/.." && pwd)"
  export CUDA_HOME="${prefix}"
  export CUDACXX="${bin}"
  export PATH="${prefix}/bin:${PATH}"
  log "CUDA_HOME=${CUDA_HOME} nvcc=$(nvcc_release "${bin}") (${bin})"
}

find_and_use_system_nvcc() {
  local p
  for p in ${NVCC_CANDIDATES}; do
    if use_nvcc "${p}"; then
      return 0
    fi
  done
  if [[ "${NVCC_STRICT}" == "1" ]]; then
    log "NVCC_STRICT=1 and no candidate matched: ${NVCC_CANDIDATES}"
    return 1
  fi
  if command -v nvcc >/dev/null 2>&1; then
    use_nvcc "$(command -v nvcc)" && return 0
  fi
  return 1
}

cuda_apt_repo_id() {
  # ubuntu + 24.04 -> ubuntu2404
  # shellcheck disable=SC1091
  . /etc/os-release
  printf '%s%s' "${ID}" "${VERSION_ID//./}"
}

fix_cuda_apt_conflicts() {
  local f
  # Packet ships cuda.list without Signed-By; cuda-keyring adds a second
  # list for the same URL. apt then refuses to read sources.
  if [[ -f /etc/apt/sources.list.d/cuda.list ]]; then
    log "removing duplicate CUDA apt source /etc/apt/sources.list.d/cuda.list"
    sudo rm -f /etc/apt/sources.list.d/cuda.list
  fi
  for f in /etc/apt/sources.list.d/cuda*.list /etc/apt/sources.list.d/cuda*.sources; do
    [[ -e "${f}" ]] || continue
    [[ "${f}" == */cuda-ubuntu*-x86_64.list ]] && continue
    if grep -q "developer.download.nvidia.com/compute/cuda" "${f}" 2>/dev/null; then
      if ! grep -q "signed-by=" "${f}"; then
        log "removing unsigned CUDA apt source ${f}"
        sudo rm -f "${f}"
      fi
    fi
  done
}

install_cuda_compiler_apt() {
  local repo_id keyring_deb
  command -v sudo >/dev/null 2>&1 || die "sudo required to install CUDA toolkit compiler"
  repo_id="$(cuda_apt_repo_id)"
  keyring_deb="${WORK_DIR}/cuda-keyring_1.1-1_all.deb"
  log "installing CUDA compiler packages: ${CUDA_APT_PACKAGES} (not full cuda-toolkit)"
  if [[ ! -f /usr/share/keyrings/cuda-archive-keyring.gpg ]]; then
    curl -fsSL -o "${keyring_deb}" \
      "https://developer.download.nvidia.com/compute/cuda/repos/${repo_id}/x86_64/cuda-keyring_1.1-1_all.deb"
    sudo dpkg -i "${keyring_deb}"
  fi
  fix_cuda_apt_conflicts
  sudo apt-get update
  # shellcheck disable=SC2086
  sudo DEBIAN_FRONTEND=noninteractive apt-get -y install ${CUDA_APT_PACKAGES}
}

find_pip_nvcc() {
  local py="${PYTHON:-${VENV_DIR}/bin/python}"
  [[ -x "${py}" ]] || return 1
  "${py}" - <<'PY'
import glob
import os
import sysconfig
sp = sysconfig.get_paths()["purelib"]
hits = sorted(
    p for p in glob.glob(os.path.join(sp, "nvidia", "**", "nvcc"), recursive=True)
    if os.path.isfile(p) and os.access(p, os.X_OK) and f"{os.sep}bin{os.sep}" in p
)
print(hits[0] if hits else "")
PY
}

ensure_sm100a_nvcc() {
  local ver pip_nvcc pip_bin shim torch_cuda
  if [[ "${FORCE_NVCC}" != "1" ]] && find_and_use_system_nvcc; then
    return 0
  fi

  log "no CUDA Toolkit 12.8+ nvcc on PATH; installing compiler via apt"
  install_cuda_compiler_apt
  if find_and_use_system_nvcc; then
    return 0
  fi

  pip_nvcc="$(find_pip_nvcc || true)"
  if [[ "${FORCE_NVCC}" == "1" || -z "${pip_nvcc}" ]]; then
    command -v uv >/dev/null 2>&1 || die "uv not on PATH; cannot install pip nvcc"
    torch_cuda="$("${PYTHON}" -c 'import torch; print(torch.version.cuda or "")' 2>/dev/null || true)"
    log "apt nvcc still missing; installing pip nvcc (torch CUDA ${torch_cuda:-unknown})"
    if [[ "${torch_cuda}" == 13* ]]; then
      uv pip install -U "nvidia-cuda-nvcc-cu13" "nvidia-cuda-nvrtc-cu13"
    else
      uv pip install -U "nvidia-cuda-nvcc-cu12>=12.8" "nvidia-cuda-nvrtc-cu12>=12.8"
    fi
    pip_nvcc="$(find_pip_nvcc || true)"
  fi
  [[ -n "${pip_nvcc}" && -x "${pip_nvcc}" ]] \
    || die "no pip nvcc after install; cannot compile TileLang sm_100a"

  pip_bin="$(dirname "${pip_nvcc}")"
  ver="$(nvcc_release "${pip_nvcc}")"
  nvcc_ge_12_8 "${ver}" || die "pip nvcc is ${ver:-unknown}; need >= 12.8 for sm_100a"

  shim="${WORK_DIR}/.cuda-sm100a"
  mkdir -p "${shim}/bin"
  ln -sfn "${pip_nvcc}" "${shim}/bin/nvcc"
  if [[ -x "${pip_bin}/ptxas" ]]; then
    ln -sfn "${pip_bin}/ptxas" "${shim}/bin/ptxas"
  fi
  if [[ -d /usr/local/cuda-12.8/include ]]; then
    ln -sfn /usr/local/cuda-12.8/include "${shim}/include"
  elif [[ -d /usr/local/cuda/include ]]; then
    ln -sfn /usr/local/cuda/include "${shim}/include"
  fi
  export CUDA_HOME="${shim}"
  export CUDACXX="${shim}/bin/nvcc"
  export PATH="${shim}/bin:${PATH}"
  log "CUDA_HOME=${CUDA_HOME} nvcc=$(nvcc_release nvcc) ($(command -v nvcc))"
}

if [[ -z "${HF_TOKEN}" ]]; then
  die "HF_TOKEN is empty. On the VM run: export HF_TOKEN=hf_your_token"
fi

need_cmd nvidia-smi
need_cmd git
need_cmd curl
need_cmd python3

GPU_COUNT="$(nvidia-smi -L | wc -l | tr -d ' ')"
log "nvidia-smi reports ${GPU_COUNT} GPU(s)"
nvidia-smi -L || true
if command -v nvcc >/dev/null 2>&1; then
  log "nvcc: $(nvcc --version | tr '\n' ' ' | sed 's/  */ /g')"
  log "VLLM_ALLREDUCE_USE_FLASHINFER=${VLLM_ALLREDUCE_USE_FLASHINFER} VLLM_USE_DEEP_GEMM=${VLLM_USE_DEEP_GEMM}"
fi
if [[ "${GPU_COUNT}" -lt "${TENSOR_PARALLEL_SIZE}" ]]; then
  die "need at least ${TENSOR_PARALLEL_SIZE} GPUs for TP=${TENSOR_PARALLEL_SIZE}; saw ${GPU_COUNT}"
fi

# 4x B200 node: use all four cards. On an 8x node this still pins 0-3.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((TENSOR_PARALLEL_SIZE - 1)))"
fi
log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

mkdir -p "${WORK_DIR}"
WORK_DIR="$(cd "${WORK_DIR}" && pwd)"
VLLM_DIR="${VLLM_DIR:-${_DETECTED_VLLM:-${WORK_DIR}/vllm}}"
MODEL_DIR="${MODEL_DIR:-${WORK_DIR}/models/GLM-5.3-Flash-BF16}"
DATASET_DIR="${DATASET_DIR:-${WORK_DIR}/datasets/Salesforce-wikitext}"
LOGITS_DIR="${VLLM_DIR}/ref_logits_$(basename "${MODEL_DIR}")_ctx${CONTEXT_LENGTH}_s${STRIDE}"
VENV_DIR="${VENV_DIR:-}"
CAPTURE_MARKER="${LOGITS_DIR}/CAPTURE_COMPLETE"
UPLOAD_MARKER="${LOGITS_DIR}/UPLOAD_COMPLETE"

log "WORK_DIR=${WORK_DIR}"
log "VLLM_DIR=${VLLM_DIR}"

# WikiText via `datasets` is small; keep it on the data volume, not $HOME.
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${WORK_DIR}/.cache/huggingface/datasets}"
mkdir -p "${HF_DATASETS_CACHE}"

has_venv() {
  [[ -x "${VENV_DIR}/bin/python" ]]
}

pick_venv() {
  local c
  for c in \
    "${VENV_DIR}" \
    "${VIRTUAL_ENV:-}" \
    "${WORK_DIR}/.venv" \
    "${VLLM_DIR}/.venv" \
    "${HOME}/glm53-flash-kld/.venv"
  do
    if [[ -n "${c}" && -x "${c}/bin/python" ]]; then
      VENV_DIR="$(cd "${c}" && pwd)"
      return 0
    fi
  done
  VENV_DIR="${WORK_DIR}/.venv"
  return 1
}

has_hf_cli() {
  command -v hf >/dev/null 2>&1 \
    && "${PYTHON:-${VENV_DIR}/bin/python}" -c "import huggingface_hub, datasets" 2>/dev/null
}

has_vllm_src() {
  [[ -d "${VLLM_DIR}/.git" ]] \
    && [[ -f "${VLLM_DIR}/examples/offline_inference/score_mode_kld.py" ]]
}

has_glm53_native() {
  [[ -f "${VLLM_DIR}/vllm/models/glm5next/nvidia/model.py" ]] || return 1
  grep -q 'Glm5NextForConditionalGeneration' \
    "${VLLM_DIR}/vllm/model_executor/models/registry.py"
}

glm53_pr_marker() {
  printf '%s' "${VLLM_DIR}/.glm53_pr_${GLM53_PR_NUMBER}.sha"
}

has_vllm_install() {
  local py="${PYTHON:-${VENV_DIR}/bin/python}"
  [[ -x "${py}" ]] || return 1
  # WORK_DIR contains a folder named vllm/ (the clone). Import from /tmp or
  # from inside the clone, never from WORK_DIR.
  ( cd /tmp && "${py}" -c "from vllm import LLM, SamplingParams" ) >/dev/null 2>&1 && return 0
  ( cd "${VLLM_DIR}" && "${py}" -c "from vllm import LLM, SamplingParams" ) >/dev/null 2>&1
}

has_model() {
  [[ -f "${MODEL_DIR}/config.json" ]] || return 1
  [[ -f "${MODEL_DIR}/model.safetensors.index.json" ]] || return 1
  local sample expected have
  sample="$(find "${MODEL_DIR}" -maxdepth 1 -name 'model-*-of-*.safetensors' | head -1 || true)"
  [[ -n "${sample}" ]] || return 1
  expected="$(basename "${sample}" | sed -n 's/.*-of-\([0-9][0-9]*\)\.safetensors$/\1/p')"
  [[ -n "${expected}" ]] || return 1
  expected=$((10#${expected}))
  have="$(find "${MODEL_DIR}" -maxdepth 1 -name 'model-*-of-*.safetensors' | wc -l | tr -d ' ')"
  [[ "${have}" -ge "${expected}" ]]
}

pick_model() {
  local c
  for c in \
    "${MODEL_DIR}" \
    "${WORK_DIR}/models/GLM-5.3-Flash-BF16" \
    "${WORK_DIR}/models/zai-org/GLM-5.3-Flash-BF16" \
    "${VLLM_DIR}/../models/GLM-5.3-Flash-BF16"
  do
    [[ -n "${c}" && -d "${c}" ]] || continue
    MODEL_DIR="${c}"
    if has_model; then
      MODEL_DIR="$(cd "${c}" && pwd)"
      return 0
    fi
  done
  MODEL_DIR="${WORK_DIR}/models/GLM-5.3-Flash-BF16"
  return 1
}

has_dataset() {
  local cfg_dir="${DATASET_DIR}/${DATASET_CONFIG}"
  [[ -f "${DATASET_DIR}/README.md" ]] || return 1
  [[ -d "${cfg_dir}" ]] || return 1
  [[ -f "${cfg_dir}/test-00000-of-00001.parquet" ]] || return 1
  [[ -f "${cfg_dir}/train-00000-of-00001.parquet" ]] || return 1
  [[ -f "${cfg_dir}/validation-00000-of-00001.parquet" ]] || return 1
}

pick_dataset() {
  local c
  for c in \
    "${DATASET_DIR}" \
    "${WORK_DIR}/datasets/Salesforce-wikitext" \
    "${WORK_DIR}/datasets/wikitext" \
    "${VLLM_DIR}/../datasets/Salesforce-wikitext"
  do
    [[ -n "${c}" && -d "${c}" ]] || continue
    DATASET_DIR="${c}"
    if has_dataset; then
      DATASET_DIR="$(cd "${c}" && pwd)"
      return 0
    fi
  done
  DATASET_DIR="${WORK_DIR}/datasets/Salesforce-wikitext"
  return 1
}

has_capture() {
  [[ -f "${CAPTURE_MARKER}" ]]
}

has_upload() {
  [[ -f "${UPLOAD_MARKER}" ]]
}

# Overlay native GLM-5.3-Flash files from PR 53906 onto the KLD clone.
# Shallow clones cannot merge that PR (needs-rebase vs main). Copy the PR
# tree instead, keeping gpu_model_runner.py so score mode is not overwritten.
apply_glm53_pr_overlay() {
  local pr_dir="${WORK_DIR}/.src/vllm-pr-${GLM53_PR_NUMBER}"
  local marker sha copied skipped
  marker="$(glm53_pr_marker)"
  mkdir -p "$(dirname "${pr_dir}")"

  if [[ -d "${pr_dir}/.git" ]]; then
    log "updating GLM-5.3 PR clone ${pr_dir}"
    git -C "${pr_dir}" fetch --depth 1 origin "${GLM53_PR_REF}"
    git -C "${pr_dir}" checkout -f FETCH_HEAD
  else
    log "fetching ${GLM53_PR_GIT} ${GLM53_PR_REF} into ${pr_dir}"
    mkdir -p "${pr_dir}"
    git -C "${pr_dir}" init -q
    git -C "${pr_dir}" remote add origin "${GLM53_PR_GIT}"
    git -C "${pr_dir}" fetch --depth 1 origin "${GLM53_PR_REF}"
    git -C "${pr_dir}" checkout -f FETCH_HEAD
  fi
  sha="$(git -C "${pr_dir}" rev-parse HEAD)"
  log "PR #${GLM53_PR_NUMBER} HEAD ${sha}"
  [[ -d "${pr_dir}/vllm/models/glm5next" ]] \
    || die "PR tree has no vllm/models/glm5next; check ${GLM53_PR_GIT} ${GLM53_PR_REF}"

  copied=0
  skipped=0
  copy_pr_path() {
    local rel="$1"
    if [[ ! -e "${pr_dir}/${rel}" ]]; then
      log "PR tree missing ${rel}; skipping"
      skipped=$((skipped + 1))
      return 0
    fi
    mkdir -p "${VLLM_DIR}/$(dirname "${rel}")"
    cp -a "${pr_dir}/${rel}" "${VLLM_DIR}/${rel}"
    copied=$((copied + 1))
  }

  # New model package: copy the whole tree so later PR files are included.
  mkdir -p "${VLLM_DIR}/vllm/models"
  rm -rf "${VLLM_DIR}/vllm/models/glm5next"
  cp -a "${pr_dir}/vllm/models/glm5next" "${VLLM_DIR}/vllm/models/glm5next"
  copied=$((copied + 1))

  # The model-runner-v2 package must be copied whole. Cherry-picking files out
  # of it mixes PR and branch versions: the PR's gpu/model_runner.py imports
  # DPSyncState from gpu/dp_utils.py, and the PR restructured gpu/spec_decode/.
  # Score mode is untouched because it lives in v1/worker/gpu_model_runner.py,
  # which is outside this directory and deliberately never overlaid.
  if [[ -d "${pr_dir}/vllm/v1/worker/gpu" ]]; then
    rm -rf "${VLLM_DIR}/vllm/v1/worker/gpu"
    cp -a "${pr_dir}/vllm/v1/worker/gpu" "${VLLM_DIR}/vllm/v1/worker/gpu"
    copied=$((copied + 1))
    log "overlaid vllm/v1/worker/gpu as a whole package"
  fi

  # Keep score mode: do not copy vllm/v1/worker/gpu_model_runner.py.
  while IFS= read -r rel; do
    [[ -z "${rel}" || "${rel}" == \#* ]] && continue
    copy_pr_path "${rel}"
  done <<'EOF'
cmake/external_projects/flashmla.cmake
cmake/external_projects/vllm_flash_attn.cmake
csrc/libtorch_stable/cache_kernels.cu
vllm/_aiter_ops.py
vllm/compilation/passes/fusion/allreduce_rms_fusion.py
vllm/config/compilation.py
vllm/config/parallel.py
vllm/config/speculative.py
vllm/config/vllm.py
vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py
vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/coordinator.py
vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py
vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py
vllm/model_executor/layers/attention/mla_attention.py
vllm/model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py
vllm/model_executor/layers/mamba/ops/scatter_states.py
vllm/model_executor/layers/mhc.py
vllm/model_executor/layers/mla.py
vllm/model_executor/layers/quantization/utils/flashinfer_utils.py
vllm/model_executor/layers/sparse_attn_indexer_kpool.py
vllm/model_executor/models/registry.py
vllm/model_executor/warmup/kernel_warmup.py
vllm/model_executor/warmup/spec_decode_rejection_warmup.py
vllm/multimodal/image.py
vllm/multimodal/video.py
vllm/multimodal/video_decoders/opencv.py
vllm/platforms/cuda.py
vllm/platforms/interface.py
vllm/third_party/flash_linear_attention/ops/fused_recurrent.py
vllm/third_party/flash_linear_attention/ops/kda.py
vllm/tilelang_utils/__init__.py
vllm/transformers_utils/config.py
vllm/transformers_utils/configs/__init__.py
vllm/transformers_utils/configs/glm5_next.py
vllm/transformers_utils/model_arch_config_convertor.py
vllm/transformers_utils/processors/__init__.py
vllm/transformers_utils/processors/glm5next.py
vllm/utils/deep_gemm.py
vllm/utils/flashinfer.py
vllm/v1/attention/backends/mla/flashattn_mla_sparse.py
vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py
vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py
vllm/v1/attention/backends/mla/flashmla_sparse.py
vllm/v1/attention/backends/mla/indexer.py
vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py
vllm/v1/attention/backends/mla/sparse_utils.py
vllm/v1/attention/backends/mla/triton_mla.py
vllm/v1/attention/backends/registry.py
vllm/v1/attention/ops/rocm_aiter_mla_sparse.py
vllm/v1/core/kv_cache_coordinator.py
vllm/v1/core/kv_cache_utils.py
vllm/v1/core/sched/scheduler.py
vllm/v1/core/single_type_kv_cache_manager.py
vllm/v1/engine/core.py
vllm/v1/kv_cache_interface.py
vllm/v1/spec_decode/llm_base_proposer.py
vllm/v1/worker/utils.py
vllm/vllm_flash_attn/flash_attn_interface.py
EOF

  log "kept local vllm/v1/worker/gpu_model_runner.py (score mode)"
  skipped=$((skipped + 1))

  has_glm53_native || die "overlay finished but Glm5Next is still not in ${VLLM_DIR}"
  printf '%s\n' "${sha}" > "${marker}"
  log "overlaid ${copied} PR paths onto ${VLLM_DIR} (skipped ${skipped}; SHA ${sha})"
}

# An in-place edit over a laggy SSH session once deleted single characters
# throughout gpu_model_runner.py, and the only symptom was a worker SyntaxError
# several minutes into model load. Git HEAD is authoritative for this file
# because the overlay deliberately never copies it, so restore it whenever it
# stops compiling and let the patch below reapply to clean content.
heal_corrupted_model_runner() {
  local rel="vllm/v1/worker/gpu_model_runner.py"
  [[ -f "${VLLM_DIR}/${rel}" ]] || die "missing ${VLLM_DIR}/${rel}"
  if "${PYTHON:-python3}" -m py_compile "${VLLM_DIR}/${rel}" 2>/dev/null; then
    printf 'ok\n'
    return 0
  fi
  git -C "${VLLM_DIR}" checkout -- "${rel}" \
    || die "${rel} does not compile and could not be restored from git"
  "${PYTHON:-python3}" -m py_compile "${VLLM_DIR}/${rel}" \
    || die "${rel} still does not compile after git restore"
  printf 'restored from git\n'
}

# Compile the whole package before loading ~600 GB of weights. A corrupted file
# otherwise surfaces only as an opaque WorkerProc startup failure.
verify_vllm_syntax() {
  local out
  if out="$("${PYTHON:-python3}" -m compileall -q "${VLLM_DIR}/vllm" 2>&1)"; then
    printf 'ok\n'
    return 0
  fi
  printf '%s\n' "${out}" >&2
  die "vLLM tree has syntax errors (above); restore those files with git checkout"
}

# PR KVBlockZeroer requires num_blocks. Overlay keeps V1 gpu_model_runner.py
# (score mode) and copies PR utils.py, so the call site must be patched even
# when the overlay itself is skipped on resume.
patch_v1_kv_block_zeroer_call() {
  local runner="${VLLM_DIR}/vllm/v1/worker/gpu_model_runner.py"
  [[ -f "${runner}" ]] || die "missing ${runner}"
  "${PYTHON:-python3}" - "${runner}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
old = """        self._kv_block_zeroer = KVBlockZeroer(
            self.device,
            attn_groups_iter=self._kv_cache_spec_attn_group_iterator(),
            kernel_block_sizes=self._kernel_block_sizes,
            runner_only_attn_layers=self.runner_only_attn_layers,
            static_forward_context=self.compilation_config.static_forward_context,
        )"""
new = """        self._kv_block_zeroer = KVBlockZeroer(
            self.device,
            attn_groups_iter=self._kv_cache_spec_attn_group_iterator(),
            kernel_block_sizes=self._kernel_block_sizes,
            runner_only_attn_layers=self.runner_only_attn_layers,
            static_forward_context=self.compilation_config.static_forward_context,
            num_blocks=self.kv_cache_config.num_blocks,
        )"""
if "num_blocks=self.kv_cache_config.num_blocks" in text and "self._kv_block_zeroer = KVBlockZeroer(" in text:
    print("already patched")
    raise SystemExit(0)
if old not in text:
    raise SystemExit(f"could not find KVBlockZeroer call site in {path}")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
print("patched")
PY
}

# score_mode_kld.py has no --moe-backend flag. Honor VLLM_MOE_BACKEND so the
# launcher can force Triton without changing the KLD argv. Always run: overlay
# skip does not refresh examples/.
patch_score_mode_kld_moe_backend_env() {
  local kld="${VLLM_DIR}/examples/offline_inference/score_mode_kld.py"
  [[ -f "${kld}" ]] || die "missing ${kld}"
  "${PYTHON:-python3}" - "${kld}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
if 'os.environ.get("VLLM_MOE_BACKEND")' in text:
    print("already patched")
    raise SystemExit(0)
old = '''    else:
        apply_eager_llm_kwargs(llm_kwargs)
        print("Deterministic (eager) mode: bit-reproducible scoring")

    print("\\nCalculating KLD...")'''
new = '''    else:
        apply_eager_llm_kwargs(llm_kwargs)
        print("Deterministic (eager) mode: bit-reproducible scoring")

    moe_backend = os.environ.get("VLLM_MOE_BACKEND")
    if moe_backend:
        llm_kwargs["moe_backend"] = moe_backend
        print(f"MoE backend override (VLLM_MOE_BACKEND): {moe_backend}")

    print("\\nCalculating KLD...")'''
if old not in text:
    raise SystemExit(f"could not find llm_kwargs eager block in {path}")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
print("patched")
PY
}

# Declare score/KLD fields on the rendered EngineInput schema so the
# renderer copy is typed. Always run: overlay does not copy inputs/.
patch_score_kld_engine_input_schema() {
  local engine="${VLLM_DIR}/vllm/inputs/engine.py"
  [[ -f "${engine}" ]] || die "missing ${engine}"
  "${PYTHON:-python3}" - "${engine}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
if "target_token_ids: NotRequired[list[int]]" in text and "reference_logits_path: NotRequired[str]" in text:
    print("already patched")
    raise SystemExit(0)
old = '''    cache_salt: NotRequired[str]
    """Optional cache salt to be used for prefix caching."""
'''
new = '''    cache_salt: NotRequired[str]
    """Optional cache salt to be used for prefix caching."""

    target_token_ids: NotRequired[list[int]]
    """Target token IDs for score mode; copied from TokensPrompt."""

    reference_logits_path: NotRequired[str]
    """Safetensors path for KLD mode; copied from TokensPrompt."""

    reference_logits_key: NotRequired[str]
    """Safetensors key for KLD mode; copied from TokensPrompt."""
'''
if old not in text:
    raise SystemExit(f"could not find _InputOptions cache_salt in {path}")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
print("patched")
PY
}

# Renderer rebuilds EngineInput from TokensPrompt and used to drop score/KLD
# metadata. Copy the three fields with `is not None` so empty target lists
# survive. Always run: overlay skip does not refresh renderers/.
patch_score_kld_renderer_fields() {
  local renderer="${VLLM_DIR}/vllm/renderers/base.py"
  [[ -f "${renderer}" ]] || die "missing ${renderer}"
  "${PYTHON:-python3}" - "${renderer}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
helper = '''    @staticmethod
    def _copy_score_kld_fields(
        prompt: TokensPrompt,
        engine_input: TokensInput | MultiModalInput,
    ) -> None:
        """Keep score/KLD TokensPrompt fields on the rendered EngineInput."""
        for key in (
            "target_token_ids",
            "reference_logits_path",
            "reference_logits_key",
        ):
            if (value := prompt.get(key)) is not None:
                engine_input[key] = value  # type: ignore[literal-required]

'''
call = "        self._copy_score_kld_fields(prompt, engine_input)\n"
cache_salt = '''        if cache_salt := prompt.get("cache_salt"):
            engine_input["cache_salt"] = cache_salt
'''
changed = False
if "def _copy_score_kld_fields(" in text:
    start = text.index("    @staticmethod\n    def _copy_score_kld_fields(")
    end = text.index("    def _process_tokens(", start)
    if text[start:end] != helper:
        text = text[:start] + helper + text[end:]
        changed = True
else:
    anchor = "    def _process_tokens(\n        self,\n        prompt: TokensPrompt,"
    if anchor not in text:
        raise SystemExit(f"could not find _process_tokens in {path}")
    text = text.replace(anchor, helper + anchor, 1)
    changed = True

if text.count(call) < 2:
    if cache_salt not in text:
        raise SystemExit(f"could not find cache_salt copy in {path}")
    text = text.replace(cache_salt, cache_salt + call)
    changed = True

if not changed:
    print("already patched")
    raise SystemExit(0)
if text.count(call) < 2:
    raise SystemExit(f"renderer score/KLD copy calls missing in {path}")
path.write_text(text, encoding="utf-8")
print("patched")
PY
}

# InputProcessor must read score/KLD metadata from the rendered decoder
# EngineInput, not the pre-render prompt argument.
patch_score_kld_input_processor() {
  local processor="${VLLM_DIR}/vllm/v1/engine/input_processor.py"
  [[ -f "${processor}" ]] || die "missing ${processor}"
  "${PYTHON:-python3}" - "${processor}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
new = '''        # Score/KLD metadata lives on the rendered decoder EngineInput.
        target_token_ids = decoder_input.get("target_token_ids")
        reference_logits_path = decoder_input.get("reference_logits_path")
        reference_logits_key = decoder_input.get("reference_logits_key")
'''
if 'target_token_ids = decoder_input.get("target_token_ids")' in text:
    print("already patched")
    raise SystemExit(0)
old = '''        # Extract target_token_ids from TokensPrompt if present
        target_token_ids: list[int] | None = None
        reference_logits_path: str | None = None
        reference_logits_key: str | None = None
        if isinstance(prompt, dict) and "prompt_token_ids" in prompt:
            prompt_dict = prompt
            target_token_ids = prompt_dict.get("target_token_ids")
            reference_logits_path = prompt_dict.get("reference_logits_path")
            reference_logits_key = prompt_dict.get("reference_logits_key")
'''
if old not in text:
    raise SystemExit(f"could not find score/KLD prompt extraction in {path}")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
print("patched")
PY
}

# GLM overlay copies scheduler.py and drops KLD prompt_logits / kld_result
# forwarding. Reapply after overlay, even on resume skip.
patch_kld_scheduler_output_plumbing() {
  local scheduler="${VLLM_DIR}/vllm/v1/core/sched/scheduler.py"
  [[ -f "${scheduler}" ]] || die "missing ${scheduler}"
  "${PYTHON:-python3}" - "${scheduler}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
already = (
    "prompt_logits_dict = model_runner_output.prompt_logits_dict" in text
    and "kld_result_dict = model_runner_output.kld_result_dict" in text
    and "new_prompt_logits=new_prompt_logits," in text
    and "kld_result=kld_result," in text
)
if already:
    print("already patched")
    raise SystemExit(0)

old_extract = """        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens"""
new_extract = """        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
        prompt_logits_dict = model_runner_output.prompt_logits_dict
        kld_result_dict = model_runner_output.kld_result_dict
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens"""
if old_extract not in text:
    raise SystemExit(f"could not find prompt_logprobs_dict extract in {path}")
text = text.replace(old_extract, new_extract, 1)

old_lookup = """            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
            if should_emit_output:"""
new_lookup = """            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
            new_prompt_logits = prompt_logits_dict.get(req_id)
            kld_result = kld_result_dict.get(req_id)
            if should_emit_output:"""
if old_lookup not in text:
    raise SystemExit(f"could not find prompt_logprobs_dict lookup in {path}")
text = text.replace(old_lookup, new_lookup, 1)

old_kwargs = """                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        pooling_output=pooler_output,"""
new_kwargs = """                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        new_prompt_logits=new_prompt_logits,
                        kld_result=kld_result,
                        pooling_output=pooler_output,"""
if old_kwargs not in text:
    raise SystemExit(f"could not find EngineCoreOutput logprobs kwargs in {path}")
text = text.replace(old_kwargs, new_kwargs, 1)
path.write_text(text, encoding="utf-8")
print("patched")
PY
}

pick_venv || true
pick_model || true
pick_dataset || true
LOGITS_DIR="${VLLM_DIR}/ref_logits_$(basename "${MODEL_DIR}")_ctx${CONTEXT_LENGTH}_s${STRIDE}"
CAPTURE_MARKER="${LOGITS_DIR}/CAPTURE_COMPLETE"
UPLOAD_MARKER="${LOGITS_DIR}/UPLOAD_COMPLETE"
log "VENV_DIR=${VENV_DIR} ($(has_venv && echo ready || echo missing))"
log "MODEL_DIR=${MODEL_DIR} ($(has_model && echo complete || echo missing/incomplete))"
log "DATASET_DIR=${DATASET_DIR} ($(has_dataset && echo complete || echo missing/incomplete))"
log "GLM53 native=$(has_glm53_native && echo yes || echo no)"
log "LOGITS_DIR=${LOGITS_DIR}"

AVAILABLE_GB="$(free_gb "${WORK_DIR}")"
log "free disk on ${WORK_DIR}: ${AVAILABLE_GB} GB"
DISK_NEED="${MIN_FREE_GB}"
if has_model; then
  DISK_NEED=200
  log "weight shards already complete; requiring ${DISK_NEED} GB free for logits"
elif [[ -f "${MODEL_DIR}/config.json" ]]; then
  DISK_NEED=50
  log "partial weights present; hf download will resume"
fi
if [[ "${AVAILABLE_GB}" -lt "${DISK_NEED}" ]]; then
  die "need ~${DISK_NEED} GB free on this volume; have ${AVAILABLE_GB} GB"
fi

# ---------------------------------------------------------------------------
# 1) uv + venv
# ---------------------------------------------------------------------------
if has_venv; then
  log "venv exists at ${VENV_DIR}; skipping create"
else
  if ! command -v uv >/dev/null 2>&1; then
    log "installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
  fi
  export PATH="${HOME}/.local/bin:${PATH}"
  need_cmd uv
  log "creating venv at ${VENV_DIR}"
  uv venv --python 3.12 --seed "${VENV_DIR}"
fi
export PATH="${HOME}/.local/bin:${PATH}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
PYTHON="${VENV_DIR}/bin/python"
export VIRTUAL_ENV="${VENV_DIR}"
log "activated venv: ${VIRTUAL_ENV} ($("${PYTHON}" -V))"

# ---------------------------------------------------------------------------
# 2) Hugging Face CLI + KLD fork + native GLM-5.3-Flash from PR 53906
# ---------------------------------------------------------------------------
if [[ "${SKIP_INSTALL}" == "1" ]]; then
  log "SKIP_INSTALL=1"
  has_venv || die "venv missing"
  has_vllm_src || die "vLLM clone missing at ${VLLM_DIR}"
else
  if has_hf_cli && [[ "${FORCE_INSTALL}" != "1" ]]; then
    log "huggingface_hub + datasets already in venv; skipping pip"
  else
    log "installing huggingface_hub (download/upload) + hf_transfer"
    uv pip install -U "huggingface_hub[cli,hf_transfer]" datasets
  fi

  # torch's JIT extension loader shells out to `ninja` to build the TileLang
  # MHC kernels during memory profiling; without it the worker dies with
  # "No such file or directory: 'ninja'".
  if command -v ninja >/dev/null 2>&1; then
    log "ninja present: $(command -v ninja)"
  else
    log "installing ninja (required by TileLang MHC JIT)"
    uv pip install -U ninja
    command -v ninja >/dev/null 2>&1 || die "ninja still not on PATH after install"
  fi

  if has_vllm_src; then
    log "vLLM clone exists at ${VLLM_DIR}; skipping git clone"
  else
    log "cloning ${VLLM_GIT_URL} (${VLLM_GIT_BRANCH})"
    git clone --branch "${VLLM_GIT_BRANCH}" --depth 1 "${VLLM_GIT_URL}" "${VLLM_DIR}"
  fi
fi
has_vllm_src || die "vLLM clone missing at ${VLLM_DIR}"

if [[ "${SKIP_GLM53_PR}" == "1" ]]; then
  log "SKIP_GLM53_PR=1"
  has_glm53_native || die "Glm5Next missing in ${VLLM_DIR}; re-run without SKIP_GLM53_PR=1"
elif [[ "${FORCE_GLM53_PR}" != "1" ]] && has_glm53_native; then
  log "native GLM-5.3-Flash already present; skipping PR #${GLM53_PR_NUMBER} overlay"
else
  apply_glm53_pr_overlay
fi
log "checking gpu_model_runner.py integrity"
heal_status="$(heal_corrupted_model_runner)"
log "gpu_model_runner.py: ${heal_status}"
# Always patch: overlay skip leaves PR utils.py + KLD gpu_model_runner.py.
log "patching V1 KVBlockZeroer(num_blocks=...) call site"
patch_status="$(patch_v1_kv_block_zeroer_call)"
log "V1 KVBlockZeroer call site: ${patch_status}"
log "patching score_mode_kld.py VLLM_MOE_BACKEND passthrough"
kld_patch_status="$(patch_score_mode_kld_moe_backend_env)"
log "score_mode_kld MoE backend env: ${kld_patch_status}"
log "patching EngineInput score/KLD schema"
engine_schema_status="$(patch_score_kld_engine_input_schema)"
log "EngineInput score/KLD schema: ${engine_schema_status}"
log "patching renderer score/KLD EngineInput fields"
renderer_patch_status="$(patch_score_kld_renderer_fields)"
log "renderer score/KLD fields: ${renderer_patch_status}"
log "patching InputProcessor score/KLD decoder_input reads"
processor_patch_status="$(patch_score_kld_input_processor)"
log "InputProcessor score/KLD reads: ${processor_patch_status}"
log "patching KLD scheduler prompt_logits/kld_result output plumbing"
scheduler_patch_status="$(patch_kld_scheduler_output_plumbing)"
log "KLD scheduler output plumbing: ${scheduler_patch_status}"
log "verifying vLLM tree compiles before loading weights"
syntax_status="$(verify_vllm_syntax)"
log "vLLM syntax check: ${syntax_status}"

if [[ "${SKIP_INSTALL}" == "1" ]]; then
  has_vllm_install || die "vLLM is not importable in the venv"
elif [[ "${FORCE_INSTALL}" != "1" ]] && has_vllm_install; then
  log "vLLM already importable from ${VLLM_DIR}; skipping pip install"
else
  log "editable install with VLLM_USE_PRECOMPILED=1 (no full CUDA compile)"
  (
    cd "${VLLM_DIR}"
    VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
  )
fi

if [[ "${SKIP_NVCC}" == "1" ]]; then
  log "SKIP_NVCC=1"
else
  ensure_sm100a_nvcc
fi

# ---------------------------------------------------------------------------
# 3) Download BF16 weights
# ---------------------------------------------------------------------------
if [[ "${SKIP_DOWNLOAD}" == "1" ]]; then
  log "SKIP_DOWNLOAD=1"
  has_model || die "model dir incomplete: ${MODEL_DIR}"
elif [[ "${FORCE_DOWNLOAD}" != "1" ]] && has_model; then
  log "BF16 weights already complete in ${MODEL_DIR}; skipping hf download"
  log "model dir size: $(du -sh "${MODEL_DIR}" | awk '{print $1}')"
else
  command -v hf >/dev/null 2>&1 || die "hf CLI not on PATH; cannot download weights"
  log "hf download ${HF_MODEL_ID} --local-dir ${MODEL_DIR}"
  mkdir -p "${MODEL_DIR}"
  hf download "${HF_MODEL_ID}" --local-dir "${MODEL_DIR}" --token "${HF_TOKEN}"
  has_model || die "download finished but weight shards look incomplete: ${MODEL_DIR}"
  log "model dir size: $(du -sh "${MODEL_DIR}" | awk '{print $1}')"
fi

# ---------------------------------------------------------------------------
# 3b) Download WikiText-2 locally (Salesforce/wikitext, wikitext-2-raw-v1)
# ---------------------------------------------------------------------------
if [[ "${SKIP_DATASET}" == "1" ]]; then
  log "SKIP_DATASET=1"
  has_dataset || die "dataset dir incomplete: ${DATASET_DIR}"
elif [[ "${FORCE_DATASET}" != "1" ]] && has_dataset; then
  log "WikiText already complete in ${DATASET_DIR}; skipping hf download"
  log "dataset dir size: $(du -sh "${DATASET_DIR}" | awk '{print $1}')"
else
  command -v hf >/dev/null 2>&1 || die "hf CLI not on PATH; cannot download dataset"
  log "hf download ${HF_DATASET_ID} --repo-type dataset --local-dir ${DATASET_DIR}"
  mkdir -p "${DATASET_DIR}"
  hf download "${HF_DATASET_ID}" \
    --repo-type dataset \
    --local-dir "${DATASET_DIR}" \
    --token "${HF_TOKEN}"
  has_dataset || die "download finished but WikiText files look incomplete: ${DATASET_DIR}"
  log "dataset dir size: $(du -sh "${DATASET_DIR}" | awk '{print $1}')"
fi

# ---------------------------------------------------------------------------
# 3a-bis) Extra patches, applied after the GLM overlay so the overlay cannot
#     silently revert them. Drop *.patch into PATCH_DIR; each is applied only
#     if it is not already present in the tree.
# ---------------------------------------------------------------------------
PATCH_DIR="${PATCH_DIR:-${HOME}/patches}"
if [[ -d "${PATCH_DIR}" ]]; then
  shopt -s nullglob
  for patch_file in "${PATCH_DIR}"/*.patch; do
    if git -C "${VLLM_DIR}" apply --reverse --check "${patch_file}" >/dev/null 2>&1; then
      log "patch already applied: $(basename "${patch_file}")"
    elif git -C "${VLLM_DIR}" apply "${patch_file}" >/dev/null 2>&1; then
      log "applied patch: $(basename "${patch_file}")"
    else
      die "failed to apply $(basename "${patch_file}"); tree does not match"
    fi
  done
  shopt -u nullglob
fi

# Sparse MLA hands the attention kernel an unstable index order, which costs one
# ULP per layer and compounds into the KLD floor. Sorting the rows fixes the
# accumulation order without changing which tokens are attended.
export VLLM_SPARSE_MLA_SORT_TOPK="${VLLM_SPARSE_MLA_SORT_TOPK:-1}"

# Full-length forward passes are not reproducible; 128-token prefill chunks are.
# Verified at 2048 tokens across two fresh processes: bit-identical logits.
# Capture and every scoring run must use the same value, since chunk size
# changes the numerics.
CAPTURE_CHUNK_TOKENS="${CAPTURE_CHUNK_TOKENS:-128}"

# ---------------------------------------------------------------------------
# 3b) Determinism gate: refuse to capture from a rig that cannot reproduce
#     itself. A non-zero noise floor is indistinguishable from quantization
#     error in every KLD later measured against these logits.
# ---------------------------------------------------------------------------
GATE_PY="${VLLM_DIR}/scripts/glm53_determinism_probe.py"
if [[ "${SKIP_GATE}" == "1" || "${SKIP_CAPTURE}" == "1" ]]; then
  log "determinism gate skipped"
elif [[ ! -f "${GATE_PY}" ]]; then
  log "determinism gate unavailable (missing ${GATE_PY}); continuing"
elif [[ "${FORCE_CAPTURE}" != "1" ]] && has_capture; then
  log "capture already complete; skipping determinism gate"
else
  log "determinism gate: lengths=${GATE_LENGTHS} repeats=${GATE_REPEATS} max-self-kld=${GATE_MAX_SELF_KLD}"
  gate_rc=0
  (
    cd "${VLLM_DIR}"
    "${VIRTUAL_ENV:-${VENV_DIR}}/bin/python" scripts/glm53_determinism_probe.py \
      --config-name capture_gate \
      --model "${MODEL_DIR}" \
      --dataset-dir "${DATASET_DIR}" \
      --dataset-config "${DATASET_CONFIG}" \
      --out-dir "${WORK_DIR}/determinism_gate" \
      --ctx "${CONTEXT_LENGTH}" \
      --stride "${STRIDE}" \
      --max-num-batched-tokens "${CAPTURE_CHUNK_TOKENS}" \
      --tp "${TENSOR_PARALLEL_SIZE}" \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
      --moe-backend "${VLLM_MOE_BACKEND}" \
      --repeats "${GATE_REPEATS}" \
      --lengths "${GATE_LENGTHS}" \
      --gate --gate-max-self-kld "${GATE_MAX_SELF_KLD}"
  ) || gate_rc=$?
  if [[ "${gate_rc}" != "0" ]]; then
    if [[ "${GATE_ENFORCE}" == "1" ]]; then
      log "gate report: ${WORK_DIR}/determinism_gate/capture_gate.json"
      log "next step: scripts/glm53_layer_bisect.py localizes the divergence"
      log "override with GATE_ENFORCE=0 to capture a knowingly noisy reference"
      die "determinism gate failed (exit ${gate_rc}); refusing to capture"
    fi
    log "WARNING: determinism gate failed (exit ${gate_rc}) but GATE_ENFORCE=0"
    log "WARNING: KLD measured against this reference has a non-zero floor"
  else
    log "determinism gate passed"
  fi
fi

# ---------------------------------------------------------------------------
# 4) KLD capture via score_mode_kld.py (exact command; run from the clone)
# ---------------------------------------------------------------------------
if [[ "${SKIP_CAPTURE}" == "1" ]]; then
  log "SKIP_CAPTURE=1"
  if ! has_capture; then
    # Provisioning-only mode: env, overlay, weights and dataset are ready and
    # there is nothing to package or upload. Not an error.
    log "no capture at ${LOGITS_DIR}; provisioning complete, stopping here"
    log "done"
    exit 0
  fi
elif [[ "${FORCE_CAPTURE}" != "1" ]] && has_capture; then
  log "capture already complete (${CAPTURE_MARKER}); skipping"
else
  KLD_PY="${VLLM_DIR}/examples/offline_inference/score_mode_kld.py"
  [[ -f "${KLD_PY}" ]] || die "missing ${KLD_PY}"
  [[ -x "${VIRTUAL_ENV:-${VENV_DIR}}/bin/python" ]] || die "venv python missing"

  log "cd ${VLLM_DIR} && score_mode_kld.py (do not run from WORK_DIR; a sibling vllm/ clone shadows import)"
  has_glm53_native || die "Glm5Next missing; re-run without SKIP_GLM53_PR=1"
  has_dataset || die "WikiText missing at ${DATASET_DIR}; re-run without SKIP_DATASET=1"
  log "dataset=${DATASET_DIR} config=${DATASET_CONFIG}"
  (
    cd "${VLLM_DIR}"
    "${VIRTUAL_ENV:-${VENV_DIR}}/bin/python" examples/offline_inference/score_mode_kld.py \
      --model "${MODEL_DIR}" \
      --reference-model "${MODEL_DIR}" \
      --dataset "${DATASET_DIR}" \
      --dataset-config "${DATASET_CONFIG}" \
      --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
      --max-num-seqs "${MAX_NUM_SEQS}" \
      --max-num-batched-tokens "${CAPTURE_CHUNK_TOKENS}" \
      --language-model-only \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  )
  WINDOW_NOW="$(find "${LOGITS_DIR}" -maxdepth 1 -name 'logits_*.safetensors' 2>/dev/null | wc -l | tr -d ' ')"
  [[ "${WINDOW_NOW}" -gt 0 ]] || die "KLD script finished but no logits in ${LOGITS_DIR}"
  printf '%s\n' "${WINDOW_NOW}" > "${CAPTURE_MARKER}"
fi

WINDOW_COUNT="$(find "${LOGITS_DIR}" -maxdepth 1 -name 'logits_*.safetensors' 2>/dev/null | wc -l | tr -d ' ')"
[[ "${WINDOW_COUNT}" -gt 0 ]] || die "no logits_*.safetensors in ${LOGITS_DIR}"
log "found ${WINDOW_COUNT} logit windows in ${LOGITS_DIR}"

cat > "${LOGITS_DIR}/README.md" <<EOF
# ${HF_REPO}

Private WikiText-2 reference logits for KLD scoring.

- **Source model:** ${HF_MODEL_ID}
- **Dataset:** ${HF_DATASET_ID} / ${DATASET_CONFIG} (local: ${DATASET_DIR})
- **Window:** context_length=${CONTEXT_LENGTH}, stride=${STRIDE}
- **Windows:** ${WINDOW_COUNT} files named \`logits_{i}.safetensors\` (key: \`logits\`)
- **Engine:** ${VLLM_GIT_URL} @ ${VLLM_GIT_BRANCH}
- **Flags:** TP=${TENSOR_PARALLEL_SIZE}, max_num_seqs=${MAX_NUM_SEQS}, language_model_only, eager, gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}
- **Reuse:** \`score_mode_kld.py --reference-logits <this-repo-or-dir> --dataset ${DATASET_DIR} --dataset-config ${DATASET_CONFIG} --context-length ${CONTEXT_LENGTH} --stride ${STRIDE}\`
EOF

# ---------------------------------------------------------------------------
# 5) Upload to Hugging Face (private), matching phaedawg/ref-logits-* models
# ---------------------------------------------------------------------------
if [[ "${SKIP_UPLOAD}" == "1" ]]; then
  log "SKIP_UPLOAD=1 (logits left at ${LOGITS_DIR})"
elif [[ "${FORCE_UPLOAD}" != "1" ]] && has_upload; then
  log "upload already complete (${UPLOAD_MARKER}); skipping"
else
  command -v hf >/dev/null 2>&1 || die "hf CLI not on PATH; cannot upload logits"
  log "creating private HF ${HF_REPO_TYPE} ${HF_REPO} (ok if it already exists)"
  hf repo create "${HF_REPO}" --repo-type "${HF_REPO_TYPE}" --private --exist-ok \
    --token "${HF_TOKEN}"
  log "uploading ${LOGITS_DIR} -> https://huggingface.co/${HF_REPO}"
  hf upload "${HF_REPO}" "${LOGITS_DIR}" . --repo-type "${HF_REPO_TYPE}" --token "${HF_TOKEN}"
  printf '%s\n' "${HF_REPO}" > "${UPLOAD_MARKER}"
  log "upload complete: https://huggingface.co/${HF_REPO}"
fi

log "done"
