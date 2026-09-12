#!/usr/bin/env bash
# Stand up a venv, install this branch of vLLM into it, and fetch the
# checkpoints a campaign names. Idempotent: re-running skips work already done.
#
# Usage:
#   bash fidelity/bootstrap.sh fidelity/campaigns/qwen3.6.json
#
# Knobs:
#   VENV        virtualenv location (default <repo>/.venv)
#   PYTHON_VER  interpreter version for uv venv (default 3.12)
#   SKIP_INSTALL=1   assume vLLM is already installed
#   SKIP_KERNEL=1    assume the patched MoE kernels are already built
#   SKIP_DOWNLOAD=1  assume every checkpoint is already local
#   BUILD_DIR        cmake build directory (default <repo>/cmake-build-release)
#   TORCH_CUDA_ARCH_LIST  restrict the kernel build to your own arch, e.g. 12.0

set -euo pipefail

CONFIG=${1:-}
if [[ -z $CONFIG || ! -f $CONFIG ]]; then
  echo "usage: bash fidelity/bootstrap.sh <campaign.json>" >&2
  exit 2
fi

REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
VENV=${VENV:-$REPO_ROOT/.venv}
PYTHON_VER=${PYTHON_VER:-3.12}

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it with:" >&2
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 2
fi

if [[ ! -x $VENV/bin/python ]]; then
  echo "=== creating venv at $VENV (python $PYTHON_VER)"
  uv venv --python "$PYTHON_VER" "$VENV"
fi
PY="$VENV/bin/python"

if [[ ${SKIP_INSTALL:-0} != 1 ]]; then
  if "$PY" -c 'import vllm' 2>/dev/null; then
    echo "=== vLLM already importable in $VENV"
  else
    echo "=== installing this branch into $VENV"
    VIRTUAL_ENV="$VENV" VLLM_USE_PRECOMPILED=1 \
      uv pip install -e "$REPO_ROOT" --torch-backend=auto
  fi
  VIRTUAL_ENV="$VENV" uv pip install --quiet 'huggingface_hub[cli]' datasets matplotlib
fi

# The install above takes upstream's precompiled extensions, so
# `_moe_C_stable_libtorch` arrives built from unpatched Marlin MoE sources. The
# fork's batch-invariant full-K reduction lives in three files under
# csrc/libtorch_stable/moe/marlin_moe_wna16/, and certification is keyed on the
# kernel's class name, which is the same either way -- so an unpatched build is
# certified and then quietly lets Marlin MoE arithmetic move with batching.
# Rebuilding is therefore part of installing rather than a tuning step, and only
# this one target, because the fork touches no other extension. Ninja no-ops when
# nothing changed. See fidelity/INSTALL.md section 4.
if [[ ${SKIP_KERNEL:-0} != 1 ]]; then
  NVCC=$(command -v nvcc || true)
  if [[ -z $NVCC && -x ${CUDA_HOME:-/usr/local/cuda}/bin/nvcc ]]; then
    NVCC=${CUDA_HOME:-/usr/local/cuda}/bin/nvcc
  fi
  if [[ -z $NVCC ]]; then
    echo "WARNING: no nvcc found, so the patched MoE kernels were not built." >&2
    echo "         Marlin MoE candidates will fail exact repeat." >&2
    echo "         See fidelity/INSTALL.md section 4." >&2
  else
    echo "=== building patched MoE kernels (_moe_C_stable_libtorch)"
    VIRTUAL_ENV="$VENV" uv pip install --quiet \
      -r "$REPO_ROOT/requirements/build/cuda.txt" --torch-backend=auto
    BUILD_DIR=${BUILD_DIR:-$REPO_ROOT/cmake-build-release}
    LAUNCHERS=()
    if command -v ccache >/dev/null 2>&1; then
      LAUNCHERS=(-DCMAKE_C_COMPILER_LAUNCHER=ccache
                 -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
                 -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache)
    fi
    # cmake and ninja come from the venv via requirements/build/cuda.txt.
    PATH="$VENV/bin:$PATH" cmake -S "$REPO_ROOT" -B "$BUILD_DIR" -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DVLLM_TARGET_DEVICE=cuda \
      -DCMAKE_CUDA_COMPILER="$NVCC" \
      -DVLLM_PYTHON_EXECUTABLE="$PY" \
      -DVLLM_PYTHON_PATH="$("$PY" -c 'import sys; print(":".join(sys.path))')" \
      -DFETCHCONTENT_BASE_DIR="$REPO_ROOT/.deps" \
      ${LAUNCHERS[@]+"${LAUNCHERS[@]}"}
    PATH="$VENV/bin:$PATH" cmake --build "$BUILD_DIR" \
      --target _moe_C_stable_libtorch
    # Each extension is its own install component (cmake/utils.cmake), and the
    # target's destination is `vllm`, so the repo root is the right prefix for an
    # editable install to see it.
    PATH="$VENV/bin:$PATH" cmake --install "$BUILD_DIR" --prefix "$REPO_ROOT" \
      --component _moe_C_stable_libtorch
  fi
fi

# The commit is part of every artifact's identity (Law 6), so make it loud here
# rather than discovering a dirty tree after a six-hour campaign.
echo "=== repo state"
git -C "$REPO_ROOT" rev-parse HEAD
if [[ -n $(git -C "$REPO_ROOT" status --porcelain) ]]; then
  echo "WARNING: working tree is dirty; the artifact will record it as such" >&2
fi

if [[ ${SKIP_DOWNLOAD:-0} != 1 ]]; then
  echo "=== resolving checkpoints"
  KLD_PYTHON="$PY" "$PY" "$REPO_ROOT/fidelity/campaign.py" download \
    --config "$CONFIG"
fi

echo
echo "ready. interpreter: $PY"
echo "next: KLD_PYTHON=$PY $PY fidelity/campaign.py all --config $CONFIG"
