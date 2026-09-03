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
#   SKIP_DOWNLOAD=1  assume every checkpoint is already local

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
