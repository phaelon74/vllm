#!/usr/bin/env bash
# Capture the host, driver, library, and checkpoint provenance that a KLD
# number is bound to. Run this once per host before any KLD run and keep the
# output directory next to the capture manifests and report JSONs.
#
# Usage:
#   scripts/kld_env_report.sh OUT_DIR [MODEL_PATH ...]
#
# Weight files are fingerprinted by name, size, and count rather than by
# hashing every byte. Set KLD_HASH_WEIGHTS=1 for full sha256 of every file
# (hours on a 50 GB checkpoint over a network mount).

set -uo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 OUT_DIR [MODEL_PATH ...]" >&2
  exit 2
fi

OUT_DIR=$1
shift
mkdir -p "$OUT_DIR/models" || exit 1

REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)

# Prefer an explicit override, then the activated venv (which need not live in
# the repo), then a repo-local .venv, and only then whatever python is on PATH.
PY=""
for candidate in \
  "${KLD_PYTHON:-}" \
  "${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}" \
  "$REPO_ROOT/.venv/bin/python"; do
  if [[ -n $candidate && -x $candidate ]]; then
    PY=$candidate
    break
  fi
done
[[ -n $PY ]] || PY=$(command -v python)
if [[ -z $PY ]]; then
  echo "no python interpreter found; set KLD_PYTHON=/path/to/venv/bin/python" >&2
  exit 2
fi
echo "interpreter: $PY"

log_cmd () {
  local name=$1
  shift
  local dest="$OUT_DIR/$name.txt"
  printf '### %s\n\n' "$*" >"$dest"
  if ! command -v "$1" >/dev/null 2>&1; then
    printf 'command not found: %s\n' "$1" >>"$dest"
    return 0
  fi
  "$@" >>"$dest" 2>&1
  local rc=$?
  printf '\n### exit=%d\n' "$rc" >>"$dest"
  return 0
}

# Host and OS.
log_cmd host-uname uname -a
log_cmd host-os cat /etc/os-release
log_cmd host-cpu lscpu
log_cmd host-memory free -g
log_cmd host-mounts df -h

# GPUs, driver, and interconnect. nvidia-smi -q carries ECC state, persistence
# mode, clock throttle reasons, and per-GPU serials; all of these can move a
# bitwise-exactness result.
log_cmd gpu-smi nvidia-smi
log_cmd gpu-smi-query nvidia-smi -q
log_cmd gpu-topology nvidia-smi topo -m
log_cmd gpu-clocks nvidia-smi --query-gpu=index,name,serial,uuid,driver_version,vbios_version,pstate,clocks.sm,clocks.max.sm,power.limit,persistence_mode,ecc.mode.current --format=csv

# Compiler and CUDA toolchain.
log_cmd toolchain-nvcc nvcc --version
log_cmd toolchain-gcc gcc --version
log_cmd toolchain-ldd ldd --version

# Repo state and installed packages.
log_cmd repo-git-log git -C "$REPO_ROOT" log -n 5 --format='%H %ad %s' --date=iso
log_cmd repo-git-status git -C "$REPO_ROOT" status --porcelain=v1 --branch
log_cmd repo-git-describe git -C "$REPO_ROOT" describe --always --dirty --tags
log_cmd pip-freeze "$PY" -m pip freeze

# Torch / vLLM runtime, including the exact fields the capture manifest binds.
# The commit goes in as an environment variable so it lands in runtime.json in
# machine-readable form; a number cannot be reproduced from prose alone.
KLD_REPO_COMMIT=$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null)
KLD_REPO_DIRTY=0
[[ -n $(git -C "$REPO_ROOT" status --porcelain 2>/dev/null) ]] && KLD_REPO_DIRTY=1
KLD_REPO_DIRTY_DIGEST=$(git -C "$REPO_ROOT" diff HEAD 2>/dev/null | sha256sum | awk '{print $1}')
KLD_REPO_ROOT=$REPO_ROOT
export KLD_REPO_COMMIT KLD_REPO_DIRTY KLD_REPO_DIRTY_DIGEST KLD_REPO_ROOT
"$PY" - >"$OUT_DIR/runtime.json" 2>"$OUT_DIR/runtime.err" <<'PYEOF'
import json
import os
import platform
import sys

PREFIXES = (
    "VLLM_", "TORCH", "PYTORCH", "CUDA", "CUBLAS", "NCCL", "TRITON",
    "FLASHINFER", "OMP_", "MKL_", "HF_", "SAFETENSORS_",
)

# The watched prefixes are deliberately broad, and HF_TOKEN shares a namespace
# with HF_HOME. A credential's value is never written to the report; that it was
# set is, because an unset token changes what a run can reach.
sys.path.insert(0, os.environ.get("KLD_REPO_ROOT") or os.getcwd())
try:
    from fidelity.redaction import redact_env
except ImportError:
    import re

    WORDS = {"TOKEN", "SECRET", "PASSWORD", "PASSWD", "PWD", "KEY", "APIKEY",
             "CREDENTIAL", "CREDENTIALS", "COOKIE", "AUTH", "AUTHORIZATION",
             "SESSION"}

    def redact_env(env):
        clean = {}
        hidden = []
        for name, value in env.items():
            if WORDS.intersection(re.split(r"[_\-.]", name.upper())):
                clean[name] = "<redacted>"
                hidden.append(name)
            else:
                clean[name] = value
        return clean, sorted(hidden)

captured_env, redacted_names = redact_env(
    {k: v for k, v in sorted(os.environ.items()) if k.startswith(PREFIXES)}
)

info = {
    "python": platform.python_version(),
    "executable": sys.executable,
    "platform": platform.platform(),
    "hostname": platform.node(),
    "vllm_commit": os.environ.get("KLD_REPO_COMMIT") or None,
    "vllm_tree_dirty": os.environ.get("KLD_REPO_DIRTY") == "1",
    "vllm_dirty_digest": os.environ.get("KLD_REPO_DIRTY_DIGEST") or None,
    "env": captured_env,
    "env_redacted": redacted_names,
    "env_redaction_policy": (
        "Values of variables whose names look like credentials are never "
        "recorded. The names are, so a reader can see what was set."
    ),
}

try:
    import torch

    info["torch"] = {
        "version": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "arch_list": torch.cuda.get_arch_list(),
        "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    if torch.cuda.is_available():
        info["torch"]["nccl"] = ".".join(str(v) for v in torch.cuda.nccl.version())
        info["devices"] = [
            {
                "index": i,
                "name": (props := torch.cuda.get_device_properties(i)).name,
                "capability": f"{props.major}.{props.minor}",
                "total_memory_gib": round(props.total_memory / 2**30, 2),
                "multi_processor_count": props.multi_processor_count,
            }
            for i in range(torch.cuda.device_count())
        ]
except Exception as exc:
    info["torch_error"] = repr(exc)

try:
    import vllm

    info["vllm"] = {"version": vllm.__version__, "path": vllm.__file__}
except Exception as exc:
    info["vllm_error"] = repr(exc)

try:
    from vllm.v1.sample.kld import capture_runtime_manifest

    info["capture_runtime_manifest"] = capture_runtime_manifest()
except Exception as exc:
    info["capture_runtime_manifest_error"] = repr(exc)

manifest = info.get("capture_runtime_manifest") or {}
if isinstance(manifest, dict):
    info.setdefault("vllm_dirty_digest", manifest.get("vllm_dirty_digest"))
    info["compiled_extensions"] = manifest.get("compiled_extensions") or {}
    info["compiled_extensions_sha256"] = manifest.get(
        "compiled_extensions_sha256"
    )
    info["flashinfer"] = manifest.get("flashinfer")
    info["determinism"] = manifest.get("determinism")

json.dump(info, sys.stdout, indent=2, default=str)
sys.stdout.write("\n")
PYEOF

fingerprint_model () {
  local path=$1
  local dest="$OUT_DIR/models/$(basename "$path").txt"
  {
    printf '### %s\n\n' "$path"
    if [[ ! -d $path ]]; then
      printf 'not a directory\n'
      return 0
    fi
    printf '## file listing (name size)\n'
    find "$path" -maxdepth 1 -type f -printf '%f %s\n' | sort
    printf '\n## listing sha256\n'
    find "$path" -maxdepth 1 -type f -printf '%f %s\n' | sort | sha256sum
    printf '\n## total bytes / file count\n'
    du -sb "$path"
    find "$path" -type f | wc -l
    printf '\n## config and tokenizer sha256\n'
    for f in config.json generation_config.json tokenizer_config.json \
      tokenizer.json vocab.json merges.txt preprocessor_config.json \
      chat_template.jinja model.safetensors.index.json; do
      [[ -f "$path/$f" ]] && sha256sum "$path/$f"
    done
    printf '\n## config.json\n'
    cat "$path/config.json" 2>/dev/null
    if [[ ${KLD_HASH_WEIGHTS:-0} == 1 ]]; then
      printf '\n## full weight sha256\n'
      find "$path" -type f -name '*.safetensors' -print0 | sort -z |
        xargs -0 -r -n1 sha256sum
    fi
  } >"$dest" 2>&1
  return 0
}

for model in "$@"; do
  fingerprint_model "$model"
done

{
  printf '# KLD environment report\n\n'
  printf -- '- captured: %s\n' "$(date -Is)"
  printf -- '- host: %s\n' "$(hostname)"
  printf -- '- kernel: %s\n' "$(uname -r)"
  printf -- '- python: %s\n' "$PY"
  printf -- '- repo: %s\n' "$REPO_ROOT"
  printf -- '- commit: %s\n' \
    "$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null)"
  printf -- '- branch: %s\n' \
    "$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null)"
  printf -- '- driver: %s\n' \
    "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null |
      head -n1)"
  printf -- '- gpus: %s\n' \
    "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null |
      sort | uniq -c | tr '\n' ';')"
  printf -- '- torch: %s\n' \
    "$("$PY" -c 'import torch; print(torch.__version__)' 2>/dev/null)"
  printf -- '- vllm: %s\n' \
    "$("$PY" -c 'import vllm; print(vllm.__version__)' 2>/dev/null)"
  printf '\n## files\n\n'
  (cd "$OUT_DIR" && find . -type f | sort | sed 's/^/- /')
} >"$OUT_DIR/summary.md" 2>&1

echo "wrote $OUT_DIR/summary.md"
