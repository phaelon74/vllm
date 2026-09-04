#!/usr/bin/env bash
# Validate exact-repeat QxQ/BxQ before a full campaign.
#
# Run on the GPU scoring host after rebuilding Marlin. Do not wrap the
# interactive smoke in `set -e`; a failed backend must be recorded, not hide
# later rows. CPU selftests can run first from any checkout.
#
# Usage:
#   scripts/validate_deterministic_qxq_bxq.sh [python]
#
# Rebuild CUDA first when Marlin sources changed:
#   python tools/generate_cmake_presets.py
#   cmake --build cmake-build-debug --target _moe_C -j
# See docs/contributing/incremental_build.md.

set -uo pipefail

PY=${1:-${KLD_PYTHON:-${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}}}
if [[ -z ${PY} || ! -x ${PY} ]]; then
  PY=$(command -v python)
fi
if [[ -z ${PY} ]]; then
  echo "no python interpreter found; pass the venv python as argv1" >&2
  exit 2
fi

echo "=== CPU fidelity selftests"
"$PY" fidelity/artifact.py selftest
"$PY" fidelity/campaign.py --selftest
"$PY" fidelity/strata.py --selftest

echo "=== canonicalize inactive-tail (CPU)"
"$PY" -m pytest tests/kernels/moe/test_moe.py::test_canonicalize_marlin_moe_token_order -q

if ! "$PY" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
  echo "CUDA unavailable; skip Marlin microprobes, backend smokes, and campaigns."
  echo "On d011sd01, rebuild Marlin then rerun this script from the venv."
  exit 0
fi

export VLLM_BATCH_INVARIANT=1
export VLLM_MOE_USE_DEEP_GEMM=0
export NCCL_DETERMINISTIC=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "=== Marlin kernel microprobes (bit-exact repeats)"
"$PY" -m pytest tests/v1/determinism/test_marlin_moe_batch_invariant.py -q \
  tests/kernels/moe/test_moe.py::test_canonicalize_marlin_moe_token_order -q

echo "=== CPU/GPU forced-route repeatability"
"$PY" -m pytest tests/model_executor/test_routed_experts_capture.py::test_forced_routing_weights_are_repeatable -q

echo "=== GDN attention batch invariance"
VLLM_TEST_MODEL="${KLD_GDN_TEST_MODEL:-Qwen/Qwen3.5-0.8B}" \
VLLM_NEEDLE_TRIALS="${KLD_GDN_TRIALS:-3}" \
VLLM_NEEDLE_BATCH_SIZE="${KLD_GDN_BATCH_SIZE:-8}" \
VLLM_MIN_PROMPT="${KLD_GDN_MIN_PROMPT:-64}" \
VLLM_MAX_PROMPT="${KLD_GDN_MAX_PROMPT:-256}" \
VLLM_NEEDLE_MAX_TOKENS="${KLD_GDN_MAX_TOKENS:-8}" \
VLLM_MAX_MODEL_LEN="${KLD_GDN_MODEL_LEN:-512}" \
VLLM_TEST_ENFORCE_EAGER=1 \
"$PY" -m pytest tests/v1/determinism/test_batch_invariance.py \
  -k "needle and GDN_ATTN and default" -q

echo
echo "Microprobes passed. Next, without set -e:"
echo "  1. One-row exact-repeat smokes for each selected backend"
echo "     (Qwen NVFP4/Marlin, Qwen FP8/Triton, each Gemma/AutoRound backend)."
echo "  2. Record any uncertified backend and do not publish it."
echo "  3. Full suites only after every active backend's smoke is exact."
echo "  4. Assemble, checksum, Law-compliant receipts, then cumulative publish."
