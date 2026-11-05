# Installing vLLM with Native Ricursive (Ripple-65B-Preview) Support

This branch adds native vLLM support for `ricursiveAI/Ripple-65B-Preview`
(`model_type: "ricursive"`, `architectures: ["RicursiveForCausalLM"]`),
including compressed-tensors W8A8 Int8 quantization. No `trust_remote_code`
is needed for config or model loading — the architecture is registered
natively.

Branch state: based on vLLM `3199780ac6` (as of 2026-11-05).

## Requirements

| Dependency | Version used in development | Notes |
|---|---|---|
| Python | 3.12 | tested on 3.12.3 |
| transformers | **5.22.1** | current stable v5; **no nightly required** |
| torch | 2.14.0 | install with `--torch-backend=auto` below |
| CUDA | 12.x | for GPU serving |
| OS | Linux x86_64 | |

No extra patches are needed for transformers. See
[Known transformers 5 caveat](#known-transformers-5-caveat) for one
compatibility detail that affects HF-side `generate()` (not vLLM serving).

## Install

### 1. Clone and set up the environment

```bash
git clone <this-fork-or-worktree-url> vllm-ricursive
cd vllm-ricursive

# uv manages the venv; do NOT use system python or bare pip
curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv --python 3.12
source .venv/bin/activate
```

### 2. Install vLLM from this branch

**GPU (recommended path):** precompiled kernels + editable install of this
branch's Python code:

```bash
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
```

This compiles only the Python layer changes (the ricursive model is pure
Python) against precompiled CUDA kernels — fastest install.

**Full source build (only if you changed C++/CUDA code):**

```bash
uv pip install -r requirements/lint.txt
uv pip install -r requirements/test/cuda.in   # test deps, optional
VLLM_USE_PRECOMPILED= uv pip install -e . --torch-backend=auto --no-build-isolation
```

**CPU-only (for the parity test on a machine without GPU):**

```bash
CPATH=<numa-include-dir> LIBRARY_PATH=<numa-lib-dir> \
  VLLM_TARGET_DEVICE=cpu uv pip install -e . --no-build-isolation --no-deps
```

CPU build needs `numa.h`. On hosts without sudo, extract the headers locally:

```bash
cd /tmp
apt-get download libnuma-dev
dpkg-deb -x libnuma-dev_*.deb numa_root
mkdir -p numa_lib && ln -s /usr/lib/x86_64-linux-gnu/libnuma.so.1 numa_lib/libnuma.so
export CPATH=/tmp/numa_root/usr/include
export LIBRARY_PATH=/tmp/numa_lib
```

### 3. Verify the install

```bash
python -c "
from vllm.model_executor.models.registry import ModelRegistry
cls = ModelRegistry.resolve_model_cls('RicursiveForCausalLM')
print(cls)
"
# Expected: (<class 'vllm.model_executor.models.ricursive.RicursiveForCausalLM'>, ...)

python -c "
from vllm.transformers_utils.config import get_config
cfg = get_config('ricursiveAI/Ripple-65B-Preview', trust_remote_code=False)
print(type(cfg).__name__, cfg.num_hidden_layers, cfg.hidden_size)
"
# Expected: RicursiveConfig 80 5120   (needs network for the config download)
```

## Serving

### BF16 full model

```bash
vllm serve ricursiveAI/Ripple-65B-Preview \
  --max-model-len 32768 \
  --tensor-parallel-size 8
```

- 65B params in BF16 ≈ 130 GB weights → needs ≥ 2×H100/A100-80G, realistically
  4–8 GPUs with `--tensor-parallel-size 4..8`.
- `sliding_window=1024` and the 4:1 sliding/full layer mix are handled by the
  hybrid KV cache automatically; no extra flags.
- `max_position_embeddings=2097152` flows through from the config; set
  `--max-model-len` to what your memory and workload support (the model card
  says "up to 1M", but every full-attention layer's KV cache at 1M context is
  enormous — size accordingly).

### W8A8 Int8 (compressed-tensors, llm-compressor Auto Round)

```bash
vllm serve /path/to/your-int8-checkpoint \
  --max-model-len 32768 \
  --tensor-parallel-size 8
```

- Quantization is auto-detected from the checkpoint's `quantization_config`;
  passing `--quantization compressed-tensors` explicitly also works.
- The W8A8 Int8 kernels are architecture-agnostic; this branch only supplies
  the correct layer names (`q_proj/k_proj/v_proj/o_proj/g_proj` unfused,
  `gate_proj/up_proj` merged into `gate_up_proj`) so per-projection quant
  targets from llm-compressor match. No special flags needed.
- Per-head Q/K RMSNorms, `value_res_lambda`, and norms are never quantized
  (they are not `LinearBase` layers).

### Quantizing the checkpoint yourself

```bash
pip install llmcompressor
python - <<'EOF'
from llmcompressor.transformers import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "ricursiveAI/Ripple-65B-Preview", trust_remote_code=True, dtype="auto"
)
tokenizer = AutoTokenizer.from_pretrained(
    "ricursiveAI/Ripple-65B-Preview", trust_remote_code=True
)
recipe = QuantizationModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"])
oneshot(model=model, tokenizer=tokenizer, recipe=recipe,
        output_dir="./ripple-65b-w8a8-int8")
EOF
```

`targets="Linear"` names layers by their HF names (`q_proj`, `g_proj`, …),
which this vLLM implementation preserves 1:1.

## Running the tests

```bash
# HF-vs-vLLM numerical parity (runs on CPU or GPU)
python -m pytest tests/models/language/generation/test_ricursive.py -v

# On CPU-only hosts the vLLM side needs a CPU build of vLLM (see step 2).
```

The test builds a tiny random 2-layer ricursive checkpoint locally (vendored
HF remote code, no network) and compares greedy logprobs between the HF
remote-code model and the native vLLM implementation. It covers embed scale,
value residual, NoPE full-attention layer, sliding-window layer, gated
attention, per-head Q/K norms, and FFN clamping.

## Known transformers 5 caveat

The published HF remote code (`modeling_ricursive.py`) has three issues under
transformers ≥ 5.17 **when calling HF `generate()`** (vLLM serving is
unaffected — vLLM does not use HF's generation loop):

1. `prepare_inputs_for_generation` does not handle `cache_position=None`.
2. It does not accept/forward `next_sequence_length` (new in transformers 5),
   so decode steps feed the full sequence again.
3. The sliding-window mask derives query positions from a stale
   `cache_position` instead of the real KV length.

The test fixture in `tests/models/language/generation/ricursive_fixtures/`
contains the corrected versions. If you need HF-side `generate()` on the full
65B model (e.g. for reference outputs), apply the same three fixes to the
downloaded remote code, or use the fixture versions as a patch.

transformers nightly is **not** required for anything in this branch.

## File map

| File | Purpose |
|---|---|
| `vllm/model_executor/models/ricursive.py` | Native model implementation |
| `vllm/transformers_utils/configs/ricursive.py` | Config class (registered for `model_type: ricursive`) |
| `vllm/model_executor/models/registry.py` | Registry entry |
| `vllm/transformers_utils/config.py`, `vllm/transformers_utils/configs/__init__.py` | Config registration |
| `tests/models/language/generation/test_ricursive.py` | HF-vs-vLLM parity test |
| `tests/models/language/generation/ricursive_fixtures/` | Tiny-checkpoint builder + corrected HF remote code |
