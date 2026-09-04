# Score Mode: Perplexity and KLD Evaluation

This page describes how to use vLLM's score mode for efficient model evaluation
via perplexity (PPL) and Kullback-Leibler divergence (KLD) computation.

## Overview

Score mode enables GPU-side extraction of log-probabilities for specific target
tokens, avoiding the overhead of transferring full vocabulary logprobs to CPU.
This makes perplexity and KLD calculations significantly faster than extracting
all logprobs and post-processing on CPU.

Score mode is implemented on both the V1 GPU model runner and the V2
`PromptLogprobsWorker`. GLM-5.3-Flash requires V2; do not pin
`VLLM_USE_V2_MODEL_RUNNER=0` for that model.

`SamplingParams` fields:

| Parameter | Type | Description |
|-----------|------|-------------|
| `score_mode` | `bool` | Extract only target token logprobs on GPU (for PPL). Requires `prompt_logprobs` to be set. |
| `return_prompt_logits` | `bool` | Return raw logits for all prompt positions (for generating reference logits). |
| `return_prompt_hidden_states` | `bool` | Return pre-LM-head hidden states. Combinable with `return_prompt_logits` for the replay probe. Mutually exclusive with `kld_mode`. |
| `kld_mode` | `bool` | Compute KL divergence on GPU against a reference capture. Mutually exclusive with `return_prompt_logits` and `return_prompt_hidden_states`. |

`TokensPrompt` fields:

| Field | Type | Description |
|-------|------|-------------|
| `target_token_ids` | `list[int]` | Target tokens for score mode (typically `prompt_token_ids[1:]`). |
| `reference_logits_path` | `str` | Path to a safetensors file with reference logits or hidden states. |
| `reference_logits_key` | `str` | Key within the file (`logits` / `hidden_states`, or legacy `logits_N`). |
| `kld_vocab_size` | `int` | Unpadded tokenizer vocab. Softmax is truncated to this width so padding rows cannot contribute. |

## Vocabulary rule

KL is `KL(reference || candidate)` in nats over the unpadded vocabulary.
Both runners call `vllm.v1.sample.kld.compute_kld_chunk`, which truncates
logits to `min(model_width, ref_width, kld_vocab_size)` **before** softmax.
Always pass `kld_vocab_size` from `tokenizer.actual_vocab_size` when available,
falling back to `tokenizer.vocab_size`. KLD mode requires a positive value.
Scoring refuses to proceed if the live tokenizer
identity or token-id hash disagrees with the capture manifest.

## Windowing

Default evaluation is **non-overlapping rows**: `--rows 100` (default) times
`--context-length 2048` (default), with stride equal to context length. That
matches Turbo/EXL3 `get_test_tokens(..., eval_len, eval_len)`.

`--stride` is a deprecated alias that restores historical overlapping windows
and prints a warning. `--score-from` (default `0`) skips a leading prefix from
**every row**; use `context_length // 2` for llama.cpp / Turbo deep-context parity.
Depth buckets are always reported so the deep-context number is a derived
view rather than the only number.

`--num-samples` controls how much dataset text is concatenated. There is no
`99 * stride` cap.

## Perplexity Calculation

Perplexity measures how well a model predicts a sequence. Lower values indicate
better predictions. Score mode computes per-token logprobs on GPU for specified
target tokens, which are then aggregated into a perplexity score.

### Quick Start

```python
from vllm import LLM, SamplingParams
from vllm.inputs.data import TokensPrompt

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", enforce_eager=True)

# Tokenize your text
tokens = llm.llm_engine.tokenizer.encode(
    "The quick brown fox jumps over the lazy dog.",
    add_special_tokens=False,
)

prompt: TokensPrompt = {
    "prompt_token_ids": tokens,
    "target_token_ids": tokens[1:],  # predict each next token
}

sampling_params = SamplingParams(
    prompt_logprobs=1,
    max_tokens=1,
    score_mode=True,
)

outputs = llm.generate([prompt], sampling_params=sampling_params)
```

For a complete non-overlapping-row implementation, see:

[examples/offline_inference/score_mode_perplexity.py](../../examples/offline_inference/score_mode_perplexity.py)

## KL Divergence Calculation

KLD measures how much a candidate model's predictions diverge from a reference
(typically full-precision) model. Lower values indicate the candidate preserves
more of the original model's behavior. Self-KLD of a model against its own
capture must be **exactly 0.0**; a non-zero floor is measurement error.

### Two-Phase Workflow

1. **Phase 1 -- Capture**: Run the reference model with
   `return_prompt_logits=True` (and optionally `return_prompt_hidden_states`)
   and write per-row safetensors plus `manifest.json`.
2. **Phase 2 -- Score**: Run the candidate with `kld_mode=True`. The engine
   loads the matching reference slice (logits, or hidden states replayed
   through a bundled teacher `lm_head.safetensors`) and returns a per-position
   `KLDResult`.

Default storage is **full logits** (`--storage logits`). `--storage hidden`
stores teacher hidden states in their original dtype plus the teacher head and
is allowed only when `--probe-replay` reports bitwise equality through the
same TP-aware `compute_logits` path used during scoring. `--storage auto` uses
hidden storage if the probe is exact
and otherwise keeps logits. If the probe is not exact, stop; logits remain
authoritative.

### Example script flags

```bash
python examples/offline_inference/score_mode_kld.py \
  --model /path/to/QUANT_MODEL \
  --reference-model /path/to/BF16_MODEL \
  --dataset wikitext --dataset-config wikitext-2-raw-v1 \
  --rows 100 --context-length 2048 --score-from 0 \
  --storage logits --probe-replay
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--rows` | 100 | Number of evaluation rows |
| `--context-length` | 2048 | Tokens per row |
| `--stride` | context-length | Deprecated overlapping stride |
| `--score-from` | 0 | Skip this many leading positions per row |
| `--storage` | logits | `logits`, `hidden`, or `auto` |
| `--probe-replay` | off | Bitwise live-logits vs hidden-state replay |
| `--decompose-head` | off | Report trunk (shared teacher head) vs deployed (student logits) KLD |
| `--report-json` | off | Persist score statistics plus teacher/student head provenance |
| `--num-samples` | all | Dataset rows concatenated before windowing |
| `--capture-only` | off | Phase 1 only |

A `GlmMoeDsaForCausalLM` checkpoint (or an explicit GLM-5.3 model path) always runs
`scripts/glm53_determinism_probe.py --gate --gate-max-self-kld 0.0` before
capture. There is no bypass. Capture refuses if the gate fails or
`VLLM_SPARSE_MLA_SORT_TOPK=1` is not explicitly set.

### Manifest

Each capture directory contains `manifest.json` with token-id SHA-256,
tokenizer vocabulary fingerprint, rows/length/stride/score-from,
`kld_vocab_size`, TP, actual V1/V2 runner, eager flag, GPU/driver/PyTorch
runtime, static and runtime LM-head state, per-file hashes, and the replay
probe result. Scoring a directory **requires** this file and aborts on any
mismatch, including a different tokenizer, runtime, model runner, token hash,
or capture-file hash.

The complete CLI workflow refuses legacy single-file references. Direct API
calls may use a single tensor file for focused tests, but then provenance is
the caller's responsibility. KLD requests also fail closed if scheduler
preemption during prompt processing would reset reference-position
accumulation; run them without competing work or KV-cache pressure.

### Result object

`RequestOutput.kld_result` is a `KLDResult` named tuple (lists, not a
`(sum, count)` pair):

- `kld_ref_to_model`, `kld_model_to_ref`
- `ref_top1_prob`, `model_top1`, `ref_top1`, `topk_agree`

Mean KLD is `kld_result.kld_sum / kld_result.kld_count`. The example script
prints mean and reverse KL, median/p90/p99/max, Turbo confidence bins
`[0, .25) [.25, .5) [.5, .75) [.75, .95) [.95, 1]`, context-depth buckets,
and top-K agreement for K = 1..5.

### Quick Start (API)

```python
from vllm import LLM, SamplingParams
from vllm.inputs.data import TokensPrompt

# Phase 1: Get reference logits
ref_llm = LLM(model="/path/to/reference_model", enforce_eager=True)

prompt: TokensPrompt = {
    "prompt_token_ids": tokens,
}

ref_params = SamplingParams(
    max_tokens=1,
    return_prompt_logits=True,
)

outputs = ref_llm.generate([prompt], sampling_params=ref_params)
ref_logits = outputs[0].prompt_logits  # [num_positions, vocab_size]

# Save ref_logits to safetensors, then unload reference model...

# Phase 2: Compute KLD
test_llm = LLM(model="/path/to/quantized_model", enforce_eager=True)

prompt_kld: TokensPrompt = {
    "prompt_token_ids": tokens,
    "reference_logits_path": "/path/to/ref_logits.safetensors",
    "reference_logits_key": "logits",
    "kld_vocab_size": getattr(
        tokenizer, "actual_vocab_size", tokenizer.vocab_size
    ),
}

kld_params = SamplingParams(
    max_tokens=1,
    kld_mode=True,
)

outputs = test_llm.generate([prompt_kld], sampling_params=kld_params)
kld = outputs[0].kld_result
mean_kld = kld.kld_sum / kld.kld_count
```

For a complete implementation that handles both phases, non-overlapping rows,
manifests, and optional hidden-state storage, see:

[examples/offline_inference/score_mode_kld.py](../../examples/offline_inference/score_mode_kld.py)

Manual verification commands live in
[docs/design/kld_manual_verification.md](../design/kld_manual_verification.md).

## Determinism

Scoring is only meaningful if the same command produces the same score every
time. Investigation on this fork found:

- **Eager execution** (`enforce_eager=True`) removes graph-level timing choices,
  but does not make every custom GPU kernel bit-reproducible. Marlin MoE, for
  example, has produced 1–2 ULP BF16 differences from identical inputs and
  routing. Paired routed scoring measures and publishes this repeatability
  floor instead of assuming it away.
- The **compiled stack** wobbles run-to-run even with every known timing-based
  selector disabled (`combo_kernels`, Inductor pointwise autotune,
  `TORCHINDUCTOR_DETERMINISTIC=1`, FlashInfer autotune). It converges to an
  attractor value only after repeated warm runs and deviates again after idle
  time. Not certifiable for scoring.
- **CUDA graphs** are numerically neutral (identical converged KLD with and
  without graph capture on the tested stack).
- Reference logits and test runs must use the **same execution stack**. An
  eager-vs-compiled baseline offset was directly measured (~0.001 KLD).
- GLM-5.3-Flash sparse MLA is nondeterministic unless
  `VLLM_SPARSE_MLA_SORT_TOPK=1`. See
  [kld_determinism_floor.md](../design/kld_determinism_floor.md).

### Rules

1. **Scoring runs eager.** The example scripts use eager mode by default. API
   users must pass `enforce_eager=True`. Paired routed scoring repeats both its
   natural and forced-natural controls because eager custom kernels may still
   be numerically nondeterministic.
2. **Never mix stacks.** References and every scored model must use the same
   execution mode on the same GPU, driver, and PyTorch build. Regenerate
   references after any of those change.
3. **`--compiled` is for speed experiments only.** It applies best-effort
   determinism settings but is **not** bit-reproducible run-to-run on the
   current stack.
4. **Use the runner the model requires.** Both V1 and V2 implement score mode
   through `compute_kld_chunk`. GLM-5.3-Flash needs V2. Do not pin V1 globally.

### Eager scoring commands

Generate reference logits and score a quant (one pass each; no extra flags):

```bash
python examples/offline_inference/score_mode_kld.py \
  --model /path/to/QUANT_MODEL \
  --reference-model /path/to/BF16_MODEL \
  --dataset wikitext --dataset-config wikitext-2-raw-v1 \
  --rows 100 --context-length 2048 \
  --tensor-parallel-size 1 --gpu-memory-utilization 0.85

python examples/offline_inference/score_mode_kld.py \
  --model /path/to/QUANT_MODEL \
  --reference-logits ./ref_BF16_MODEL_rows100_ctx2048_s2048 \
  --dataset wikitext --dataset-config wikitext-2-raw-v1 \
  --rows 100 --context-length 2048 \
  --tensor-parallel-size 1 --gpu-memory-utilization 0.85
```

Perplexity uses the same default (eager); see
[score_mode_perplexity.py](../../examples/offline_inference/score_mode_perplexity.py).

### API usage (eager, deterministic)

```python
llm = LLM(
    model=...,
    enforce_eager=True,
    enable_prefix_caching=False,
)
```

### Verify reference logits are byte-identical

Generate references twice into separate directories, then diff hashes:

```bash
diff <(cd ref_eager_a && sha256sum logits_*.safetensors) \
     <(cd ref_eager_b && sha256sum logits_*.safetensors)
```

An empty diff confirms single-pass reference generation is safe.

### Compiled mode (`--compiled`, not for authoritative scoring)

Pass `--compiled` to the example scripts to enable `torch.compile` with
best-effort settings (combo kernels off, Inductor autotune off,
`TORCHINDUCTOR_DETERMINISTIC=1`, FlashInfer autotune off). This is faster but
**not** bit-reproducible run-to-run. Evidence: repeated runs converge to an
attractor then deviate after idle gaps; disabling CUDA graphs does not change
the converged value.

### Rounding noise vs accuracy

Quantization effects (~1e-2 KLD) dwarf kernel rounding noise (~1e-7). Compiled
wobble corrupts **reproducibility**, not model accuracy. Argmax token flips from
kernel rounding occur only at model-declared ties (top-1/top-2 logit gap
under ~1e-3). Verify with the forensic diagnostic:

[examples/offline_inference/score_mode_argmax_diag.py](../../examples/offline_inference/score_mode_argmax_diag.py)

Generate logits under default (compiled wobble) or `--deterministic` (eager
ground truth), then `compare` two directories.

## Constraints

- `score_mode` requires `prompt_logprobs` to be set.
- `kld_mode` is mutually exclusive with `return_prompt_logits` and
  `return_prompt_hidden_states`.
- `kld_mode` requires `reference_logits_path` and `reference_logits_key` in
  the prompt. Pass `kld_vocab_size` to exclude padding vocab.
- Hidden-state scoring requires `lm_head.safetensors` beside the capture and
  a bitwise-exact replay probe recorded in the manifest.
- Prefix caching should be disabled (`enable_prefix_caching=False`) for
  accurate evaluation results.
