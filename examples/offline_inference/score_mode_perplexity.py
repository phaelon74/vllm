#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Perplexity using vLLM score mode.

Concatenates corpus text, tokenizes, then scores non-overlapping rows of
``--context-length`` tokens (Turbo/EXL3 windowing). ``--stride`` smaller
than context length reproduces historical overlapping windows and is not
the EXL3 default.

Usage:
    python examples/offline_inference/score_mode_perplexity.py \\
        --model /path/to/model \\
        --dataset wikitext --dataset-config wikitext-2-raw-v1
"""

import argparse
import logging
import math
import os
import time
from typing import Any

from datasets import load_dataset

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

logger = logging.getLogger(__name__)

# Best-effort compiled determinism config (--compiled only). See
# score_mode_kld.py and docs/features/score_mode.md.
DETERMINISTIC_COMPILATION_CONFIG: dict[str, Any] = {
    "inductor_compile_config": {
        "combo_kernels": False,
        "benchmark_combo_kernel": False,
        "triton.autotune_pointwise": False,
        "max_autotune": False,
        "coordinate_descent_tuning": False,
        "benchmark_fusion": False,
    },
}


def apply_deterministic_env() -> None:
    """Best-effort: disable timing-based autotuners (--compiled only)."""
    os.environ.setdefault("TORCHINDUCTOR_DETERMINISTIC", "1")
    os.environ.setdefault("VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE", "0")
    os.environ.setdefault("VLLM_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING", "0")
    try:
        import torch._inductor.config as inductor_config

        if hasattr(inductor_config, "deterministic"):
            inductor_config.deterministic = True
    except ImportError:
        pass


def apply_compiled_llm_kwargs(llm_kwargs: dict[str, Any]) -> None:
    """Apply best-effort compiled determinism settings (--compiled only)."""
    apply_deterministic_env()
    llm_kwargs["compilation_config"] = DETERMINISTIC_COMPILATION_CONFIG
    llm_kwargs["enable_flashinfer_autotune"] = False


def apply_eager_llm_kwargs(llm_kwargs: dict[str, Any]) -> None:
    """Apply guaranteed bit-reproducible eager execution (default)."""
    llm_kwargs["enforce_eager"] = True


def _extract_logprobs_from_window(
    output_prompt_logprobs: list,
    window_tokens: list[int],
) -> tuple[float, int]:
    """Extract log-probabilities for each target position in a window.

    Returns (logprob_sum, count) for the window.
    """
    if not output_prompt_logprobs:
        raise ValueError("prompt_logprobs is None or empty")

    if len(output_prompt_logprobs) != len(window_tokens):
        raise ValueError(
            f"prompt_logprobs length ({len(output_prompt_logprobs)}) "
            f"does not match window length ({len(window_tokens)})"
        )

    window_sum = 0.0
    window_count = 0
    for i in range(1, len(output_prompt_logprobs)):
        logprobs_dict = output_prompt_logprobs[i]
        if logprobs_dict:
            actual_token = window_tokens[i]
            if actual_token in logprobs_dict:
                window_sum += logprobs_dict[actual_token].logprob
                window_count += 1

    return window_sum, window_count


def calculate_perplexity(
    llm: LLM,
    texts: list[str],
    context_length: int,
    stride: int,
    rows: int,
    num_samples: int | None = None,
) -> tuple[float, int]:
    """Score non-overlapping (or strided) rows and return perplexity.

    Concatenates texts, tokenizes once, then evaluates ``rows`` windows of
    ``context_length``. Default stride equals context_length (no overlap).
    """
    from vllm.v1.sample.kld import iter_eval_rows

    logprob_sum = 0.0
    logprob_count = 0

    samples_to_process = texts[:num_samples] if num_samples else texts
    concatenated_text = "\n\n".join(samples_to_process)

    tokens = llm.llm_engine.tokenizer.encode(
        concatenated_text, add_special_tokens=False
    )
    windows = iter_eval_rows(tokens, context_length, stride, rows)
    sampling_params = SamplingParams(
        prompt_logprobs=1,
        max_tokens=1,
        score_mode=True,
    )
    windows_processed = 0
    for window_tokens in windows:
        if len(window_tokens) < 2:
            continue
        windows_processed += 1
        prompt: TokensPrompt = {
            "prompt_token_ids": window_tokens,
            "target_token_ids": window_tokens[1:],
        }
        outputs = llm.generate([prompt], sampling_params=sampling_params)
        window_sum, window_count = _extract_logprobs_from_window(
            outputs[0].prompt_logprobs, window_tokens
        )
        logprob_sum += window_sum
        logprob_count += window_count
        if windows_processed % 100 == 0:
            print(
                f"Processed {windows_processed} windows, "
                f"{logprob_count} tokens evaluated"
            )

    if logprob_count == 0:
        raise ValueError("No valid tokens found for perplexity calculation")

    logger.debug(
        "Evaluation complete: %d windows, %d tokens",
        windows_processed,
        logprob_count,
    )
    mean_log_prob = logprob_sum / logprob_count
    perplexity = math.exp(-mean_log_prob)
    return perplexity, logprob_count


def load_dataset_texts(
    dataset_name: str,
    dataset_config: str | None = None,
    split: str | None = None,
) -> list[str]:
    """Load and extract text from a HuggingFace dataset.

    Supports datasets with "text" fields, chat-format "messages" fields,
    or falls back to the first string field found.
    """
    if split is None:
        for candidate_split in ["test", "train", "validation"]:
            try:
                if dataset_config:
                    dataset = load_dataset(
                        dataset_name, dataset_config, split=candidate_split
                    )
                else:
                    dataset = load_dataset(dataset_name, split=candidate_split)
                split = candidate_split
                break
            except Exception:
                continue

        if split is None:
            raise ValueError(
                f"Could not load dataset {dataset_name} with any split "
                "(test/train/validation)"
            )

    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config, split=split)
    else:
        dataset = load_dataset(dataset_name, split=split)

    texts = []
    for example in dataset:
        if "text" in example:
            text = example["text"]
            if text and text.strip():
                texts.append(text)
        elif "messages" in example:
            messages = example["messages"]
            if isinstance(messages, list):
                text = "\n".join(
                    msg.get("content", "") for msg in messages if isinstance(msg, dict)
                )
                if text and text.strip():
                    texts.append(text)
        else:
            for key, value in example.items():
                if isinstance(value, str) and value.strip():
                    texts.append(value)
                    break

    if not texts:
        raise ValueError(f"No valid text found in dataset {dataset_name}")

    return texts


def main():
    parser = argparse.ArgumentParser(
        description="Calculate perplexity using vLLM's score mode"
    )
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        help="Quantization method (e.g., 'awq', 'gptq', 'compressed-tensors')",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., 'wikitext')",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default=None,
        help="Dataset configuration (e.g., 'wikitext-2-raw-v1')",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of dataset rows to concatenate (default: all)",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=100,
        help="Number of evaluation rows (default: 100)",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=2048,
        help="Tokens per row (default: 2048)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="DEPRECATED overlapping stride. Default is context-length.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallelism size (default: 1)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.30,
        help="GPU memory utilization (default: 0.30)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading model",
    )
    parser.add_argument(
        "--compiled",
        action="store_true",
        help="Use torch.compile with best-effort determinism settings. "
        "Faster but NOT bit-reproducible run-to-run; for speed experiments "
        "only. Eager mode is the default.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    print(f"Loading dataset: {args.dataset}")
    texts = load_dataset_texts(args.dataset, args.dataset_config)
    print(f"Loaded {len(texts)} text samples")

    llm_kwargs: dict[str, Any] = {
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": args.trust_remote_code,
        "enable_prefix_caching": False,
        "max_model_len": args.context_length * 2,
    }

    if args.quantization:
        llm_kwargs["quantization"] = args.quantization

    if args.compiled:
        apply_compiled_llm_kwargs(llm_kwargs)
        print(
            "WARNING: --compiled mode is NOT bit-reproducible run-to-run. "
            "Use default eager mode for authoritative perplexity scoring."
        )
    else:
        apply_eager_llm_kwargs(llm_kwargs)
        print("Deterministic (eager) mode: bit-reproducible scoring")

    print(f"Initializing LLM with model: {args.model}")
    llm = LLM(model=args.model, **llm_kwargs)

    if args.stride is None:
        stride = args.context_length
    else:
        stride = args.stride
        print(
            "WARNING: --stride is deprecated overlapping windowing. "
            "Omit it for non-overlapping rows (EXL3/Turbo)."
        )

    print("\nCalculating perplexity...")
    print(f"  Context length: {args.context_length}")
    print(f"  Stride: {stride}")
    print(f"  Rows: {args.rows}")
    print(f"  Samples: {args.num_samples or len(texts)}")

    start_time = time.time()
    perplexity, total_tokens = calculate_perplexity(
        llm,
        texts,
        args.context_length,
        stride,
        args.rows,
        args.num_samples,
    )
    elapsed_time = time.time() - start_time

    print("\nResults:")
    print(f"  Perplexity: {perplexity:.4f}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Time elapsed: {elapsed_time:.2f} seconds")
    print(f"  Tokens/second: {total_tokens / elapsed_time:.2f}")


if __name__ == "__main__":
    main()
