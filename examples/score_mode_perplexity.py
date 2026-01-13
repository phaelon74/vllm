#!/usr/bin/env python3
"""
Perplexity calculation script using vLLM's score mode.

This script implements EXL3-compatible sliding window perplexity calculation,
evaluating all tokens in each window (including overlaps) for accurate comparison.

Usage:
    python examples/score_mode_perplexity.py \
        --model /path/to/model \
        --dataset wikitext \
        --dataset-config wikitext-2-raw-v1 \
        --num-samples 100 \
        --context-length 2048 \
        --stride 512
"""

import argparse
import time
from typing import Any

from datasets import load_dataset
from vllm import LLM, SamplingParams
from vllm.inputs.data import TokensPrompt


def calculate_perplexity(
    llm: LLM,
    texts: list[str],
    context_length: int,
    stride: int,
    num_samples: int | None = None,
) -> tuple[float, int]:
    """
    Calculate perplexity using EXL3-compatible sliding window approach.

    Args:
        llm: Initialized vLLM LLM instance
        texts: List of text samples to evaluate
        context_length: Maximum context length for each window
        stride: Stride between windows (overlap = context_length - stride)
        num_samples: Maximum number of samples to process (None = all)

    Returns:
        Tuple of (perplexity, total_tokens)
    """
    total_nll = 0.0
    total_tokens = 0

    samples_to_process = texts[:num_samples] if num_samples else texts

    for sample_idx, text in enumerate(samples_to_process):
        # Tokenize the text
        tokenizer = llm.llm_engine.tokenizer.tokenizer
        tokens = tokenizer.encode(text, add_special_tokens=False)

        if len(tokens) < 2:
            continue

        # Process with sliding windows
        for start_idx in range(0, len(tokens) - 1, stride):
            # Create window: [start_idx : start_idx + context_length]
            end_idx = min(start_idx + context_length, len(tokens))
            window_tokens = tokens[start_idx:end_idx]

            if len(window_tokens) < 2:
                continue

            # Target tokens are all tokens after the first one
            # For position i, we want the logprob of token at position i+1
            target_token_ids = window_tokens[1:]

            # Create prompt with target_token_ids for score mode
            prompt: TokensPrompt = {
                "prompt_token_ids": window_tokens,
                "target_token_ids": target_token_ids,
            }

            # Use score_mode for efficient logprob extraction
            sampling_params = SamplingParams(
                prompt_logprobs=1,  # Request prompt logprobs
                max_tokens=0,  # Don't generate any tokens
                score_mode=True,  # Enable score mode for GPU-side extraction
            )

            # Generate (this will only compute prompt logprobs, no generation)
            outputs = llm.generate([prompt], sampling_params=sampling_params)

            # Extract logprobs from output
            output = outputs[0]
            if output.prompt_logprobs:
                # prompt_logprobs[0] is None (position 0 has no logprobs)
                # prompt_logprobs[1] contains logprobs for window_tokens[1]
                # prompt_logprobs[i] contains logprobs for window_tokens[i]
                # Start from index 1 (skip None at index 0)
                for i in range(1, len(output.prompt_logprobs)):
                    logprobs_dict = output.prompt_logprobs[i]
                    if logprobs_dict:
                        actual_token = window_tokens[i]
                        if actual_token in logprobs_dict:
                            logprob = logprobs_dict[actual_token].logprob
                            total_nll += -logprob
                            total_tokens += 1

        if (sample_idx + 1) % 10 == 0:
            print(f"Processed {sample_idx + 1}/{len(samples_to_process)} samples")

    if total_tokens == 0:
        raise ValueError("No valid tokens found for perplexity calculation")

    perplexity = (total_nll / total_tokens).exp()
    return perplexity.item(), total_tokens


def load_dataset_texts(
    dataset_name: str,
    dataset_config: str | None = None,
    split: str | None = None,
) -> list[str]:
    """
    Load and extract text from a HuggingFace dataset.

    Args:
        dataset_name: Name of the dataset (e.g., "wikitext", "neuralmagic/LLM_compression_calibration")
        dataset_config: Optional dataset configuration
        split: Optional split name (auto-detected if None)

    Returns:
        List of text strings
    """
    # Try to auto-detect split if not provided
    if split is None:
        for candidate_split in ["test", "train", "validation"]:
            try:
                if dataset_config:
                    dataset = load_dataset(dataset_name, dataset_config, split=candidate_split)
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

    # Load the dataset
    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config, split=split)
    else:
        dataset = load_dataset(dataset_name, split=split)

    texts = []
    for example in dataset:
        # Handle different dataset formats
        if "text" in example:
            text = example["text"]
            if text and text.strip():
                texts.append(text)
        elif "messages" in example:
            # Handle chat format datasets
            messages = example["messages"]
            if isinstance(messages, list):
                # Concatenate all messages
                text = "\n".join(
                    msg.get("content", "") for msg in messages if isinstance(msg, dict)
                )
                if text and text.strip():
                    texts.append(text)
        else:
            # Try to find any string field
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
        help="Dataset name (e.g., 'wikitext', 'neuralmagic/LLM_compression_calibration')",
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
        help="Number of samples to process (default: all)",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=2048,
        help="Context length for each window (default: 2048)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=512,
        help="Stride between windows (default: 512)",
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

    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset}")
    texts = load_dataset_texts(args.dataset, args.dataset_config)
    print(f"Loaded {len(texts)} text samples")

    # Initialize LLM with score mode optimizations
    llm_kwargs: dict[str, Any] = {
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": args.trust_remote_code,
        "enable_prefix_caching": False,  # Disable prefix caching for accurate perplexity
        "max_model_len": args.context_length * 2,  # Set reasonable max_model_len
    }

    if args.quantization:
        llm_kwargs["quantization"] = args.quantization

    print(f"Initializing LLM with model: {args.model}")
    llm = LLM(model=args.model, **llm_kwargs)

    print(f"\nCalculating perplexity...")
    print(f"  Context length: {args.context_length}")
    print(f"  Stride: {args.stride}")
    print(f"  Samples: {args.num_samples or len(texts)}")

    start_time = time.time()
    perplexity, total_tokens = calculate_perplexity(
        llm,
        texts,
        args.context_length,
        args.stride,
        args.num_samples,
    )
    elapsed_time = time.time() - start_time

    print(f"\nResults:")
    print(f"  Perplexity: {perplexity:.4f}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Time elapsed: {elapsed_time:.2f} seconds")
    print(f"  Tokens/second: {total_tokens / elapsed_time:.2f}")


if __name__ == "__main__":
    main()
