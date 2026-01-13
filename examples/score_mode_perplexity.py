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
import math
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
    debug: bool = False,
) -> tuple[float, int]:
    """
    Calculate perplexity using EXL3-compatible sliding window approach.

    Args:
        llm: Initialized vLLM LLM instance
        texts: List of text samples to evaluate
        context_length: Maximum context length for each window
        stride: Stride between windows (overlap = context_length - stride)
        num_samples: Maximum number of samples to process (None = all)
        debug: Enable debug logging for first few windows

    Returns:
        Tuple of (perplexity, total_tokens)
    """
    total_nll = 0.0
    total_tokens = 0

    samples_to_process = texts[:num_samples] if num_samples else texts

    # EXL3 approach: Join all texts with "\n\n" separator before tokenizing
    # This matches EXL3's dataset preparation exactly
    concatenated_text = "\n\n".join(samples_to_process)
    
    # Tokenize the entire concatenated text as one sequence
    tokens = llm.llm_engine.tokenizer.encode(concatenated_text, add_special_tokens=False)

    if len(tokens) < 2:
        raise ValueError("Not enough tokens after concatenation")
    
    if debug:
        print(f"Total tokens after concatenation: {len(tokens)}")
        print(f"First 20 tokens: {tokens[:20]}")
        print(f"Last 20 tokens: {tokens[-20:]}")

    # Process with sliding windows (matching EXL3's pattern exactly)
    # EXL3: for a in range(0, num_tokens - eval_len, eval_stride):
    #       b = a + eval_len
    #       seqs.append(eval_tokens[:, a:b])
    # This creates windows of exactly eval_len tokens
    num_tokens = len(tokens)
    windows_processed = 0
    # EXL3's exact pattern: range(0, num_tokens - eval_len, eval_stride)
    # This ensures all windows are exactly context_length tokens
    for start_idx in range(0, num_tokens - context_length, stride):
            # Create window: [start_idx : start_idx + context_length]
            # EXL3: eval_tokens[:, a:b] where b = a + eval_len (exactly eval_len tokens)
            end_idx = start_idx + context_length
            window_tokens = tokens[start_idx:end_idx]
            
            # All windows should be exactly context_length tokens (guaranteed by range)
            assert len(window_tokens) == context_length, f"Window length mismatch: {len(window_tokens)} != {context_length}"

            if len(window_tokens) < 2:
                continue
            
            windows_processed += 1
            
            if debug and windows_processed <= 3:
                print(f"\nWindow {windows_processed}:")
                print(f"  Start index: {start_idx}, End index: {end_idx}")
                print(f"  Window tokens (first 10): {window_tokens[:10]}")
                print(f"  Target tokens (first 10): {target_token_ids[:10]}")

            # EXL3 approach: target_ids = input_ids[:, 1:]
            # Target tokens are all tokens after the first one
            # We evaluate positions 1 through len-1 (all tokens except first)
            target_token_ids = window_tokens[1:]

            # Create prompt with target_token_ids for score mode
            prompt: TokensPrompt = {
                "prompt_token_ids": window_tokens,
                "target_token_ids": target_token_ids,
            }

            # Use score_mode for efficient logprob extraction
            # Note: max_tokens must be at least 1, but we only use prompt logprobs
            sampling_params = SamplingParams(
                prompt_logprobs=1,  # Request prompt logprobs
                max_tokens=1,  # Required to be >= 1, but we only use prompt logprobs
                score_mode=True,  # Enable score mode for GPU-side extraction
            )

            # Generate (this will only compute prompt logprobs, no generation)
            outputs = llm.generate([prompt], sampling_params=sampling_params)

            # Extract logprobs from output
            # EXL3 evaluates ALL tokens in the window (positions 1 through len-1)
            output = outputs[0]
            if output.prompt_logprobs:
                # prompt_logprobs[0] is None (position 0 has no logprobs)
                # prompt_logprobs[1] contains logprobs for window_tokens[1] given context [window_tokens[0]]
                # prompt_logprobs[i] contains logprobs for window_tokens[i] given context [window_tokens[0:i]]
                # We evaluate positions 1 through len(window_tokens)-1 (matching EXL3's logits[:, :-1])
                window_nll = 0.0
                window_token_count = 0
                for i in range(1, len(output.prompt_logprobs)):
                    logprobs_dict = output.prompt_logprobs[i]
                    if logprobs_dict:
                        actual_token = window_tokens[i]
                        if actual_token in logprobs_dict:
                            logprob = logprobs_dict[actual_token].logprob
                            # EXL3: logprob_sum += target_log_probs.sum().item()
                            # We accumulate negative log probabilities
                            nll = -logprob
                            total_nll += nll
                            total_tokens += 1
                            window_nll += nll
                            window_token_count += 1
                            
                            if debug and windows_processed <= 3 and i <= 5:
                                print(f"    Position {i}: token={actual_token}, logprob={logprob:.6f}, nll={nll:.6f}")
                        elif debug and windows_processed <= 3:
                            print(f"    WARNING: Position {i}: token {actual_token} not in logprobs_dict. Keys: {list(logprobs_dict.keys())[:5]}")
                    elif debug and windows_processed <= 3:
                        print(f"    WARNING: Position {i}: logprobs_dict is None or empty")
                
                if debug and windows_processed <= 3:
                    print(f"  Window {windows_processed} summary: {window_token_count} tokens, avg_nll={window_nll/window_token_count if window_token_count > 0 else 0:.6f}")
            
            # Progress logging
            if windows_processed % 100 == 0:
                print(f"Processed {windows_processed} windows, {total_tokens} tokens evaluated")

    if total_tokens == 0:
        raise ValueError("No valid tokens found for perplexity calculation")

    avg_nll = total_nll / total_tokens
    perplexity = math.exp(avg_nll)
    return perplexity, total_tokens


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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging to compare with EXL3",
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
        debug=args.debug,
    )
    elapsed_time = time.time() - start_time

    print(f"\nResults:")
    print(f"  Perplexity: {perplexity:.4f}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Time elapsed: {elapsed_time:.2f} seconds")
    print(f"  Tokens/second: {total_tokens / elapsed_time:.2f}")


if __name__ == "__main__":
    main()
