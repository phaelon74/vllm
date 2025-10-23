#!/usr/bin/env python3
"""
Example: Computing perplexity using vLLM's score_mode.

This script demonstrates how to use score_mode to compute perplexity
for quantized models (e.g., compressed-tensors format) without 
decompressing weights to FP16.

The key innovation is that score_mode:
1. Sets max_tokens=0 (no generation)
2. Enables prompt_logprobs=-1 (returns exact logprobs for ALL vocab tokens)
3. Computes exact log probabilities for each prompt token

This allows accurate perplexity measurement on quantized models.
"""

import argparse
import math
from typing import List
import numpy as np
from tqdm import tqdm

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


def calculate_perplexity(
    llm: LLM,
    token_ids: List[int],
    context_length: int = 2048,
    stride: int = 512,
) -> float:
    """
    Calculate perplexity using EXL3-compatible sliding window approach.
    
    EXL3 methodology (direct from developer):
    1. Tokenize the entire test set
    2. Create overlapping windows:
       - input_ids[0] = tokenized_set[0:2048]
       - input_ids[1] = tokenized_set[512:2560]
       - input_ids[2] = tokenized_set[1024:3072]
       - etc.
    3. For EACH window (treated as a NEW sequence):
       - Get logits = model.forward(input_ids[i])
       - logprobs = log_softmax(logits)
       - Gather target token logprobs (skip position 0 with no context)
       - Sum up gathered logprobs
    4. Final: ppl = exp(-mean of ALL logprobs from ALL windows)
    
    Example with 5000 tokens (context=2048, stride=512):
        Window 0: tokens [   0:2048]  → evaluate positions [1:2048] (2047 evaluations)
        Window 1: tokens [ 512:2560]  → evaluate positions [1:2048] (2047 evaluations)
        Window 2: tokens [1024:3072]  → evaluate positions [1:2048] (2047 evaluations)
        ...
        Tokens in overlapping regions are evaluated multiple times.
        Final perplexity = exp(-mean(all logprobs))
    
    Args:
        llm: vLLM LLM instance
        token_ids: List of token IDs to evaluate
        context_length: Maximum context length for each window
        stride: Stride for sliding window (how many tokens to advance)
        
    Returns:
        Perplexity value
    """
    # Create sampling params with score_mode
    sampling_params = SamplingParams(
        score_mode=True,
        temperature=0.0,
    )
    
    # Accumulate negative log-likelihoods across ALL windows
    # Each window is treated as an independent sequence
    total_nll = 0.0  # Sum of -logprob for all evaluated tokens
    total_tokens = 0  # Count of all evaluated tokens (across all windows)
    
    # Handle case where sequence is shorter than context_length
    if len(token_ids) <= context_length:
        # Just use the whole sequence
        num_windows = 1
        windows = [(0, len(token_ids))]
    else:
        # Slide window across the sequence
        num_windows = (len(token_ids) - context_length) // stride + 1
        windows = []
        for i in range(num_windows):
            start_idx = i * stride
            end_idx = min(start_idx + context_length, len(token_ids))
            windows.append((start_idx, end_idx))
    
    for window_idx in tqdm(range(num_windows), desc="Computing perplexity"):
        start_idx, end_idx = windows[window_idx]
        window_tokens = token_ids[start_idx:end_idx]
        
        if len(window_tokens) < 2:
            # Need at least 2 tokens (context + target)
            continue
        
        # Prepare target_token_ids for optimization (skip position 0, no context for prediction)
        target_token_ids = window_tokens[1:]  # Ground-truth tokens to evaluate
        
        # Get logprobs for this window (with optimization: only extract targets)
        outputs = llm.generate(
            prompts=[TokensPrompt(
                prompt_token_ids=window_tokens,
                target_token_ids=target_token_ids,  # NEW: Enable fast path
            )],
            sampling_params=sampling_params,
        )
        
        output = outputs[0]
        
        # EXL3-compatible: Treat each window as a NEW independent sequence
        # Evaluate ALL positions in this window (except position 0 which has no context)
        # Tokens in overlapping regions will be evaluated multiple times across different windows
        # All evaluations are summed and averaged: ppl = exp(-mean(all logprobs))
        
        if output.prompt_logprobs:
            # prompt_logprobs[0] is None (position 0 has no logprobs)
            # prompt_logprobs[1] contains logprobs for window_tokens[1]
            # prompt_logprobs[i] contains logprobs for window_tokens[i]
            # Start from index 1 (skip None at index 0)
            
            # DEBUG: Verify correct token alignment for first window
            if window_idx == 0 and len(output.prompt_logprobs) > 5:
                print(f"\n[DEBUG] First window verification:")
                print(f"  window_tokens[0:5] = {window_tokens[:5]}")
                print(f"  len(prompt_logprobs) = {len(output.prompt_logprobs)}")
                for i in range(1, min(6, len(output.prompt_logprobs))):
                    logprobs_dict = output.prompt_logprobs[i]
                    expected_token = window_tokens[i]
                    if logprobs_dict:
                        top_token = max(logprobs_dict.items(), key=lambda x: x[1].logprob)
                        print(f"  Position {i}: expecting token {expected_token}, "
                              f"got dict with keys {list(logprobs_dict.keys())[:3]}, "
                              f"top token = {top_token[0]} (logprob={top_token[1].logprob:.4f})")
            
            for i in range(1, len(output.prompt_logprobs)):
                logprobs_dict = output.prompt_logprobs[i]
                if logprobs_dict:
                    actual_token = window_tokens[i]
                    if actual_token in logprobs_dict:
                        logprob = logprobs_dict[actual_token].logprob
                        total_nll += -logprob
                        total_tokens += 1
    
    if total_tokens == 0:
        raise ValueError("No valid tokens found for perplexity calculation")
    
    # Final perplexity calculation (EXL3 formula)
    # ppl = exp(-mean(all logprobs))
    avg_nll = total_nll / total_tokens  # mean of negative log probs
    perplexity = math.exp(avg_nll)       # exp(-mean(logprobs))
    
    return perplexity


def main():
    parser = argparse.ArgumentParser(
        description="Calculate perplexity using vLLM score_mode"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name or path (can be a quantized model)",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        help="Quantization method (e.g., 'compressed-tensors')",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Text to evaluate (if not provided, uses a sample)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset to load (e.g., 'wikitext', 'wikitext-103-raw-v1')",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default=None,
        help="Dataset configuration (e.g., 'wikitext-2-raw-v1', 'wikitext-103-raw-v1')",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to use from dataset (default: all)",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=2048,
        help="Context length for evaluation",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=512,
        help="Stride for sliding window",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Maximum model length",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size for multi-GPU",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization (0.0 to 1.0)",
    )
    parser.add_argument(
        "--disable-custom-all-reduce",
        action="store_true",
        help="Disable custom all-reduce",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code",
    )
    
    args = parser.parse_args()
    
    # Initialize model
    print(f"Loading model: {args.model}")
    llm_kwargs = {
        "model": args.model,
        "enforce_eager": True,  # For simplicity in example
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enable_prefix_caching": False,  # CRITICAL: Disable prefix caching to avoid OOM
        # CRITICAL: Limit max_model_len to avoid allocating massive KV cache
        # Use context_length * 2 to give plenty of headroom for internal buffers
        # (model's max is 131K, but we only need ~2048-4096!)
        "max_model_len": args.context_length * 2,
    }
    
    if args.quantization:
        llm_kwargs["quantization"] = args.quantization
    
    # User can still override max_model_len if they want
    if args.max_model_len:
        llm_kwargs["max_model_len"] = args.max_model_len
    
    if args.disable_custom_all_reduce:
        llm_kwargs["disable_custom_all_reduce"] = True
    
    if args.trust_remote_code:
        llm_kwargs["trust_remote_code"] = True
    
    print(f"Model config:")
    print(f"  Quantization: {args.quantization}")
    print(f"  Tensor parallel size: {args.tensor_parallel_size}")
    print(f"  GPU memory utilization: {args.gpu_memory_utilization}")
    print(f"  Max model length: {args.max_model_len}")
    
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    
    # Get text to evaluate
    if args.text:
        text = args.text
        print(f"Using provided text ({len(args.text)} characters)")
    elif args.dataset:
        # Load dataset
        try:
            from datasets import load_dataset
            
            # Load dataset with optional config
            if args.dataset_config:
                print(f"Loading dataset: {args.dataset} (config: {args.dataset_config})")
                dataset = load_dataset(args.dataset, args.dataset_config, split="test")
            else:
                print(f"Loading dataset: {args.dataset}")
                dataset = load_dataset(args.dataset, split="test")
            
            print(f"Loaded {len(dataset)} examples from dataset")
            
            # Determine how many samples to use
            num_samples = args.num_samples if args.num_samples else len(dataset)
            num_samples = min(num_samples, len(dataset))
            
            # Concatenate text samples
            # For wikitext, join with double newline to separate articles
            text = "\n\n".join(dataset["text"][:num_samples])
            print(f"Using {num_samples} samples ({len(text)} characters)")
            
        except ImportError:
            print("ERROR: Please install datasets library:")
            print("  pip install datasets")
            return
        except Exception as e:
            print(f"ERROR loading dataset: {e}")
            print("\nFor WikiText-2, use:")
            print("  --dataset wikitext --dataset-config wikitext-2-raw-v1")
            print("\nFor WikiText-103, use:")
            print("  --dataset wikitext --dataset-config wikitext-103-raw-v1")
            return
    else:
        print("ERROR: Must provide either --text or --dataset")
        print("\nExamples:")
        print("  --text 'Your text here'")
        print("  --dataset wikitext --dataset-config wikitext-2-raw-v1")
        return
    
    # Tokenize
    print("\nTokenizing...")
    token_ids = tokenizer.encode(text)
    print(f"Total tokens: {len(token_ids)}")
    
    # Calculate perplexity
    print(f"\n{'='*70}")
    print(f"PERPLEXITY EVALUATION")
    print(f"{'='*70}")
    print(f"Context length: {args.context_length}")
    print(f"Stride: {args.stride}")
    print(f"Total tokens to evaluate: {len(token_ids)}")
    print(f"{'='*70}\n")
    
    perplexity = calculate_perplexity(
        llm,
        token_ids,
        context_length=args.context_length,
        stride=args.stride,
    )
    
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"Model: {args.model}")
    print(f"Quantization: {args.quantization}")
    print(f"Dataset: {args.dataset} ({args.dataset_config if args.dataset_config else 'default'})")
    print(f"Total tokens evaluated: {len(token_ids)}")
    print(f"\n>>> Perplexity: {perplexity:.4f} <<<")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

