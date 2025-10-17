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
    Calculate perplexity using sliding window approach.
    
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
    
    total_nll = 0.0
    total_tokens = 0
    
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
    
    for i in tqdm(range(num_windows), desc="Computing perplexity"):
        start_idx, end_idx = windows[i]
        window_tokens = token_ids[start_idx:end_idx]
        
        if len(window_tokens) < 2:
            # Need at least 2 tokens (context + target)
            continue
        
        # Get logprobs for this window
        outputs = llm.generate(
            prompts=[TokensPrompt(prompt_token_ids=window_tokens)],
            sampling_params=sampling_params,
        )
        
        output = outputs[0]
        
        # Calculate NLL for tokens in this window
        # Skip first token (no context) in the first window
        # For subsequent windows, skip tokens that were already evaluated
        if len(token_ids) <= context_length:
            # Single window case - evaluate all but first token
            start_eval = 1
            end_eval = len(window_tokens)
        else:
            # Multi-window case
            start_eval = 1 if i == 0 else stride
            end_eval = len(window_tokens) if end_idx == len(token_ids) else len(window_tokens)
        
        if output.prompt_logprobs:
            for j in range(start_eval, end_eval):
                if j < len(output.prompt_logprobs) and output.prompt_logprobs[j]:
                    actual_token = window_tokens[j]
                    if actual_token in output.prompt_logprobs[j]:
                        logprob = output.prompt_logprobs[j][actual_token].logprob
                        total_nll += -logprob
                        total_tokens += 1
    
    if total_tokens == 0:
        raise ValueError("No valid tokens found for perplexity calculation")
    
    avg_nll = total_nll / total_tokens
    perplexity = math.exp(avg_nll)
    
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
    }
    
    if args.quantization:
        llm_kwargs["quantization"] = args.quantization
    
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

