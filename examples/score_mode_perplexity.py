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
    
    # Slide window across the sequence
    num_windows = (len(token_ids) - context_length) // stride + 1
    
    for i in tqdm(range(num_windows), desc="Computing perplexity"):
        start_idx = i * stride
        end_idx = min(start_idx + context_length, len(token_ids))
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
        # Skip first token (no context) and last token if we're not at the end
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
        help="Dataset to load (e.g., 'wikitext-2-raw-v1')",
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
    
    args = parser.parse_args()
    
    # Initialize model
    print(f"Loading model: {args.model}")
    llm_kwargs = {
        "model": args.model,
        "enforce_eager": True,  # For simplicity in example
    }
    
    if args.quantization:
        llm_kwargs["quantization"] = args.quantization
    
    if args.max_model_len:
        llm_kwargs["max_model_len"] = args.max_model_len
    
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    
    # Get text to evaluate
    if args.text:
        text = args.text
    elif args.dataset:
        # Load dataset
        try:
            from datasets import load_dataset
            dataset = load_dataset(args.dataset, split="test")
            # Concatenate all text
            text = "\n\n".join(dataset["text"][:100])  # Use first 100 examples
            print(f"Loaded {len(dataset)} examples from {args.dataset}")
        except ImportError:
            print("Please install datasets: pip install datasets")
            return
    else:
        # Use sample text
        text = """
        The quick brown fox jumps over the lazy dog. This is a sample text
        for demonstrating perplexity calculation using vLLM's score_mode.
        With score_mode enabled, we can compute exact log probabilities for
        all tokens in the vocabulary, which is essential for accurate
        perplexity measurement on quantized models.
        """
    
    # Tokenize
    print("\nTokenizing...")
    token_ids = tokenizer.encode(text)
    print(f"Total tokens: {len(token_ids)}")
    
    # Calculate perplexity
    print(f"\nCalculating perplexity with:")
    print(f"  Context length: {args.context_length}")
    print(f"  Stride: {args.stride}")
    
    perplexity = calculate_perplexity(
        llm,
        token_ids,
        context_length=args.context_length,
        stride=args.stride,
    )
    
    print(f"\n{'='*50}")
    print(f"Perplexity: {perplexity:.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

