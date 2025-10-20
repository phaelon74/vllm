#!/usr/bin/env python3
"""
Memory-optimized perplexity calculation using score_mode.

This version is optimized to use less VRAM by processing smaller windows
and being more aggressive about memory cleanup.
"""

import argparse
import math
from typing import List
import gc
import torch
from tqdm import tqdm

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


def calculate_perplexity_optimized(
    llm: LLM,
    token_ids: List[int],
    context_length: int = 512,  # Smaller default
    stride: int = 256,
) -> tuple[float, int]:
    """
    Memory-optimized perplexity calculation.
    
    Args:
        llm: vLLM LLM instance
        token_ids: List of token IDs to evaluate
        context_length: Maximum context length for each window
        stride: Stride for sliding window
        
    Returns:
        (perplexity, num_tokens_evaluated)
    """
    sampling_params = SamplingParams(
        score_mode=True,
        temperature=0.0,
    )
    
    total_nll = 0.0
    total_tokens = 0
    
    # Handle case where sequence is shorter than context_length
    if len(token_ids) <= context_length:
        num_windows = 1
        windows = [(0, len(token_ids))]
    else:
        num_windows = (len(token_ids) - context_length) // stride + 1
        windows = []
        for i in range(num_windows):
            start_idx = i * stride
            end_idx = min(start_idx + context_length, len(token_ids))
            windows.append((start_idx, end_idx))
    
    print(f"Processing {num_windows} windows...")
    
    for i in tqdm(range(num_windows), desc="Computing perplexity"):
        start_idx, end_idx = windows[i]
        window_tokens = token_ids[start_idx:end_idx]
        
        if len(window_tokens) < 2:
            continue
        
        # Get logprobs for this window
        outputs = llm.generate(
            prompts=[TokensPrompt(prompt_token_ids=window_tokens)],
            sampling_params=sampling_params,
        )
        
        output = outputs[0]
        
        # Determine which tokens to evaluate
        if len(token_ids) <= context_length:
            start_eval = 1
            end_eval = len(window_tokens)
        else:
            start_eval = 1 if i == 0 else stride
            end_eval = len(window_tokens) if end_idx == len(token_ids) else len(window_tokens)
        
        # Calculate NLL for tokens in this window
        if output.prompt_logprobs:
            for j in range(start_eval, end_eval):
                if j < len(output.prompt_logprobs) and output.prompt_logprobs[j]:
                    actual_token = window_tokens[j]
                    if actual_token in output.prompt_logprobs[j]:
                        logprob = output.prompt_logprobs[j][actual_token].logprob
                        total_nll += -logprob
                        total_tokens += 1
        
        # Aggressive memory cleanup
        del outputs
        del output
        torch.cuda.empty_cache()
        gc.collect()
    
    if total_tokens == 0:
        raise ValueError("No valid tokens found for perplexity calculation")
    
    avg_nll = total_nll / total_tokens
    perplexity = math.exp(avg_nll)
    
    return perplexity, total_tokens


def main():
    parser = argparse.ArgumentParser(
        description="Memory-optimized perplexity calculation with vLLM score_mode"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--quantization", type=str, default=None)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--dataset-config", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--context-length", type=int, default=512, help="Context length (default: 512 for memory efficiency)")
    parser.add_argument("--stride", type=int, default=256, help="Stride (default: 256)")
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--disable-custom-all-reduce", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    
    args = parser.parse_args()
    
    # Initialize model
    print(f"Loading model: {args.model}")
    llm_kwargs = {
        "model": args.model,
        "enforce_eager": True,
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
    print(f"  Context length: {args.context_length}")
    print(f"  Stride: {args.stride}")
    
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    
    # Get text to evaluate
    if args.text:
        text = args.text
        print(f"Using provided text ({len(args.text)} characters)")
    elif args.dataset:
        try:
            from datasets import load_dataset
            
            if args.dataset_config:
                print(f"Loading dataset: {args.dataset} (config: {args.dataset_config})")
                dataset = load_dataset(args.dataset, args.dataset_config, split="test")
            else:
                print(f"Loading dataset: {args.dataset}")
                dataset = load_dataset(args.dataset, split="test")
            
            print(f"Loaded {len(dataset)} examples from dataset")
            
            num_samples = args.num_samples if args.num_samples else len(dataset)
            num_samples = min(num_samples, len(dataset))
            
            text = "\n\n".join(dataset["text"][:num_samples])
            print(f"Using {num_samples} samples ({len(text)} characters)")
            
        except ImportError:
            print("ERROR: Please install datasets: pip install datasets")
            return
        except Exception as e:
            print(f"ERROR loading dataset: {e}")
            return
    else:
        print("ERROR: Must provide either --text or --dataset")
        return
    
    # Tokenize
    print("\nTokenizing...")
    token_ids = tokenizer.encode(text)
    print(f"Total tokens: {len(token_ids)}")
    
    # Calculate perplexity
    print(f"\n{'='*70}")
    print(f"MEMORY-OPTIMIZED PERPLEXITY EVALUATION")
    print(f"{'='*70}")
    
    perplexity, num_evaluated = calculate_perplexity_optimized(
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
    print(f"Context length: {args.context_length}")
    print(f"Stride: {args.stride}")
    print(f"Total tokens in dataset: {len(token_ids)}")
    print(f"Tokens evaluated: {num_evaluated}")
    print(f"\n>>> Perplexity: {perplexity:.4f} <<<")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

