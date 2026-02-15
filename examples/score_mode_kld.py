#!/usr/bin/env python3
"""
KLD (Kullback-Leibler Divergence) calculation script using vLLM's score mode.

This script implements EXL3-compatible sliding window KLD calculation,
comparing a model under test against reference logits (e.g., from a
full-precision model) to measure quantization quality. All KL math is
computed on GPU when reference logits are provided.

Usage:
    # Two-phase: reference model + test model
    python examples/score_mode_kld.py \
        --model /path/to/quantized_model \
        --reference-model /path/to/reference_model \
        --dataset wikitext \
        --dataset-config wikitext-2-raw-v1 \
        --context-length 2048 \
        --stride 512

    # Using pre-saved reference logits
    python examples/score_mode_kld.py \
        --model /path/to/quantized_model \
        --reference-logits /path/to/reference_logits.safetensors \
        --dataset wikitext \
        --dataset-config wikitext-2-raw-v1 \
        --context-length 2048 \
        --stride 512
"""

import argparse
import hashlib
import json
import math
import os
import time
from typing import Any

import torch
import torch.nn.functional as F
from datasets import load_dataset
from safetensors.torch import save_file, safe_open

from vllm import LLM, SamplingParams
from vllm.inputs.data import TokensPrompt


def load_dataset_texts(
    dataset_name: str,
    dataset_config: str | None = None,
    split: str | None = None,
) -> list[str]:
    """Load and extract text from a HuggingFace dataset."""
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
                    msg.get("content", "")
                    for msg in messages
                    if isinstance(msg, dict)
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


def _dict_hash(x: dict) -> str:
    key = str(json.dumps(x, sort_keys=True))
    return hashlib.sha256(key.encode()).hexdigest()


def calculate_kld(
    llm: LLM,
    texts: list[str],
    context_length: int,
    stride: int,
    reference_logits_path: str | None = None,
    reference_model_path: str | None = None,
    llm_kwargs: dict[str, Any] | None = None,
    num_samples: int | None = None,
    debug: bool = False,
) -> tuple[float, int]:
    """
    Calculate KLD using EXL3-compatible sliding window approach.

    Args:
        llm: Initialized vLLM LLM instance (test model)
        texts: List of text samples to evaluate
        context_length: Maximum context length for each window
        stride: Stride between windows
        reference_logits_path: Path to safetensors file with reference logits
        reference_model_path: Path to reference model (for Phase 1)
        llm_kwargs: Kwargs for initializing reference model
        num_samples: Maximum number of samples to process (None = all)
        debug: Enable debug logging

    Returns:
        Tuple of (mean_kld, total_positions)
    """
    kld_sum = 0.0
    kld_count = 0

    samples_to_process = texts[:num_samples] if num_samples else texts
    concatenated_text = "\n\n".join(samples_to_process)
    tokens = llm.llm_engine.tokenizer.encode(
        concatenated_text, add_special_tokens=False
    )

    if len(tokens) < 2:
        raise ValueError("Not enough tokens after concatenation")

    max_tokens_for_eval = context_length + 99 * stride
    if len(tokens) > max_tokens_for_eval:
        tokens = tokens[:max_tokens_for_eval]

    num_tokens = len(tokens)

    # Phase 1: Generate reference logits if reference_model_path provided
    if reference_model_path is not None:
        data_spec = {
            "dataset": "wikitext",
            "context_length": context_length,
            "stride": stride,
        }
        ref_logits_file = reference_logits_path or os.path.join(
            os.getcwd(),
            f"ref_logits_{_dict_hash(data_spec)}.safetensors",
        )
        reference_logits_path = ref_logits_file
        if not os.path.exists(ref_logits_file):
            print(f"Phase 1: Generating reference logits from {reference_model_path}")
            ref_llm = LLM(model=reference_model_path, **(llm_kwargs or {}))
            ref_logits_dict = {}
            window_idx = 0
            for start_idx in range(
                0, num_tokens - context_length + stride, stride
            ):
                end_idx = start_idx + context_length
                if end_idx > num_tokens:
                    break
                window_tokens = tokens[start_idx:end_idx]
                if len(window_tokens) < 2:
                    continue
                target_token_ids = window_tokens[1:]
                prompt: TokensPrompt = {
                    "prompt_token_ids": window_tokens,
                    "target_token_ids": target_token_ids,
                }
                sampling_params = SamplingParams(
                    prompt_logprobs=1,
                    max_tokens=1,
                    return_prompt_logits=True,
                )
                outputs = ref_llm.generate([prompt], sampling_params=sampling_params)
                out = outputs[0]
                if out.prompt_logits is not None:
                    ref_logits_dict[f"logits_{window_idx}"] = out.prompt_logits
                    window_idx += 1
            save_file(ref_logits_dict, ref_logits_file)
            del ref_llm
            print(f"Saved reference logits to {ref_logits_file}")

    if reference_logits_path is None:
        raise ValueError(
            "Either --reference-logits or --reference-model must be provided"
        )
    if not os.path.exists(reference_logits_path):
        raise FileNotFoundError(
            f"Reference logits file not found: {reference_logits_path}"
        )

    # Phase 2: Compute KLD using test model with reference logits
    print("Phase 2: Computing KLD...")
    window_idx = 0
    for start_idx in range(0, num_tokens - context_length + stride, stride):
        end_idx = start_idx + context_length
        if end_idx > num_tokens:
            break
        window_tokens = tokens[start_idx:end_idx]
        if len(window_tokens) < 2:
            continue

        target_token_ids = window_tokens[1:]
        ref_key = f"logits_{window_idx}"

        prompt: TokensPrompt = {
            "prompt_token_ids": window_tokens,
            "target_token_ids": target_token_ids,
            "reference_logits_path": reference_logits_path,
            "reference_logits_key": ref_key,
        }

        sampling_params = SamplingParams(
            prompt_logprobs=1,
            max_tokens=1,
            kld_mode=True,
        )

        outputs = llm.generate([prompt], sampling_params=sampling_params)
        out = outputs[0]

        if out.kld_result is not None:
            win_kld_sum, win_kld_count = out.kld_result
            kld_sum += win_kld_sum
            kld_count += win_kld_count
        else:
            # Fallback: get model logits, load ref, compute KLD in script
            sampling_params_fallback = SamplingParams(
                prompt_logprobs=1,
                max_tokens=1,
                return_prompt_logits=True,
            )
            prompt_fallback: TokensPrompt = {
                "prompt_token_ids": window_tokens,
                "target_token_ids": target_token_ids,
            }
            outputs = llm.generate(
                [prompt_fallback], sampling_params=sampling_params_fallback
            )
            out = outputs[0]
            if out.prompt_logits is not None:
                model_logits = out.prompt_logits
                with safe_open(
                    reference_logits_path,
                    framework="pt",
                    device="cpu",
                ) as f:
                    ref_logits = f.get_tensor(ref_key)
                device = model_logits.device
                ref_logits = ref_logits.to(device)
                vs = min(model_logits.shape[-1], ref_logits.shape[-1])
                probs_model = F.softmax(
                    model_logits[..., :vs].float() + 1e-10, dim=-1
                )
                probs_ref = F.softmax(
                    ref_logits[..., :vs].float() + 1e-10, dim=-1
                )
                kld_per_pos = F.kl_div(
                    torch.log(probs_model + 1e-10),
                    probs_ref,
                    reduction="none",
                ).sum(dim=-1)
                kld_sum += kld_per_pos.sum().item()
                kld_count += kld_per_pos.numel()

        window_idx += 1
        if debug and window_idx <= 3:
            print(f"Window {window_idx}: kld_sum={kld_sum}, kld_count={kld_count}")

    if kld_count == 0:
        raise ValueError("No valid positions for KLD calculation")

    mean_kld = kld_sum / kld_count
    return mean_kld, kld_count


def main():
    parser = argparse.ArgumentParser(
        description="Calculate KLD using vLLM's score mode"
    )
    parser.add_argument("--model", type=str, required=True, help="Path to test model")
    parser.add_argument(
        "--reference-model",
        type=str,
        default=None,
        help="Path to reference model (generates ref logits if needed)",
    )
    parser.add_argument(
        "--reference-logits",
        type=str,
        default=None,
        help="Path to pre-saved reference logits (safetensors)",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        help="Quantization method (e.g., 'awq', 'gptq')",
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
        help="Enable debug logging",
    )

    args = parser.parse_args()

    if args.reference_model is None and args.reference_logits is None:
        parser.error("Either --reference-model or --reference-logits is required")

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

    print(f"Initializing LLM with model: {args.model}")
    llm = LLM(model=args.model, **llm_kwargs)

    print("\nCalculating KLD...")
    print(f"  Context length: {args.context_length}")
    print(f"  Stride: {args.stride}")
    print(f"  Samples: {args.num_samples or len(texts)}")

    start_time = time.time()
    mean_kld, total_positions = calculate_kld(
        llm,
        texts,
        args.context_length,
        args.stride,
        reference_logits_path=args.reference_logits,
        reference_model_path=args.reference_model,
        llm_kwargs=llm_kwargs,
        num_samples=args.num_samples,
        debug=args.debug,
    )
    elapsed_time = time.time() - start_time

    print("\nResults:")
    print(f"  Mean KLD: {mean_kld:.6f}")
    print(f"  Total positions: {total_positions}")
    print(f"  Time elapsed: {elapsed_time:.2f} seconds")
    print(f"  Positions/second: {total_positions / elapsed_time:.2f}")


if __name__ == "__main__":
    main()
