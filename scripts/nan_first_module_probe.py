#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Name the first module whose forward output goes non-finite.

A model that produces NaN logits at every position has a definite first
offender, and reading weight-preparation code cannot find it: the padding
geometry of a quantized layer can be self-consistent on paper and still be
wrong in the kernel. This installs a forward hook on every module, runs one
short prompt, and reports the modules whose inputs were finite but whose
output was not. The first such module is the fault; everything after it is
downstream contamination.

The engine must run in-process for hooks to survive, so this forces
VLLM_ENABLE_V1_MULTIPROCESSING=0 before importing vLLM.
"""

from __future__ import annotations

import argparse
import json
import os

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import torch  # noqa: E402
from scan_checkpoint_nonfinite import nonfinite_count, nonfinite_mask  # noqa: E402

from vllm import LLM, SamplingParams  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--quantization", default=None)
    p.add_argument("--tokens", type=int, default=64)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument(
        "--report",
        type=int,
        default=12,
        help="how many offending modules to print",
    )
    p.add_argument(
        "--checkpoint-keys",
        default="router",
        help="list checkpoint tensor names containing this substring",
    )
    p.add_argument(
        "--prompt-logprobs",
        action="store_true",
        help="score logits at every prompt position, as KLD scoring does",
    )
    p.add_argument(
        "--moe-backend",
        default=None,
        help="override kernel_config.moe_backend, e.g. marlin or emulation",
    )
    p.add_argument(
        "--context-file",
        nargs="+",
        default=(),
        help=(
            "suite context JSON files to replay instead of a synthetic prompt; "
            "real token content decides which experts route"
        ),
    )
    return p.parse_args()


def _tensors(value) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (list, tuple)):
        return [t for v in value for t in _tensors(v)]
    if isinstance(value, dict):
        return [t for v in value.values() for t in _tensors(v)]
    return []


def _bad(tensors: list[torch.Tensor]) -> bool:
    return any(t.is_floating_point() and nonfinite_count(t) for t in tensors)


def _first_bad_row(tensors: list[torch.Tensor]) -> int | None:
    """Index of the first row along dim 0 holding a non-finite value.

    Prefill activations are token-major, so this names the token position where
    a fault begins. A fault that starts partway through and persists implicates
    position-dependent state -- an attention window or chunk boundary -- while a
    scattered one implicates the tokens themselves and the experts they route
    to.
    """
    for tensor in tensors:
        if not tensor.is_floating_point() or tensor.dim() == 0:
            continue
        rows = nonfinite_mask(tensor).reshape(tensor.shape[0], -1).any(dim=1)
        hits = rows.nonzero()
        if hits.numel():
            return int(hits[0])
    return None


def report_checkpoint_keys(path: str, needle: str) -> None:
    """List checkpoint tensor names matching ``needle``, from the index."""
    index = os.path.join(path, "model.safetensors.index.json")
    if not os.path.isfile(index):
        print(f"=== no safetensors index at {index}")
        return
    with open(index, encoding="utf-8") as handle:
        names = json.load(handle).get("weight_map", {})
    matches = sorted({n for n in names if needle in n})
    print(f"=== {len(matches)} checkpoint tensor(s) containing {needle!r}")
    for name in matches[:20]:
        print(f"  ckpt  {name}")


def report_unloaded_params(model: torch.nn.Module) -> int:
    """Name parameters holding non-finite values after weight loading.

    A parameter the loader never wrote keeps its ``torch.empty`` contents, so
    a non-finite parameter is almost always a checkpoint key that did not map
    to this module. Checking before the forward separates that from a kernel
    that computes a NaN.
    """
    bad = [
        name
        for name, param in model.named_parameters()
        if param.is_floating_point() and nonfinite_count(param)
    ]
    print(f"=== {len(bad)} parameter(s) non-finite after load")
    for name in bad[:20]:
        print(f"  UNLOADED  {name}")
    return len(bad)


def _describe(module: torch.nn.Module) -> str:
    bits = [type(module).__name__]
    for name in ("qweight", "weight_packed", "weight", "scales", "weight_scale"):
        param = getattr(module, name, None)
        if isinstance(param, torch.Tensor):
            bits.append(f"{name}{tuple(param.shape)}")
    quant = getattr(module, "quant_method", None)
    if quant is not None:
        bits.append(f"quant={type(quant).__name__}")
    return " ".join(bits)


def probe(model: torch.nn.Module, events: list[dict]) -> list:
    handles: list = []

    def make_hook(name: str):
        def hook(module, args, output):
            outputs = _tensors(output)
            events.append(
                {
                    "name": name,
                    "module": module,
                    "in_bad": _bad(_tensors(args)),
                    "out_bad": _bad(outputs),
                    "row": _first_bad_row(outputs),
                }
            )

        return hook

    for name, module in model.named_modules():
        if name:
            handles.append(module.register_forward_hook(make_hook(name)))
    return handles


def _prompts(args: argparse.Namespace, llm: LLM) -> list[tuple[str, list[int]]]:
    """Label and token ids for each prompt to probe.

    Real suite contexts are preferred over synthetic text because expert
    routing is content-dependent: a repeated sentence exercises a narrow slice
    of the experts and can miss a fault that diverse text hits.
    """
    if args.context_file:
        prompts = []
        for path in args.context_file:
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
            prompts.append((os.path.basename(path), payload["tokens"]))
        return prompts

    sentence = "The rain in Spain falls mainly on the plain. "
    repeats = max(1, args.tokens // 4)
    ids = llm.get_tokenizer().encode(sentence * repeats)
    while len(ids) < args.tokens:
        repeats *= 2
        ids = llm.get_tokenizer().encode(sentence * repeats)
    return [("synthetic repeated text", ids[: args.tokens])]


def main() -> int:
    args = parse_args()
    extra = {}
    if args.moe_backend:
        extra["kernel_config"] = {"moe_backend": args.moe_backend}
    llm = LLM(
        model=args.model,
        quantization=args.quantization,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        enforce_eager=True,
        trust_remote_code=True,
        **extra,
    )

    report_checkpoint_keys(args.model, args.checkpoint_keys)
    llm.apply_model(report_unloaded_params)

    events: list[dict] = []
    llm.apply_model(lambda model: probe(model, events))

    print(f"=== probing {args.model}")
    failures = 0
    for label, ids in _prompts(args, llm):
        events.clear()
        print(f"=== {len(ids)} tokens from {label}")
        llm.generate(
            {"prompt_token_ids": ids},
            sampling_params=SamplingParams(
                max_tokens=1,
                temperature=0.0,
                prompt_logprobs=0 if args.prompt_logprobs else None,
            ),
        )
        failures += _report(events, args.report)
    return 1 if failures else 0


def _report(events: list[dict], report: int) -> int:
    offenders = [e for e in events if e["out_bad"]]
    origins = [e for e in offenders if not e["in_bad"]]
    print(f"=== {len(events)} module calls, {len(offenders)} with non-finite output")

    if not offenders:
        print("PASS: every module output was finite")
        return 0

    print(f"=== {len(origins)} module(s) turned finite inputs into non-finite output")
    for event in origins[:report]:
        print(
            f"  ORIGIN  {event['name']} at row {event['row']}: "
            f"{_describe(event['module'])}"
        )
    if not origins:
        print("  none; the first non-finite value entered before any hooked module")
    print("=== first outputs to go non-finite, in execution order")
    for event in offenders[:report]:
        flag = "in-bad " if event["in_bad"] else "in-ok  "
        print(
            f"  {flag} {event['name']} at row {event['row']}: "
            f"{type(event['module']).__name__}"
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
