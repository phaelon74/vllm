#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Materialize a quantize-dequantize variant of a BF16 checkpoint.

A mean KLD against a quantized checkpoint answers "what did this quantization
cost", which on a mixture-of-experts model is two questions wearing one number:
the router now selects different experts, and the experts themselves compute
less precisely. A routing flip replaces the function that produced a token
rather than perturbing it, so it dominates any mean and hides the expert-
precision signal underneath.

Separating them needs a checkpoint where exactly one component carries
quantization error. Quantize-dequantize gives that: round a tensor through the
target format and store the result back in BF16. The weights are numerically
what the quantized model would use, while the file is an ordinary BF16
checkpoint, so vLLM loads it with no quantized kernels and the whole fidelity
pipeline - frozen suite, hidden-state capture, replay probe, every law - applies
unchanged. No engine patch, no expert-selection override.

Building the router-only and expert-only variants turns one ambiguous number
into a 2x2:

    BF16 routes x BF16 experts     the zero baseline
    BF16 routes x QDQ  experts     what expert precision costs
    QDQ  routes x BF16 experts     what router precision costs
    QDQ  routes x QDQ  experts     what a deployment pays

Those do not add up, and they are not supposed to: once a token is routed
elsewhere, degrading the expert it no longer uses costs nothing.

What this does not measure: the deployed kernel. QDQ is the quantization
*scheme* evaluated in BF16 arithmetic. Comparing a QDQ variant against the real
quantized checkpoint isolates what the kernel itself contributes, which is a
different and also useful number.

Granularity has to match the checkpoint being explained, so `--match` reads the
scheme and block size off a real quantized checkpoint instead of asking anyone to
remember them. The same inspection reports which components that checkpoint
actually quantized, which is worth knowing first: if it leaves the router in
BF16, router-weight precision costs nothing by construction and only three of the
four cells exist.

Usage:
    # What does the real FP8 checkpoint do?
    python fidelity/qdq.py --inspect /models/Qwen3.6-35B-A3B-FP8

    # Variants that imitate it, one component at a time.
    python fidelity/qdq.py --model /models/Qwen3.6-35B-A3B \\
        --out /models/Qwen3.6-35B-A3B-qdq-experts \\
        --components experts --match /models/Qwen3.6-35B-A3B-FP8

    python fidelity/qdq.py --selftest
"""

import argparse
import json
import os
import re
import shutil
import sys
from typing import Any

# Names are matched against the checkpoint's own parameter names. The router is
# `mlp.gate`, which is a different tensor from `mlp.gate_proj`; matching on the
# word "gate" would quantize both and destroy the separation this tool exists to
# create, so every pattern is anchored.
COMPONENT_PATTERNS: dict[str, tuple[str, ...]] = {
    "router": (
        r"\.mlp\.gate\.weight$",
        r"\.mlp\.router\.weight$",
        r"\.block_sparse_moe\.gate\.weight$",
        r"\.feed_forward\.router\.weight$",
    ),
    "experts": (
        r"\.mlp\.experts\.",
        r"\.block_sparse_moe\.experts\.",
        r"\.feed_forward\.experts\.",
    ),
    "shared_expert": (
        r"\.mlp\.shared_expert\.",
        r"\.mlp\.shared_experts\.",
    ),
    "attention": (r"\.self_attn\..*\.weight$",),
    "dense_mlp": (
        r"\.mlp\.(gate_proj|up_proj|down_proj|gate_up_proj)\.weight$",
    ),
}

# Quantizing these would change what the fidelity comparison means: the head is
# held constant so trunk and head effects stay separable (Law 8), and the
# embedding is not a matmul weight in the same sense.
NEVER = (
    r"\.?embed_tokens\.weight$",
    r"^lm_head\.weight$",
    r"\.lm_head\.weight$",
)

COMPONENTS = tuple(COMPONENT_PATTERNS)


def _matches(name: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, name) for pattern in patterns)


def classify(name: str, selected: tuple[str, ...]) -> str | None:
    """Which selected component a tensor belongs to, or None to copy verbatim.

    `dense_mlp` is checked last because an expert projection also ends in
    `down_proj.weight`; the expert patterns must claim it first.
    """
    if _matches(name, NEVER):
        return None
    order = ("router", "experts", "shared_expert", "attention", "dense_mlp")
    for component in order:
        if component in selected and _matches(name, COMPONENT_PATTERNS[component]):
            return component
    return None


def quantize_dequantize(tensor: Any, scheme: str, block_size: int) -> Any:
    """Round a weight through `scheme` and return it in its original dtype.

    A 3-D tensor is a stack of per-expert matrices; each is scaled on its own
    amax, matching how a real quantizer treats a fused expert weight. Doing it
    over the whole stack would let one loud expert set the scale for all of
    them, which no quantizer does.
    """
    import torch

    if tensor.ndim == 3:
        return torch.stack(
            [
                quantize_dequantize(tensor[i], scheme, block_size)
                for i in range(tensor.shape[0])
            ]
        )
    if tensor.ndim != 2:
        raise ValueError(
            f"expected a 2-D or 3-D weight, got shape {tuple(tensor.shape)}"
        )

    if scheme == "nvfp4":
        from vllm.model_executor.layers.quantization.utils import (
            nvfp4_emulation_utils as nvfp4,
        )
        from vllm.platforms import current_platform

        float8_max = torch.finfo(torch.float8_e4m3fn).max
        if tensor.shape[-1] % block_size:
            raise ValueError(
                f"last dimension {tensor.shape[-1]} is not a multiple of the "
                f"{block_size}-element block; this weight cannot be NVFP4 "
                f"quantized as-is"
            )
        device = tensor.device
        amax = tensor.abs().max().to(torch.float32)
        if float(amax) == 0.0:
            return tensor.clone()
        global_scale = (float8_max * nvfp4.FLOAT4_E2M1_MAX / amax).to(torch.float32)
        if not current_platform.is_cuda_alike() and device.type != "cpu":
            tensor = tensor.cpu()
        converted = nvfp4.ref_nvfp4_quant_dequant(tensor, global_scale, block_size)
        return converted.to(device)

    if scheme == "mxfp8":
        # vLLM's reference implementation, not its fused kernel: this tool
        # measures what a format costs, and the kernel's contribution is a
        # separate question answered by comparing against the real checkpoint.
        from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
            MXFP8_BLOCK_SIZE,
            _mxfp8_e4m3_quantize_torch,
            dequant_mxfp8_to_bf16,
        )

        if tensor.shape[-1] % MXFP8_BLOCK_SIZE:
            raise ValueError(
                f"last dimension {tensor.shape[-1]} is not a multiple of the "
                f"MXFP8 block of {MXFP8_BLOCK_SIZE}"
            )
        values, scales = _mxfp8_e4m3_quantize_torch(tensor, False)
        return dequant_mxfp8_to_bf16(values, scales).to(tensor.dtype)

    if scheme.startswith("fp8"):
        return _fp8_quant_dequant(tensor, scheme, block_size)

    raise ValueError(f"unknown scheme {scheme!r}")


def _fp8_quant_dequant(tensor: Any, scheme: str, block_size: int) -> Any:
    """Round through E4M3 at the granularity the scheme names.

    Granularity is not a detail. A per-tensor scale lets the single largest
    weight in a matrix set the step size for every other weight in it, which is
    materially worse than a per-channel or per-block scale. Measuring a variant
    at one granularity and comparing it against a checkpoint quantized at
    another attributes the difference to the wrong cause, so this must match the
    checkpoint being explained - see `--match`.
    """
    import torch

    finfo = torch.finfo(torch.float8_e4m3fn)
    source = tensor.to(torch.float32)

    if scheme == "fp8_per_tensor":
        scale = (source.abs().max() / finfo.max).clamp(min=1e-12)
    elif scheme == "fp8_per_channel":
        scale = (source.abs().amax(dim=-1, keepdim=True) / finfo.max).clamp(min=1e-12)
    elif scheme == "fp8_block":
        rows, cols = source.shape
        if rows % block_size or cols % block_size:
            raise ValueError(
                f"shape {(rows, cols)} is not a multiple of the "
                f"{block_size}x{block_size} block; a real block quantizer pads "
                f"here, and padding changes the scales, so refusing to guess"
            )
        blocked = source.reshape(
            rows // block_size, block_size, cols // block_size, block_size
        )
        scale = (
            blocked.abs().amax(dim=(1, 3), keepdim=True) / finfo.max
        ).clamp(min=1e-12)
        rounded = (blocked / scale).clamp(finfo.min, finfo.max)
        rounded = rounded.to(torch.float8_e4m3fn).to(torch.float32)
        return (rounded * scale).reshape(rows, cols).to(tensor.dtype)
    else:
        raise ValueError(f"unknown scheme {scheme!r}")

    rounded = (source / scale).clamp(finfo.min, finfo.max)
    rounded = rounded.to(torch.float8_e4m3fn).to(torch.float32)
    return (rounded * scale).to(tensor.dtype)


SCALE_SUFFIXES = ("weight_scale", "weight_scale_inv", "weight_scale_2", "scales")


def _scale_shapes(model: str) -> dict[str, tuple[int, ...]]:
    """Map each quantized weight to its scale's shape, reading headers only."""
    from safetensors import safe_open

    found: dict[str, tuple[int, ...]] = {}
    for name in sorted(os.listdir(model)):
        if not name.endswith(".safetensors"):
            continue
        with safe_open(os.path.join(model, name), framework="pt", device="cpu") as f:
            for key in f.keys():
                for suffix in SCALE_SUFFIXES:
                    if key.endswith("." + suffix):
                        weight = key[: -len(suffix)] + "weight"
                        found[weight] = tuple(f.get_slice(key).get_shape())
                        break
    return found


def inspect(model: str) -> dict[str, Any]:
    """Report what a quantized checkpoint actually quantized, and how finely.

    Two questions this answers before any GPU time is spent. Whether the router
    is quantized at all - if it is not, router-precision cost is zero by
    construction and any routing divergence comes from upstream activations
    instead. And what granularity to match, so a QDQ variant is comparable to
    this checkpoint rather than to a scheme nobody deployed.
    """
    from safetensors import safe_open

    with open(os.path.join(model, "config.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    quant = config.get("quantization_config") or {}

    scales = _scale_shapes(model)
    weight_shapes: dict[str, tuple[int, ...]] = {}
    for name in sorted(os.listdir(model)):
        if not name.endswith(".safetensors"):
            continue
        with safe_open(os.path.join(model, name), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key in scales:
                    weight_shapes[key] = tuple(f.get_slice(key).get_shape())

    granularity: str | None = None
    block: int | None = None
    for weight, scale_shape in scales.items():
        shape = weight_shapes.get(weight)
        if not shape or len(shape) < 2:
            continue
        elements = 1
        for dim in scale_shape:
            elements *= dim
        if elements == 1:
            granularity = "fp8_per_tensor"
        elif elements == shape[0]:
            granularity = "fp8_per_channel"
        else:
            granularity = "fp8_block"
            if len(scale_shape) >= 2 and scale_shape[-1]:
                block = max(1, round(shape[-1] / scale_shape[-1]))
        break

    coverage: dict[str, dict[str, int]] = {}
    for component in COMPONENTS:
        coverage[component] = {"weights": 0, "quantized": 0}
    for name in sorted(set(weight_shapes) | set(scales)):
        component = classify(name, COMPONENTS)
        if component is None:
            continue
        coverage[component]["quantized"] += 1
    for name in sorted(os.listdir(model)):
        if not name.endswith(".safetensors"):
            continue
        with safe_open(os.path.join(model, name), framework="pt", device="cpu") as f:
            for key in f.keys():
                if not key.endswith(".weight"):
                    continue
                component = classify(key, COMPONENTS)
                if component is not None:
                    coverage[component]["weights"] += 1

    return {
        "model": os.path.abspath(model),
        "quant_method": quant.get("quant_method"),
        "declared": {
            key: quant.get(key)
            for key in ("fmt", "activation_scheme", "weight_block_size", "strategy")
            if quant.get(key) is not None
        },
        "detected_scheme": granularity,
        "detected_block": block,
        "coverage": coverage,
    }


def render_inspection(report: dict[str, Any]) -> str:
    lines = [
        f"{os.path.basename(report['model'])}",
        f"  quant_method: {report['quant_method'] or 'none declared'}",
    ]
    for key, value in (report["declared"] or {}).items():
        lines.append(f"  {key}: {value}")
    lines.append(f"  detected scheme: {report['detected_scheme'] or 'unquantized'}")
    if report["detected_block"]:
        lines.append(f"  detected block: {report['detected_block']}")
    lines.append("  component coverage (quantized / total weights):")
    for component, counts in report["coverage"].items():
        if not counts["weights"] and not counts["quantized"]:
            continue
        total = counts["weights"]
        done = counts["quantized"]
        verdict = "all" if done and done >= total else ("none" if not done else "some")
        lines.append(f"    {component}: {done} / {total} ({verdict})")
    router = report["coverage"].get("router") or {}
    if router.get("weights") and not router.get("quantized"):
        lines += [
            "",
            "  The router is NOT quantized in this checkpoint. Router-weight "
            "precision therefore costs exactly zero here, and a Q x B cell "
            "would measure nothing. Any routing divergence comes from the "
            "activations reaching the router, not from the router itself.",
        ]
    return "\n".join(lines)


def _error_stats(before: Any, after: Any) -> dict[str, float]:
    import torch

    diff = (after.to(torch.float32) - before.to(torch.float32)).abs()
    denominator = before.to(torch.float32).pow(2).mean().sqrt()
    rms = diff.pow(2).mean().sqrt()
    return {
        "max_abs_err": float(diff.max()),
        "rms_err": float(rms),
        "relative_rms": float(rms / denominator) if float(denominator) else 0.0,
    }


def _assert_unquantized(model: str) -> dict[str, Any]:
    """A QDQ variant of an already-quantized checkpoint would be meaningless."""
    path = os.path.join(model, "config.json")
    if not os.path.isfile(path):
        raise SystemExit(f"{model} has no config.json")
    with open(path, encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("quantization_config"):
        raise SystemExit(
            f"{model} is already quantized. Build QDQ variants from the BF16 "
            f"checkpoint, so that exactly one component carries error."
        )
    return config


def convert(
    model: str,
    out: str,
    selected: tuple[str, ...],
    scheme: str,
    block_size: int,
    device: str,
) -> dict[str, Any]:
    """Copy a checkpoint, rounding the selected components through `scheme`."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    _assert_unquantized(model)
    os.makedirs(out, exist_ok=True)

    shards = sorted(
        name for name in os.listdir(model) if name.endswith(".safetensors")
    )
    if not shards:
        raise SystemExit(f"no .safetensors files in {model}")

    for name in sorted(os.listdir(model)):
        source = os.path.join(model, name)
        if name.endswith(".safetensors") or not os.path.isfile(source):
            continue
        shutil.copy2(source, os.path.join(out, name))

    rollup: dict[str, dict[str, Any]] = {
        component: {
            "tensors": 0,
            "parameters": 0,
            "max_abs_err": 0.0,
            "worst_relative_rms": 0.0,
            "names": [],
        }
        for component in selected
    }
    untouched = 0
    for shard in shards:
        with safe_open(os.path.join(model, shard), framework="pt", device="cpu") as f:
            metadata = f.metadata() or {}
            tensors: dict[str, Any] = {}
            for name in f.keys():
                tensor = f.get_tensor(name)
                component = classify(name, selected)
                if component is None or not torch.is_floating_point(tensor):
                    tensors[name] = tensor
                    untouched += 1
                    continue
                staged = tensor.to(device) if device != "cpu" else tensor
                converted = quantize_dequantize(staged, scheme, block_size)
                stats = _error_stats(staged, converted)
                tensors[name] = converted.to("cpu")
                entry = rollup[component]
                entry["tensors"] += 1
                entry["parameters"] += int(tensor.numel())
                entry["max_abs_err"] = max(entry["max_abs_err"], stats["max_abs_err"])
                entry["worst_relative_rms"] = max(
                    entry["worst_relative_rms"], stats["relative_rms"]
                )
                entry["names"].append(name)
        save_file(tensors, os.path.join(out, shard), metadata=metadata)
        print(f"  wrote {shard}")

    for component, entry in rollup.items():
        if not entry["tensors"]:
            raise SystemExit(
                f"component {component!r} matched no tensor in {model}. The "
                f"checkpoint names its weights differently; add its pattern to "
                f"COMPONENT_PATTERNS rather than publishing a variant that "
                f"quantized nothing."
            )

    manifest = {
        "source_model": os.path.abspath(model),
        "scheme": scheme,
        "block_size": block_size if scheme in ("nvfp4", "fp8_block") else None,
        "components": list(selected),
        "device": device,
        "tensors_untouched": untouched,
        "components_detail": rollup,
        "vllm_version": _vllm_version(),
        "note": (
            "Quantize-dequantize variant stored in the source dtype. Weights are "
            "numerically what the quantized model would use; arithmetic is not. "
            "Comparing this against the real quantized checkpoint isolates the "
            "kernel's contribution."
        ),
    }
    with open(
        os.path.join(out, "qdq-manifest.json"), "w", encoding="utf-8", newline="\n"
    ) as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def _vllm_version() -> str | None:
    try:
        import vllm

        return getattr(vllm, "__version__", None)
    except ImportError:
        return None


def selftest() -> int:
    """Prove the properties the 2x2 depends on, on tensors small enough to check."""
    import torch

    torch.manual_seed(0)
    weight = torch.randn(64, 256, dtype=torch.bfloat16)

    schemes = ("fp8_per_tensor", "fp8_per_channel", "fp8_block", "mxfp8", "nvfp4")
    error: dict[str, float] = {}
    for scheme in schemes:
        block = 16 if scheme == "nvfp4" else 64
        once = quantize_dequantize(weight, scheme, block)
        twice = quantize_dequantize(once, scheme, block)
        assert once.dtype == weight.dtype, f"{scheme} changed dtype"
        assert not torch.equal(once, weight), f"{scheme} was a no-op"
        assert torch.equal(once, twice), (
            f"{scheme} is not idempotent; a second rounding moved the weights, "
            f"so the variant is not a fixed point of the format"
        )
        relative = _error_stats(weight, once)["relative_rms"]
        assert relative < 0.5, f"{scheme} relative rms {relative} is implausible"
        error[scheme] = relative
        print(f"  {scheme}: idempotent, relative rms {relative:.4f}")

    # The ladder has to be ordered, or a scheme comparison built on it is
    # measuring the harness. Four mantissa bits cannot beat eight.
    assert error["nvfp4"] > error["mxfp8"], (
        f"nvfp4 rms {error['nvfp4']} is not worse than mxfp8 "
        f"{error['mxfp8']}; the rounding is wrong somewhere"
    )
    assert error["fp8_per_channel"] <= error["fp8_per_tensor"], (
        "a per-channel scale cannot be worse than one scale for the matrix"
    )
    print("  ladder is ordered: nvfp4 > mxfp8, per-channel <= per-tensor")

    stacked = torch.randn(4, 32, 64, dtype=torch.bfloat16)
    stacked[0] *= 1000.0
    per_expert = quantize_dequantize(stacked, "nvfp4", 16)
    alone = quantize_dequantize(stacked[1], "nvfp4", 16)
    assert torch.equal(per_expert[1], alone), (
        "a fused expert stack must be scaled per expert; expert 1 changed when "
        "expert 0 was loud, so one expert is setting the scale for all of them"
    )
    print("  fused stacks are scaled per expert")

    names = {
        "model.layers.0.mlp.gate.weight": "router",
        "model.layers.0.mlp.experts.3.down_proj.weight": "experts",
        "model.layers.0.mlp.experts.3.gate_proj.weight": "experts",
        "model.layers.0.mlp.shared_expert.up_proj.weight": "shared_expert",
        "model.layers.0.self_attn.q_proj.weight": "attention",
        "model.layers.0.mlp.gate_proj.weight": "dense_mlp",
        "model.embed_tokens.weight": None,
        "lm_head.weight": None,
        "model.layers.0.input_layernorm.weight": None,
    }
    for name, expected in names.items():
        actual = classify(name, COMPONENTS)
        assert actual == expected, f"{name}: classified {actual}, expected {expected}"
    # The separation the whole tool exists for: selecting the router must leave
    # every expert and projection weight alone.
    assert classify("model.layers.0.mlp.gate_proj.weight", ("router",)) is None
    expert_gate = "model.layers.0.mlp.experts.0.gate_proj.weight"
    assert classify(expert_gate, ("router",)) is None
    assert classify("model.layers.0.mlp.gate.weight", ("experts",)) is None
    print("  router and gate_proj do not alias; components are disjoint")
    print("selftest passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="source BF16 checkpoint directory")
    parser.add_argument("--out", help="destination directory for the variant")
    parser.add_argument(
        "--components",
        default=None,
        help="comma-separated: " + ", ".join(COMPONENTS),
    )
    parser.add_argument(
        "--scheme",
        default=None,
        choices=(
            "fp8_per_tensor",
            "fp8_per_channel",
            "fp8_block",
            "mxfp8",
            "nvfp4",
        ),
        help="rounding scheme; omit and pass --match to copy it from a "
        "quantized checkpoint",
    )
    parser.add_argument(
        "--match",
        default=None,
        help="a real quantized checkpoint whose scheme, block size, and "
        "component coverage this variant should imitate",
    )
    parser.add_argument(
        "--inspect",
        default=None,
        help="report what a quantized checkpoint quantized, and exit",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="block edge; defaults to 128 for fp8_block and 16 for nvfp4",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="where to round; defaults to cuda when available",
    )
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        return selftest()
    if args.inspect:
        print(render_inspection(inspect(args.inspect)))
        return 0
    if not (args.model and args.out and args.components):
        parser.error("--model, --out, and --components are required")

    selected = tuple(
        part.strip() for part in args.components.split(",") if part.strip()
    )
    unknown = [part for part in selected if part not in COMPONENTS]
    if unknown:
        parser.error(f"unknown component(s): {', '.join(unknown)}")

    scheme = args.scheme
    block_size = args.block_size
    matched: dict[str, Any] | None = None
    if args.match:
        matched = inspect(args.match)
        if scheme is None:
            scheme = matched["detected_scheme"]
            if scheme is None:
                parser.error(
                    f"{args.match} carries no weight scales, so there is no "
                    f"scheme to match; pass --scheme explicitly"
                )
            print(f"matched scheme {scheme} from {os.path.basename(args.match)}")
        if block_size is None and matched["detected_block"]:
            block_size = matched["detected_block"]
    if scheme is None:
        parser.error("pass --scheme, or --match a quantized checkpoint")
    if block_size is None:
        block_size = 16 if scheme == "nvfp4" else 128

    device = args.device
    if device is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    if matched:
        for component in selected:
            counts = matched["coverage"].get(component) or {}
            if counts.get("weights") and not counts.get("quantized"):
                print(
                    f"WARNING  {os.path.basename(args.match)} does not quantize "
                    f"{component}, so this variant carries error the deployed "
                    f"checkpoint does not have"
                )

    print(
        f"QDQ {scheme} on {', '.join(selected)} "
        f"({os.path.basename(args.model)} -> {os.path.basename(args.out)}) "
        f"on {device}"
    )
    manifest = convert(
        args.model, args.out, selected, scheme, block_size, device
    )
    if matched:
        manifest["matched"] = {
            "model": matched["model"],
            "detected_scheme": matched["detected_scheme"],
            "detected_block": matched["detected_block"],
            "coverage": matched["coverage"],
        }
        with open(
            os.path.join(args.out, "qdq-manifest.json"),
            "w",
            encoding="utf-8",
            newline="\n",
        ) as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
    for component, entry in manifest["components_detail"].items():
        print(
            f"  {component}: {entry['tensors']} tensors, "
            f"{entry['parameters'] / 1e9:.2f}B parameters, worst relative rms "
            f"{entry['worst_relative_rms']:.4f}"
        )
    print(f"  {manifest['tensors_untouched']} tensors copied unchanged")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
