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
import hashlib
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
        # Gemma 4 hangs the router off the layer and splits it into a projection
        # and two 1-D scales. Only the projection is a matmul operand.
        r"\.router\.proj\.weight$",
    ),
    "experts": (
        r"\.mlp\.experts\.",
        r"\.block_sparse_moe\.experts\.",
        r"\.feed_forward\.experts\.",
        # Gemma 4 has no MoE wrapper module: the fused stacks sit directly on the
        # layer as `layers.N.experts.gate_up_proj`. Some exporters reintroduce a
        # wrapper of their own when they unfuse the stack per expert.
        r"\.layers\.\d+\.experts\.",
        r"\.moe\.experts\.",
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
# embedding is not a matmul weight in the same sense. Normalization weights are
# excluded because no quantizer touches them - they are 1-D gains, not matmul
# operands - and counting them as attention coverage misreports a fully
# quantized projection set as partial.
NEVER = (
    r"\.?embed_tokens\.weight$",
    r"^lm_head\.weight$",
    r"\.lm_head\.weight$",
    r"norm\.weight$",
    # Scoring runs --language-model-only, so a vision tower contributes nothing
    # to the logits being compared. Its `self_attn` projections would otherwise
    # be swept into the attention cell, whose cost the run cannot observe.
    r"^model\.vision_tower\.",
    r"^model\.embed_vision\.",
    r"^vision_tower\.",
)

COMPONENTS = tuple(COMPONENT_PATTERNS)
# An inspection is cached beside the checkpoint and in the work tree, so a
# reading taken by an older inspector outlives the code that produced it. Bump
# this whenever detection or classification changes and every cache that
# predates the change is re-read instead of trusted.
INSPECT_VERSION = 4
# Variant-path suffix; full hex lives on inspect.json and the QDQ manifest.
MATCH_DIGEST_LEN = 12


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


INT4_SCHEME_RE = re.compile(r"^int4_g(-1|\d+)_(sym|asym)(_desc_act)?$")
KNOWN_SCHEMES = (
    "fp8_per_tensor",
    "fp8_per_channel",
    "fp8_block",
    "mxfp8",
    "nvfp4",
)


def parse_int4_scheme(scheme: str) -> tuple[int, bool] | None:
    """Group size and symmetry, or None if `scheme` is not an int4 format name.

    `desc_act` is a scheme name the rounder refuses: those group boundaries
    permute along K, so a contiguous-group round is a different format.
    """
    match = INT4_SCHEME_RE.fullmatch(scheme)
    if match is None:
        return None
    if match.group(3):
        raise ValueError(
            f"{scheme} uses act-order (desc_act): group boundaries permute "
            "along K, so a contiguous-group round is not that format"
        )
    return int(match.group(1)), match.group(2) == "sym"


def scheme_arg(value: str) -> str:
    """Accept the named FP8/NVFP4 schemes or an `int4_g<G>_(sym|asym)` format."""
    if value in KNOWN_SCHEMES or INT4_SCHEME_RE.fullmatch(value):
        return value
    known = ", ".join(KNOWN_SCHEMES)
    raise argparse.ArgumentTypeError(
        f"unknown scheme {value!r}; want {known} or int4_g<G>_(sym|asym)"
    )


def _int4_quant_dequant(tensor: Any, group_size: int, symmetric: bool) -> Any:
    """Round a 2-D HF weight [out, in] through grouped int4, return as-stored."""
    import torch
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        quantize_weights,
    )
    from vllm.scalar_type import scalar_types

    size_k = tensor.shape[-1]
    if group_size != -1 and size_k % group_size:
        raise ValueError(
            f"last dimension {size_k} is not a multiple of the "
            f"{group_size}-element group; a real int4 quantizer pads "
            "here, and padding changes the scales, so refusing to guess"
        )
    # quantize_weights groups along dim 0 and expects [size_k, size_n].
    transposed = tensor.t().to(torch.float32).contiguous()
    w_ref, _, _, _ = quantize_weights(
        transposed,
        scalar_types.uint4b8 if symmetric else scalar_types.uint4,
        group_size,
        zero_points=not symmetric,
    )
    return w_ref.t().contiguous().to(tensor.dtype)


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
        # ref_nvfp4_quant_dequant dispatches on the platform, not on the tensor,
        # so on a CUDA host it always enters a Triton kernel that cannot read a
        # CPU pointer. Weights are loaded on the CPU here, so move them.
        work = device
        if current_platform.is_cuda_alike() and device.type == "cpu":
            work = torch.device("cuda")
        converted = nvfp4.ref_nvfp4_quant_dequant(
            tensor.to(work), global_scale.to(work), block_size
        )
        return converted.to(device=device, dtype=tensor.dtype)

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

    int4 = parse_int4_scheme(scheme)
    if int4 is not None:
        group_size, symmetric = int4
        return _int4_quant_dequant(tensor, group_size, symmetric)

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
# Resolution prefers `.weight` when both names exist. Rewrite matches
# longest suffix first so `.qweight` is not parsed as a `.weight` key.
PACKED_WEIGHT_SUFFIXES = ("weight", "qweight", "weight_packed")
_PACKED_REWRITE_SUFFIXES = ("weight_packed", "qweight", "weight")

# A pack may keep its experts fused the way the reference does, in which case the
# operand carries no `.weight` at all: `experts.gate_up_proj_packed` beside
# `experts.gate_up_proj_scale`. The canonical name is then the bare stack, which
# is exactly what the reference calls it. Leaves match `_FUSED_EXPERT_MEMBERS`.
_FUSED_STACK_LEAVES = "gate_up_proj|gate_proj|up_proj|down_proj"
_FUSED_STACK_OPERAND = re.compile(
    rf"^(?P<stem>.+\.experts\.(?:{_FUSED_STACK_LEAVES}))_packed$"
)
_FUSED_STACK_SCALE = re.compile(
    rf"^(?P<stem>.+\.experts\.(?:{_FUSED_STACK_LEAVES}))"
    rf"_(?:scale|scales|scale_inv|scale_2|global_scale)$"
)


def _as_weight_name(key: str) -> str | None:
    """Canonical operand name for a packed or unpacked matmul weight."""
    for suffix in _PACKED_REWRITE_SUFFIXES:
        if key.endswith("." + suffix):
            return key[: -len(suffix)] + "weight"
    hit = _FUSED_STACK_OPERAND.match(key)
    return hit.group("stem") if hit is not None else None


def _is_scale_key(key: str) -> bool:
    """Whether `key` names a weight scale rather than a matmul operand."""
    if any(key.endswith("." + suffix) for suffix in SCALE_SUFFIXES):
        return True
    return _FUSED_STACK_SCALE.match(key) is not None


def _operand_from_scale_key(scale_key: str, keys: set[str]) -> str | None:
    """The packed or unpacked weight a scale tensor belongs to, or None."""
    for suffix in SCALE_SUFFIXES:
        if not scale_key.endswith("." + suffix):
            continue
        base = scale_key[: -len(suffix)]
        for packed in PACKED_WEIGHT_SUFFIXES:
            candidate = base + packed
            if candidate in keys:
                return candidate
        return None
    hit = _FUSED_STACK_SCALE.match(scale_key)
    if hit is not None:
        candidate = hit.group("stem") + "_packed"
        if candidate in keys:
            return candidate
    return None


def unloadable_reason(model: str) -> str | None:
    """Why vLLM will crash or refuse this checkpoint, or None if unknown.

    Runs on config.json only. The AMD Quark W4A16 export stores AWQ metadata in
    ``algo_config`` as a list of dicts; ``QuarkConfig.apply_vllm_mapper`` then
    feeds that list to ``WeightsMapper.apply_list``, which assumes strings and
    raises ``AttributeError: 'dict' object has no attribute 'endswith'`` inside
    EngineCore. Installing ``amd-quark`` does not change that: the crash is in
    vLLM's mapper, and this tree has no INT4 W4A16 QuarkScheme anyway.
    """
    path = os.path.join(model, "config.json")
    try:
        with open(path, encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        return f"config.json unreadable: {exc}"
    return unloadable_reason_from_config(config)


def _padded_group_reason(config: dict[str, Any], quant: dict[str, Any]) -> str | None:
    """Why a grouped int4 pack of this model will not load, or None.

    A grouped quantizer needs the reduction dimension to be a multiple of its
    group size. When it is not, the exporter pads and stores one group more than
    the dimension really carries, while vLLM allocates for the unpadded width and
    the expert weight loader fails copying a scale of the wrong length. The
    dimensions are declared, so this is answerable before a download.
    """
    group = _as_int(_first_present(quant, ("group_size", "q_group_size")))
    if not group or group <= 0:
        return None
    text = config.get("text_config")
    section = text if isinstance(text, dict) else config
    for key in ("moe_intermediate_size", "intermediate_size"):
        width = _as_int(section.get(key))
        if width and width % group:
            return (
                f"{key} {width} is not a multiple of the declared group size "
                f"{group}, so the exporter padded it to "
                f"{(width // group + 1) * group} while vLLM allocates {width}; "
                f"the expert weight loader fails copying a scale one group too "
                f"long. This is the checkpoint's geometry, not a missing package"
            )
    return None


def unloadable_reason_from_config(config: dict[str, Any]) -> str | None:
    quant = config.get("quantization_config")
    if not isinstance(quant, dict):
        return None
    method = str(quant.get("quant_method") or "").lower()
    grouped = (
        "auto-round",
        "auto_round",
        "gptq",
        "gptq_marlin",
        "awq",
        "awq_marlin",
    )
    if method in grouped:
        return _padded_group_reason(config, quant)
    if method != "quark":
        return None
    if "export" not in quant:
        return (
            "quark checkpoint has no quantization_config.export; "
            "vLLM refuses to load it"
        )
    for key, value in quant.items():
        if isinstance(value, list) and any(
            not isinstance(item, str) for item in value
        ):
            return (
                f"quark field {key!r} is a list of non-strings; "
                "vLLM WeightsMapper.apply_list crashes on load "
                "(not a missing amd-quark package)"
            )
    weight = (quant.get("global_quant_config") or {}).get("weight") or {}
    dtype = str(weight.get("dtype") or "").lower()
    if dtype == "int4":
        qscheme = weight.get("qscheme") or "unspecified"
        return (
            f"quark {dtype} {qscheme} is not a vLLM QuarkScheme "
            "(no W4A16 INT4 path); installing amd-quark will not help"
        )
    return None


def _quark_declared_scheme(quant: dict[str, Any]) -> str | None:
    if str(quant.get("quant_method") or "").lower() != "quark":
        return None
    weight = (quant.get("global_quant_config") or {}).get("weight") or {}
    dtype = weight.get("dtype")
    if not dtype:
        return None
    qscheme = weight.get("qscheme")
    return f"quark_{dtype}" + (f"_{qscheme}" if qscheme else "")


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_present(mapping: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return None


def _int4_scheme_name(
    group_size: int, symmetric: bool, desc_act: bool = False
) -> str:
    polarity = "sym" if symmetric else "asym"
    name = f"int4_g{group_size}_{polarity}"
    if desc_act:
        name += "_desc_act"
    return name


def _int4_from_bits(
    bits: Any,
    group_size: Any,
    *,
    symmetric: bool,
    algorithm: str,
    desc_act: bool = False,
) -> dict[str, Any] | None:
    width = _as_int(bits)
    group = _as_int(group_size)
    if width != 4 or group is None:
        return None
    return {
        "scheme": _int4_scheme_name(group, symmetric, desc_act),
        "group_size": group,
        "algorithm": algorithm,
    }


def _int4_from_compressed_tensors(quant: dict[str, Any]) -> dict[str, Any] | None:
    groups = quant.get("config_groups")
    if not isinstance(groups, dict):
        return None
    for group in groups.values():
        if not isinstance(group, dict):
            continue
        weights = group.get("weights")
        if not isinstance(weights, dict):
            continue
        kind = str(weights.get("type") or "int").lower()
        if kind not in ("int", "integer"):
            continue
        found = _int4_from_bits(
            weights.get("num_bits"),
            weights.get("group_size"),
            symmetric=bool(weights.get("symmetric", True)),
            algorithm="round_to_nearest",
        )
        if found is not None:
            return found
    return None


def quant_sections(model: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    """Every place a checkpoint might declare its quantization, in priority order.

    Vendors disagree about where the declaration lives and what nests it: inside
    `config.json`, under `text_config`, in a `hf_quant_config.json` sidecar, and
    sometimes under a `quantization` child of any of those. Reading one location
    means a checkpoint that declares itself plainly can still be misread from its
    tensor shapes, so gather them all and let the first that resolves win.
    """
    sections: list[dict[str, Any]] = []

    def add(candidate: Any) -> None:
        if not isinstance(candidate, dict) or not candidate:
            return
        sections.append(candidate)
        inner = candidate.get("quantization")
        if isinstance(inner, dict) and inner:
            merged = dict(inner)
            # The method usually sits outside the section that names the algorithm.
            merged.setdefault("quant_method", candidate.get("quant_method"))
            sections.append(merged)

    add(config.get("quantization_config"))
    text = config.get("text_config")
    if isinstance(text, dict):
        add(text.get("quantization_config"))
    add(_sidecar_quant_config(model))
    return [section for section in sections if section]


def _two_level_scaled(model: str) -> bool:
    """Whether any operand carries two levels of weight scale.

    NVFP4 is alone among the formats here in scaling twice: an FP8 scale per
    16-element block and one scalar over the tensor. Every FP8 granularity and
    every int4 grouping scales once. Reading the structure rather than the
    paperwork is what makes detection survive a vendor's choice of config key.
    """
    from safetensors import safe_open

    keys: set[str] = set()
    for name in sorted(os.listdir(model)):
        if not name.endswith(".safetensors"):
            continue
        with safe_open(os.path.join(model, name), framework="pt", device="cpu") as f:
            keys.update(f.keys())
    seconds = {
        key for key in keys
        if key.endswith(".weight_scale_2") or key.endswith(".weight_global_scale")
    }
    for key in seconds:
        stem = key.rsplit(".", 1)[0]
        if f"{stem}.weight_scale" in keys:
            return True
    return False


def _sidecar_quant_config(model: str) -> dict[str, Any]:
    """A quantization config kept beside config.json rather than inside it.

    NVIDIA's modelopt exports declare the format in `hf_quant_config.json` and
    leave `config.json` silent about it. Read only as a fallback, so a config
    that declares its own format is never second-guessed by a stale sidecar.
    """
    path = os.path.join(model, "hf_quant_config.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    inner = payload.get("quantization")
    quant = inner if isinstance(inner, dict) else payload
    if not quant.get("quant_method"):
        quant = dict(quant)
        quant["quant_method"] = "modelopt"
    return quant


def _nvfp4_declared_scheme(quant: dict[str, Any]) -> dict[str, Any] | None:
    """NVFP4 declared by a config, or None.

    NVFP4 is a 4-bit float on a 16-element block, so it is neither an int4
    grouping nor an FP8 granularity and no other reader here recognizes it. Left
    undetected, a compressed-tensors pack reports no scheme at all and a modelopt
    pack is mistaken for per-tensor FP8, because its scalar `weight_scale_2`
    global scale looks exactly like one.
    """
    if not isinstance(quant, dict):
        return None
    method = str(quant.get("quant_method") or "").lower()
    group: Any = None
    algorithm: str | None = None

    inner = quant.get("quantization")
    section = inner if isinstance(inner, dict) else quant
    algo = str(section.get("quant_algo") or quant.get("quant_algo") or "").upper()
    fmt = str(quant.get("format") or "").lower()

    # `quant_algo` names the algorithm unambiguously, so honour it whatever the
    # method says - some exporters omit `quant_method` entirely, and gating on it
    # left a plainly self-describing NVFP4 pack to be read from its shapes.
    if "NVFP4" in algo:
        group = section.get("group_size") or quant.get("group_size")
        algorithm = "modelopt" if method != "compressed-tensors" else "round_to_nearest"
    elif method == "modelopt" or (not method and "nvfp4" in fmt):
        if "nvfp4" not in fmt:
            return None
        group = section.get("group_size")
        algorithm = "modelopt"
    elif method == "compressed-tensors":
        if "nvfp4" not in str(quant.get("format") or "").lower():
            groups = quant.get("config_groups")
            if not isinstance(groups, dict):
                return None
            for entry in groups.values():
                weights = (entry or {}).get("weights")
                if not isinstance(weights, dict):
                    continue
                if (
                    str(weights.get("type") or "").lower() == "float"
                    and _as_int(weights.get("num_bits")) == 4
                ):
                    group = weights.get("group_size")
                    break
            else:
                return None
        else:
            for entry in (quant.get("config_groups") or {}).values():
                weights = (entry or {}).get("weights")
                if isinstance(weights, dict):
                    group = weights.get("group_size")
                    break
        algorithm = "round_to_nearest"
    else:
        return None

    return {
        "scheme": "nvfp4",
        "group_size": _as_int(group) or 16,
        "algorithm": algorithm,
    }


def _int4_declared_scheme(quant: dict[str, Any]) -> dict[str, Any] | None:
    """Format name, group size, and algorithm from a packed int4 config.

    Cells are named for the format (`int4_g128_asym`), not the vendor method.
    `desc_act` permutes group boundaries along K, so that flag is kept in the
    scheme name for the rounder to refuse rather than mislabel.
    """
    if not isinstance(quant, dict):
        return None
    method = str(quant.get("quant_method") or "").lower()
    if method in ("awq", "awq_marlin"):
        zero_point = _first_present(quant, ("zero_point",))
        has_zero_point = True if zero_point is None else bool(zero_point)
        return _int4_from_bits(
            _first_present(quant, ("w_bit", "bits")),
            _first_present(quant, ("q_group_size", "group_size")),
            symmetric=not has_zero_point,
            algorithm="awq",
        )
    if method in ("gptq", "gptq_marlin", "gptq_marlin_24"):
        return _int4_from_bits(
            quant.get("bits"),
            quant.get("group_size"),
            symmetric=bool(quant.get("sym", True)),
            algorithm="gptq",
            desc_act=bool(quant.get("desc_act", False)),
        )
    if method in ("auto-round", "auto_round"):
        return _int4_from_bits(
            quant.get("bits"),
            quant.get("group_size"),
            symmetric=bool(quant.get("sym", True)),
            algorithm="autoround",
            desc_act=bool(quant.get("desc_act", False)),
        )
    if method == "compressed-tensors":
        return _int4_from_compressed_tensors(quant)
    return None


def _refuse_mixed_bit_widths(quant: dict[str, Any]) -> None:
    """Refuse a checkpoint that declares two quantized widths in one config.

    An AutoRound-style `extra_config` names per-module overrides. A 16-bit or
    wider entry names a module left alone, which carries no scale and drops
    out of the matched set on its own. A narrower width that disagrees with
    the global one is a second format, and one cell rounding both at the
    global scheme would over-round the coarser half while reporting an exact
    match.
    """
    global_width = _as_int(quant.get("bits"))
    offenders: dict[str, int] = {}
    for module, override in (quant.get("extra_config") or {}).items():
        if not isinstance(override, dict):
            continue
        width = _as_int(override.get("bits"))
        if width is None or width >= 16:
            continue
        if global_width is None or width != global_width:
            offenders[str(module)] = width
    if not offenders:
        return
    module, width = sorted(offenders.items())[0]
    extra = f" and {len(offenders) - 1} more" if len(offenders) > 1 else ""
    raise SystemExit(
        f"{module} is declared at {width} bits against a global "
        f"{global_width}{extra}. A cell carries one scheme, so matching this "
        f"checkpoint needs per-tensor schemes, not just per-tensor names."
    )


def _scale_shapes(model: str) -> dict[str, tuple[int, ...]]:
    """Map each quantized weight to its scale's shape, reading headers only.

    Scale tensors and the packed operand they belong to can live in different
    shards, so names are gathered across the checkpoint before resolving.
    """
    from safetensors import safe_open

    all_keys: set[str] = set()
    scale_shapes: dict[str, tuple[int, ...]] = {}
    for name in sorted(os.listdir(model)):
        if not name.endswith(".safetensors"):
            continue
        with safe_open(os.path.join(model, name), framework="pt", device="cpu") as f:
            for key in f.keys():
                all_keys.add(key)
                if _is_scale_key(key):
                    scale_shapes[key] = tuple(f.get_slice(key).get_shape())
    found: dict[str, tuple[int, ...]] = {}
    for scale_key, shape in sorted(scale_shapes.items()):
        operand = _operand_from_scale_key(scale_key, all_keys)
        if operand is None:
            continue
        # An operand can carry both a per-block scale and a scalar global scale.
        # The finer one describes the granularity; letting shard order decide
        # would report a 16-element block as per-tensor FP8.
        if operand in found and _elements(found[operand]) >= _elements(shape):
            continue
        found[operand] = shape
    return found


def _elements(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total


def scheme_block_size(scheme: str, block_size: int | None) -> int | None:
    """The edge a scheme's scales are laid out on, or None if it has no grid.

    This is the scheme's own convention, which is what a ladder rung must use.
    A rung's width cannot come from the candidate beside it: the ladder exists
    to hold the weight set and the grid fixed while only the format changes.
    """
    group = parse_int4_scheme(scheme)
    if group is not None:
        size = group[0]
        return None if size == -1 else size
    if scheme == "fp8_block":
        return block_size or 128
    if scheme == "nvfp4":
        return block_size or 16
    if scheme == "mxfp8":
        return 32
    return None


def unexpressible_reason(
    model: str,
    selected: tuple[str, ...],
    scheme: str,
    block_size: int | None = None,
) -> str | None:
    """Why `scheme` cannot round `model`'s selected weights, or None if it can.

    A blocked or grouped scheme lays its scales on a fixed grid, and a weight
    whose shape is not a multiple of that grid cannot carry them. A real
    quantizer pads, which changes the scales and so changes what is being
    measured, so the ladder reports such a rung as absent with this reason
    rather than guessing at a padding the deployed checkpoint never used.
    `fp8_block` is square, so it constrains both dimensions; every other grid
    applies along the reduction dimension only.
    """
    edge = scheme_block_size(scheme, block_size)
    if edge is None:
        return None
    for name, shape in sorted(_component_shapes(model, selected).items()):
        if len(shape) < 2:
            continue
        dims = shape[-2:] if scheme == "fp8_block" else shape[-1:]
        offenders = [dim for dim in dims if dim % edge]
        if offenders:
            return (
                f"{scheme} lays scales on a {edge}-wide grid and "
                f"{os.path.basename(name)} is {'x'.join(str(d) for d in shape)}, "
                f"whose {offenders[0]} is not a multiple of {edge}"
            )
    return None


def names_sha256(names: list[str] | set[str]) -> str:
    """Stable digest of a tensor-name set, independent of input order."""
    return hashlib.sha256("\n".join(sorted(names)).encode("utf-8")).hexdigest()


def weights_identity(model: str) -> str:
    """Content digest of every safetensors shard at ``model``.

    The digest includes shard names and every file byte, including safetensors
    headers. Same-geometry checkpoints with different tensor values therefore
    cannot share a Law 16 identity.
    """
    identity = hashlib.sha256()
    found = False
    for fname in sorted(os.listdir(model)):
        if not fname.endswith(".safetensors"):
            continue
        found = True
        path = os.path.join(model, fname)
        encoded_name = fname.encode("utf-8")
        identity.update(len(encoded_name).to_bytes(4, "big"))
        identity.update(encoded_name)
        identity.update(os.path.getsize(path).to_bytes(16, "big"))
        with open(path, "rb") as handle:
            while chunk := handle.read(16 * 1024 * 1024):
                identity.update(chunk)
    if not found:
        raise SystemExit(f"no safetensors weights at {model} to bind a score to")
    return identity.hexdigest()


def weights_bytes(model: str) -> int:
    """Total size of the checkpoint's safetensors shards on disk.

    What the quantization actually costs to store and to load, including the
    scales and zero points the format needs, which is the number a reader
    comparing two formats wants and cannot derive from the scheme name.
    """
    return sum(
        os.path.getsize(os.path.join(model, name))
        for name in os.listdir(model)
        if name.endswith(".safetensors")
    )


def inspect_for_disk(report: dict[str, Any]) -> dict[str, Any]:
    """Inspect JSON without the name list, which is tens of thousands of strings.

    `inspect.json` is copied into assembled artifacts. The digest lets a cell
    prove it matched the same set without publishing every name.
    """
    out = {key: value for key, value in report.items() if key != "quantized_names"}
    out["quantized_names_sha256"] = names_sha256(report.get("quantized_names") or [])
    return out


def _component_shapes(
    model: str, selected: tuple[str, ...]
) -> dict[str, tuple[int, ...]]:
    """Canonical names in `model` that classify into `selected`, with shapes.

    Headers only; no tensor data is read.
    """
    from safetensors import safe_open

    shapes: dict[str, tuple[int, ...]] = {}
    if not os.path.isdir(model):
        return shapes
    for fname in sorted(os.listdir(model)):
        if not fname.endswith(".safetensors"):
            continue
        with safe_open(
            os.path.join(model, fname), framework="pt", device="cpu"
        ) as handle:
            for key in handle.keys():
                canonical = _as_weight_name(key) or key
                if classify(canonical, selected) is not None:
                    shapes[canonical] = tuple(handle.get_slice(key).get_shape())
    return shapes


def _component_names(model: str, selected: tuple[str, ...]) -> set[str]:
    """Canonical `.weight` names in `model` that classify into `selected`."""
    return set(_component_shapes(model, selected))


_FUSED_EXPERT_MEMBERS: dict[str, tuple[str, ...]] = {
    "gate_up_proj": ("gate_proj", "up_proj"),
    "gate_proj": ("gate_proj",),
    "up_proj": ("up_proj",),
    "down_proj": ("down_proj",),
}

_EXPERT_STACK = re.compile(r"^(?P<prefix>.+\.experts)\.(?P<leaf>[^.]+)$")

# The module a checkpoint hangs its experts off is the exporter's choice, not a
# property of the model: the same weight is `layers.N.experts.<e>...` in one pack
# and `layers.N.moe.experts.<e>...` in another. Comparing a pack's names against
# the reference's own means erasing that choice from both sides first.
_EXPERT_WRAPPER = re.compile(
    r"\.(?:moe|mlp|block_sparse_moe|feed_forward)\.experts\."
)


def canonical_expert_name(name: str) -> str:
    """`name` with any MoE wrapper module between the layer and `experts`
    removed, so two exporters' names for one weight compare equal."""
    return _EXPERT_WRAPPER.sub(".experts.", name)


def _fused_constituents(name: str, shape: tuple[int, ...]) -> set[str] | None:
    """Per-expert names carried by a fused expert stack, or None if not one.

    A reference that stores experts fused holds `experts.gate_up_proj` with
    shape `[num_experts, ...]` where a quantized pack names the same weights
    `experts.<e>.gate_proj.weight`. Enumerating the stack's members is what
    lets a per-tensor cell decide whether a pack covers the whole stack.
    """
    if len(shape) != 3:
        return None
    hit = _EXPERT_STACK.match(name)
    if hit is None:
        return None
    members = _FUSED_EXPERT_MEMBERS.get(hit.group("leaf"))
    if members is None:
        return None
    prefix = hit.group("prefix")
    return {
        canonical_expert_name(f"{prefix}.{index}.{member}.weight")
        for index in range(shape[0])
        for member in members
    }


def resolve_matched(
    reference: str, selected: tuple[str, ...], matched: set[str] | list[str]
) -> set[str]:
    """Reference tensor names that carry the weights named in `matched`.

    Names present in the reference resolve to themselves. A per-expert name
    resolves to the fused stack that holds it. A stack the pack covers only in
    part is refused: rounding it whole would over-round the experts the pack
    left alone, and rounding it in part is not a tensor-level operation. Both
    sides are compared with any MoE wrapper module erased, so a pack that keeps
    its experts under `moe.` still resolves against a reference that does not.
    """
    wanted = {
        canonical_expert_name(name)
        for name in matched
        if classify(name, selected) is not None
    }
    shapes = _component_shapes(reference, selected)
    resolved = {
        name for name in shapes if canonical_expert_name(name) in wanted
    }
    remaining = wanted - {canonical_expert_name(name) for name in resolved}
    partial: list[tuple[str, int, int]] = []
    for name, shape in sorted(shapes.items()):
        members = _fused_constituents(name, shape)
        if not members:
            continue
        covered = members & remaining
        if not covered:
            continue
        if covered != members:
            partial.append((name, len(covered), len(members)))
            continue
        resolved.add(name)
        remaining -= covered
    if partial:
        name, covered, total = partial[0]
        extra = f" and {len(partial) - 1} more" if len(partial) > 1 else ""
        raise SystemExit(
            f"{name} is a fused stack of {total} expert weights and the "
            f"checkpoint quantized {covered} of them{extra}. Rounding the "
            f"stack whole would over-round the rest, so this cell cannot "
            f"match the checkpoint exactly."
        )
    if remaining:
        missing = sorted(remaining)
        extra = f" and {len(missing) - 1} more" if len(missing) > 1 else ""
        raise SystemExit(
            f"{len(missing)} matched weight(s) have no counterpart in "
            f"{reference}: {missing[0]}{extra}"
        )
    return resolved


def match_digest(
    reference: str,
    inspection: dict[str, Any],
    components: tuple[str, ...] | list[str],
) -> str | None:
    """Short digest of the reference tensors a cell would round, or None when
    that is every tensor the components own.

    Returning None is what keeps a fully-covered cell on the path (and the
    report) a component-wide conversion already wrote. The digest covers
    reference names, not the pack's, so a fused and an unfused pack that touch
    the same weights land on the same cell.
    """
    selected = tuple(components)
    resolved = resolve_matched(
        reference, selected, inspection.get("quantized_names") or []
    )
    if resolved == _component_names(reference, selected):
        return None
    return names_sha256(resolved)[:MATCH_DIGEST_LEN]


def inspect(model: str) -> dict[str, Any]:
    """Report what a quantized checkpoint actually quantized, and how finely.

    Two questions this answers before any GPU time is spent. Whether the router
    is quantized at all - if it is not, router-precision cost is zero by
    construction and any routing divergence comes from upstream activations
    instead. And what granularity to match, so a QDQ variant is comparable to
    this checkpoint rather than to a scheme nobody deployed.
    """
    from safetensors import safe_open

    config_path = os.path.join(model, "config.json")
    if not os.path.isfile(config_path):
        raise SystemExit(f"no config.json at {model}")
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)
    sections = quant_sections(model, config)
    quant = sections[0] if sections else {}
    reason = unloadable_reason_from_config(config)
    quark_scheme = next(
        (s for s in map(_quark_declared_scheme, sections) if s is not None), None
    )
    _refuse_mixed_bit_widths(quant)
    nvfp4 = next(
        (s for s in map(_nvfp4_declared_scheme, sections) if s is not None), None
    )
    int4 = next(
        (s for s in map(_int4_declared_scheme, sections) if s is not None), None
    )
    if nvfp4 is None and int4 is None and quark_scheme is None:
        # No config anywhere named a format. Two levels of weight scale is NVFP4's
        # structural signature, and believing it here is what stops the packed
        # operand's half-width last dim from reading as an 8-wide FP8 block.
        if _two_level_scaled(model):
            nvfp4 = {
                "scheme": "nvfp4",
                "group_size": 16,
                "algorithm": "round_to_nearest",
            }
            print(f"{os.path.basename(model)} declares no format; "
                  f"two-level weight scales identify it as nvfp4")

    scales = _scale_shapes(model)
    packed_scales = any(
        name.endswith(".qweight") or name.endswith("_packed") for name in scales
    )
    weight_shapes: dict[str, tuple[int, ...]] = {}
    operand_names: set[str] = set()
    for name in sorted(os.listdir(model)):
        if not name.endswith(".safetensors"):
            continue
        with safe_open(os.path.join(model, name), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key in scales:
                    weight_shapes[key] = tuple(f.get_slice(key).get_shape())
                canonical = _as_weight_name(key)
                if canonical is not None:
                    operand_names.add(canonical)

    granularity: str | None = None
    block: int | None = None
    # Packed int4 scale/weight shapes are not FP8 granularities. A qweight's
    # packed last dim would otherwise look like a block scale.
    if not packed_scales:
        for weight, scale_shape in sorted(scales.items()):
            shape = weight_shapes.get(weight)
            if not shape or len(shape) < 2:
                continue
            elements = _elements(scale_shape)
            if elements == 1:
                granularity = "fp8_per_tensor"
            elif elements == shape[0]:
                granularity = "fp8_per_channel"
            else:
                granularity = "fp8_block"
                if len(scale_shape) >= 2 and scale_shape[-1]:
                    block = max(1, round(shape[-1] / scale_shape[-1]))
            break

    detected_scheme = quark_scheme
    detected_block = block
    quant_algorithm: str | None = None
    if detected_scheme is None and nvfp4 is not None:
        detected_scheme = nvfp4["scheme"]
        detected_block = nvfp4["group_size"]
        quant_algorithm = nvfp4["algorithm"]
    elif detected_scheme is None and int4 is not None:
        detected_scheme = int4["scheme"]
        detected_block = int4["group_size"]
        quant_algorithm = int4["algorithm"]
    elif detected_scheme is None:
        detected_scheme = granularity

    coverage: dict[str, dict[str, int]] = {}
    for component in COMPONENTS:
        coverage[component] = {"weights": 0, "quantized": 0}
    quantized_names: set[str] = set()
    for name in {_as_weight_name(key) or key for key in scales}:
        component = classify(name, COMPONENTS)
        if component is None:
            continue
        coverage[component]["quantized"] += 1
        quantized_names.add(name)
    for name in operand_names:
        component = classify(name, COMPONENTS)
        if component is not None:
            coverage[component]["weights"] += 1

    return {
        "model": os.path.abspath(model),
        "inspect_version": INSPECT_VERSION,
        "weights_sha256": weights_identity(model),
        "weights_bytes": weights_bytes(model),
        "quant_method": quant.get("quant_method"),
        "declared": {
            key: quant.get(key)
            for key in ("fmt", "activation_scheme", "weight_block_size", "strategy")
            if quant.get(key) is not None
        },
        "detected_scheme": detected_scheme,
        "detected_block": detected_block,
        "quant_algorithm": quant_algorithm,
        "coverage": coverage,
        "quantized_names": sorted(quantized_names),
        "unloadable_reason": reason,
    }


def render_inspection(report: dict[str, Any]) -> str:
    lines = [
        f"{os.path.basename(report['model'])}",
        f"  quant_method: {report['quant_method'] or 'none declared'}",
    ]
    for key, value in (report["declared"] or {}).items():
        lines.append(f"  {key}: {value}")
    if report.get("weights_bytes"):
        lines.append(
            f"  weights on disk: {report['weights_bytes'] / 2**30:.2f} GiB"
        )
    lines.append(f"  detected scheme: {report['detected_scheme'] or 'unquantized'}")
    if report.get("detected_block") is not None:
        lines.append(f"  detected block: {report['detected_block']}")
    if report.get("quant_algorithm"):
        lines.append(f"  quant algorithm: {report['quant_algorithm']}")
    lines.append("  component coverage (quantized / total weights):")
    for component, counts in report["coverage"].items():
        if not counts["weights"] and not counts["quantized"]:
            continue
        total = counts["weights"]
        done = counts["quantized"]
        verdict = "all" if done and done >= total else ("none" if not done else "some")
        lines.append(f"    {component}: {done} / {total} ({verdict})")
    if report.get("unloadable_reason"):
        lines += ["", f"  will not load: {report['unloadable_reason']}"]
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
    only: set[str] | None = None,
) -> dict[str, Any]:
    """Copy a checkpoint, rounding the selected components through `scheme`.

    `only` restricts rounding to that name set (canonical `.weight` names as
    the quantized pack spells them). Each is resolved to the reference tensor
    that carries it, which for a fused expert stack is one tensor per many
    names. A name with no counterpart, or a stack the pack covers only in
    part, is a hard error: the cell would otherwise look like a match while
    rounding nothing, or more than the pack did.
    """
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    _assert_unquantized(model)
    if only is not None:
        only = resolve_matched(model, selected, only)
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
                # A 1-D parameter is a gain, a bias, or a scale, never a matmul
                # operand, and no quantizer rounds one. Copying it verbatim keeps
                # an over-broad pattern from turning into a crash 2 shards into a
                # 26-shard conversion.
                if (
                    component is None
                    or tensor.ndim < 2
                    or not torch.is_floating_point(tensor)
                ):
                    tensors[name] = tensor
                    untouched += 1
                    continue
                if only is not None and name not in only:
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

    note = (
        "Quantize-dequantize variant stored in the source dtype. Weights are "
        "numerically what the quantized model would use; arithmetic is not. "
        "Comparing this against the real quantized checkpoint isolates the "
        "kernel's contribution."
    )
    if scheme.startswith("int4_"):
        note += (
            " This cell matched the int4 format (bit width, group size, "
            "symmetry), not a vendor calibration algorithm such as AWQ or GPTQ."
        )
    tensors_rounded = sum(entry["tensors"] for entry in rollup.values())
    manifest = {
        "source_model": os.path.abspath(model),
        "scheme": scheme,
        "block_size": (
            block_size
            if scheme in ("nvfp4", "fp8_block") or scheme.startswith("int4_")
            else None
        ),
        "components": list(selected),
        "device": device,
        "tensors_untouched": untouched,
        "tensors_rounded": tensors_rounded,
        "match_mode": "per_tensor" if only is not None else "per_component",
        "quantized_names_sha256": (
            names_sha256(only) if only is not None else None
        ),
        "components_detail": rollup,
        "vllm_version": _vllm_version(),
        "note": note,
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
    import tempfile

    amd_quark = {
        "quantization_config": {
            "quant_method": "quark",
            "algo_config": [
                {"name": "awq", "scaling_layers": [{"inp": "mlp.gate_proj"}]}
            ],
            "export": {"kv_cache_group": [], "pack_method": "reorder"},
            "global_quant_config": {
                "weight": {"dtype": "int4", "qscheme": "per_group"}
            },
        }
    }
    reason = unloadable_reason_from_config(amd_quark)
    assert reason is not None and "algo_config" in reason, reason
    assert "amd-quark" in reason
    fp8_quark = {
        "quantization_config": {
            "quant_method": "quark",
            "export": {"kv_cache_group": ["*k_proj"], "pack_method": "reorder"},
            "exclude": ["lm_head"],
            "global_quant_config": {
                "weight": {"dtype": "fp8", "qscheme": "per_tensor"}
            },
        }
    }
    assert unloadable_reason_from_config(fp8_quark) is None

    # Gemma 4's 704-wide experts do not divide by AutoRound's 128 group.
    gemma_autoround = {
        "text_config": {"moe_intermediate_size": 704, "intermediate_size": 2112},
        "quantization_config": {"quant_method": "auto-round", "group_size": 128},
    }
    reason = unloadable_reason_from_config(gemma_autoround)
    assert reason is not None and "moe_intermediate_size 704" in reason, reason
    assert "768" in reason, reason
    for group in (32, 64):
        loadable = {
            "text_config": {"moe_intermediate_size": 704, "intermediate_size": 2112},
            "quantization_config": {"quant_method": "awq", "group_size": group},
        }
        assert unloadable_reason_from_config(loadable) is None, group

    awq_cfg = {
        "quant_method": "awq",
        "bits": 4,
        "group_size": 128,
        "zero_point": True,
    }
    awq = _int4_declared_scheme(awq_cfg)
    assert awq is not None
    assert awq["scheme"] == "int4_g128_asym", awq
    assert awq["group_size"] == 128
    assert awq["algorithm"] == "awq"
    awq_sym = _int4_declared_scheme({**awq_cfg, "zero_point": False})
    assert awq_sym is not None and awq_sym["scheme"] == "int4_g128_sym"
    gptq = _int4_declared_scheme(
        {
            "quant_method": "gptq",
            "bits": 4,
            "group_size": 128,
            "sym": True,
            "desc_act": False,
        }
    )
    assert gptq is not None and gptq["scheme"] == "int4_g128_sym"
    assert gptq["algorithm"] == "gptq"
    gptq_act = _int4_declared_scheme(
        {
            "quant_method": "gptq",
            "bits": 4,
            "group_size": 64,
            "sym": True,
            "desc_act": True,
        }
    )
    assert gptq_act is not None
    assert gptq_act["scheme"] == "int4_g64_sym_desc_act"
    ct_int4 = _int4_declared_scheme(
        {
            "quant_method": "compressed-tensors",
            "config_groups": {
                "group_0": {
                    "weights": {
                        "num_bits": 4,
                        "group_size": 128,
                        "symmetric": False,
                        "type": "int",
                    }
                }
            },
        }
    )
    assert ct_int4 is not None
    assert ct_int4["scheme"] == "int4_g128_asym"
    assert ct_int4["algorithm"] == "round_to_nearest"
    ct_fp8 = _int4_declared_scheme(
        {
            "quant_method": "compressed-tensors",
            "config_groups": {
                "group_0": {
                    "weights": {
                        "num_bits": 8,
                        "strategy": "tensor",
                        "type": "float",
                    }
                }
            },
        }
    )
    assert ct_fp8 is None
    autoround = _int4_declared_scheme(
        {
            "quant_method": "auto-round",
            "bits": 4,
            "group_size": 128,
            "sym": True,
        }
    )
    assert autoround is not None and autoround["algorithm"] == "autoround"
    eight_bit = {"quant_method": "awq", "bits": 8, "group_size": 128}
    assert _int4_declared_scheme(eight_bit) is None
    print("  declared int4 schemes: awq/gptq/ct/autoround")

    # NVFP4 is a 4-bit float on a 16-element block: neither an int4 grouping nor
    # an FP8 granularity. Undetected, compressed-tensors packs report no scheme
    # and modelopt packs are mistaken for per-tensor FP8.
    for label, quant in (
        (
            "compressed-tensors format",
            {
                "quant_method": "compressed-tensors",
                "format": "nvfp4-pack-quantized",
                "config_groups": {
                    "group_0": {
                        "weights": {"num_bits": 4, "type": "float", "group_size": 16}
                    }
                },
            },
        ),
        (
            "compressed-tensors groups",
            {
                "quant_method": "compressed-tensors",
                "config_groups": {
                    "group_0": {
                        "weights": {"num_bits": 4, "type": "float", "group_size": 16}
                    }
                },
            },
        ),
        (
            "modelopt nested",
            {
                "quant_method": "modelopt",
                "quantization": {"quant_algo": "NVFP4", "group_size": 16},
            },
        ),
        (
            "modelopt flat",
            {"quant_method": "modelopt", "quant_algo": "NVFP4"},
        ),
    ):
        found = _nvfp4_declared_scheme(quant)
        assert found is not None, label
        assert found["scheme"] == "nvfp4", label
        assert found["group_size"] == 16, (label, found)
        # Never an algorithm the artifact treats as calibrated: NVFP4 packs here
        # round to nearest, and claiming calibration would credit work not done.
        assert found["algorithm"] in ("round_to_nearest", "modelopt"), label
    # An int4 compressed-tensors pack and an FP8 config are not NVFP4.
    assert _nvfp4_declared_scheme(
        {
            "quant_method": "compressed-tensors",
            "config_groups": {
                "group_0": {
                    "weights": {"num_bits": 4, "type": "int", "group_size": 128}
                }
            },
        }
    ) is None
    assert _nvfp4_declared_scheme(
        {"quant_method": "modelopt", "quantization": {"quant_algo": "FP8"}}
    ) is None
    assert _nvfp4_declared_scheme({"quant_method": "fp8"}) is None
    # An exporter that names the algorithm but omits the method still resolves;
    # gating on the method is what let a self-describing pack be read from shapes.
    no_method = _nvfp4_declared_scheme({"quant_algo": "NVFP4", "group_size": 16})
    assert no_method is not None and no_method["group_size"] == 16, no_method
    assert _nvfp4_declared_scheme(
        {"quantization": {"quant_algo": "nvfp4"}}
    ) is not None
    assert scheme_block_size("nvfp4", 16) == 16
    # NVIDIA declares the format beside config.json, not inside it. Undetected,
    # the packed weight's half-width last dim makes its 16-wide scale look like
    # an 8-wide FP8 block.
    with tempfile.TemporaryDirectory() as sidecar:
        with open(
            os.path.join(sidecar, "hf_quant_config.json"),
            "w", encoding="utf-8", newline="\n",
        ) as handle:
            json.dump(
                {
                    "producer": {"name": "modelopt"},
                    "quantization": {"quant_algo": "NVFP4", "group_size": 16},
                },
                handle,
            )
        found = _nvfp4_declared_scheme(_sidecar_quant_config(sidecar))
        assert found is not None and found["scheme"] == "nvfp4", found
        assert found["group_size"] == 16, found
        # The sidecar is reached through the same gathering the inspector uses.
        gathered = quant_sections(sidecar, {"quantization_config": {}})
        assert any(
            _nvfp4_declared_scheme(section) is not None for section in gathered
        ), gathered
    with tempfile.TemporaryDirectory() as bare:
        assert _sidecar_quant_config(bare) == {}
        assert quant_sections(bare, {}) == []
    # A declaration nested under text_config is found without a special case.
    nested = quant_sections(
        os.devnull + "-absent",
        {"text_config": {"quantization_config": {"quant_algo": "NVFP4"}}},
    )
    assert any(_nvfp4_declared_scheme(s) is not None for s in nested), nested
    print("  NVFP4 is detected from compressed-tensors and modelopt configs")
    print("  NVFP4 is detected from a hf_quant_config.json sidecar")

    # A 16-bit override names a module left alone, which is how AutoRound
    # spells "int4-mixed"; every such module carries no scale and drops out of
    # the matched set already. An 8-bit override is a second format.
    kept_in_bf16 = {
        "quant_method": "auto-round",
        "bits": 4,
        "group_size": 128,
        "extra_config": {
            "model.layers.0.mlp.gate": {"bits": 16},
            "model.layers.0.self_attn.q_proj": {"bits": 16, "data_type": "fp"},
        },
    }
    _refuse_mixed_bit_widths(kept_in_bf16)
    try:
        _refuse_mixed_bit_widths(
            {
                "quant_method": "auto-round",
                "bits": 4,
                "group_size": 128,
                "extra_config": {
                    "model.layers.0.self_attn.q_proj": {"bits": 8}
                },
            }
        )
    except SystemExit as exc:
        assert "per-tensor schemes" in str(exc)
    else:
        raise AssertionError("two quantized widths in one config must refuse")
    print("  mixed widths: 16-bit overrides pass, 8-bit beside 4-bit refuses")

    keys = {
        "model.layers.0.self_attn.q_proj.qweight",
        "model.layers.0.self_attn.q_proj.qzeros",
        "model.layers.0.self_attn.q_proj.scales",
    }
    assert (
        _operand_from_scale_key(
            "model.layers.0.self_attn.q_proj.scales", keys
        )
        == "model.layers.0.self_attn.q_proj.qweight"
    )
    packed = {
        "model.layers.0.mlp.experts.0.down_proj.weight_packed",
        "model.layers.0.mlp.experts.0.down_proj.weight_scale",
    }
    assert (
        _operand_from_scale_key(
            "model.layers.0.mlp.experts.0.down_proj.weight_scale", packed
        )
        == "model.layers.0.mlp.experts.0.down_proj.weight_packed"
    )
    fp8_keys = {
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.q_proj.weight_scale_inv",
    }
    assert (
        _operand_from_scale_key(
            "model.layers.0.self_attn.q_proj.weight_scale_inv", fp8_keys
        )
        == "model.layers.0.self_attn.q_proj.weight"
    )
    assert (
        _as_weight_name("model.layers.0.self_attn.q_proj.qweight")
        == "model.layers.0.self_attn.q_proj.weight"
    )
    print("  packed scale operands resolve to qweight/weight_packed")

    assert parse_int4_scheme("int4_g32_asym") == (32, False)
    assert parse_int4_scheme("int4_g128_sym") == (128, True)
    assert parse_int4_scheme("int4_g-1_sym") == (-1, True)
    assert parse_int4_scheme("fp8_block") is None
    assert scheme_arg("int4_g32_asym") == "int4_g32_asym"
    try:
        parse_int4_scheme("int4_g32_sym_desc_act")
    except ValueError as exc:
        assert "desc_act" in str(exc)
    else:
        raise AssertionError("desc_act must be refused, not rounded")
    print("  int4 scheme names parse; desc_act is refused")

    import torch

    torch.manual_seed(0)
    try:
        quantize_dequantize(
            torch.zeros(8, 30, dtype=torch.bfloat16), "int4_g32_asym", 32
        )
    except ValueError as exc:
        assert "multiple" in str(exc)
    else:
        raise AssertionError("non-multiple K must be refused")

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
    print(f"  ladder is ordered: nvfp4 {error['nvfp4']:.4f} > mxfp8 "
          f"{error['mxfp8']:.4f}")

    def _assert_int4_rung(source: Any, scheme: str) -> float:
        parsed = parse_int4_scheme(scheme)
        assert parsed is not None, scheme
        group, _ = parsed
        once = quantize_dequantize(source, scheme, group)
        twice = quantize_dequantize(once, scheme, group)
        assert once.dtype == source.dtype, f"{scheme} changed dtype"
        assert not torch.equal(once, source), f"{scheme} was a no-op"
        relative = _error_stats(source, once)["relative_rms"]
        assert relative < 0.5, f"{scheme} relative rms {relative} is implausible"
        if torch.equal(once, twice):
            print(f"  {scheme}: idempotent, relative rms {relative:.4f}")
            return relative
        # Scales and zero-points are recomputed from the tensor. After one
        # round the extrema are reconstruction grid points snapped to bf16,
        # so a second min/max is a different affine map and values can move
        # by a bin, not a bf16 ulp of the stored weight.
        second = _error_stats(once, twice)["relative_rms"]
        moved = float((twice.float() - once.float()).abs().max())
        assert second < 0.5 * relative, (
            f"{scheme} second pass rms {second:.4f} is not well below the "
            f"format rms {relative:.4f} (max abs {moved:.6g}); the variant "
            "is not a near-fixed-point of the format"
        )
        print(
            f"  {scheme}: near-fixed-point, relative rms {relative:.4f}; "
            f"second pass {second:.4f} (max abs {moved:.6g}) because "
            "int4 min/max scales are recomputed from the bf16 snapshot"
        )
        return relative

    for scheme in ("int4_g32_asym", "int4_g128_asym", "int4_g128_sym"):
        _assert_int4_rung(weight, scheme)

    # Groups run along K (the last dim of an HF weight). A loud 32-wide
    # slice poisons a 128-wide scale and leaves a 32-wide scale untouched.
    along_k = weight.clone()
    along_k[:, :32] *= 1e6
    quiet_k = slice(32, 128)
    g32 = _error_stats(
        along_k[:, quiet_k],
        quantize_dequantize(along_k, "int4_g32_asym", 32)[:, quiet_k],
    )["relative_rms"]
    g128 = _error_stats(
        along_k[:, quiet_k],
        quantize_dequantize(along_k, "int4_g128_asym", 128)[:, quiet_k],
    )["relative_rms"]
    assert g32 < 0.5 * g128, (
        f"on the quiet K-slice beside a loud group, g32 {g32} did not "
        f"clearly beat g128 {g128}, which is the case group size exists for"
    )
    print(f"  int4 group size: quiet K-slice g32 {g32:.4f} vs g128 {g128:.4f}")

    # A zero point buys a step, and the size of that win is bounded by the
    # codebooks. On a group spanning [min, max] with min < 0 < max,
    # asymmetric (uint4) spreads 15 codes across max-min, while symmetric
    # (uint4b8, levels -8..7) must fit max into 7 positive codes:
    #
    #   asym step = (max - min) / 15      sym step = max(max/7, |min|/8)
    #
    # The best possible ratio is therefore 7/15 ~ 0.47, approached only as
    # min -> 0-. A [-1, 10] ramp predicts (11/15)/(10/7) = 0.513, which is
    # what this measures - so a 0.5 bar would be demanding the format's
    # ceiling rather than testing the zero point. 0.7 is a clear win with
    # room for bf16 storage and endpoint effects.
    group_k = 128
    ramp = torch.linspace(-1.0, 10.0, group_k, dtype=weight.dtype)
    shifted = ramp.repeat(weight.shape[-1] // group_k).unsqueeze(0)
    shifted = shifted.expand_as(weight).contiguous()
    asym = _error_stats(
        shifted, quantize_dequantize(shifted, "int4_g128_asym", 128)
    )["relative_rms"]
    sym = _error_stats(
        shifted, quantize_dequantize(shifted, "int4_g128_sym", 128)
    )["relative_rms"]
    assert asym < 0.7 * sym, (
        f"on a [-1, 10] range, asymmetric {asym} did not clearly beat "
        f"symmetric {sym}; the zero point should buy a step near the "
        f"7/15 codebook ratio"
    )
    print(
        f"  int4 zero point: [-1, 10] asym {asym:.4f} vs sym {sym:.4f} "
        f"(ratio {asym / sym:.3f}, codebook predicts 0.513)"
    )

    # FP8 granularity is nearly free on a well-conditioned matrix, because E4M3
    # carries an exponent per element: the relative step follows the mantissa,
    # not the scale, so a finer scale buys almost nothing and the small residual
    # difference is which side of a binade each value lands on. Granularity earns
    # its keep only when one row's dynamic range does not fit the format.
    spread = abs(error["fp8_per_channel"] - error["fp8_per_tensor"])
    assert spread < 0.1 * error["fp8_per_tensor"], (
        f"per-channel {error['fp8_per_channel']} and per-tensor "
        f"{error['fp8_per_tensor']} differ by more than 10% on a well-"
        f"conditioned matrix, which neither granularity should manage"
    )

    # With one row six orders of magnitude louder, a single scale drives the rest
    # of the matrix into E4M3's subnormals and a per-row scale does not.
    outlier = weight.clone()
    outlier[0] *= 1e6
    # Measured on the quiet rows alone. Over the whole matrix the outlier row
    # dominates the norm, so a metric relative to it would report that losing
    # every other row costs nothing.
    quiet = slice(1, None)
    by_tensor = _error_stats(
        outlier[quiet], quantize_dequantize(outlier, "fp8_per_tensor", 64)[quiet]
    )["relative_rms"]
    by_channel = _error_stats(
        outlier[quiet], quantize_dequantize(outlier, "fp8_per_channel", 64)[quiet]
    )["relative_rms"]
    assert by_channel < 0.5 * by_tensor, (
        f"on the quiet rows of a matrix with one outlier row, per-channel "
        f"{by_channel} did not clearly beat per-tensor {by_tensor}, which is the "
        f"case granularity exists for"
    )
    print(f"  granularity: within 10% when well conditioned; on quiet rows "
          f"beside an outlier, per-channel {by_channel:.4f} vs per-tensor "
          f"{by_tensor:.4f}")

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
    packed_names = {
        "model.layers.0.self_attn.q_proj.qweight": "attention",
        "model.layers.0.mlp.experts.0.down_proj.qweight": "experts",
        "model.layers.0.mlp.gate.qweight": "router",
        "model.layers.0.mlp.gate_proj.weight_packed": "dense_mlp",
    }
    for name, expected in packed_names.items():
        actual = classify(_as_weight_name(name), COMPONENTS)
        assert actual == expected, f"{name}: classified {actual}, expected {expected}"
    print("  router and gate_proj do not alias; components are disjoint")

    # Gemma 4 puts no MoE wrapper between the layer and its experts, names the
    # router `router.proj`, keeps a dense MLP in every layer beside the expert
    # block, and ships a vision tower whose attention the scored graph never
    # touches. Every one of these was silently unmatched.
    gemma = {
        "model.language_model.layers.0.experts.gate_up_proj": "experts",
        "model.language_model.layers.0.experts.down_proj": "experts",
        "model.language_model.layers.0.router.proj.weight": "router",
        "model.language_model.layers.0.mlp.gate_proj.weight": "dense_mlp",
        "model.language_model.layers.0.mlp.down_proj.weight": "dense_mlp",
        "model.language_model.layers.0.self_attn.q_proj.weight": "attention",
        "model.language_model.layers.0.self_attn.k_norm.weight": None,
        "model.language_model.layers.0.router.scale": None,
        "model.language_model.layers.0.router.per_expert_scale": None,
        "model.language_model.layers.0.layer_scalar": None,
        "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight": None,
        "model.vision_tower.encoder.layers.0.mlp.gate_proj.linear.weight": None,
        "model.embed_vision.embedding_projection.weight": None,
    }
    for name, expected in gemma.items():
        actual = classify(name, COMPONENTS)
        assert actual == expected, f"{name}: classified {actual}, expected {expected}"
    assert _fused_constituents(
        "model.language_model.layers.0.experts.gate_up_proj", (128, 1408, 2816)
    ) == {
        f"model.language_model.layers.0.experts.{e}.{leaf}.weight"
        for e in range(128)
        for leaf in ("gate_proj", "up_proj")
    }
    # An exporter that unfuses Gemma's stack may put the experts back under a
    # wrapper the reference does not have.
    wrapped = "model.language_model.layers.0.moe.experts.7.gate_proj.weight"
    assert classify(wrapped, COMPONENTS) == "experts"
    assert canonical_expert_name(wrapped) == (
        "model.language_model.layers.0.experts.7.gate_proj.weight"
    )
    assert canonical_expert_name(
        "model.layers.0.mlp.experts.3.down_proj.weight"
    ) == "model.layers.0.experts.3.down_proj.weight"
    print("  Gemma 4 experts, router, dense MLP, and vision tower classify")

    # A pack that keeps its experts fused names the operand with no `.weight`.
    # Canonicalizing it to the bare stack is what lets it match the reference.
    stack = "model.language_model.layers.0.experts.gate_up_proj"
    assert _as_weight_name(stack + "_packed") == stack
    assert _as_weight_name(stack + "_scale") is None, "a scale is not an operand"
    assert _is_scale_key(stack + "_scale")
    assert not _is_scale_key(stack + "_packed")
    assert _operand_from_scale_key(
        stack + "_scale", {stack + "_packed"}
    ) == stack + "_packed"
    assert classify(_as_weight_name(stack + "_packed"), COMPONENTS) == "experts"
    # The dotted conventions must keep resolving as before.
    q = "model.layers.0.self_attn.q_proj"
    assert _as_weight_name(q + ".weight_packed") == q + ".weight"
    assert _as_weight_name(q + ".qweight") == q + ".weight"
    assert _operand_from_scale_key(
        q + ".weight_scale", {q + ".weight_packed"}
    ) == q + ".weight_packed"
    print("  fused experts.<leaf>_packed resolves to the reference's own stack")

    for scheme, edge in (
        ("fp8_block", 128),
        ("nvfp4", 16),
        ("mxfp8", 32),
        ("int4_g64_sym", 64),
        ("int4_g-1_sym", None),
        ("fp8_per_channel", None),
        ("fp8_per_tensor", None),
    ):
        assert scheme_block_size(scheme, None) == edge, scheme

    from safetensors.torch import load_file, save_file

    with tempfile.TemporaryDirectory() as root:
        save_file(
            {
                "model.layers.0.self_attn.q_proj.qweight": torch.zeros(
                    64, 8, dtype=torch.int32
                ),
                "model.layers.0.self_attn.q_proj.scales": torch.zeros(
                    64, 8, dtype=torch.float16
                ),
                "model.layers.0.mlp.gate.weight": torch.zeros(
                    32, 32, dtype=torch.bfloat16
                ),
                "lm_head.weight": torch.zeros(32, 32, dtype=torch.bfloat16),
            },
            os.path.join(root, "model.safetensors"),
        )
        with open(
            os.path.join(root, "config.json"), "w", encoding="utf-8", newline="\n"
        ) as handle:
            json.dump(
                {
                    "quantization_config": {
                        "quant_method": "awq",
                        "w_bit": 4,
                        "q_group_size": 128,
                        "zero_point": True,
                    }
                },
                handle,
            )
        report = inspect(root)
        assert report["detected_scheme"] == "int4_g128_asym", report
        assert report["detected_block"] == 128
        assert report["quant_algorithm"] == "awq"
        attn = report["coverage"]["attention"]
        assert attn["weights"] == 1 and attn["quantized"] == 1, attn
        router = report["coverage"]["router"]
        assert router["weights"] == 1 and router["quantized"] == 0, router
        rendered = render_inspection(report)
        assert "int4_g128_asym" in rendered
        assert "quant algorithm: awq" in rendered
        assert report["quantized_names"] == [
            "model.layers.0.self_attn.q_proj.weight"
        ]
        disk = inspect_for_disk(report)
        assert "quantized_names" not in disk
        assert disk["quantized_names_sha256"] == names_sha256(
            report["quantized_names"]
        )
        assert report["weights_bytes"] == weights_bytes(root)
        assert report["weights_bytes"] > 0
        assert "weights on disk" in rendered
        identity = report["weights_sha256"]
        assert identity == weights_identity(root), "identity is not stable"
        assert disk["weights_sha256"] == identity, "identity must survive to disk"
        # A pack that differs only in one tensor's dtype is a different pack.
        with tempfile.TemporaryDirectory() as other:
            save_file(
                {
                    "model.layers.0.self_attn.q_proj.qweight": torch.zeros(
                        64, 8, dtype=torch.int32
                    ),
                    "model.layers.0.self_attn.q_proj.scales": torch.zeros(
                        64, 8, dtype=torch.float32
                    ),
                    "model.layers.0.mlp.gate.weight": torch.zeros(
                        32, 32, dtype=torch.bfloat16
                    ),
                    "lm_head.weight": torch.zeros(32, 32, dtype=torch.bfloat16),
                },
                os.path.join(other, "model.safetensors"),
            )
            assert weights_identity(other) != identity, "dtype drift went unseen"
        with tempfile.TemporaryDirectory() as other:
            tensors = {
                name: tensor.clone()
                for name, tensor in load_file(
                    os.path.join(root, "model.safetensors")
                ).items()
            }
            first = next(iter(tensors.values()))
            first.view(-1)[0] = 1
            save_file(tensors, os.path.join(other, "model.safetensors"))
            assert weights_identity(other) != identity, "content drift went unseen"
    print("  inspect reads packed AWQ scales and coverage denominators")

    # Gemma 4's moe_intermediate_size is 704, which no 128-wide grid divides.
    # fp8_block therefore has no rung on it, while every narrower grid does.
    with tempfile.TemporaryDirectory() as root:
        save_file(
            {
                "model.layers.0.experts.gate_up_proj": torch.zeros(
                    4, 1408, 2816, dtype=torch.bfloat16
                ),
                "model.layers.0.experts.down_proj": torch.zeros(
                    4, 2816, 704, dtype=torch.bfloat16
                ),
            },
            os.path.join(root, "model.safetensors"),
        )
        blocked = unexpressible_reason(root, ("experts",), "fp8_block")
        assert blocked and "704" in blocked and "128" in blocked, blocked
        for fine in ("nvfp4", "mxfp8", "int4_g64_sym", "fp8_per_channel"):
            assert unexpressible_reason(root, ("experts",), fine) is None, fine
    print("  a ladder rung the reference's shapes cannot carry is refused early")
    print("  weights_identity binds a checkpoint to its tensors and shards")

    from safetensors import safe_open

    with tempfile.TemporaryDirectory() as root:
        ref = os.path.join(root, "ref")
        out = os.path.join(root, "out")
        os.makedirs(ref)
        kept = "model.layers.0.mlp.experts.0.gate_proj.weight"
        rounded = "model.layers.0.mlp.experts.1.gate_proj.weight"
        original_kept = torch.linspace(
            -1, 1, 32 * 32, dtype=torch.bfloat16
        ).reshape(32, 32)
        original_rounded = torch.linspace(
            -2, 2, 32 * 32, dtype=torch.bfloat16
        ).reshape(32, 32)
        save_file(
            {
                kept: original_kept.clone(),
                rounded: original_rounded.clone(),
                "lm_head.weight": torch.zeros(32, 32, dtype=torch.bfloat16),
            },
            os.path.join(ref, "model.safetensors"),
        )
        with open(
            os.path.join(ref, "config.json"), "w", encoding="utf-8", newline="\n"
        ) as handle:
            json.dump({"hidden_size": 32}, handle)
        manifest = convert(
            ref, out, ("experts",), "fp8_per_tensor", 128, "cpu",
            only={rounded},
        )
        assert manifest["match_mode"] == "per_tensor"
        assert manifest["tensors_rounded"] == 1
        with safe_open(
            os.path.join(out, "model.safetensors"), framework="pt"
        ) as handle:
            assert torch.equal(handle.get_tensor(kept), original_kept)
            assert not torch.equal(handle.get_tensor(rounded), original_rounded)
        try:
            convert(
                ref,
                os.path.join(root, "drift"),
                ("experts",),
                "fp8_per_tensor",
                128,
                "cpu",
                only={"model.layers.0.mlp.experts.99.gate_proj.weight"},
            )
        except SystemExit as exc:
            assert "no counterpart" in str(exc)
        else:
            raise AssertionError("name drift must be refused")
        full = {"quantized_names": [kept, rounded]}
        assert match_digest(ref, full, ("experts",)) is None
        part = {"quantized_names": [rounded]}
        digest = match_digest(ref, part, ("experts",))
        assert digest is not None and len(digest) == MATCH_DIGEST_LEN
        assert digest == match_digest(ref, part, ("experts",))

        # A reference that fuses experts names two stacks where a pack names
        # six per-expert weights. Without resolution every pack looks like
        # total name drift, which is how four int4 cells came to refuse.
        fused = os.path.join(root, "fused")
        os.makedirs(fused)
        stem = "model.layers.0.mlp.experts"
        save_file(
            {
                f"{stem}.gate_up_proj": torch.randn(2, 4, 8, dtype=torch.bfloat16),
                f"{stem}.down_proj": torch.randn(2, 8, 4, dtype=torch.bfloat16),
            },
            os.path.join(fused, "model.safetensors"),
        )
        with open(
            os.path.join(fused, "config.json"), "w", encoding="utf-8", newline="\n"
        ) as handle:
            json.dump({"hidden_size": 8}, handle)
        per_expert = {
            f"{stem}.{index}.{proj}.weight"
            for index in range(2)
            for proj in ("gate_proj", "up_proj", "down_proj")
        }
        assert resolve_matched(fused, ("experts",), per_expert) == {
            f"{stem}.gate_up_proj",
            f"{stem}.down_proj",
        }
        assert match_digest(fused, {"quantized_names": sorted(per_expert)},
                            ("experts",)) is None
        # The same weights under a wrapper the reference lacks must land on the
        # same stacks, and therefore on the same cell.
        wrapped_experts = {
            name.replace(".mlp.experts.", ".moe.experts.") for name in per_expert
        }
        assert resolve_matched(fused, ("experts",), wrapped_experts) == {
            f"{stem}.gate_up_proj",
            f"{stem}.down_proj",
        }
        assert match_digest(
            fused, {"quantized_names": sorted(wrapped_experts)}, ("experts",)
        ) is None
        try:
            resolve_matched(
                fused, ("experts",), per_expert - {f"{stem}.1.up_proj.weight"}
            )
        except SystemExit as exc:
            assert "fused stack" in str(exc)
        else:
            raise AssertionError("a partly covered fused stack must be refused")
        down_only = {f"{stem}.{index}.down_proj.weight" for index in range(2)}
        fused_manifest = convert(
            fused, os.path.join(root, "fused-out"), ("experts",),
            "fp8_per_tensor", 128, "cpu", only=down_only,
        )
        assert fused_manifest["tensors_rounded"] == 1, (
            "rounding both experts' down_proj must round the one stack that "
            "holds them, and leave gate_up_proj alone"
        )
        match_dir = os.path.join(root, "match")
        os.makedirs(match_dir)
        save_file(
            {
                rounded.replace(".weight", ".qweight"): torch.zeros(
                    32, 8, dtype=torch.int32
                ),
                rounded.replace(".weight", ".scales"): torch.zeros(
                    32, 1, dtype=torch.float16
                ),
            },
            os.path.join(match_dir, "model.safetensors"),
        )
        with open(
            os.path.join(match_dir, "config.json"),
            "w",
            encoding="utf-8",
            newline="\n",
        ) as handle:
            json.dump(
                {
                    "quantization_config": {
                        "quant_method": "awq",
                        "w_bit": 4,
                        "q_group_size": 128,
                        "zero_point": True,
                    }
                },
                handle,
            )
        assert check_names(ref, match_dir) == 0
    print("  per-tensor convert leaves excluded tensors bit-identical")
    print("selftest passed")
    return 0


def check_names(reference: str, match: str) -> int:
    """Refuse (exit 1) if MATCH names weights the reference cannot resolve."""
    inspection = inspect(match)
    quantized = inspection.get("quantized_names") or []
    try:
        resolved = resolve_matched(reference, COMPONENTS, quantized)
    except SystemExit as exc:
        print(f"{os.path.basename(match)}: {exc}")
        return 1
    total = len(_component_names(reference, COMPONENTS))
    print(
        f"{os.path.basename(match)}: {len(quantized)} quantized names resolve "
        f"to {len(resolved)} of {total} reference tensors in "
        f"{os.path.basename(reference)}"
    )
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
        type=scheme_arg,
        help="rounding scheme: "
        + ", ".join(KNOWN_SCHEMES)
        + ", or int4_g<G>_(sym|asym); omit and pass --match to copy it "
        "from a quantized checkpoint",
    )
    parser.add_argument(
        "--match",
        default=None,
        help="a real quantized checkpoint whose scheme, block size, and "
        "component coverage this variant should imitate",
    )
    parser.add_argument(
        "--per-tensor",
        action="store_true",
        help="round only the tensors --match actually quantized, not every "
        "weight whose name matches a selected component",
    )
    parser.add_argument(
        "--expect-match-digest",
        default=None,
        help="refuse if the per-tensor match digest of --match against "
        "--model is not this value",
    )
    parser.add_argument(
        "--inspect",
        default=None,
        help="report what a quantized checkpoint quantized, and exit",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="write the inspect report as JSON (with --inspect)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="block or group edge; defaults to 128 for fp8_block, 16 for "
        "nvfp4, and the group size in an int4_g<G> scheme name",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="where to round; defaults to cuda when available",
    )
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument(
        "--check-names",
        nargs=2,
        metavar=("REFERENCE", "MATCH"),
        help="exit non-zero if MATCH's quantized names are absent from "
        "REFERENCE",
    )
    args = parser.parse_args()

    if args.selftest:
        return selftest()
    if args.check_names:
        return check_names(args.check_names[0], args.check_names[1])
    if args.inspect:
        report = inspect(args.inspect)
        print(render_inspection(report))
        if args.json_out:
            os.makedirs(
                os.path.dirname(os.path.abspath(args.json_out)) or ".",
                exist_ok=True,
            )
            with open(args.json_out, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(inspect_for_disk(report), handle, indent=2)
                handle.write("\n")
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
        inherited_scheme = scheme is None
        if inherited_scheme:
            scheme = matched["detected_scheme"]
            if scheme is None:
                parser.error(
                    f"{args.match} carries no weight scales, so there is no "
                    f"scheme to match; pass --scheme explicitly"
                )
            print(f"matched scheme {scheme} from {os.path.basename(args.match)}")
        # A block edge belongs to the scheme it was measured on. Inheriting an
        # int4 group size into an explicitly named fp8_block produces a
        # different format under the same label, and a ladder whose rung width
        # depends on the candidate beside it compares nothing.
        if block_size is None and inherited_scheme:
            block_size = matched.get("detected_block")
    if scheme is None:
        parser.error("pass --scheme, or --match a quantized checkpoint")
    if block_size is None:
        block_size = scheme_block_size(scheme, None) or 128

    device = args.device
    if device is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.per_tensor and not args.match:
        parser.error("--per-tensor requires --match")
    if args.expect_match_digest and not args.match:
        parser.error("--expect-match-digest requires --match")

    only: set[str] | None = None
    if args.per_tensor:
        assert matched is not None
        only = {
            name
            for name in matched.get("quantized_names") or []
            if classify(name, selected) is not None
        }
        if not only:
            raise SystemExit(
                f"{os.path.basename(args.match)} quantizes none of "
                f"{', '.join(selected)}"
            )
        print(
            f"  matching {len(only)} tensors from "
            f"{os.path.basename(args.match)}"
        )
    elif matched:
        for component in selected:
            counts = matched["coverage"].get(component) or {}
            if counts.get("weights") and not counts.get("quantized"):
                print(
                    f"WARNING  {os.path.basename(args.match)} does not quantize "
                    f"{component}, so this variant carries error the deployed "
                    f"checkpoint does not have"
                )

    if args.expect_match_digest:
        assert matched is not None
        computed = match_digest(args.model, matched, selected)
        if computed != args.expect_match_digest:
            raise SystemExit(
                f"match digest {computed} != expected "
                f"{args.expect_match_digest}"
            )

    print(
        f"QDQ {scheme} on {', '.join(selected)} "
        f"({os.path.basename(args.model)} -> {os.path.basename(args.out)}) "
        f"on {device}"
    )
    manifest = convert(
        args.model, args.out, selected, scheme, block_size, device, only=only
    )
    if matched:
        manifest["matched"] = {
            "model": matched["model"],
            "detected_scheme": matched["detected_scheme"],
            "detected_block": matched["detected_block"],
            "quant_algorithm": matched.get("quant_algorithm"),
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
