#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Locate the module that first diverges between two identical GLM-5.3 runs.

Established by scripts/glm53_determinism_probe.py on 4xB200, TP=4, BF16, eager:

* A 129-token prompt is bit-identical across runs in every configuration
  tested (baseline, VLLM_BATCH_INVARIANT, no FlashInfer autotune,
  torch.use_deterministic_algorithms, chunked and single-chunk prefill).
* A 130-token prompt differs in exactly one row - row 128 - by ~0.2 logits
  with zero argmax flips. Longer prompts amplify from there: 63 rows differ
  at length 192, 383 at 512, 1919 at 2048, reaching ~8 logits and ~40 flips.
* The divergence is per-invocation, not per-process: runs 2v3 and 2v4 differ
  as much as 1v2, so it is not launch-time kernel selection.
* No NaNs appear under torch.utils.deterministic.fill_uninitialized_memory,
  so it is not a read of uninitialized memory.

130 tokens with a single divergent row is therefore the cheapest exact
reproducer available. This script runs it twice, captures every decoder
layer's output via forward hooks inside the TP workers, and reports the first
layer whose output differs. It then re-hooks the submodules of that layer to
name the specific operation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import regex as re

# The hook installer crosses two serialization boundaries to reach the workers.
# Client -> EngineCore goes through MsgpackEncoder, which only cloudpickles
# functions when this is set, and the decoding side checks it too, so it must
# be set before vLLM spawns anything. Nothing untrusted crosses this boundary:
# the payload is a function from this file.
#
# EngineCore -> workers goes through shm_broadcast, which uses plain pickle and
# only cloudpickles the *method*. That rules out LLM.apply_model(), which passes
# the function in args; these helpers are sent as the method instead.
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from glm53_determinism_probe import build_llm, build_tokens
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        f"glm53_determinism_probe.py must sit next to this script: {exc}"
    ) from exc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    p.add_argument("--out-dir", default="determinism_probe")
    p.add_argument(
        "--length",
        type=int,
        default=130,
        help="Prompt length. 130 is the minimal reproducer; 129 is "
        "the negative control and must show no divergence.",
    )
    p.add_argument("--tp", type=int, default=4)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    p.add_argument("--moe-backend", default="triton")
    p.add_argument(
        "--repeats",
        type=int,
        default=2,
        help="Runs to compare. 2 is enough; 3 cross-checks.",
    )
    p.add_argument(
        "--control",
        action="store_true",
        help="Also run length 129 first as a negative control.",
    )
    p.add_argument(
        "--fingerprint-mqa",
        action="store_true",
        help="Hash every tensor argument the sparse MLA kernel "
        "receives and report which ones differ between runs. "
        "The kernel is deterministic standalone, so this "
        "distinguishes a nondeterministic kernel from an "
        "engine that feeds it different arguments.",
    )
    p.add_argument(
        "--pattern",
        default=None,
        help="Hook every module matching this regex and run a "
        "single round, instead of the two-phase layer-then-"
        "submodule search. Use this when the divergence is "
        "stochastic: the two-phase search compares a fresh "
        "pair of runs per phase, so a fault that strikes a "
        "different layer each time makes the phases disagree. "
        "One pattern covering every layer plus the operations "
        "of interest measures them all in the same pair.",
    )
    return p.parse_args()


# --------------------------------------------------------------------------
# Worker-side functions. These are cloudpickled and executed inside each TP
# worker, so they must be self-contained and keep state on the model object.
# --------------------------------------------------------------------------
def _install_hooks(model, pattern: str) -> list[str]:
    import regex as _re
    import torch as _torch

    state = getattr(model, "_bisect", None)
    if state is not None:
        for handle in state["handles"]:
            handle.remove()
    # "order" records the sequence in which hooks first fire, which is
    # execution order. Sorting by name instead would report layer 3's mlp
    # ahead of its self_attn purely because "m" < "s".
    state = {
        "slot": 0,
        "runs": {},
        "handles": [],
        "seen": {},
        "order": {},
        "counter": 0,
    }
    model._bisect = state

    regex = _re.compile(pattern)
    names: list[str] = []

    def make_hook(name: str):
        def hook(_module, _inputs, output):
            tensor = output
            while isinstance(tensor, (tuple, list)) and tensor:
                tensor = tensor[0]
            if not isinstance(tensor, _torch.Tensor):
                return
            slot = state["slot"]
            key = (slot, name)
            # Only the first call per slot: that is the prefill pass.
            if key in state["seen"]:
                return
            state["seen"][key] = True
            if name not in state["order"]:
                state["order"][name] = state["counter"]
                state["counter"] += 1
            state["runs"].setdefault(slot, {})[name] = (
                tensor.detach().float().cpu().clone()
            )

        return hook

    for name, module in model.named_modules():
        if regex.search(name):
            state["handles"].append(module.register_forward_hook(make_hook(name)))
            names.append(name)
    return names


def _set_slot(model, slot: int) -> int:
    state = model._bisect
    state["slot"] = slot
    return slot


def _compare(model, slot_a: int, slot_b: int) -> list[dict[str, Any]]:
    import torch as _torch

    state = model._bisect
    run_a = state["runs"].get(slot_a, {})
    run_b = state["runs"].get(slot_b, {})
    results = []
    for name in run_a:
        if name not in run_b:
            continue
        order = state["order"].get(name, -1)
        a, b = run_a[name], run_b[name]
        if a.shape != b.shape:
            results.append({"module": name, "order": order, "shape_mismatch": True})
            continue
        diff = (a - b).abs()
        flat = diff.reshape(diff.shape[0], -1) if diff.ndim > 1 else diff[:, None]
        per_row = flat.amax(dim=-1)
        nz = (per_row > 0).nonzero().flatten()
        results.append(
            {
                "module": name,
                "order": order,
                "rows": int(a.shape[0]),
                "identical": bool(_torch.equal(a, b)),
                "max_abs": float(diff.max()),
                "num_differing_rows": int(nz.numel()),
                "first_differing_row": int(nz[0]) if nz.numel() else None,
            }
        )
    return results


def _clear_hooks(model) -> int:
    state = getattr(model, "_bisect", None)
    if state is None:
        return 0
    for handle in state["handles"]:
        handle.remove()
    n = len(state["handles"])
    model._bisect = None
    return n


# Sent as collective_rpc's `method`, so they receive the worker and must take
# only plainly picklable arguments.
def _wrap_sparse_mla_kernel(model) -> bool:
    """Fingerprint every argument the sparse MLA kernel actually receives.

    The kernel reproduces itself bit-for-bit when called standalone with this
    model's shapes, at every row count and regardless of where the KV lives, so
    if the engine's result varies then its arguments must vary. Wrapping
    FlashInfer's entry point rather than vLLM's call site records whatever is
    passed without depending on which overlay revision is installed;
    ``forward_mqa`` imports the symbol per call, so the patched attribute wins.
    """
    import hashlib

    import flashinfer.decode as fid
    import torch as _torch

    state = model._bisect
    if getattr(fid, "_glm53_wrapped", False):
        fid._glm53_state = state
        return True

    original = fid.trtllm_batch_decode_with_kv_cache_mla

    def digest(tensor) -> str | None:
        if not isinstance(tensor, _torch.Tensor):
            return None
        # Reinterpret as bytes before leaving torch: numpy has no bfloat16.
        raw = tensor.detach().contiguous().flatten().view(_torch.uint8)
        return hashlib.blake2b(raw.cpu().numpy().tobytes(), digest_size=8).hexdigest()

    def wrapper(*args, **kwargs):
        active = getattr(fid, "_glm53_state", None)
        if active is None:
            return original(*args, **kwargs)
        record: dict[str, Any] = {}
        for name, value in kwargs.items():
            if not isinstance(value, _torch.Tensor):
                continue
            # The paged KV cache is far too large to hash whole; fingerprint
            # only the rows this call can actually read.
            if name == "kv_cache":
                table = kwargs.get("block_tables")
                if isinstance(table, _torch.Tensor):
                    idx = table.detach().reshape(-1)
                    idx = idx[idx >= 0].unique()
                    rows = value.detach().reshape(-1, value.shape[-1])
                    record["kv_referenced"] = digest(rows[idx.long()])
                continue
            record[name] = digest(value)
            # A different order of the same indices changes accumulation order;
            # different indices mean different tokens or different physical
            # slots. Sorting separates those two causes.
            if name == "block_tables":
                record["block_tables_sorted"] = digest(
                    value.detach().sort(dim=-1).values
                )
                record["block_tables_row0"] = value.detach().reshape(-1)[:24].tolist()
        out = original(*args, **kwargs)
        tensor = out[0] if isinstance(out, tuple) else out
        record["OUTPUT"] = digest(tensor)
        active.setdefault("calls", {}).setdefault(active["slot"], []).append(record)
        return out

    fid.trtllm_batch_decode_with_kv_cache_mla = wrapper
    fid._glm53_wrapped = True
    fid._glm53_state = state
    return True


def _rpc_install(worker, pattern: str, fingerprint_mqa: bool = False) -> list[str]:
    model = worker.get_model()
    names = _install_hooks(model, pattern)
    if fingerprint_mqa:
        _wrap_sparse_mla_kernel(model)
    return names


def _rpc_set_slot(worker, slot: int) -> int:
    return _set_slot(worker.get_model(), slot)


def _rpc_compare(worker, slot_a: int, slot_b: int) -> list[dict[str, Any]]:
    return _compare(worker.get_model(), slot_a, slot_b)


def _rpc_compare_calls(worker, slot_a: int, slot_b: int) -> list[dict[str, Any]]:
    """Per kernel invocation, which arguments differ between the two runs."""
    state = worker.get_model()._bisect
    calls = state.get("calls", {})
    run_a, run_b = calls.get(slot_a, []), calls.get(slot_b, [])
    results = []
    for index, (a, b) in enumerate(zip(run_a, run_b)):
        differing = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
        if differing:
            entry = {"call": index, "differing": differing}
            if index == 0:
                entry["row0_a"] = a.get("block_tables_row0")
                entry["row0_b"] = b.get("block_tables_row0")
            results.append(entry)
    return [
        {
            "num_calls_a": len(run_a),
            "num_calls_b": len(run_b),
            "divergent_calls": results,
        }
    ]


def _rpc_dump_calls(worker, slot: int) -> list[dict[str, Any]]:
    """Raw kernel-argument fingerprints, for comparing across processes."""
    return worker.get_model()._bisect.get("calls", {}).get(slot, [])


def _rpc_clear(worker) -> int:
    import flashinfer.decode as fid

    fid._glm53_state = None
    return _clear_hooks(worker.get_model())


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
LAYER_PATTERN = r"\.layers\.\d+$"


def bisect_round(
    llm, tokens: list[int], pattern: str, repeats: int, fingerprint_mqa: bool = False
) -> dict[str, Any]:
    """Hook every module matching pattern, run repeats times, compare runs."""
    from glm53_determinism_probe import capture

    names = llm.collective_rpc(_rpc_install, args=(pattern, fingerprint_mqa))
    hooked = len(names[0]) if names else 0
    print(f"[hook] pattern {pattern!r} matched {hooked} modules per rank", flush=True)
    if not hooked:
        return {"pattern": pattern, "hooked": 0, "modules": []}

    logits = []
    for slot in range(repeats):
        llm.collective_rpc(_rpc_set_slot, args=(slot,))
        logits.append(capture(llm, tokens))

    per_rank = llm.collective_rpc(_rpc_compare, args=(0, 1))
    call_diffs = (
        llm.collective_rpc(_rpc_compare_calls, args=(0, 1)) if fingerprint_mqa else None
    )
    llm.collective_rpc(_rpc_clear)

    # Aggregate across ranks: a module is divergent if any rank saw a diff.
    # Keep the worst magnitude and the *earliest* row separately. Taking both
    # from whichever rank had the larger magnitude hides an earlier divergence
    # on another rank, which misreports where the chain starts.
    merged: dict[str, dict[str, Any]] = {}
    for rank, entries in enumerate(per_rank):
        for entry in entries:
            name = entry["module"]
            prev = merged.get(name)
            if prev is None:
                merged[name] = {**entry, "worst_rank": rank}
                continue
            if entry.get("max_abs", 0.0) > prev.get("max_abs", 0.0):
                prev.update(
                    {
                        k: v
                        for k, v in entry.items()
                        if k not in ("first_differing_row",)
                    }
                )
                prev["worst_rank"] = rank
            prev["identical"] = prev.get("identical", True) and entry.get(
                "identical", True
            )
            rows = [
                r
                for r in (
                    prev.get("first_differing_row"),
                    entry.get("first_differing_row"),
                )
                if r is not None
            ]
            prev["first_differing_row"] = min(rows) if rows else None
            prev["num_differing_rows"] = max(
                prev.get("num_differing_rows", 0), entry.get("num_differing_rows", 0)
            )

    modules = sorted(merged.values(), key=lambda e: e.get("order", -1))
    divergent = [m for m in modules if not m.get("identical", True)]
    # Hooks only record tensor outputs, so a module returning None, a dataclass
    # or an in-place result is silently absent from the comparison. Name those
    # explicitly: an unchecked module is a gap in the causal chain, not a pass.
    unchecked = sorted(
        set(names[0] if names else []) - set(m["module"] for m in modules)
    )
    logit_diff = (logits[0] - logits[1]).abs()
    return {
        "pattern": pattern,
        "hooked": hooked,
        "modules": modules,
        "unchecked": unchecked,
        "first_divergent": divergent[0]["module"] if divergent else None,
        "num_divergent": len(divergent),
        "logits_identical": bool(logit_diff.max().item() == 0.0),
        "logits_max_abs": float(logit_diff.max()),
        "mqa_calls": call_diffs,
    }


def report(round_result: dict[str, Any]) -> None:
    for entry in round_result["modules"]:
        if entry.get("shape_mismatch"):
            print(f"  {entry['module']:<60} SHAPE MISMATCH", flush=True)
            continue
        if entry["identical"]:
            continue
        print(
            f"  #{entry.get('order', -1):<4} {entry['module']:<58} "
            f"max|d|={entry['max_abs']:.3e} "
            f"rows={entry['num_differing_rows']}/{entry['rows']} "
            f"first_row={entry['first_differing_row']} "
            f"rank={entry['worst_rank']}",
            flush=True,
        )
    unchecked = round_result.get("unchecked") or []
    if unchecked:
        print(
            f"[unchecked] {len(unchecked)} hooked modules produced no tensor "
            f"output and were not compared:",
            flush=True,
        )
        for name in unchecked:
            print(f"    {name}", flush=True)
    for rank, payload in enumerate(round_result.get("mqa_calls") or []):
        summary = payload[0] if isinstance(payload, list) else payload
        divergent_calls = summary["divergent_calls"]
        print(
            f"[mqa rank{rank}] {summary['num_calls_a']} vs "
            f"{summary['num_calls_b']} kernel calls, "
            f"{len(divergent_calls)} with differing arguments",
            flush=True,
        )
        for call in divergent_calls[:8]:
            print(
                f"    call #{call['call']:<4} differs in: "
                f"{', '.join(call['differing'])}",
                flush=True,
            )
            if call.get("row0_a") is not None:
                print(f"      run A block_tables[:24]={call['row0_a']}", flush=True)
                print(f"      run B block_tables[:24]={call['row0_b']}", flush=True)
    first = round_result["first_divergent"]
    print(
        f"[result] first divergent module (execution order): "
        f"{first or 'NONE (all identical)'}",
        flush=True,
    )
    print(
        f"[result] final logits identical: "
        f"{round_result['logits_identical']} "
        f"max|d|={round_result['logits_max_abs']:.3e}",
        flush=True,
    )


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    token_args = SimpleNamespace(
        model=args.model,
        dataset_dir=args.dataset_dir,
        dataset_config=args.dataset_config,
        ctx=2048,
        stride=0,
    )
    tokens = build_tokens(token_args)

    llm_args = SimpleNamespace(
        model=args.model,
        tp=args.tp,
        ctx=2048,
        gpu_memory_utilization=args.gpu_memory_utilization,
        moe_backend=args.moe_backend,
        max_num_batched_tokens=None,
        disable_flashinfer_autotune=False,
        enforce_eager=True,
    )
    llm = build_llm(llm_args)

    results: dict[str, Any] = {"args": vars(args), "rounds": []}

    if args.control:
        print(
            "\n=== negative control: length 129 (expect no divergence) ===", flush=True
        )
        control = bisect_round(llm, tokens[:129], LAYER_PATTERN, args.repeats)
        report(control)
        results["control_129"] = control
        # Hidden row 128 exists at 129 tokens, but prompt logits only cover
        # rows 0..127, so a divergence confined to row 128 is expected here and
        # is exactly why the logits stay identical. Only an earlier row, or
        # divergent logits, contradicts the position-128 story.
        early = [
            m
            for m in control["modules"]
            if not m.get("identical", True)
            and (m.get("first_differing_row") or 128) < 128
        ]
        if early or not control["logits_identical"]:
            print(
                "[warn] the 129-token control diverged before row 128; the "
                "position-128 boundary is not the whole story",
                flush=True,
            )
        else:
            print(
                "[control] divergence confined to hidden row 128 and absent "
                "from the logits, as expected at this length",
                flush=True,
            )

    window = tokens[: args.length]

    if args.pattern:
        print(
            f"\n=== single round at length {args.length}: {args.pattern} ===",
            flush=True,
        )
        single = bisect_round(
            llm,
            window,
            args.pattern,
            args.repeats,
            fingerprint_mqa=args.fingerprint_mqa,
        )
        report(single)
        results["rounds"].append(single)
        print(
            f"\n[done] first divergent module in one pair of runs: "
            f"{single['first_divergent'] or 'NONE'}",
            flush=True,
        )
        path = out_dir / "layer_bisect.json"
        path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"wrote {path}", flush=True)
        return 0

    print(f"\n=== phase 1: decoder layers at length {args.length} ===", flush=True)
    phase1 = bisect_round(
        llm, window, LAYER_PATTERN, args.repeats, fingerprint_mqa=args.fingerprint_mqa
    )
    report(phase1)
    results["rounds"].append(phase1)

    culprit = phase1["first_divergent"]
    if culprit is None:
        print(
            "\n[done] no decoder layer diverged. Either this length is "
            "below the boundary or the divergence lives after the last "
            "layer (norm / lm_head / sampler).",
            flush=True,
        )
    else:
        pattern = rf"^{re.escape(culprit)}\..+"
        print(f"\n=== phase 2: submodules of {culprit} ===", flush=True)
        phase2 = bisect_round(llm, window, pattern, args.repeats)
        report(phase2)
        results["rounds"].append(phase2)
        results["culprit_layer"] = culprit
        results["culprit_submodule"] = phase2["first_divergent"]
        print(
            f"\n[done] narrowed to {phase2['first_divergent'] or culprit}", flush=True
        )

    path = out_dir / "layer_bisect.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
