#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Isolate GLM-5.3's sparse MLA attention kernel and test it for determinism.

scripts/glm53_layer_bisect.py localized the engine's nondeterminism to
``layers.3.self_attn.mla_attn.mla_attn`` with bit-identical inputs and an
identical top-k index set, differing by exactly one BF16 ULP. For this model no
MLA prefill backend is available, so every token including prefill goes through
``trtllm_batch_decode_with_kv_cache_mla`` one query row at a time.

This script calls that kernel directly with GLM-5.3's shapes. No model weights,
no engine, no TP: it loads in seconds, so the knobs vLLM currently leaves at
default (``backend``, ``cute_dsl_impl``, ``enable_pdl``) can be swept cheaply to
find a configuration that reproduces itself bit-for-bit.

The valid-count sweep is the key measurement. In the engine, divergence appears
at query row 128 and never before, so a kernel whose output is stable at a valid
KV count of 128 and unstable at 129 reproduces the engine fault in isolation and
confirms the kernel as the sole source.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--num-heads",
        type=int,
        default=32,
        help="Query heads this rank owns (total heads / TP).",
    )
    p.add_argument("--kv-lora-rank", type=int, default=512)
    p.add_argument("--qk-nope-head-dim", type=int, default=256)
    p.add_argument("--qk-rope-head-dim", type=int, default=0)
    p.add_argument(
        "--topk",
        type=int,
        default=2048,
        help="sparse_mla_top_k, i.e. the model's index_topk.",
    )
    p.add_argument(
        "--page-size",
        type=int,
        default=64,
        help="KV cache page size. trtllm-gen requires 32 or 64.",
    )
    p.add_argument("--num-pages", type=int, default=512)
    p.add_argument(
        "--rows-sweep",
        default="1,64,129,130,192,512",
        help="Prompt lengths to sweep. Each length submits that "
        "many query rows in one call, row i attending to i+1 "
        "keys, which is how a prefill reaches this kernel. "
        "Row count matters: the cubin is a MultiCtasKv "
        "variant, so a single row gives the scheduler nothing "
        "to split and cannot expose a cross-CTA combine.",
    )
    p.add_argument("--repeats", type=int, default=4)
    p.add_argument(
        "--backends",
        default="auto",
        help="Comma-separated backend values to try, e.g. auto,trtllm-gen,cute-dsl.",
    )
    p.add_argument(
        "--cute-dsl-impls",
        default="auto",
        help="Comma-separated cute_dsl_impl values to try.",
    )
    p.add_argument(
        "--enable-pdl", default="none", help="Comma-separated: none,true,false."
    )
    p.add_argument(
        "--no-top-k-lens",
        dest="top_k_lens",
        action="store_false",
        help="Omit sparse_mla_top_k_lens. The native "
        "qk_rope_head_dim=0 TRTLLM-GEN path rejects the call "
        "without it, so it is passed by default.",
    )
    p.add_argument(
        "--seq-lens-mode",
        choices=("valid", "topk", "none"),
        default="valid",
        help="What to put in seq_lens: the per-row valid KV count, "
        "the full top-k width, or nothing.",
    )
    p.add_argument(
        "--placement-test",
        action="store_true",
        help="Also run each length with the same logical KV moved "
        "to different physical slots. The math is unchanged, "
        "so any difference means results depend on where the "
        "KV cache manager happened to place the blocks.",
    )
    p.add_argument("--workspace-mb", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="sparse_mla_kernel_probe.json")
    return p.parse_args()


def make_placement(
    args: argparse.Namespace, num_logical: int, gen: torch.Generator, permute: bool
) -> torch.Tensor:
    """Map logical token t to the physical KV slot holding it.

    The engine's top-k rows are logical positions translated through the block
    table, so the same logical content can sit at different physical slots in
    two runs if the KV cache manager hands out different blocks. Only the
    placement changes; the attention math is identical either way.
    """
    num_slots = args.num_pages * args.page_size
    if not permute:
        return torch.arange(num_logical, dtype=torch.long)
    return torch.randperm(num_slots, generator=gen)[:num_logical]


def build_inputs(
    args: argparse.Namespace,
    rows: int,
    device: torch.device,
    placement: torch.Tensor | None = None,
) -> dict[str, Any]:
    """A whole prefill in one call: `rows` query rows, row i seeing i+1 keys.

    ``block_tables`` holds global token indices with -1 padding beyond each
    row's valid count, and both ``seq_lens`` and ``sparse_mla_top_k_lens``
    carry that count, matching what
    ``triton_convert_req_index_to_global_index`` hands the kernel.
    """
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    head_dim = args.kv_lora_rank + args.qk_rope_head_dim

    num_slots = args.num_pages * args.page_size
    num_logical = min(rows, args.topk, num_slots)
    if placement is None:
        placement = make_placement(args, num_logical, gen, permute=False)

    query = torch.randn(
        (rows, 1, args.num_heads, head_dim), generator=gen, dtype=torch.float32
    ).to(device=device, dtype=torch.bfloat16)

    # Same logical KV content, written to whatever slots placement dictates.
    kv_logical = torch.randn(
        (num_logical, head_dim), generator=gen, dtype=torch.float32
    ).to(torch.bfloat16)
    flat = torch.zeros((num_slots, head_dim), dtype=torch.bfloat16)
    flat[placement] = kv_logical
    kv_cache = flat.view(args.num_pages, args.page_size, head_dim).to(device)

    counts = [min(i + 1, num_logical) for i in range(rows)]
    indices = torch.full((rows, 1, args.topk), -1, dtype=torch.int32)
    for i, count in enumerate(counts):
        indices[i, 0, :count] = placement[:count].to(torch.int32)
    block_tables = indices.to(device)

    counts_t = torch.tensor(counts, dtype=torch.int32)
    if args.seq_lens_mode == "none":
        seq_lens = None
    elif args.seq_lens_mode == "valid":
        seq_lens = counts_t.to(device)
    else:
        seq_lens = torch.full((rows,), args.topk, dtype=torch.int32, device=device)
    top_k_lens = counts_t.to(device) if args.top_k_lens else None
    return {
        "query": query,
        "kv_cache": kv_cache,
        "block_tables": block_tables,
        "seq_lens": seq_lens,
        "sparse_mla_top_k_lens": top_k_lens,
    }


def run_once(
    args: argparse.Namespace,
    inputs: dict[str, Any],
    workspace: torch.Tensor,
    backend: str,
    cute_dsl_impl: str,
    enable_pdl: bool | None,
) -> torch.Tensor:
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

    out = trtllm_batch_decode_with_kv_cache_mla(
        query=inputs["query"],
        kv_cache=inputs["kv_cache"].unsqueeze(1),
        workspace_buffer=workspace,
        qk_nope_head_dim=args.qk_nope_head_dim,
        kv_lora_rank=args.kv_lora_rank,
        qk_rope_head_dim=args.qk_rope_head_dim,
        block_tables=inputs["block_tables"],
        seq_lens=inputs["seq_lens"],
        max_seq_len=args.topk,
        bmm1_scale=1.0,
        bmm2_scale=1.0,
        sparse_mla_top_k=args.topk,
        sparse_mla_top_k_lens=inputs["sparse_mla_top_k_lens"],
        backend=backend,
        cute_dsl_impl=cute_dsl_impl,
        enable_pdl=enable_pdl,
    )
    if isinstance(out, tuple):
        out = out[0]
    torch.accelerator.synchronize()
    return out.detach().float().cpu().clone()


def compare(runs: list[torch.Tensor]) -> dict[str, Any]:
    worst = 0.0
    identical = True
    first_row: int | None = None
    num_rows = 0
    for i, j in itertools.combinations(range(len(runs)), 2):
        a, b = runs[i], runs[j]
        diff = (a - b).abs()
        worst = max(worst, diff.max().item())
        identical = identical and torch.equal(a, b)
        per_row = diff.reshape(diff.shape[0], -1).amax(dim=-1)
        nz = (per_row > 0).nonzero().flatten()
        if nz.numel():
            num_rows = max(num_rows, int(nz.numel()))
            row = int(nz[0])
            first_row = row if first_row is None else min(first_row, row)
    return {
        "bitwise_identical": identical,
        "max_abs_diff": worst,
        "first_differing_row": first_row,
        "num_differing_rows": num_rows,
    }


def probe(
    args: argparse.Namespace,
    workspace: torch.Tensor,
    device: torch.device,
    backend: str,
    cute_dsl_impl: str,
    enable_pdl: bool | None,
) -> dict[str, Any]:
    label = f"backend={backend} cute_dsl_impl={cute_dsl_impl} enable_pdl={enable_pdl}"
    print(f"\n=== {label} ===", flush=True)
    row_counts = [int(x) for x in args.rows_sweep.split(",") if x.strip()]
    entries: list[dict[str, Any]] = []
    for rows in row_counts:
        gen = torch.Generator(device="cpu").manual_seed(args.seed + 1)
        num_logical = min(rows, args.topk, args.num_pages * args.page_size)
        placements = [make_placement(args, num_logical, gen, permute=False)]
        if args.placement_test:
            placements.append(make_placement(args, num_logical, gen, permute=True))
        try:
            runs = []
            for placement in placements:
                inputs = build_inputs(args, rows, device, placement)
                runs.extend(
                    run_once(
                        args, inputs, workspace, backend, cute_dsl_impl, enable_pdl
                    )
                    for _ in range(args.repeats)
                )
        except Exception as exc:  # noqa: BLE001 - report and keep sweeping
            print(
                f"  rows={rows:<5} UNSUPPORTED: {type(exc).__name__}: {exc}", flush=True
            )
            entries.append({"rows": rows, "error": f"{exc}"})
            # A rejected configuration fails identically for every row count.
            break
        entry = {"rows": rows, **compare(runs)}
        entries.append(entry)
        print(
            f"  rows={rows:<5} identical={entry['bitwise_identical']} "
            f"first_row={entry['first_differing_row']} "
            f"differing={entry['num_differing_rows']}/{rows} "
            f"max|d|={entry['max_abs_diff']:.4e}",
            flush=True,
        )
    ok = [e for e in entries if "error" not in e]
    return {
        "backend": backend,
        "cute_dsl_impl": cute_dsl_impl,
        "enable_pdl": enable_pdl,
        "entries": entries,
        "supported": bool(ok),
        "deterministic": bool(ok) and all(e["bitwise_identical"] for e in ok),
    }


def parse_pdl(text: str) -> list[bool | None]:
    values: list[bool | None] = []
    for item in text.split(","):
        item = item.strip().lower()
        if not item:
            continue
        values.append(None if item == "none" else item == "true")
    return values or [None]


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("this probe needs a CUDA device")
    device = torch.device("cuda:0")
    print(
        f"device: {torch.cuda.get_device_name(0)} "
        f"capability={torch.cuda.get_device_capability(0)}",
        flush=True,
    )

    import flashinfer

    print(f"flashinfer: {flashinfer.__version__}", flush=True)

    # The docstring requires a zero-initialized workspace for kernels that keep
    # semaphore state, and vLLM allocates it once per process with torch.zeros.
    workspace = torch.zeros(
        args.workspace_mb * 1024 * 1024, dtype=torch.int8, device=device
    )

    results: list[dict[str, Any]] = []
    for backend in [b.strip() for b in args.backends.split(",") if b.strip()]:
        for impl in [i.strip() for i in args.cute_dsl_impls.split(",") if i.strip()]:
            for pdl in parse_pdl(args.enable_pdl):
                results.append(probe(args, workspace, device, backend, impl, pdl))

    print("\n=== summary ===", flush=True)
    for r in results:
        if not r["supported"]:
            state = "unsupported"
        elif r["deterministic"]:
            state = "DETERMINISTIC"
        else:
            bad = next(
                (
                    e
                    for e in r["entries"]
                    if "error" not in e and not e["bitwise_identical"]
                ),
                None,
            )
            state = (
                f"nondeterministic from rows={bad['rows']} "
                f"(first row {bad['first_differing_row']})"
            )
        print(
            f"  backend={r['backend']:<12} impl={r['cute_dsl_impl']:<12} "
            f"pdl={str(r['enable_pdl']):<5} {state}",
            flush=True,
        )

    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
