#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture prompt logits once and exit, for cross-process comparison.

Repeating a capture inside one process is not a determinism test for this
model: the KV block allocator hands out different physical blocks on the second
request, and the sparse MLA kernel's result depends on where the KV lives. Two
fresh processes replaying the same prompt sequence allocate identically, which
is what the reference-capture and scoring workflow actually does.
"""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    p.add_argument("--length", type=int, default=130)
    p.add_argument("--tp", type=int, default=4)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    p.add_argument("--moe-backend", default="triton")
    p.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help="Chunk prefill to this many tokens per forward pass. "
        "Small batches keep GEMMs out of the split-K regime "
        "where reductions accumulate through atomics.",
    )
    p.add_argument("--out", required=True, help="Path to write logits (.pt)")
    p.add_argument(
        "--fingerprint-out",
        default=None,
        help="Also write sparse MLA kernel argument fingerprints "
        "as JSON, so two processes can be diffed.",
    )
    return p.parse_args()


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path.home()))
    from glm53_determinism_probe import build_llm, build_tokens, capture

    args = parse_args()
    tokens = build_tokens(
        SimpleNamespace(
            model=args.model,
            dataset_dir=args.dataset_dir,
            dataset_config=args.dataset_config,
            ctx=2048,
            stride=0,
        )
    )
    llm = build_llm(
        SimpleNamespace(
            model=args.model,
            tp=args.tp,
            ctx=2048,
            gpu_memory_utilization=args.gpu_memory_utilization,
            moe_backend=args.moe_backend,
            max_num_batched_tokens=args.max_num_batched_tokens,
            disable_flashinfer_autotune=False,
            enforce_eager=True,
        )
    )

    if args.fingerprint_out:
        from glm53_layer_bisect import _rpc_install

        # Hook no modules; the pattern only exists to initialise worker state
        # that the kernel wrapper records into.
        llm.collective_rpc(_rpc_install, args=(r"^$", True))

    logits = capture(llm, tokens[: args.length])
    torch.save(logits, args.out)
    print(f"wrote {args.out} shape={tuple(logits.shape)}", flush=True)

    if args.fingerprint_out:
        import json

        from glm53_layer_bisect import _rpc_dump_calls

        per_rank = llm.collective_rpc(_rpc_dump_calls, args=(0,))
        Path(args.fingerprint_out).write_text(
            json.dumps(per_rank, indent=2), encoding="utf-8"
        )
        print(
            f"wrote {args.fingerprint_out} calls_per_rank={[len(r) for r in per_rank]}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
