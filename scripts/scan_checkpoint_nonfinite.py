#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Name checkpoint tensors that already hold non-finite or zero-scale values.

A model whose logits go NaN at only some positions is usually carrying the
NaN in its weights rather than computing it: tokens routed to one damaged
expert go non-finite while the rest of the batch stays clean. Reading the
kernel cannot distinguish that from a preparation bug, but reading the
checkpoint can.

Two encodings matter for FP4 exports. The packed weights are integer-coded
and cannot represent NaN, so a NaN must live in a scale, and NVFP4 keeps its
per-block scales in FP8 e4m3 -- which has a NaN encoding that a conversion
run over a zero or infinite amax will happily emit. A zero scale is just as
fatal and stays finite, so it is counted separately: it annihilates a whole
block, and any kernel that takes a reciprocal turns it into an infinity.

Runs on CPU against the safetensors shards, so it needs no GPU and never
builds the model.
"""

from __future__ import annotations

import argparse
import glob
import os

import torch
from safetensors import safe_open

# safetensors dtype tags that can encode a NaN. The packed FP4/int weights are
# excluded deliberately: they have no non-finite encoding, so materializing
# them would cost gigabytes to prove nothing.
_FLOAT_TAGS = {"F8_E5M2", "F8_E4M3", "F8_E4M3FN", "F16", "BF16", "F32", "F64"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("paths", nargs="+", help="checkpoint directories to scan")
    p.add_argument(
        "--report",
        type=int,
        default=12,
        help="how many offending tensors to print per checkpoint",
    )
    return p.parse_args()


def nonfinite_mask(tensor: torch.Tensor) -> torch.Tensor:
    """Mark non-finite elements, including in FP8 encodings.

    ``torch.isfinite`` has no kernel for the float8 variants at all, so the FP8
    cases are decided on their bit patterns instead of by casting.
    """
    if tensor.dtype == torch.float8_e4m3fn:
        return (tensor.view(torch.uint8) & 0x7F) == 0x7F
    if tensor.dtype == torch.float8_e5m2:
        return (tensor.view(torch.uint8) & 0x7F) >= 0x7C
    return ~torch.isfinite(tensor)


def nonfinite_count(tensor: torch.Tensor) -> int:
    return int(nonfinite_mask(tensor).sum())


def _zero_count(tensor: torch.Tensor) -> int:
    if tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return int(((tensor.view(torch.uint8) & 0x7F) == 0).sum())
    return int((tensor == 0).sum())


def scan(path: str, report: int) -> int:
    shards = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    print(f"=== {path}")
    if not shards:
        print("  no safetensors shards found")
        return 0

    bad: list[str] = []
    zeroed: list[str] = []
    scanned = 0
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if handle.get_slice(key).get_dtype() not in _FLOAT_TAGS:
                    continue
                tensor = handle.get_tensor(key)
                scanned += 1
                count = nonfinite_count(tensor)
                if count:
                    bad.append(f"{key} {tuple(tensor.shape)} {count} non-finite")
                if "scale" in key:
                    zeros = _zero_count(tensor)
                    if zeros:
                        zeroed.append(
                            f"{key} {tuple(tensor.shape)} {zeros} zero of "
                            f"{tensor.numel()}"
                        )

    print(f"  scanned {scanned} float tensor(s) across {len(shards)} shard(s)")
    print(f"  {len(bad)} tensor(s) hold non-finite values")
    for line in bad[:report]:
        print(f"    NONFINITE  {line}")
    print(f"  {len(zeroed)} scale tensor(s) hold zeros")
    for line in zeroed[:report]:
        print(f"    ZEROSCALE  {line}")
    return len(bad)


def main() -> int:
    args = parse_args()
    return 1 if sum(scan(p, args.report) for p in args.paths) else 0


if __name__ == "__main__":
    raise SystemExit(main())
