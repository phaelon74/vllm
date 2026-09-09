# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which NVFP4 modules a published checkpoint left without an activation scale.

Reads safetensors headers over HTTP range requests, so it answers for a
checkpoint that was released after scoring without pulling its shards back. Use
it to decide whether an existing result carries a substitution nobody disclosed.

    python3 scripts/scan_remote_nvfp4_scale_keys.py \\
        unsloth/gemma-4-26B-A4B-it-NVFP4@20df0542b1a86ce19f495ac2eca2c7c12bce82f9

A module is counted as NVFP4 when it carries a ``weight_scale``. Absence of an
activation scale is only a gap when the checkpoint's other modules have one: a
weight-only W4A16 export has none anywhere and that is correct.
"""

import argparse
import collections
import json
import sys

ACTIVATION_KEYS = ("input_scale", "input_global_scale")


def shard_names(repo: str, revision: str | None) -> list[str]:
    from huggingface_hub import HfApi

    return sorted(
        entry.path
        for entry in HfApi().list_repo_tree(
            repo, revision=revision, recursive=True
        )
        if entry.path.endswith(".safetensors")
    )


def header_keys(repo: str, revision: str, name: str) -> list[str]:
    """Tensor names from one shard, reading only its header."""
    import requests

    url = f"https://huggingface.co/{repo}/resolve/{revision}/{name}"
    first = requests.get(url, headers={"Range": "bytes=0-7"}, timeout=60)
    first.raise_for_status()
    length = int.from_bytes(first.content[:8], "little")
    rest = requests.get(
        url, headers={"Range": f"bytes=8-{8 + length - 1}"}, timeout=60
    )
    rest.raise_for_status()
    return [key for key in json.loads(rest.content) if key != "__metadata__"]


def scan(repo: str, revision: str) -> int:
    print(f"\n=== {repo}@{revision[:12]}")
    modules: dict[str, set[str]] = collections.defaultdict(set)
    for name in shard_names(repo, revision):
        for key in header_keys(repo, revision, name):
            module, _, leaf = key.rpartition(".")
            if module:
                modules[module].add(leaf)

    quantized = {
        module: leaves
        for module, leaves in modules.items()
        if "weight_scale" in leaves
    }
    # Experts are already covered by the routed disclosure; the open question is
    # everything else, where a fill would have gone unreported entirely.
    dense = {m: v for m, v in quantized.items() if "experts" not in m}
    with_scale = {
        m for m, v in dense.items() if any(k in v for k in ACTIVATION_KEYS)
    }
    print(f"  quantized modules      : {len(quantized)}")
    print(f"  dense (non-expert)     : {len(dense)}")
    print(f"  dense with act scale   : {len(with_scale)}")
    if not dense:
        print("  -> no dense NVFP4 modules; nothing the dense walk would fill")
        return 0
    if not with_scale:
        print("  -> weight-only dense: no activation scales anywhere, as expected")
        return 0
    missing = sorted(set(dense) - with_scale)
    if not missing:
        print("  -> every dense NVFP4 module carries its own activation scale")
        return 0
    print(f"  GAP: {len(missing)} dense module(s) short an activation scale")
    for module in missing[:20]:
        print(f"    {module}")
    if len(missing) > 20:
        print(f"    ... and {len(missing) - 20} more")
    return len(missing)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", nargs="+", metavar="REPO@REVISION")
    args = parser.parse_args()
    gaps = 0
    for target in args.target:
        repo, _, revision = target.partition("@")
        if not revision:
            raise SystemExit(f"{target}: pin a revision, as repo@revision")
        gaps += scan(repo, revision)
    print(
        "\nno undisclosed dense gaps"
        if not gaps
        else f"\n{gaps} dense module(s) would have been filled without disclosure"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
