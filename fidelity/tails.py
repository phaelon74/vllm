#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ask whether a mean KLD describes a distribution or a handful of positions.

A mean of 0.017 can mean two unrelated things: every position drifted slightly,
or a few hundred positions out of two million changed completely and the rest
agree. The first is precision loss. The second is a structural divergence -
on a mixture-of-experts checkpoint, most likely a routing flip, where the
candidate selected a different expert and so computed a different function.

The distinction matters because a mean that is mostly tail cannot rank two
candidates on precision: the tail masks the precision signal entirely. This tool
measures how much of a mean lives in its tail, and whether the tail coincides
with positions where the top-1 prediction changed, which is the cheap observable
proxy for a structural flip.

It reads a dump written by `score_mode_kld.py --dump-positions` and computes no
model output of its own, so it costs seconds and can be re-run freely.

Usage:
    python fidelity/tails.py --dump positions.safetensors
    python fidelity/tails.py --dump positions.safetensors --label 35B-A3B-FP8 \\
        --json tails.json
"""

import argparse
import json
import os
from typing import Any

import numpy as np

# Fractions of the scored positions to attribute the mean to, smallest first.
TAIL_FRACTIONS = (0.0001, 0.001, 0.01, 0.05, 0.10)

# Absolute KLD levels. 0.1 nat is a visible change in a token's distribution;
# above 1.0 the two models substantially disagree about what comes next.
THRESHOLDS = (0.01, 0.1, 1.0, 5.0)


def load_dump(path: str) -> dict[str, np.ndarray]:
    from safetensors.numpy import load_file

    data = load_file(path)
    if "kld" not in data:
        raise SystemExit(
            f"{path} has no 'kld' array; was it written by --dump-positions?"
        )
    return data


def concentration(kld: np.ndarray) -> list[dict[str, Any]]:
    """How much of the total each top fraction of positions contributes."""
    total = float(kld.sum())
    n = kld.size
    ordered = np.sort(kld)[::-1]
    out: list[dict[str, Any]] = []
    for fraction in TAIL_FRACTIONS:
        count = max(1, int(round(n * fraction)))
        head = float(ordered[:count].sum())
        rest = ordered[count:]
        out.append(
            {
                "fraction": fraction,
                "positions": count,
                "share_of_total": head / total if total else float("nan"),
                "mean_excluding": float(rest.mean()) if rest.size else None,
            }
        )
    return out


def by_agreement(kld: np.ndarray, agree: np.ndarray) -> dict[str, Any]:
    """Split the mean by whether the top-1 prediction survived.

    A position whose argmax changed is where a structural difference shows up
    first, so this is the cheapest available separation of "drifted" from
    "changed", with no extra instrumentation.
    """
    matched = agree.astype(bool)
    flipped = ~matched
    total = float(kld.sum())
    n = kld.size
    return {
        "matched_positions": int(matched.sum()),
        "flipped_positions": int(flipped.sum()),
        "flipped_fraction": float(flipped.sum()) / n if n else float("nan"),
        "matched_mean": float(kld[matched].mean()) if matched.any() else None,
        "flipped_mean": float(kld[flipped].mean()) if flipped.any() else None,
        "flipped_share_of_total": (
            float(kld[flipped].sum()) / total if total else float("nan")
        ),
    }


def analyze(data: dict[str, np.ndarray], label: str) -> dict[str, Any]:
    kld = np.asarray(data["kld"], dtype=np.float64)
    n = kld.size
    total = float(kld.sum())
    report: dict[str, Any] = {
        "label": label,
        "positions": n,
        "mean": float(kld.mean()),
        "median": float(np.median(kld)),
        "p99": float(np.quantile(kld, 0.99)),
        "p999": float(np.quantile(kld, 0.999)),
        "max": float(kld.max()),
        "concentration": concentration(kld),
        "thresholds": [
            {
                "above": level,
                "positions": int((kld > level).sum()),
                "fraction": float((kld > level).sum()) / n,
                "share_of_total": (
                    float(kld[kld > level].sum()) / total if total else float("nan")
                ),
            }
            for level in THRESHOLDS
        ],
    }
    if "top1_agree" in data:
        report["agreement"] = by_agreement(kld, data["top1_agree"])
    return report


def render(report: dict[str, Any]) -> str:
    lines = [
        f"# Tail concentration — {report['label']}",
        "",
        f"{report['positions']} scored positions, mean "
        f"{report['mean']:.8f}, median {report['median']:.8f}, "
        f"max {report['max']:.6f}.",
        "",
        "## Where the mean lives",
        "",
        "| Top fraction | Positions | Share of total KLD | Mean excluding them |",
        "|---|---|---|---|",
    ]
    for row in report["concentration"]:
        excluding = row["mean_excluding"]
        lines.append(
            f"| {row['fraction'] * 100:g}% | {row['positions']} | "
            f"{row['share_of_total'] * 100:.1f}% | "
            f"{'n/a' if excluding is None else f'{excluding:.8f}'} |"
        )
    lines += [
        "",
        "## Positions above a level",
        "",
        "| KLD above | Positions | Fraction | Share of total |",
        "|---|---|---|---|",
    ]
    for row in report["thresholds"]:
        lines.append(
            f"| {row['above']} | {row['positions']} | "
            f"{row['fraction'] * 100:.3f}% | {row['share_of_total'] * 100:.1f}% |"
        )
    agreement = report.get("agreement")
    if agreement:
        matched = agreement["matched_mean"]
        flipped = agreement["flipped_mean"]
        lines += [
            "",
            "## Split by top-1 survival",
            "",
            f"- top-1 changed at {agreement['flipped_positions']} positions "
            f"({agreement['flipped_fraction'] * 100:.2f}%), carrying "
            f"{agreement['flipped_share_of_total'] * 100:.1f}% of the total",
            f"- mean where top-1 held: "
            f"{'n/a' if matched is None else f'{matched:.8f}'}",
            f"- mean where top-1 changed: "
            f"{'n/a' if flipped is None else f'{flipped:.8f}'}",
        ]
    lines += ["", _verdict(report), ""]
    return "\n".join(lines)


def _verdict(report: dict[str, Any]) -> str:
    """State what the numbers imply, so the table is not left to interpretation."""
    one_percent = next(
        row for row in report["concentration"] if row["fraction"] == 0.01
    )
    share = one_percent["share_of_total"]
    without = one_percent["mean_excluding"]
    if share >= 0.75:
        reading = (
            f"Tail-dominated: 1% of positions carry {share * 100:.0f}% of the "
            f"mean. This mean cannot rank candidates on precision — without "
            f"that 1% it is {without:.8f}, a different number in kind. Find "
            f"what those positions have in common before quoting the mean."
        )
    elif share >= 0.40:
        reading = (
            f"Mixed: 1% of positions carry {share * 100:.0f}% of the mean. "
            f"Report the trimmed mean {without:.8f} alongside it; a precision "
            f"comparison between candidates should use the trimmed number."
        )
    else:
        reading = (
            f"Broad: 1% of positions carry {share * 100:.0f}% of the mean, so "
            f"the mean describes the distribution rather than its outliers."
        )
    return f"## Reading\n\n{reading}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dump", required=True, help="safetensors from --dump-positions"
    )
    parser.add_argument("--label", default=None, help="name for the report heading")
    parser.add_argument("--json", default=None, help="also write the raw numbers here")
    parser.add_argument("--out", default=None, help="write the markdown here")
    args = parser.parse_args()

    label = args.label or os.path.basename(args.dump)
    report = analyze(load_dump(args.dump), label)
    text = render(report)
    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        print(f"wrote {args.out}")
    else:
        print(text)
    if args.json:
        with open(args.json, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
