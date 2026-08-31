#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attribute a KLD mean to the kinds of text it was measured on.

The suite is stratified - encyclopedic reference, worked mathematics, source
code, dialogue, Chinese, structured data - because a candidate does not lose
fidelity uniformly. A single mean says how far two models are apart; it cannot
say that the gap is concentrated in code and tool calls while prose is
untouched, which is the difference between a candidate that is safe for a given
workload and one that is not.

This joins the per-context records in a report against the suite manifest that
produced them. It reads no model output, so it costs seconds and can be re-run
over every cell of an attribution ladder.

Two cautions are built into the output rather than left to the reader. A
stratum's KLD depends on how predictable its text is to begin with, so the
reference's own top-1 probability is reported beside every mean: a stratum that
diverges more where the reference was also less certain is a weaker finding than
one that diverges more where the reference was confident. And a stratum holds
tens of contexts, not hundreds, so the spread across contexts is reported
alongside the mean.

Usage:
    python fidelity/strata.py --suite /mnt/kld/suites/qwen3.6-1024x2048-v1 \\
        --cell deployed=library/Qwen3.6-27B/Qwen3.6-27B-FP8/report.json \\
        --out strata.md --json strata.json

    python fidelity/strata.py --suite SUITE \\
        --cell deployed=deployed/report.json \\
        --cell fp8_block=ladder/experts-fp8_block/report.json \\
        --cell nvfp4=ladder/experts-nvfp4/report.json --out strata.md

    python fidelity/strata.py --selftest
"""

import argparse
import json
import os
from typing import Any

GROUPINGS = ("stratum", "source_key")


def load_report(path: str) -> list[dict[str, Any]]:
    """Read the per-context records a report carries.

    Raises:
        SystemExit: if the report predates per-context recording, since the
            answer is a rescore rather than a weaker join.
    """
    with open(path, encoding="utf-8") as handle:
        report = json.load(handle)
    records = report.get("per_context")
    if not records:
        raise SystemExit(
            f"{path} has no per_context records, so its mean cannot be "
            f"attributed to a stratum. Rescore with a build that records them."
        )
    missing = [r for r in records if r.get("context_id") is None]
    if missing:
        raise SystemExit(
            f"{path}: {len(missing)} of {len(records)} records carry no "
            f"context_id, which means the run did not read a frozen suite. "
            f"Per-domain attribution requires suite rows (Law 3)."
        )
    return records


def load_suite(suite: str) -> tuple[dict[int, dict[str, Any]], dict[str, str]]:
    """Return contexts keyed by id, and the human label for each stratum."""
    path = suite
    if os.path.isdir(suite):
        path = os.path.join(suite, "suite-manifest.json")
    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    contexts = {int(c["context_id"]): c for c in manifest["contexts"]}
    labels = {
        s["key"]: s.get("label") or s["key"] for s in manifest.get("strata", [])
    }
    return contexts, labels


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    return float(ordered[min(len(ordered) - 1, int(q * len(ordered)))])


def _weighted(records: list[dict[str, Any]], field: str, positions: int) -> float:
    if not positions:
        return float("nan")
    total = sum(float(r[field]) * int(r["positions"]) for r in records)
    return total / positions


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine per-context records without pretending they are one sample.

    The mean is position-weighted, so it matches what a run over only these
    contexts would have reported. Everything else describes the spread across
    contexts, because a stratum whose mean comes from one bad document is a
    different finding from one where every document moved.
    """
    positions = sum(int(r["positions"]) for r in records)
    means = [float(r["mean_kld"]) for r in records]
    tails = [float(r["p99_kld"]) for r in records]
    worst = max(records, key=lambda r: float(r["mean_kld"]))
    return {
        "contexts": len(records),
        "positions": positions,
        "mean_kld": _weighted(records, "mean_kld", positions),
        "median_context_kld": _percentile(means, 0.5),
        "p90_context_kld": _percentile(means, 0.9),
        "worst_context_kld": float(worst["mean_kld"]),
        "worst_context_id": worst.get("context_id"),
        "median_context_p99": _percentile(tails, 0.5),
        "max_kld": max(float(r["max_kld"]) for r in records),
        "mean_ref_top1_prob": _weighted(records, "mean_ref_top1_prob", positions),
        "top1_agreement": _weighted(records, "top1_agreement", positions),
    }


def group_cell(
    records: list[dict[str, Any]],
    contexts: dict[int, dict[str, Any]],
    field: str,
) -> dict[str, dict[str, Any]]:
    """Aggregate one cell's records by a suite field, such as stratum or source."""
    buckets: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        context = contexts.get(int(record["context_id"]))
        if context is None:
            raise SystemExit(
                f"context {record['context_id']} is scored but is not in the "
                f"suite manifest; the report and the suite do not match"
            )
        buckets.setdefault(str(context[field]), []).append(record)
    return {key: aggregate(rows) for key, rows in buckets.items()}


def analyze(
    cells: dict[str, list[dict[str, Any]]],
    contexts: dict[int, dict[str, Any]],
    labels: dict[str, str],
) -> dict[str, Any]:
    """Build the per-group table for every cell, ranked by the first cell.

    The first cell is the one being reported on; the others are context, so the
    ranking follows the first and the rest are shown against it.
    """
    names = list(cells)
    primary = names[0]
    overall = {name: aggregate(records) for name, records in cells.items()}
    groups: dict[str, list[dict[str, Any]]] = {}
    for field in GROUPINGS:
        grouped = {
            name: group_cell(records, contexts, field)
            for name, records in cells.items()
        }
        rows: list[dict[str, Any]] = []
        base = overall[primary]["mean_kld"]
        for key, summary in grouped[primary].items():
            rows.append(
                {
                    "key": key,
                    "label": labels.get(key, key),
                    "relative_to_run": (
                        summary["mean_kld"] / base if base else float("nan")
                    ),
                    "cells": {
                        name: grouped[name][key]
                        for name in names
                        if key in grouped[name]
                    },
                }
            )
        rows.sort(key=lambda r: r["cells"][primary]["mean_kld"], reverse=True)
        groups[field] = rows
    return {
        "cells": names,
        "primary": primary,
        "overall": overall,
        "groups": groups,
    }


def _rank_table(
    rows: list[dict[str, Any]], primary: str, title: str, column: str = "Domain"
) -> list[str]:
    lines = [
        f"## {title}",
        "",
        f"| {column} | Contexts | Ref top-1 | Mean KLD | x run | Median ctx "
        "| p90 ctx | Worst ctx |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        cell = row["cells"][primary]
        lines.append(
            f"| {row['label']} | {cell['contexts']} | "
            f"{cell['mean_ref_top1_prob'] * 100:.1f}% | "
            f"{cell['mean_kld']:.8f} | {row['relative_to_run']:.2f} | "
            f"{cell['median_context_kld']:.8f} | "
            f"{cell['p90_context_kld']:.8f} | "
            f"{cell['worst_context_kld']:.6f} |"
        )
    return lines + [""]


def _ladder_table(rows: list[dict[str, Any]], names: list[str]) -> list[str]:
    lines = [
        "## The same domains across the ladder",
        "",
        "Each column is a separate measurement of the same contexts, so a "
        "domain that moves between columns is one the format treats differently.",
        "",
        "| Domain | " + " | ".join(names) + " |",
        "|---" * (len(names) + 1) + "|",
    ]
    for row in rows:
        values = []
        for name in names:
            cell = row["cells"].get(name)
            values.append("n/a" if cell is None else f"{cell['mean_kld']:.8f}")
        lines.append(f"| {row['label']} | " + " | ".join(values) + " |")
    return lines + [""]


def _reading(report: dict[str, Any]) -> list[str]:
    """State what the ranking supports, and what it does not."""
    primary = report["primary"]
    rows = report["groups"]["stratum"]
    run = report["overall"][primary]
    worst, best = rows[0], rows[-1]
    worst_cell, best_cell = worst["cells"][primary], best["cells"][primary]
    spread = (
        worst_cell["mean_kld"] / best_cell["mean_kld"]
        if best_cell["mean_kld"]
        else float("nan")
    )
    notes = [
        f"- Weakest domain: **{worst['label']}** at {worst_cell['mean_kld']:.8f}, "
        f"{worst['relative_to_run']:.2f}x the run mean of {run['mean_kld']:.8f} "
        f"over {worst_cell['contexts']} context(s).",
        f"- Strongest domain: **{best['label']}** at {best_cell['mean_kld']:.8f}, "
        f"{best['relative_to_run']:.2f}x the run mean. The spread across "
        f"domains is {spread:.1f}x.",
    ]

    confidence_gap = worst_cell["mean_ref_top1_prob"] - run["mean_ref_top1_prob"]
    if confidence_gap < -0.05:
        notes.append(
            f"- Read with care: the reference was itself less certain on "
            f"{worst['label']} (top-1 "
            f"{worst_cell['mean_ref_top1_prob'] * 100:.1f}% against "
            f"{run['mean_ref_top1_prob'] * 100:.1f}% overall). Part of this "
            f"gap is the text being harder to predict, not the candidate being "
            f"worse on it."
        )
    elif confidence_gap > 0.05:
        notes.append(
            f"- This is a strong finding: the reference was *more* certain on "
            f"{worst['label']} (top-1 "
            f"{worst_cell['mean_ref_top1_prob'] * 100:.1f}% against "
            f"{run['mean_ref_top1_prob'] * 100:.1f}% overall) and the candidate "
            f"still diverged most there, so predictability does not explain it."
        )

    if worst_cell["median_context_kld"] and (
        worst_cell["worst_context_kld"] / worst_cell["median_context_kld"] >= 4.0
    ):
        notes.append(
            f"- {worst['label']} is driven by outlier contexts: its worst "
            f"context is {worst_cell['worst_context_kld']:.6f} against a median "
            f"of {worst_cell['median_context_kld']:.8f} (context "
            f"{worst_cell['worst_context_id']}). Read the documents before "
            f"treating the domain as weak."
        )

    names = report["cells"]
    if len(names) > 1:
        worst_by_cell = {}
        for name in names:
            ranked = sorted(
                (r for r in rows if name in r["cells"]),
                key=lambda r: r["cells"][name]["mean_kld"],
                reverse=True,
            )
            if ranked:
                worst_by_cell[name] = ranked[0]["label"]
        agreed = len(set(worst_by_cell.values())) == 1
        if agreed:
            notes.append(
                f"- Every cell is weakest on the same domain "
                f"({next(iter(worst_by_cell.values()))}), so the ranking is a "
                f"property of the model and the text rather than of one format."
            )
        else:
            detail = ", ".join(f"{k}: {v}" for k, v in worst_by_cell.items())
            notes.append(
                f"- The cells disagree about which domain is weakest "
                f"({detail}), which means the format choice, not the model, "
                f"decides where fidelity is lost."
            )
    return ["## Reading", ""] + notes + [""]


def render(report: dict[str, Any], label: str) -> str:
    primary = report["primary"]
    run = report["overall"][primary]
    lines = [
        f"# Fidelity by domain - {label}",
        "",
        f"{run['contexts']} contexts, {run['positions']} scored positions, mean "
        f"{run['mean_kld']:.8f}, reference top-1 "
        f"{run['mean_ref_top1_prob'] * 100:.1f}%, top-1 agreement "
        f"{run['top1_agreement'] * 100:.4f}%.",
        "",
        "`x run` is the domain's mean divided by the run's mean, so 1.00 is a "
        "domain that degrades exactly as much as the model as a whole. `Median "
        "ctx` and `p90 ctx` are the spread of per-context means within the "
        "domain.",
        "",
    ]
    lines += _rank_table(report["groups"]["stratum"], primary, "Ranked by domain")
    if len(report["cells"]) > 1:
        lines += _ladder_table(report["groups"]["stratum"], report["cells"])
    lines += _rank_table(
        report["groups"]["source_key"],
        primary,
        "Ranked by source dataset",
        column="Source",
    )
    lines += _reading(report)
    return "\n".join(lines)


def _parse_cell(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise SystemExit(
            f"--cell wants NAME=path/to/report.json, got {spec!r}"
        )
    name, path = spec.split("=", 1)
    if not name or not path:
        raise SystemExit(f"--cell wants NAME=path/to/report.json, got {spec!r}")
    return name, path


def selftest() -> int:
    """Check the join and the weighting on data whose answer is known by hand."""
    contexts = {
        0: {"context_id": 0, "stratum": "code", "source_key": "github"},
        1: {"context_id": 1, "stratum": "code", "source_key": "stackv2"},
        2: {"context_id": 2, "stratum": "prose", "source_key": "wikipedia"},
    }
    labels = {"code": "Source code", "prose": "Prose"}

    def record(cid: int, mean: float, positions: int) -> dict[str, Any]:
        return {
            "row": cid,
            "context_id": cid,
            "positions": positions,
            "mean_kld": mean,
            "median_kld": mean,
            "p99_kld": mean * 3,
            "max_kld": mean * 10,
            "mean_ref_top1_prob": 0.9,
            "top1_agreement": 0.95,
        }

    # Unequal lengths, so an unweighted mean would give 0.15 where the
    # position-weighted answer is 0.10.
    cells = {
        "deployed": [record(0, 0.2, 100), record(1, 0.1, 300), record(2, 0.01, 400)],
        "nvfp4": [record(0, 0.4, 100), record(1, 0.2, 300), record(2, 0.05, 400)],
    }
    report = analyze(cells, contexts, labels)
    code = report["groups"]["stratum"][0]
    if code["key"] != "code":
        print(f"FAIL  ranking put {code['key']} first, expected code")
        return 1
    expected = (0.2 * 100 + 0.1 * 300) / 400
    if abs(code["cells"]["deployed"]["mean_kld"] - expected) > 1e-12:
        print(
            f"FAIL  code mean is {code['cells']['deployed']['mean_kld']}, "
            f"expected the position-weighted {expected}"
        )
        return 1
    if code["cells"]["deployed"]["contexts"] != 2:
        print("FAIL  the two code contexts did not group together")
        return 1
    sources = {row["key"] for row in report["groups"]["source_key"]}
    if sources != {"github", "stackv2", "wikipedia"}:
        print(f"FAIL  source grouping produced {sources}")
        return 1
    text = render(report, "selftest")
    for needle in ("Source code", "Ranked by source dataset", "## Reading"):
        if needle not in text:
            print(f"FAIL  rendered report is missing {needle!r}")
            return 1
    if "n/a" in text:
        print("FAIL  a cell went missing from the ladder table")
        return 1
    print("PASS  grouping, position weighting, and rendering")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", help="suite directory or suite-manifest.json")
    parser.add_argument(
        "--cell",
        action="append",
        default=[],
        metavar="NAME=REPORT",
        help="a report to attribute; the first is the one being reported on",
    )
    parser.add_argument("--label", default=None, help="name for the report heading")
    parser.add_argument("--out", default=None, help="write the markdown here")
    parser.add_argument("--json", default=None, help="write the raw numbers here")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        return selftest()
    if not args.suite or not args.cell:
        parser.error("--suite and at least one --cell are required")

    contexts, labels = load_suite(args.suite)
    cells = {}
    for spec in args.cell:
        name, path = _parse_cell(spec)
        cells[name] = load_report(path)
    report = analyze(cells, contexts, labels)
    text = render(report, args.label or report["primary"])
    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text + "\n")
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
