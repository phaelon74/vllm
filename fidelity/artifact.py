#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Render Local Inference Lab distribution-fidelity artifacts.

Subcommands:
    onepager    one candidate's report.md from its report, manifest, and receipt
    leaderboard rank every compliant result sharing one comparability key
    checksums   sha256sum-compatible manifest over an assembled artifact

The renderers read only published JSON, so an artifact can be rebuilt from its
own contents without a GPU.
"""

import argparse
import csv
import hashlib
import json
import os
import sys
from typing import Any

LAWS_VERSION = 1
PROGRAM = "Local Inference Lab"


def _load(path: str | None) -> dict[str, Any] | None:
    if not path or not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _short(value: Any, width: int = 12) -> str:
    text = str(value)
    return text[:width] if len(text) > width else text


def _kld(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.8f}"


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{float(value) * 100:.4f}%"


def _table(rows: list[tuple[str, ...]], header: tuple[str, ...]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join("---" for _ in header) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _headline(report: dict[str, Any], label: str) -> list[str]:
    """Mean and its tail, together, per Law 9."""
    out = [f"# {label}: distribution fidelity", ""]
    out.append(
        f"**Mean KLD(reference || candidate) = {_kld(report.get('mean_kld'))}** "
        f"over {report.get('num_positions')} scored positions."
    )
    out.append("")
    out.extend(
        _table(
            [
                (
                    _kld(report.get("mean_kld")),
                    _kld(report.get("median_kld")),
                    _kld(report.get("p90_kld")),
                    _kld(report.get("p99_kld")),
                    _kld(report.get("max_kld")),
                    _pct(report.get("top1_agreement")),
                )
            ],
            ("Mean", "Median", "p90", "p99", "Max", "Top-1 agreement"),
        )
    )
    out.append("")
    out.append(
        f"Reverse direction, KLD(candidate || reference): "
        f"{_kld(report.get('mean_kld_reverse'))}."
    )
    out.append("")
    return out


def _identity(
    report: dict[str, Any],
    manifest: dict[str, Any],
    receipt: dict[str, Any],
    runtime_env: dict[str, Any],
) -> list[str]:
    """Everything that bounds the number's comparability, per Law 10."""
    runtime = manifest.get("runtime") or {}
    caps = receipt.get("comparability_key") or {}
    rows = [
        ("Reference checkpoint", str(manifest.get("reference_model"))),
        ("Reference config SHA-256", _short(manifest.get("reference_config_sha256"))),
        ("Candidate checkpoint", str(report.get("student_model"))),
        ("Suite", str(caps.get("suite_id") or "run-time tokenization (Law 3 gap)")),
        ("Suite token SHA-256", _short(manifest.get("token_sha256"), 16)),
        ("Capture manifest SHA-256", _short(report.get("capture_manifest_sha256"), 16)),
        ("Tokenizer", str((manifest.get("tokenizer") or {}).get("name_or_path"))),
        ("Scored vocabulary", str(report.get("kld_vocab_size"))),
        ("Declared vocabulary", str(manifest.get("declared_vocab_size"))),
        (
            "Geometry",
            f"{manifest.get('rows')} rows x {manifest.get('context_length')} tokens, "
            f"stride {manifest.get('stride')}, score_from {manifest.get('score_from')}",
        ),
        ("Reference storage", str(manifest.get("storage"))),
        (
            "Model runner",
            "V2" if report.get("model_runner_v2") else "V1",
        ),
        ("Tensor parallel", str(manifest.get("tensor_parallel_size"))),
        ("Eager enforced", str(manifest.get("enforce_eager"))),
        ("Prefix caching", str(manifest.get("enable_prefix_caching"))),
        ("max_num_seqs", str(manifest.get("max_num_seqs"))),
        ("vLLM", str((runtime_env.get("vllm") or {}).get("version"))),
        ("vLLM commit", _short(runtime_env.get("vllm_commit"), 12)),
        ("torch", str(runtime.get("torch"))),
        ("Driver", str(runtime.get("driver"))),
        ("GPUs", ", ".join(runtime.get("gpu_names") or []) or "n/a"),
        ("Laws version", str(receipt.get("laws_version", LAWS_VERSION))),
        ("Partition", str(receipt.get("partition"))),
    ]
    return ["## Identity", "", *_table(rows, ("Item", "Value")), ""]


def _profiles(report: dict[str, Any]) -> list[str]:
    """Depth and confidence profiles, per Law 9."""
    out: list[str] = []
    depth = report.get("depth_buckets") or []
    if depth:
        rows = [
            (
                f"{b.get('depth_lo')}\u2013{b.get('depth_hi')}",
                str(b.get("n")),
                _kld(b.get("mean_kld")),
            )
            for b in depth
        ]
        out += ["## Error by context depth", ""]
        out += _table(rows, ("Position range", "Positions", "Mean KLD"))
        out.append("")
    conf = report.get("confidence_buckets") or []
    if conf:
        rows = [
            (
                f"[{b.get('lo'):.2f}, {b.get('hi'):.2f})",
                str(b.get("n")),
                f"{float(b.get('frac', 0.0)) * 100:.1f}%",
                _kld(b.get("mean_kld")),
            )
            for b in conf
        ]
        out += ["## Error by reference confidence", ""]
        out += _table(rows, ("Reference top-1 probability", "Positions", "Share",
                             "Mean KLD"))
        out.append("")
    topk = report.get("topk_agreement") or {}
    if topk:
        keys = sorted(topk, key=lambda k: int(k))
        rows = [tuple(f"K={k}" for k in keys), tuple(_pct(topk[k]) for k in keys)]
        out += ["## Top-K set agreement", ""]
        out += _table([rows[1]], rows[0])
        out.append("")
    return out


def _head_split(report: dict[str, Any], manifest: dict[str, Any]) -> list[str]:
    """Trunk versus deployed, per Law 8."""
    rows = [
        ("Trunk (candidate hidden states, reference head)",
         _kld(report.get("trunk_mean_kld"))),
        ("Deployed (candidate's own head)", _kld(report.get("deployed_mean_kld"))),
        ("Head-associated delta (not additive)", _kld(report.get("head_delta_kld"))),
        ("Reference head", str((manifest.get("lm_head") or {}).get("runtime"))),
        ("Candidate head", str(report.get("student_lm_head"))),
    ]
    return ["## Trunk versus head", "", *_table(rows, ("Component", "Value")), ""]


def _compliance(receipt: dict[str, Any], baseline: dict[str, Any] | None) -> list[str]:
    """Law-by-law status, with deviations rendered in place per Law 13."""
    out = ["## Law compliance", ""]
    if baseline is not None:
        out.append(
            f"Zero baseline: reference against itself scored "
            f"**{_kld(baseline.get('mean_kld'))}** over "
            f"{baseline.get('num_positions')} positions."
        )
        out.append("")
    rows = [
        (
            str(f.get("law")),
            str(f.get("title")),
            str(f.get("status", "")).upper(),
            str(f.get("detail", "")),
        )
        for f in receipt.get("findings") or []
    ]
    out += _table(rows, ("Law", "Name", "Status", "Detail"))
    out.append("")
    overrides = [
        f for f in receipt.get("findings") or [] if f.get("status") == "override"
    ]
    if overrides:
        out += ["### Recorded deviations", ""]
        for finding in overrides:
            approval = finding.get("approval") or {}
            out.append(
                f"- **Law {finding.get('law')} ({finding.get('title')})** "
                f"overridden by {approval.get('approver')} at "
                f"{approval.get('timestamp')}. Justification: "
                f"{str(approval.get('justification')).rstrip('.')}. "
                f"Underlying finding: {finding.get('detail')}."
            )
        out.append("")
    return out


def _scope() -> list[str]:
    return [
        "## Scope",
        "",
        "This artifact measures teacher-forced next-token distribution fidelity at "
        "the stated context length, over the stated token suite, through the stated "
        "runtime. It does not measure free-running generation, benchmark accuracy, "
        "instruction following, long-context behavior beyond the suite's context "
        "length, multimodal behavior, tool use, or throughput.",
        "",
        "Per Law 10, these numbers are comparable only to numbers produced from the "
        "same suite, geometry, and runtime identity. A KLD threshold borrowed from "
        "another model, corpus, tokenizer, or serving stack carries no verdict here.",
        "",
    ]


def render_onepager(
    report: dict[str, Any],
    manifest: dict[str, Any],
    receipt: dict[str, Any],
    runtime_env: dict[str, Any],
    baseline: dict[str, Any] | None,
    label: str,
) -> str:
    """Render the candidate one-pager."""
    parts = _headline(report, label)
    baseline_kld = receipt.get("zero_baseline_kld")
    if baseline_kld not in (0.0, None):
        floor = receipt.get("nondeterminism_floor")
        parts += [
            f"> **The zero baseline was not met.** The reference scored "
            f"{_kld(baseline_kld)} against itself, so this campaign carries a "
            f"nondeterminism floor of {_kld(floor)} measured from three repeat "
            f"captures. Every mean below sits on top of that floor, and a "
            f"difference no larger than it is not a result. Approved as a Law 1 "
            f"deviation; see Recorded deviations.",
            "",
        ]
    if not receipt.get("compliant", False):
        parts += [
            f"> **NOT LAW-COMPLIANT.** Failed law(s): "
            f"{receipt.get('failed_laws')}. This result must not be published as a "
            f"{PROGRAM} fidelity measurement.",
            "",
        ]
    parts += _identity(report, manifest, receipt, runtime_env)
    parts += _head_split(report, manifest)
    parts += _profiles(report)
    parts += _compliance(receipt, baseline)
    parts += _scope()
    return "\n".join(parts).rstrip() + "\n"


def collect_results(results_root: str) -> list[dict[str, Any]]:
    """Find every result in a library tree, at any depth.

    A result is any directory holding both ``report.json`` and
    ``compliance.json``, so this works for a flat directory of candidates and for
    the published ``<model>/<quant>/`` library layout alike. The label is the
    path relative to the root, which makes the model visible in the leaderboard.
    """
    found: list[dict[str, Any]] = []
    for dirpath, dirnames, filenames in os.walk(results_root):
        dirnames.sort()
        if not {"report.json", "compliance.json"} <= set(filenames):
            continue
        report = _load(os.path.join(dirpath, "report.json"))
        receipt = _load(os.path.join(dirpath, "compliance.json"))
        if report is None or receipt is None:
            continue
        rel = os.path.relpath(dirpath, results_root).replace(os.sep, "/")
        found.append(
            {
                "label": rel if rel != "." else os.path.basename(dirpath),
                "path": dirpath,
                "report": report,
                "receipt": receipt,
            }
        )
    return sorted(found, key=lambda item: item["label"])


def _grouping_key(receipt: dict[str, Any]) -> str:
    """Stable digest of the comparability key, so rankings never mix identities."""
    caps = receipt.get("comparability_key") or {}
    payload = json.dumps(caps, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def render_leaderboard(results: list[dict[str, Any]]) -> tuple[str, list[list[Any]]]:
    """Rank compliant results, one table per comparability key (Law 10)."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in results:
        groups.setdefault(_grouping_key(item["receipt"]), []).append(item)

    lines = [f"# {PROGRAM}: distribution-fidelity leaderboard", ""]
    csv_rows: list[list[Any]] = []
    for key, members in sorted(groups.items()):
        caps = members[0]["receipt"].get("comparability_key") or {}
        lines.append(f"## Comparability group `{key[:12]}`")
        lines.append("")
        lines.append(
            f"Reference `{_short(caps.get('reference_config_sha256'), 12)}`, "
            f"suite `{caps.get('suite_id')}`, "
            f"{caps.get('rows')} x {caps.get('context_length')} tokens, "
            f"score_from {caps.get('score_from')}, "
            f"vocabulary {caps.get('kld_vocab_size')}, "
            f"TP{caps.get('tensor_parallel_size')}, "
            f"runner {'V2' if caps.get('model_runner_v2') else 'V1'}, "
            f"torch {caps.get('torch')}, driver {caps.get('driver')}, "
            f"laws v{caps.get('laws_version')}."
        )
        lines.append("")
        ranked = sorted(
            members, key=lambda m: m["report"].get("mean_kld") or float("inf")
        )
        rows = []
        for item in ranked:
            report = item["report"]
            compliant = item["receipt"].get("compliant", False)
            status = "yes" if compliant else "NO"
            overridden = item["receipt"].get("overridden_laws") or []
            if compliant and overridden:
                status = f"yes (Law {', '.join(str(o) for o in overridden)} override)"
            rows.append(
                (
                    item["label"],
                    _kld(report.get("mean_kld")),
                    _kld(report.get("median_kld")),
                    _kld(report.get("p99_kld")),
                    _kld(report.get("max_kld")),
                    _pct(report.get("top1_agreement")),
                    str(report.get("num_positions")),
                    status,
                )
            )
            csv_rows.append([key[:12], item["label"], report.get("mean_kld"),
                             report.get("median_kld"), report.get("p99_kld"),
                             report.get("max_kld"), report.get("top1_agreement"),
                             report.get("num_positions"), compliant])
        lines += _table(
            rows,
            ("Candidate", "Mean KLD", "Median", "p99", "Max", "Top-1", "Positions",
             "Law-compliant"),
        )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n", csv_rows


def write_checksums(root: str, out_path: str) -> int:
    """Write a ``sha256sum --check`` compatible manifest over ``root``."""
    entries: list[tuple[str, str]] = []
    out_name = os.path.basename(out_path)
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            if rel == out_name:
                continue
            entries.append((rel, _sha256_file(full)))
    entries.sort()
    with open(out_path, "w", encoding="utf-8", newline="\n") as handle:
        for rel, digest in entries:
            handle.write(f"{digest}  {rel}\n")
    return len(entries)


def _write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    print(f"wrote {path}")


def _cmd_onepager(args: argparse.Namespace) -> int:
    report = _load(args.report)
    manifest = _load(args.manifest)
    receipt = _load(args.receipt)
    if report is None or manifest is None or receipt is None:
        print("report, manifest, and receipt are all required", file=sys.stderr)
        return 2
    runtime_env = {}
    if args.env_dir:
        runtime_env = _load(os.path.join(args.env_dir, "runtime.json")) or {}
    label = args.label or os.path.basename(str(report.get("student_model", "candidate")))
    text = render_onepager(
        report, manifest, receipt, runtime_env, _load(args.self_report), label
    )
    _write(args.out, text)
    return 0


def _cmd_leaderboard(args: argparse.Namespace) -> int:
    results = collect_results(args.results_root)
    if not results:
        print(f"no results with report.json + compliance.json under "
              f"{args.results_root}", file=sys.stderr)
        return 2
    text, csv_rows = render_leaderboard(results)
    _write(args.out, text)
    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        with open(args.csv, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["group", "candidate", "mean_kld", "median_kld",
                             "p99_kld", "max_kld", "top1_agreement", "positions",
                             "law_compliant"])
            writer.writerows(csv_rows)
        print(f"wrote {args.csv}")
    return 0


def _cmd_checksums(args: argparse.Namespace) -> int:
    out = args.out or os.path.join(args.root, "checksums.txt")
    count = write_checksums(args.root, out)
    print(f"wrote {out} covering {count} files")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    one = sub.add_parser("onepager", help="render one candidate's report.md")
    one.add_argument("--report", required=True)
    one.add_argument("--manifest", required=True)
    one.add_argument("--receipt", required=True, help="compliance receipt JSON")
    one.add_argument("--self-report", help="zero-baseline report JSON")
    one.add_argument("--env-dir")
    one.add_argument("--label")
    one.add_argument("--out", required=True)
    one.set_defaults(func=_cmd_onepager)

    board = sub.add_parser("leaderboard", help="rank results by comparability group")
    board.add_argument("--results-root", required=True)
    board.add_argument("--out", required=True)
    board.add_argument("--csv")
    board.set_defaults(func=_cmd_leaderboard)

    sums = sub.add_parser("checksums", help="hash every file in an artifact")
    sums.add_argument("--root", required=True)
    sums.add_argument("--out")
    sums.set_defaults(func=_cmd_checksums)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
