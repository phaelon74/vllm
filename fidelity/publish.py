#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Publish a Local Inference Lab fidelity library to the Hugging Face Hub.

Two destinations, because the artifacts are large and the answers are small:

    <namespace>/<model>-distribution-fidelity-<rows>x<context>-v<n>
        One dataset repo per reference model, carrying the token suite, the
        reusable reference tensors, the LM head, the environment report, the
        zero-baseline proof, and every candidate's receipts. Tens of gigabytes.

    <namespace>/<index-name>
        One small repo carrying every one-pager, the leaderboard, and the laws,
        so a reader can find a model's mean KLD without downloading tensors.

Nothing uploads unless it is law-compliant and its checksums verify.

Usage:
    python fidelity/publish.py --library /mnt/kld/library --namespace your-hf-name
    python fidelity/publish.py --library /mnt/kld/library --namespace your-hf-name \\
        --dry-run
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INDEX = "distribution-fidelity-index"


def _load(path: str) -> dict[str, Any] | None:
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def verify_checksums(root: str) -> list[str]:
    """Verify ``checksums.txt``. Returns the list of problems, empty when clean."""
    path = os.path.join(root, "checksums.txt")
    if not os.path.isfile(path):
        return ["checksums.txt is absent"]
    problems: list[str] = []
    listed: set[str] = set()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            expected, _, rel = line.partition("  ")
            listed.add(rel)
            target = os.path.join(root, rel.replace("/", os.sep))
            if not os.path.isfile(target):
                problems.append(f"listed but missing: {rel}")
                continue
            digest = hashlib.sha256()
            with open(target, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                problems.append(f"hash mismatch: {rel}")
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            rel = os.path.relpath(
                os.path.join(dirpath, name), root
            ).replace(os.sep, "/")
            if rel != "checksums.txt" and rel not in listed:
                problems.append(f"present but unlisted: {rel}")
    return problems


def model_roots(library: str) -> list[str]:
    """Directories that look like a model root: they hold a checksums.txt."""
    roots = []
    for entry in sorted(os.listdir(library)):
        path = os.path.join(library, entry)
        if os.path.isdir(path) and os.path.isfile(os.path.join(path, "checksums.txt")):
            roots.append(path)
    return roots


def candidates_of(model_root: str) -> list[tuple[str, dict[str, Any]]]:
    """Candidate folders under a model root, with their compliance receipts."""
    found = []
    for entry in sorted(os.listdir(model_root)):
        path = os.path.join(model_root, entry)
        receipt = _load(os.path.join(path, "compliance.json"))
        if receipt is not None:
            found.append((entry, receipt))
    return found


def repo_name(model: str, receipt: dict[str, Any], version: int) -> str:
    """Name the dataset repo after the identity that bounds its numbers."""
    caps = receipt.get("comparability_key") or {}
    rows = caps.get("rows") or "?"
    context = caps.get("context_length") or "?"
    return f"{model}-distribution-fidelity-{rows}x{context}-v{version}".lower()


def gate(
    library: str, skip_noncompliant: bool
) -> tuple[list[dict[str, Any]], list[str]]:
    """Decide what may publish. Refuses rather than publishing a bad number.

    Returns the publishable plans and the refusal reasons. A refusal is reported
    in the exit status even when other models publish successfully, so an
    automated caller cannot mistake a partial publish for a clean one.
    """
    plans: list[dict[str, Any]] = []
    hard_stop: list[str] = []
    for root in model_roots(library):
        model = os.path.basename(root)
        problems = verify_checksums(root)
        if problems:
            hard_stop.append(
                f"{model}: checksums do not verify ({len(problems)} problem(s)); "
                f"first: {problems[0]}"
            )
            continue
        entries = candidates_of(root)
        if not entries:
            hard_stop.append(f"{model}: no candidate carries a compliance receipt")
            continue
        bad = [name for name, receipt in entries if not receipt.get("compliant")]
        if bad and not skip_noncompliant:
            hard_stop.append(
                f"{model}: not law-compliant: {', '.join(bad)}. Fix them, or pass "
                f"--skip-noncompliant to publish only the compliant candidates."
            )
            continue
        good = [(name, r) for name, r in entries if r.get("compliant")]
        if not good:
            hard_stop.append(f"{model}: no compliant candidate to publish")
            continue
        plans.append(
            {
                "model": model,
                "root": root,
                "candidates": good,
                "excluded": bad,
                "repo": repo_name(model, good[0][1], 1),
            }
        )
    for line in hard_stop:
        print(f"REFUSED  {line}", file=sys.stderr)
    if hard_stop and not plans:
        raise SystemExit("nothing may publish")
    return plans, hard_stop


def build_index(library: str, plans: list[dict[str, Any]], namespace: str) -> str:
    """Stage the small, findable index: one-pagers, leaderboard, laws."""
    staging = tempfile.mkdtemp(prefix="lil-index-")
    shutil.copy2(os.path.join(HERE, "LAWS.md"), os.path.join(staging, "LAWS.md"))
    for name in ("leaderboard.md", "leaderboard.csv"):
        src = os.path.join(library, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(staging, name))

    lines = [
        "# Local Inference Lab: distribution-fidelity index",
        "",
        "Every published one-pager, with a pointer to the full artifact that "
        "backs it. Each artifact repo carries the token suite and the reusable "
        "reference tensors, so a third party can score a new candidate without "
        "loading the reference checkpoint.",
        "",
        "Read [`LAWS.md`](LAWS.md) before comparing anything here. Numbers are "
        "comparable only within a single artifact's suite, geometry, and runtime "
        "identity.",
        "",
    ]
    for plan in plans:
        dataset = f"{namespace}/{plan['repo']}"
        lines += [
            f"## {plan['model']}",
            "",
            f"Artifact: [`{dataset}`](https://huggingface.co/datasets/{dataset})",
            "",
        ]
        for name, receipt in plan["candidates"]:
            src = os.path.join(plan["root"], name, "report.md")
            dst = os.path.join(staging, "models", plan["model"], f"{name}.md")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
            mean = receipt.get("mean_kld")
            rel = f"models/{plan['model']}/{name}.md"
            mean_text = "n/a" if mean is None else f"{float(mean):.8f}"
            lines.append(f"- [{name}]({rel}) — mean KLD {mean_text}")
        if plan["excluded"]:
            lines += [
                "",
                f"Withheld as not law-compliant: {', '.join(plan['excluded'])}.",
            ]
        lines.append("")
    with open(os.path.join(staging, "README.md"), "w", encoding="utf-8",
              newline="\n") as handle:
        handle.write("\n".join(lines).rstrip() + "\n")
    return staging


def upload(local: str, repo_id: str, private: bool, message: str) -> None:
    """Create the dataset repo if needed and upload a folder to it."""
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is not installed in this interpreter. Run "
            "fidelity/bootstrap.sh, or: uv pip install 'huggingface_hub[cli]'"
        ) from exc
    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=local,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=message,
    )
    print(f"published https://huggingface.co/datasets/{repo_id}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, help="assembled library root")
    parser.add_argument(
        "--namespace",
        default=os.environ.get("LIL_HF_NAMESPACE"),
        help="Hugging Face user or org; defaults to $LIL_HF_NAMESPACE",
    )
    parser.add_argument(
        "--index-name",
        default=DEFAULT_INDEX,
        help=f"repo holding the one-pagers and leaderboard (default {DEFAULT_INDEX})",
    )
    parser.add_argument("--private", action="store_true")
    parser.add_argument(
        "--skip-noncompliant",
        action="store_true",
        help="publish a model's compliant candidates and withhold the rest",
    )
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="publish only the small index, not the multi-gigabyte artifacts",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.namespace:
        raise SystemExit(
            "no --namespace given and $LIL_HF_NAMESPACE is unset. This is "
            "deliberate: the publication namespace is never guessed."
        )

    plans, refused = gate(args.library, args.skip_noncompliant)
    for plan in plans:
        print(
            f"PUBLISH  {plan['model']} -> {args.namespace}/{plan['repo']} "
            f"({len(plan['candidates'])} candidate(s)"
            + (f", withholding {len(plan['excluded'])}" if plan["excluded"] else "")
            + ")"
        )

    index = build_index(args.library, plans, args.namespace)
    index_repo = f"{args.namespace}/{args.index_name}"
    print(f"INDEX    {index_repo} staged at {index}")

    if args.dry_run:
        print("\ndry run: nothing uploaded")
        return 1 if refused else 0

    if not args.index_only:
        for plan in plans:
            upload(
                plan["root"],
                f"{args.namespace}/{plan['repo']}",
                args.private,
                f"Publish {plan['model']} distribution-fidelity artifact",
            )
    upload(index, index_repo, args.private, "Update distribution-fidelity index")
    shutil.rmtree(index, ignore_errors=True)
    if refused:
        print(f"\n{len(refused)} model(s) were refused", file=sys.stderr)
    return 1 if refused else 0


if __name__ == "__main__":
    sys.exit(main())
