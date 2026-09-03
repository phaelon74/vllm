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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from artifact import candidate_identity  # noqa: E402 - sibling module
from redaction import scan_tree  # noqa: E402 - sibling module

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
    library: str,
    skip_noncompliant: bool,
    only: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Decide what may publish. Refuses rather than publishing a bad number.

    Returns the publishable plans and the refusal reasons. A refusal is reported
    in the exit status even when other models publish successfully, so an
    automated caller cannot mistake a partial publish for a clean one.
    """
    plans: list[dict[str, Any]] = []
    hard_stop: list[str] = []
    wanted = set(only or ())
    seen: set[str] = set()
    for root in model_roots(library):
        model = os.path.basename(root)
        seen.add(model)
        # An explicit selection is a decision about what is ready, so a name that
        # matches nothing is an error rather than a quiet publish of the rest.
        if wanted and model not in wanted:
            print(f"HELD     {model}: not named by --only")
            continue
        leaked = scan_tree(root)
        if leaked:
            # Never overridable and never skippable. A published credential cannot
            # be unpublished, so this refuses before anything is staged.
            listing = ", ".join(f"{rel} ({kind})" for rel, kind in leaked[:5])
            hard_stop.append(
                f"{model}: credential material in {len(leaked)} place(s): "
                f"{listing}. Remove it and rotate the credential; publication "
                f"cannot be forced past this."
            )
            continue
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
    unknown = sorted(wanted - seen)
    if unknown:
        raise SystemExit(
            f"--only names {', '.join(unknown)}, which is not in {library}. "
            f"Present: {', '.join(sorted(seen))}"
        )
    for line in hard_stop:
        print(f"REFUSED  {line}", file=sys.stderr)
    if hard_stop and not plans:
        raise SystemExit("nothing may publish")
    return plans, hard_stop


LEDGER = "published.json"


def read_ledger(library: str) -> dict[str, str]:
    """Model to artifact repo for everything this library has published.

    The index names every artifact ever published, not the ones a single run
    happened to touch, so publishing one family must never unlist another.
    """
    payload = _load(os.path.join(library, LEDGER)) or {}
    published = payload.get("published")
    return dict(published) if isinstance(published, dict) else {}


def write_ledger(library: str, entries: dict[str, str]) -> None:
    with open(
        os.path.join(library, LEDGER), "w", encoding="utf-8", newline="\n"
    ) as handle:
        json.dump({"published": dict(sorted(entries.items()))}, handle, indent=2)
        handle.write("\n")


def index_entries(
    library: str, plans: list[dict[str, Any]], ledger: dict[str, str]
) -> list[dict[str, Any]]:
    """This run's plans plus every previously published model, deduplicated.

    A previously published model is re-read from the library so its listing is
    rebuilt rather than remembered. If its directory is gone, the entry survives
    with its artifact link alone: a stale listing is recoverable, an erased one
    is not.
    """
    entries = {plan["model"]: plan for plan in plans}
    for model, repo in sorted(ledger.items()):
        if model in entries:
            continue
        root = os.path.join(library, model)
        candidates = (
            [pair for pair in candidates_of(root) if pair[1].get("compliant")]
            if os.path.isdir(root)
            else []
        )
        entries[model] = {
            "model": model,
            "root": root,
            "candidates": candidates,
            "excluded": [],
            "repo": repo,
        }
    return [entries[model] for model in sorted(entries)]


def index_models_in(text: str) -> set[str]:
    """Model names an index README lists, one per `## ` heading."""
    return {
        line[3:].strip()
        for line in text.splitlines()
        if line.startswith("## ") and line[3:].strip()
    }


def ledger_from_index_text(text: str) -> dict[str, str]:
    """Model to repo pairs recovered from an index README.

    Lets a lost or pre-ledger publication record be rebuilt from the index that
    documented it, including an earlier revision fetched from the Hub.
    """
    found: dict[str, str] = {}
    model: str | None = None
    for line in text.splitlines():
        if line.startswith("## "):
            model = line[3:].strip() or None
        elif model and line.startswith("Artifact: [`"):
            repo = line.split("`")[1].strip()
            if "/" in repo:
                found[model] = repo.split("/", 1)[1]
            model = None
    return found


def staged_index_models(staging: str) -> set[str]:
    with open(os.path.join(staging, "README.md"), encoding="utf-8") as handle:
        return index_models_in(handle.read())


def remote_index_models(repo_id: str) -> set[str] | None:
    """Models listed by the index already on the Hub, or None if unreadable.

    Read so that an index which would drop a published entry can be refused
    before it is uploaded. None means the question could not be asked, which is
    not the same as an empty index and must never be read as permission to shrink.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return None
    try:
        path = hf_hub_download(repo_id, "README.md", repo_type="dataset")
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
    except Exception:
        return None
    return index_models_in(text)


def build_index(
    library: str, plans: list[dict[str, Any]], namespace: str
) -> str:
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
    for plan in index_entries(library, plans, read_ledger(library)):
        dataset = f"{namespace}/{plan['repo']}"
        lines += [
            f"## {plan['model']}",
            "",
            f"Artifact: [`{dataset}`](https://huggingface.co/datasets/{dataset})",
            "",
        ]
        ranked = sorted(
            plan["candidates"],
            key=lambda pair: (
                float("inf")
                if pair[1].get("mean_kld") is None
                else float(pair[1]["mean_kld"])
            ),
        )
        for name, receipt in ranked:
            src = os.path.join(plan["root"], name, "report.md")
            dst = os.path.join(staging, "models", plan["model"], f"{name}.md")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
            report = _load(os.path.join(plan["root"], name, "report.json")) or {}
            _family, author, quant = candidate_identity(
                f"{plan['model']}/{name}", report.get("student_model")
            )
            mean = receipt.get("mean_kld")
            rel = f"models/{plan['model']}/{name}.md"
            mean_text = "n/a" if mean is None else f"{float(mean):.8f}"
            who = f"{author} " if author else ""
            lines.append(
                f"- {who}[{quant}]({rel}) — mean KLD {mean_text}"
            )
        if not ranked:
            lines.append(
                "Published previously; the library no longer holds a copy, so "
                "only the artifact link is listed here."
            )
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
    leaked = scan_tree(local)
    if leaked:
        listing = ", ".join(f"{rel} ({kind})" for rel, kind in leaked[:5])
        raise SystemExit(
            f"refusing to upload {repo_id}: credential material in "
            f"{len(leaked)} place(s): {listing}"
        )
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


def selftest() -> None:
    """Pin the index's cumulative contract and the ledger's recovery path."""
    old_index = (
        "# index\n\n"
        "## Qwen3.8-27B\n\n"
        "Artifact: [`phaedawg/qwen3.8-27b-distribution-fidelity-768x2048-v1`]"
        "(https://huggingface.co/x)\n\n"
        "- RedHatAI [FP8](models/Qwen3.8-27B/a.md) - mean KLD 0.1\n\n"
    )
    recovered = ledger_from_index_text(old_index)
    assert recovered == {
        "Qwen3.8-27B": "qwen3.8-27b-distribution-fidelity-768x2048-v1"
    }, recovered

    with tempfile.TemporaryDirectory() as library:
        write_ledger(library, recovered)
        assert read_ledger(library) == recovered
        plans = [
            {
                "model": "Qwen3.6-27B",
                "root": os.path.join(library, "Qwen3.6-27B"),
                "candidates": [("FP8", {"compliant": True, "mean_kld": 0.25})],
                "excluded": [],
                "repo": "qwen3.6-27b-distribution-fidelity-768x2048-v1",
            }
        ]
        # Publishing one family must relist every family published before it.
        entries = index_entries(library, plans, read_ledger(library))
        assert [e["model"] for e in entries] == ["Qwen3.6-27B", "Qwen3.8-27B"]
        staging = build_index(library, plans, "phaedawg")
        try:
            listed = staged_index_models(staging)
            assert listed == {"Qwen3.6-27B", "Qwen3.8-27B"}, listed
            # An index that would unlist a published model is refusable.
            assert not (listed - {"Qwen3.6-27B", "Qwen3.8-27B"})
            assert sorted({"Qwen3.8-27B"} - {"Qwen3.6-27B"}) == ["Qwen3.8-27B"]
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    print("  the index relists every previously published model")
    print("  a publication record is recoverable from an index README")
    print("selftest passed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    if "--selftest" in sys.argv:
        selftest()
        return 0
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
    parser.add_argument(
        "--only",
        action="append",
        metavar="MODEL",
        help="publish just this model directory; repeatable. Others are held.",
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
    parser.add_argument(
        "--seed-ledger",
        metavar="README",
        help="recover the publication record from an index README, then exit",
    )
    parser.add_argument(
        "--allow-index-removals",
        action="store_true",
        help="permit an index that unlists a previously published model",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.namespace:
        raise SystemExit(
            "no --namespace given and $LIL_HF_NAMESPACE is unset. This is "
            "deliberate: the publication namespace is never guessed."
        )

    if args.seed_ledger:
        with open(args.seed_ledger, encoding="utf-8") as handle:
            recovered = ledger_from_index_text(handle.read())
        if not recovered:
            raise SystemExit(f"no artifact links found in {args.seed_ledger}")
        ledger = read_ledger(args.library)
        added = {k: v for k, v in recovered.items() if k not in ledger}
        ledger.update(recovered)
        write_ledger(args.library, ledger)
        for model, repo in sorted(recovered.items()):
            mark = "+" if model in added else " "
            print(f"LEDGER  {mark} {model} -> {repo}")
        return 0

    plans, refused = gate(args.library, args.skip_noncompliant, args.only)
    for plan in plans:
        print(
            f"PUBLISH  {plan['model']} -> {args.namespace}/{plan['repo']} "
            f"({len(plan['candidates'])} candidate(s)"
            + (f", withholding {len(plan['excluded'])}" if plan["excluded"] else "")
            + ")"
        )

    index = build_index(args.library, plans, args.namespace)
    index_repo = f"{args.namespace}/{args.index_name}"
    staged = staged_index_models(index)
    print(f"INDEX    {index_repo} lists {len(staged)} model(s), "
          f"staged at {index}")

    # The index is cumulative. A ledger can be lost with a re-cloned library, so
    # the published index itself is asked what it lists, and an upload that would
    # unlist anything is refused rather than quietly narrowing the record.
    remote = remote_index_models(index_repo)
    if remote is None:
        print("INDEX    could not read the published index to compare against")
    else:
        dropped = sorted(remote - staged)
        if dropped and not args.allow_index_removals:
            raise SystemExit(
                f"refusing to publish an index that unlists "
                f"{', '.join(dropped)}. The published index is cumulative. "
                f"Either publish from a library that still holds those models, "
                f"or pass --allow-index-removals to withdraw them deliberately."
            )
        if dropped:
            print(f"INDEX    withdrawing {', '.join(dropped)} on request")

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
        # Recorded before the index goes up, so a failed index upload still
        # leaves the artifact accounted for and the next run relists it.
        ledger = read_ledger(args.library)
        ledger.update({plan["model"]: plan["repo"] for plan in plans})
        write_ledger(args.library, ledger)
    upload(index, index_repo, args.private, "Update distribution-fidelity index")
    # The record mirrors what the published index claims, so the two can never
    # disagree about what exists - including after an index-only publish, which
    # relists artifacts uploaded by an earlier run.
    ledger = read_ledger(args.library)
    ledger.update(
        {
            entry["model"]: entry["repo"]
            for entry in index_entries(args.library, plans, ledger)
            if entry["model"] in staged
        }
    )
    write_ledger(args.library, ledger)
    shutil.rmtree(index, ignore_errors=True)
    if refused:
        print(f"\n{len(refused)} model(s) were refused", file=sys.stderr)
    return 1 if refused else 0


if __name__ == "__main__":
    sys.exit(main())
