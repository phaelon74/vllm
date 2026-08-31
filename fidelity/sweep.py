#!/usr/bin/env python3
"""Report and reclaim campaign scratch space that the published library does not use.

The library is the index. A work file earns its keep by being byte-identical to
something published, or by being a cache the next campaign would otherwise pay
for again. Everything else is scratch from a run that no longer has a reader.

Matching is by content digest rather than by path, because a report is copied
into the library under a different name than it carries in the work tree, and a
run assembled from the wrong config leaves files whose names look perfectly
plausible. Nothing under the library is ever a deletion candidate.

    python fidelity/sweep.py --library /path/to/library
    python fidelity/sweep.py --library /path/to/library --delete-stale
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GIB = 1024.0**3


def digest(path: str) -> str | None:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                h.update(block)
    except OSError:
        return None
    return h.hexdigest()


def tree_stats(path: str) -> tuple[int, int]:
    """Bytes this tree occupies, and how many of them are shared with elsewhere.

    Counted per inode rather than per name. Assembly publishes the reference by
    hard link when the work directory and the library sit on one filesystem, so
    summing file sizes double-counts tens of gigabytes and makes a cache that
    costs nothing extra look like the largest thing on the disk. Shared bytes
    survive deletion here because the library still names them.
    """
    total = 0
    shared = 0
    seen: set[tuple[int, int]] = set()
    for root, _, names in os.walk(path):
        for name in names:
            try:
                st = os.lstat(os.path.join(root, name))
            except OSError:
                continue
            key = (st.st_dev, st.st_ino)
            if key in seen:
                continue
            seen.add(key)
            total += st.st_size
            if st.st_nlink > 1:
                shared += st.st_size
    return total, shared


def tree_size(path: str) -> int:
    return tree_stats(path)[0]


def published_digests(library: str) -> set[str]:
    """Every content digest the library publishes, at any depth."""
    seen: set[str] = set()
    for root, _, names in os.walk(library):
        for name in names:
            found = digest(os.path.join(root, name))
            if found:
                seen.add(found)
    return seen


def published_state(library: str) -> tuple[list[str], list[str]]:
    """Candidate directories that carry a receipt, split by whether it passed.

    A sweep is only as trustworthy as the index it sweeps against. If a
    published candidate is non-compliant or missing its receipt then the library
    is mid-flight, "the latest run" is not yet a fact, and deleting the scratch
    behind it would remove the only way to finish.
    """
    compliant: list[str] = []
    broken: list[str] = []
    for receipt in sorted(glob.glob(os.path.join(library, "*", "*", "report.json"))):
        cand = os.path.dirname(receipt)
        path = os.path.join(cand, "compliance.json")
        try:
            with open(path, encoding="utf-8") as handle:
                ok = bool(json.load(handle).get("compliant"))
        except (OSError, json.JSONDecodeError):
            broken.append(cand)
            continue
        (compliant if ok else broken).append(cand)
    return compliant, broken


def classify(work: str, published: set[str]) -> dict[str, object]:
    """What one work tree holds, relative to the published index."""
    reports = sorted(glob.glob(os.path.join(work, "reports", "*.json")))
    live = [p for p in reports if digest(p) in published]
    orphans = [p for p in reports if p not in live]

    captures = [
        p
        for p in sorted(glob.glob(os.path.join(work, "captures", "*")))
        if os.path.isdir(p)
    ]
    # A variant whose weights are gone is already pruned: its QDQ manifest is
    # provenance for a published cell and weighs nothing.
    variants = [
        p
        for p in sorted(glob.glob(os.path.join(work, "variants", "*")))
        if os.path.isdir(p) and glob.glob(os.path.join(p, "*.safetensors"))
    ]
    return {
        "work": work,
        "exists": os.path.isdir(work),
        "size": tree_size(work) if os.path.isdir(work) else 0,
        "reports": reports,
        "live": live,
        "orphans": orphans,
        "captures": captures,
        "variants": variants,
        # No live report means nothing in this tree was ever published from here.
        # That is the shape a run assembled from the wrong config leaves behind.
        "stale": os.path.isdir(work) and not live,
    }


def work_dirs(configs: list[str]) -> dict[str, list[str]]:
    """Work directory to the configs that claim it."""
    claimed: dict[str, list[str]] = {}
    for path in configs:
        try:
            with open(path, encoding="utf-8") as handle:
                work = json.load(handle).get("work")
        except (OSError, json.JSONDecodeError):
            continue
        if work:
            claimed.setdefault(os.path.abspath(work), []).append(
                os.path.basename(path)
            )
    return claimed


def gib(size: int) -> str:
    return f"{size / GIB:8.2f} GiB"


def describe(tree: dict, owners: list[str]) -> None:
    work = str(tree["work"])
    if not tree["exists"]:
        print(f"  {work}\n    absent; claimed by {', '.join(owners)}")
        return
    state = "STALE" if tree["stale"] else "live"
    print(f"  [{state}] {work}  {gib(int(tree['size']))}")
    print(f"    claimed by: {', '.join(owners) or 'no config'}")
    print(
        f"    reports: {len(tree['reports'])} "
        f"({len(tree['live'])} published, {len(tree['orphans'])} unpublished)"
    )
    if tree["stale"]:
        print("    nothing here was published; the whole tree is scratch")
        return
    captures = [str(p) for p in tree["captures"]]  # type: ignore[union-attr]
    variants = [str(p) for p in tree["variants"]]  # type: ignore[union-attr]
    if captures:
        stats = [tree_stats(p) for p in captures]
        size = sum(s for s, _ in stats)
        shared = sum(h for _, h in stats)
        plural = "dir" if len(captures) == 1 else "dirs"
        print(f"    captures: {len(captures)} {plural} {gib(size)}  (reusable cache)")
        if shared:
            print(
                f"      {gib(shared)} of that is hard-linked to the published "
                f"library and would not be freed by deleting it"
            )
    if variants:
        size = sum(tree_size(p) for p in variants)
        print(f"    variants: {len(variants)} unpruned {gib(size)}  (rebuildable)")
    for path in tree["orphans"]:  # type: ignore[union-attr]
        print(f"      unpublished report: {os.path.basename(str(path))}")


def remove(path: str, live: bool) -> int:
    """Delete a file or tree, returning the bytes actually reclaimed.

    Bytes another name still points at are not reclaimed by unlinking this one,
    so a hard-linked reference does not count toward the total.
    """
    if os.path.isdir(path):
        total, shared = tree_stats(path)
        size = total - shared
    else:
        st = os.lstat(path)
        size = 0 if st.st_nlink > 1 else st.st_size
    if live:
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
    return size


def prune_variant(path: str, live: bool) -> int:
    """Drop a variant's weights, keeping its QDQ manifest as provenance."""
    freed = 0
    for name in sorted(os.listdir(path)):
        if name == "qdq-manifest.json":
            continue
        target = os.path.join(path, name)
        if os.path.isfile(target):
            freed += remove(target, live)
    return freed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--library", required=True, help="published library root")
    ap.add_argument(
        "--config",
        action="append",
        default=[],
        help="campaign config; repeatable, defaults to fidelity/campaigns/*.json",
    )
    ap.add_argument(
        "--work-root",
        help="also scan this directory for work trees no config claims",
    )
    ap.add_argument(
        "--delete-stale",
        action="store_true",
        help="remove work trees that published nothing",
    )
    ap.add_argument(
        "--prune-variants",
        action="store_true",
        help="drop QDQ weights in live trees, keeping each manifest",
    )
    ap.add_argument(
        "--delete-captures",
        action="store_true",
        help="remove reference captures in live trees; forces a recapture later",
    )
    args = ap.parse_args()

    library = os.path.abspath(args.library)
    if not os.path.isdir(library):
        print(f"no library at {library}", file=sys.stderr)
        return 2

    configs = args.config or sorted(
        glob.glob(os.path.join(HERE, "campaigns", "*.json"))
    )
    claimed = work_dirs(configs)
    if args.work_root:
        for path in sorted(glob.glob(os.path.join(args.work_root, "*"))):
            if os.path.isdir(path):
                claimed.setdefault(os.path.abspath(path), [])

    compliant, broken = published_state(library)
    print(f"library: {library}")
    print(f"  {len(compliant)} compliant candidate(s) published")
    for cand in broken:
        print(f"  !!! no passing receipt: {cand}")
    acting = args.delete_stale or args.prune_variants or args.delete_captures
    if broken and acting:
        print(
            "\nrefusing to delete: the library has a candidate without a passing "
            "receipt, so the latest run is not settled yet. Finish or remove it "
            "first.",
            file=sys.stderr,
        )
        return 1

    print("\nindexing published content...")
    published = published_digests(library)
    print(f"  {len(published)} distinct published file digests")

    print("\nwork trees:")
    trees = []
    for work, owners in sorted(claimed.items()):
        if work == library or work.startswith(library + os.sep):
            continue
        tree = classify(work, published)
        trees.append(tree)
        describe(tree, owners)

    freed = 0
    for tree in trees:
        work = str(tree["work"])
        if tree["stale"] and tree["exists"]:
            freed += remove(work, args.delete_stale)
            print(
                f"{'removed' if args.delete_stale else 'would remove'} stale tree "
                f"{work}"
            )
            continue
        if args.prune_variants or not acting:
            for path in tree["variants"]:  # type: ignore[union-attr]
                freed += prune_variant(str(path), args.prune_variants)
                print(
                    f"{'pruned' if args.prune_variants else 'would prune'} weights "
                    f"in {path}"
                )
        if args.delete_captures:
            for path in tree["captures"]:  # type: ignore[union-attr]
                freed += remove(str(path), True)
                print(f"removed capture {path}")

    verb = "reclaimed" if acting else "reclaimable"
    print(f"\n{verb}: {gib(freed)}")
    if not acting:
        print(
            "dry run; nothing was deleted. Add --delete-stale to drop trees that "
            "published nothing, --prune-variants to drop rebuildable QDQ weights, "
            "--delete-captures to drop the reusable reference cache."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
