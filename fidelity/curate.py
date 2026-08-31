#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Curate a campaign's candidate list from the Hugging Face Hub.

A popularity-ranked sweep is how eleven quantized variants of one reference get
chosen without hand-googling. The Hub also hosts GGUF packs, MLX conversions,
and fine-tunes wearing a quant label, none of which are a precision comparison
against the reference. This tool filters to vLLM-loadable quantizations of the
unmodified base, ranks what remains by downloads, pins each repo to a commit,
and writes a campaign config `campaign.py` already knows how to load.

    python fidelity/curate.py
    python fidelity/curate.py --base Qwen/Qwen3.8-27B --picks \\
        fidelity/campaigns/picks/qwen3.8-27b.json --out \\
        fidelity/campaigns/qwen3.8-27b.json

With no `--base`, the tool asks for the reference, how candidates should be
chosen, where weights live, and whether to lease them. Flags still pre-answer
those prompts.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
USER_AGENT = "vllm-fidelity-curate/1"
HUB = "https://huggingface.co/api/models"

# Hub tags that mean "not a vLLM checkpoint".
REJECT_TAGS = frozenset(
    {
        "gguf",
        "mlx",
        "llama.cpp",
        "imatrix",
        "bitsandbytes",
        "ggml",
    }
)
# Name fragments of derivative weights. Matched as `-fragment` so an org named
# `unsloth` is not rejected for publishing a real quantization.
REJECT_NAME = (
    "-uncensored",
    "-abliterated",
    "-abliterate",
    "-heretic",
    "-obliterat",
    "-dflash",
    "-ninfer",
    "-whittle",
    "-apex",
    "-xyz",
    "-gguf",
    "-mlx",
    "bnb-4bit",
    "bnb-8bit",
    "-reap-",
    "-abliterated-",
)

FORMAT_ORDER = ("fp8", "nvfp4", "int4", "int8")
DEFAULT_MODELS_ROOT = "/media/fmodels2"
KNOWN_MODEL_ROOTS = ("/media/fmodels2", "/media/fmodels")


def _get(url: str) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def hub_search(query: str, limit: int = 200) -> list[dict[str, Any]]:
    params = urllib.parse.urlencode(
        {
            "search": query,
            "sort": "downloads",
            "direction": -1,
            "limit": limit,
        }
    )
    return _get(f"{HUB}?{params}")


def hub_model(repo: str, blobs: bool = False) -> dict[str, Any]:
    url = f"{HUB}/{urllib.parse.quote(repo, safe='/')}"
    if blobs:
        url += "?blobs=true"
    return _get(url)


def classify_format(model: dict[str, Any]) -> str | None:
    """Map a Hub card to fp8 / nvfp4 / int4 / int8, or None if unrecognized.

    FP8 is tested before INT8 so a `w8a8-fp8` card stays fp8.
    """
    tags = model.get("tags") or []
    blob = " ".join([model.get("id") or "", *tags]).lower()
    if any(t in tags for t in REJECT_TAGS):
        return None
    if "-gguf" in blob or blob.endswith("gguf") or "-mlx" in blob:
        return None
    if "nvfp4" in blob or re.search(r"(^|[^a-z])fp4([^0-9]|$)", blob):
        return "nvfp4"
    if re.search(r"(^|[^a-z])fp8([^0-9]|$)", blob):
        return "fp8"
    if any(
        token in blob
        for token in ("awq", "int4", "w4a16", "gptq", "autoround", "auto-round")
    ):
        return "int4"
    if any(token in blob for token in ("int8", "w8a8", "w8a16")):
        return "int8"
    return None


def rejected_name(repo_id: str) -> str | None:
    lower = repo_id.lower()
    _, _, name = lower.partition("/")
    haystack = f"-{name}" if not name.startswith("-") else name
    for marker in REJECT_NAME:
        if marker in haystack or marker in lower:
            return marker
    return None


def is_quant_of(model: dict[str, Any], base: str) -> bool:
    tags = model.get("tags") or []
    return (
        f"base_model:quantized:{base}" in tags
        or f"base_model:{base}" in tags
    ) and (model.get("id") != base)


def accept(model: dict[str, Any], base: str) -> str | None:
    """Why this card is rejected, or None if it is a usable candidate."""
    repo = model.get("id") or ""
    tags = set(model.get("tags") or [])
    hit = tags & REJECT_TAGS
    if hit:
        return f"tag {sorted(hit)[0]}"
    marker = rejected_name(repo)
    if marker:
        return f"name {marker}"
    if not is_quant_of(model, base):
        return "not a quantization of this base"
    if classify_format(model) is None:
        return "unrecognized format"
    return None


def unique_name(repo: str, taken: set[str]) -> str:
    """A campaign label: basename, or org-basename on collision."""
    org, _, name = repo.partition("/")
    if name not in taken:
        return name
    return f"{org}-{name}"


def candidate_entry(
    repo: str, sha: str, models_root: str, taken: set[str]
) -> dict[str, str]:
    org, _, name = repo.partition("/")
    label = unique_name(repo, taken)
    taken.add(label)
    taken.add(name)
    return {
        "name": label,
        "path": os.path.join(models_root, org, name).replace("\\", "/"),
        "hf_repo": repo,
        "revision": sha,
    }


def load_picks(path: str) -> list[str]:
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    if isinstance(raw, list):
        return [str(item) for item in raw]
    order = FORMAT_ORDER
    repos: list[str] = []
    for key in order:
        repos.extend(str(item) for item in raw.get(key) or [])
    extra = [k for k in raw if k not in order]
    for key in extra:
        if isinstance(raw[key], list):
            repos.extend(str(item) for item in raw[key])
    return repos


def resolve_repos(repos: list[str], base: str) -> list[dict[str, Any]]:
    """Fetch each repo's card, filter, and pin the commit."""
    resolved: list[dict[str, Any]] = []
    for repo in repos:
        try:
            card = hub_model(repo)
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            json.JSONDecodeError,
        ) as exc:
            print(f"skip {repo}: {exc}", file=sys.stderr)
            continue
        card.setdefault("id", repo)
        reason = accept(card, base)
        if reason:
            print(f"skip {repo}: {reason}", file=sys.stderr)
            continue
        sha = card.get("sha")
        if not sha:
            print(f"skip {repo}: Hub card has no sha", file=sys.stderr)
            continue
        resolved.append(card)
    return resolved


def sibling_bytes(card: dict[str, Any]) -> int | None:
    siblings = card.get("siblings")
    if not siblings:
        return None
    total = 0
    any_size = False
    for item in siblings:
        size = item.get("size")
        if size is None:
            continue
        any_size = True
        total += int(size)
    return total if any_size else None


def gib(nbytes: int | None) -> str:
    if nbytes is None:
        return "?"
    return f"{nbytes / (1 << 30):.1f}G"


def inventory_candidates(base: str) -> dict[str, list[dict[str, Any]]]:
    """Accepted Hub cards for this base, bucketed by format, ranked by downloads."""
    query = base.split("/")[-1]
    found = hub_search(query)
    buckets: dict[str, list[dict[str, Any]]] = {}
    seen: set[str] = set()
    for card in found:
        reason = accept(card, base)
        if reason:
            continue
        fmt = classify_format(card)
        if fmt is None:
            continue
        repo = card.get("id")
        if not repo or repo in seen:
            continue
        try:
            detail = hub_model(repo, blobs=True)
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            json.JSONDecodeError,
        ):
            continue
        if not detail.get("sha"):
            continue
        detail.setdefault("id", repo)
        seen.add(repo)
        buckets.setdefault(fmt, []).append(detail)
    return buckets


def fill_slots(
    buckets: dict[str, list[dict[str, Any]]], slots: dict[str, int]
) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    for fmt, cap in slots.items():
        chosen.extend(buckets.get(fmt, [])[:cap])
    return chosen


def even_split(
    total: int, formats: list[str], available: dict[str, int]
) -> dict[str, int]:
    """Divide total across formats; remainder goes to the more populous buckets."""
    formats = [fmt for fmt in formats if available.get(fmt, 0) > 0]
    if not formats or total <= 0:
        return {}
    base = total // len(formats)
    slots = {fmt: min(base, available[fmt]) for fmt in formats}
    leftover = total - sum(slots.values())
    ranked = sorted(formats, key=lambda fmt: available[fmt], reverse=True)
    while leftover > 0:
        progressed = False
        for fmt in ranked:
            if leftover <= 0:
                break
            if slots[fmt] < available[fmt]:
                slots[fmt] += 1
                leftover -= 1
                progressed = True
        if not progressed:
            break
    return slots


def search_candidates(
    base: str, slots: dict[str, int]
) -> list[dict[str, Any]]:
    """Fill per-format slots from a Hub search, ranked by downloads."""
    return fill_slots(inventory_candidates(base), slots)


def build_campaign(
    base: str,
    cards: list[dict[str, Any]],
    *,
    name: str,
    library: str,
    work: str,
    models_root: str,
    suite_dir: str | None,
    reference_path: str | None,
    fetch: str = "lease",
) -> dict[str, Any]:
    org, _, model_name = base.partition("/")
    ref_path = reference_path or os.path.join(models_root, org, model_name)
    reference: dict[str, str] = {
        "path": ref_path.replace("\\", "/"),
        "hf_repo": base,
    }
    try:
        ref_card = hub_model(base)
        if ref_card.get("sha"):
            reference["revision"] = ref_card["sha"]
    except (
        urllib.error.URLError,
        urllib.error.HTTPError,
        json.JSONDecodeError,
    ):
        pass
    taken: set[str] = {model_name}
    candidates = [
        candidate_entry(card["id"], card["sha"], models_root, taken)
        for card in cards
    ]
    campaign: dict[str, Any] = {
        "name": name,
        "library": library,
        "work": work,
        "suite_partition": "analysis",
        "partition": "analysis",
        "context_length": 2048,
        "score_from": 0,
        "runner_v2": False,
        "storage": "hidden",
        "max_num_seqs": 1,
        "fetch": fetch,
        "models": [
            {
                "name": model_name,
                "reference": reference,
                "candidates": candidates,
            }
        ],
    }
    if suite_dir:
        campaign["suite_dir"] = suite_dir
    return campaign


def parse_slots(raw: str) -> dict[str, int]:
    slots: dict[str, int] = {}
    for part in raw.split(","):
        key, _, value = part.partition("=")
        key = key.strip()
        if not key:
            continue
        slots[key] = int(value)
    return slots


def find_local_reference(base: str, models_root: str) -> list[str]:
    org, _, name = base.partition("/")
    hits: list[str] = []
    seen: set[str] = set()
    for root in (models_root, *KNOWN_MODEL_ROOTS):
        path = os.path.join(root, org, name).replace("\\", "/")
        if path in seen:
            continue
        seen.add(path)
        if os.path.isfile(os.path.join(path, "config.json")):
            hits.append(path)
    return hits


def describe_inventory(buckets: dict[str, list[dict[str, Any]]]) -> str:
    parts: list[str] = []
    seen: set[str] = set()
    for fmt in (*FORMAT_ORDER, *buckets):
        if fmt in seen or fmt not in buckets:
            continue
        seen.add(fmt)
        parts.append(f"{len(buckets[fmt])} {fmt}")
    return ", ".join(parts) if parts else "nothing that survived the filter"


def print_review(cards: list[dict[str, Any]]) -> None:
    print()
    print(f"{'#':>3}  {'format':<7} {'downloads':>10} {'size':>6}  {'sha':<12}  repo")
    for index, card in enumerate(cards, 1):
        fmt = classify_format(card) or "?"
        downloads = int(card.get("downloads") or 0)
        sha = (card.get("sha") or "")[:12]
        print(
            f"{index:3d}  {fmt:<7} {downloads:10d} {gib(sibling_bytes(card)):>6}  "
            f"{sha:<12}  {card.get('id')}"
        )


def drop_rows(cards: list[dict[str, Any]], raw: str) -> list[dict[str, Any]]:
    drop: set[int] = set()
    for part in raw.replace(" ", "").split(","):
        if not part:
            continue
        drop.add(int(part))
    return [card for index, card in enumerate(cards, 1) if index not in drop]


def disk_totals(
    cards: list[dict[str, Any]], ref_bytes: int | None
) -> tuple[int | None, int | None]:
    if ref_bytes is None:
        return None, None
    sizes: list[int] = []
    for card in cards:
        size = sibling_bytes(card)
        if size is None:
            return None, None
        sizes.append(size)
    total = ref_bytes + sum(sizes)
    peak = ref_bytes + (max(sizes) if sizes else 0)
    return total, peak


def write_campaign(path: str, campaign: dict[str, Any]) -> None:
    parent = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(campaign, handle, indent=2)
        handle.write("\n")


def print_next_steps(out: str) -> None:
    print()
    print("next:")
    print(f"  python fidelity/campaign.py download --config {out}")
    print(f"  python fidelity/campaign.py score --config {out}")
    print(f"  python fidelity/campaign.py assemble --config {out}")
    print("  # after an interrupted lease run:")
    print(f"  python fidelity/campaign.py release --config {out}")


def _ask(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    while True:
        try:
            raw = input(f"{prompt}{suffix}: ").strip()
        except EOFError as exc:
            raise SystemExit("stdin closed") from exc
        if raw:
            return raw
        if default is not None:
            return default
        print("a value is required")


def _ask_yes_no(prompt: str, default: bool = True) -> bool:
    hint = "Y/n" if default else "y/N"
    while True:
        try:
            raw = input(f"{prompt} [{hint}]: ").strip().lower()
        except EOFError as exc:
            raise SystemExit("stdin closed") from exc
        if not raw:
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("please answer yes or no")


def _resolve_base(raw: str) -> tuple[str, dict[str, Any]]:
    try:
        card = hub_model(raw, blobs=True)
    except (
        urllib.error.URLError,
        urllib.error.HTTPError,
        json.JSONDecodeError,
    ) as exc:
        raise ValueError(str(exc)) from exc
    repo = card.get("id") or raw
    sha = card.get("sha") or ""
    print(f"resolved {repo} @{sha[:12] or '?'}")
    return repo, card


def _choose_candidates(base: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.picks:
        return resolve_repos(load_picks(args.picks), base)

    mode = _ask(
        "Do you have a specific candidate you want to evaluate? "
        "Multiple candidates? Or have the pipeline pull X candidates "
        "from Hugging Face to review? [one / multiple / pull]",
        default="pull",
    ).lower()
    if mode in ("one", "1", "single"):
        repo = _ask("Candidate Hub repo (org/name)")
        return resolve_repos([repo], base)
    if mode in ("multiple", "multi", "many"):
        first = _ask(
            "Hub repo, or a path to a picks JSON (blank line ends a list)"
        )
        if os.path.isfile(first):
            repos = load_picks(first)
        else:
            repos = [first]
            while True:
                try:
                    line = input("  repo: ").strip()
                except EOFError:
                    break
                if not line:
                    break
                repos.append(line)
        return resolve_repos(repos, base)

    while True:
        raw = _ask("How many candidates should the pipeline pull?")
        try:
            count = int(raw)
        except ValueError:
            print("enter an integer")
            continue
        if count > 0:
            break
        print("enter a positive integer")

    print("searching the Hub...")
    buckets = inventory_candidates(base)
    print(f"found {describe_inventory(buckets)}")
    available = {fmt: len(cards) for fmt, cards in buckets.items()}
    present = [fmt for fmt in FORMAT_ORDER if available.get(fmt, 0)]
    present.extend(fmt for fmt in available if fmt not in FORMAT_ORDER)
    if args.slots:
        slots = parse_slots(args.slots)
    else:
        suggestion = ",".join(present[:2] or ["nvfp4", "int4"])
        split = _ask(
            f"Split {count} evenly across which formats? "
            f"(e.g. {suggestion}) or enter per-format counts "
            "(e.g. nvfp4=3,int4=3)",
            default=",".join(present) if present else suggestion,
        )
        if "=" in split:
            slots = parse_slots(split)
        else:
            formats = [part.strip() for part in split.split(",") if part.strip()]
            slots = even_split(count, formats, available)
    print("slots: " + ", ".join(f"{key}={value}" for key, value in slots.items()))
    return fill_slots(buckets, slots)


def wizard(args: argparse.Namespace) -> int:
    while True:
        raw = _ask("What is the base model we are working with?")
        try:
            base, ref_card = _resolve_base(raw)
            break
        except ValueError as exc:
            print(f"could not resolve {raw!r}: {exc}")

    cards = _choose_candidates(base, args)
    if not cards:
        print("no candidates survived the filter", file=sys.stderr)
        return 1

    models_root = args.models_root or _ask(
        "Where should the weights live?", default=DEFAULT_MODELS_ROOT
    )
    org, _, model_name = base.partition("/")
    example = os.path.join(models_root, org, model_name).replace("\\", "/")
    print(f"reference will land at {example}")

    reference_path = args.reference_path
    if not reference_path:
        hits = find_local_reference(base, models_root)
        if hits:
            print("already on disk:")
            for path in hits:
                print(f"  {path}")
            if _ask_yes_no(f"Reuse {hits[0]} instead of re-downloading?", default=True):
                reference_path = hits[0]

    if args.fetch:
        fetch = args.fetch
    else:
        lease = _ask_yes_no(
            "Delete each candidate's weights after scoring?", default=True
        )
        fetch = "lease" if lease else "upfront"

    while True:
        print_review(cards)
        total, peak = disk_totals(cards, sibling_bytes(ref_card))
        if total is not None:
            print(
                f"total download {gib(total)}; peak under leasing {gib(peak)} "
                "(reference + one candidate)"
            )
        else:
            print("Hub did not report sizes for every repo; totals unknown")
        action = _ask(
            "Drop rows by number (e.g. 3,5), or press enter to accept",
            default="",
        )
        if not action:
            break
        try:
            trimmed = drop_rows(cards, action)
        except ValueError:
            print("enter comma-separated row numbers")
            continue
        if not trimmed:
            print("that would leave zero candidates")
            continue
        cards = trimmed

    if not _ask_yes_no("Write this campaign?", default=True):
        print("aborted")
        return 1

    slug = base.split("/")[-1].lower()
    out = args.out or _ask(
        "Campaign JSON to write",
        default=os.path.join(HERE, "campaigns", f"{slug}.json"),
    )
    work = args.work or (
        "/media/fmodels2/Local-Inference-Lab/work/" + slug
    )
    campaign = build_campaign(
        base,
        cards,
        name=args.name or f"{slug}-fidelity",
        library=args.library,
        work=work,
        models_root=models_root,
        suite_dir=args.suite_dir,
        reference_path=reference_path,
        fetch=fetch,
    )
    write_campaign(out, campaign)
    print(f"wrote {out} with {len(cards)} candidate(s)")
    for cand in campaign["models"][0]["candidates"]:
        print(f"  {cand['name']:48s} {cand['hf_repo']} @{cand['revision'][:12]}")
    print_next_steps(out)
    return 0


def selftest() -> int:
    gguf = {
        "id": "unsloth/Qwen3.8-27B-GGUF",
        "tags": ["gguf", "base_model:quantized:Qwen/Qwen3.8-27B"],
    }
    assert accept(gguf, "Qwen/Qwen3.8-27B") == "tag gguf"
    fine_tune = {
        "id": "orcarouter/Qwen3.8-27B-Uncensored-NVFP4",
        "tags": ["nvfp4", "base_model:quantized:Qwen/Qwen3.8-27B"],
    }
    assert accept(fine_tune, "Qwen/Qwen3.8-27B") == "name -uncensored"
    official = {
        "id": "Qwen/Qwen3.8-27B-FP8",
        "tags": ["fp8", "base_model:quantized:Qwen/Qwen3.8-27B"],
    }
    assert accept(official, "Qwen/Qwen3.8-27B") is None
    assert classify_format(official) == "fp8"
    nvfp4 = {
        "id": "RadixArk/Qwen3.8-27B-NVFP4",
        "tags": ["NVFP4", "base_model:quantized:Qwen/Qwen3.8-27B"],
    }
    assert classify_format(nvfp4) == "nvfp4"
    int4 = {
        "id": "RedHatAI/Qwen3.8-27B-INT4",
        "tags": ["int4", "compressed-tensors", "base_model:quantized:Qwen/Qwen3.8-27B"],
    }
    assert classify_format(int4) == "int4"
    int8 = {
        "id": "org/Qwen3.8-27B-W8A8",
        "tags": ["int8", "w8a8", "base_model:quantized:Qwen/Qwen3.8-27B"],
    }
    assert classify_format(int8) == "int8"
    mixed = {
        "id": "org/Qwen3.8-27B-W8A8-FP8",
        "tags": ["fp8", "w8a8", "base_model:quantized:Qwen/Qwen3.8-27B"],
    }
    assert classify_format(mixed) == "fp8"
    taken: set[str] = set()
    a = candidate_entry("unsloth/Qwen3.8-27B-NVFP4", "abc", "/m", taken)
    b = candidate_entry("RadixArk/Qwen3.8-27B-NVFP4", "def", "/m", taken)
    assert a["name"] == "Qwen3.8-27B-NVFP4"
    assert b["name"] == "RadixArk-Qwen3.8-27B-NVFP4"
    assert even_split(5, ["nvfp4", "int4"], {"nvfp4": 14, "int4": 9}) == {
        "nvfp4": 3,
        "int4": 2,
    }
    assert even_split(6, ["nvfp4", "int4"], {"nvfp4": 14, "int4": 9}) == {
        "nvfp4": 3,
        "int4": 3,
    }
    filled = fill_slots(
        {"fp8": [official], "int8": [int8], "nvfp4": [nvfp4]},
        {"int8": 1, "fp8": 1},
    )
    assert [card["id"] for card in filled] == [int8["id"], official["id"]]
    dropped = drop_rows([official, nvfp4, int4], "2")
    assert [card["id"] for card in dropped] == [official["id"], int4["id"]]
    print("selftest passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", help="reference Hub repo, e.g. Qwen/Qwen3.8-27B")
    parser.add_argument(
        "--picks",
        help="JSON list of repos, or an object with fp8/nvfp4/int4 arrays",
    )
    parser.add_argument(
        "--slots",
        default=None,
        help="per-format caps when searching, e.g. fp8=1,nvfp4=5,int4=5",
    )
    parser.add_argument("--out", help="campaign JSON to write")
    parser.add_argument("--name", help="campaign name; default derived from --base")
    parser.add_argument(
        "--library",
        default="/media/fmodels2/Local-Inference-Lab/library",
    )
    parser.add_argument(
        "--work",
        default=None,
        help="work directory; default work/<base-name> under the library parent",
    )
    parser.add_argument("--models-root", default=None)
    parser.add_argument("--suite-dir", default=None)
    parser.add_argument("--reference-path", default=None)
    parser.add_argument(
        "--fetch",
        choices=("lease", "upfront"),
        default=None,
        help="lease downloads one candidate at a time and deletes it after scoring",
    )
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        return selftest()
    if not args.base:
        return wizard(args)
    if not args.out:
        parser.error(
            "--out is required (or run without --base for the interactive curator)"
        )

    slots = parse_slots(args.slots or "fp8=1,nvfp4=5,int4=5")
    if args.picks:
        repos = load_picks(args.picks)
        cards = resolve_repos(repos, args.base)
    else:
        cards = search_candidates(args.base, slots)

    if not cards:
        print("no candidates survived the filter", file=sys.stderr)
        return 1

    slug = args.base.split("/")[-1].lower()
    work = args.work or (
        "/media/fmodels2/Local-Inference-Lab/work/" + slug
    )
    campaign = build_campaign(
        args.base,
        cards,
        name=args.name or f"{slug}-fidelity",
        library=args.library,
        work=work,
        models_root=args.models_root or DEFAULT_MODELS_ROOT,
        suite_dir=args.suite_dir,
        reference_path=args.reference_path,
        fetch=args.fetch or "lease",
    )
    write_campaign(args.out, campaign)
    print(f"wrote {args.out} with {len(cards)} candidate(s)")
    for cand in campaign["models"][0]["candidates"]:
        print(f"  {cand['name']:48s} {cand['hf_repo']} @{cand['revision'][:12]}")
    print_next_steps(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
