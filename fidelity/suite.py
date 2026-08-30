#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mint a frozen token-ID evaluation suite for distribution-fidelity work.

Replicates the Kimi K3 distribution-fidelity recipe — the same sources at the same
pinned revisions, the same ten allocation strata and counts, the same dedup,
leakage scan, and analysis/qualification split — but emits token IDs for the
tokenizer you name. Token IDs are not portable across tokenizers, so a suite must
be minted per tokenizer family; the methodology is what transfers. Where the
reference cannot be followed exactly, the recipe's ``upstream.deviations`` records
what changed and why.

The output is the evaluation input itself (Law 3): candidates consume the stored
IDs directly. Retokenizing source text does not reproduce the suite.

Usage:
    python fidelity/suite.py selftest
    python fidelity/suite.py probe

    python fidelity/suite.py build \\
        --recipe fidelity/suites/recipe-v1.json \\
        --tokenizer /media/fmodels/Qwen/Qwen3.6-27B \\
        --out /mnt/kld/suites/qwen3.6-1024x2048-v1

    python fidelity/suite.py verify --suite /mnt/kld/suites/qwen3.6-1024x2048-v1
"""

import argparse
import hashlib
import json
import os
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator

HERE = os.path.dirname(os.path.abspath(__file__))
MERSENNE_PRIME = (1 << 61) - 1
FINGERPRINT_DOMAIN = "local-inference-lab/dataset-identity/v1"
_WHITESPACE = re.compile(r"\s+")


def sha256_tokens(tokens: list[int]) -> str:
    """Hash a token sequence the way the scorer and capture manifest do."""
    payload = ",".join(str(t) for t in tokens).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _digest_int(*parts: str) -> int:
    payload = "|".join(parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def dataset_fingerprint(source: dict[str, Any]) -> str:
    """SHA-256 over the domain-separated dataset identity.

    Domain separation keeps a dataset name from colliding with a revision or a
    split when they are concatenated.
    """
    parts = [
        FINGERPRINT_DOMAIN,
        f"dataset={source['dataset']}",
        f"revision={source['revision']}",
        f"config={source.get('config') or ''}",
        f"split={source.get('split') or ''}",
    ]
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def normalize_for_scan(text: str) -> str:
    """Casefold, NFKC-normalize, and collapse whitespace for overlap testing."""
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", text).casefold()).strip()


def minhash(tokens: list[int], permutations: int, shingle: int) -> list[int]:
    """MinHash signature over token shingles, for near-duplicate detection.

    Implemented directly rather than pulled from a dependency so the signature is
    a stable, documented function of the token IDs: a suite rebuilt years later
    must dedup identically.
    """
    if len(tokens) < shingle:
        grams = [tuple(tokens)]
    else:
        grams = [
            tuple(tokens[i: i + shingle])
            for i in range(len(tokens) - shingle + 1)
        ]
    hashed = [
        int.from_bytes(
            hashlib.sha1(",".join(map(str, g)).encode("utf-8")).digest()[:8], "big"
        )
        for g in set(grams)
    ]
    signature = []
    for p in range(permutations):
        a = _digest_int("minhash-a", str(p)) | 1
        b = _digest_int("minhash-b", str(p))
        signature.append(min(((a * h + b) % MERSENNE_PRIME) for h in hashed))
    return signature


def jaccard(left: list[int], right: list[int]) -> float:
    matches = sum(1 for a, b in zip(left, right) if a == b)
    return matches / len(left) if left else 0.0


@dataclass
class Document:
    """One candidate context: a coherent source unit and its chosen window."""

    source_key: str
    cluster_id: str
    representation: str
    content_sha256: str
    normalized_text: str
    tokens: list[int]
    token_offset: int
    total_tokens: int
    title: str | None = None
    path: str | None = None
    signature: list[int] = field(default_factory=list)

    @property
    def token_sha256(self) -> str:
        return sha256_tokens(self.tokens)


def _render_conversation(value: Any) -> str:
    """Render a chat record as role-tagged turns.

    The rendering is part of the extraction policy and is recorded as such: a
    different rendering produces different tokens and therefore a different suite.
    """
    if not isinstance(value, list):
        return ""
    turns = []
    for turn in value:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role", "")).strip()
        content = str(turn.get("content", "")).strip()
        if content:
            turns.append(f"{role}: {content}" if role else content)
    return "\n\n".join(turns)


def _field(record: Any, dotted: str | None) -> Any:
    """Read a possibly nested record field, addressed as ``metadata.path``."""
    if not dotted:
        return None
    value: Any = record
    for part in dotted.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _field_any(record: Any, spec: str | list[str] | None) -> Any:
    """Read the first field present out of one name or a list of alternatives.

    Datasets in one lineage carry the same content under different column names -
    ``content`` here, ``text`` there, ``max_stars_repo_path`` versus ``path`` - and
    a gated repository cannot be inspected before a build. Listing the
    alternatives keeps a rename from being a build failure. The first present name
    wins, so the choice is deterministic for a given corpus.
    """
    if spec is None or isinstance(spec, str):
        return _field(record, spec)
    for name in spec:
        value = _field(record, name)
        if value not in (None, ""):
            return value
    return None


def _text_field(record: dict[str, Any], spec: str | list[str] | None) -> str | None:
    value = _field_any(record, spec)
    return str(value) if value not in (None, "") else None


def extract(record: dict[str, Any], source: dict[str, Any]) -> str:
    extractor = source.get("extractor", "plain")
    raw = _field_any(record, source["text_field"])
    if extractor == "conversation":
        return _render_conversation(raw)
    return raw if isinstance(raw, str) else ""


def _resolve_data_files(source: dict[str, Any]) -> list[str]:
    """Expand a source's data-file spec into a sorted, deterministic list.

    Some Hub repositories are reachable only through a loader script, which newer
    ``datasets`` refuses to execute. Naming their data files directly reads the
    same bytes at the same pinned revision that the script would have read.
    """
    revision = source["revision"]
    template = source.get("data_files_template")
    if template:
        shards = int(source["data_files_shards"])
        files = [
            template.format(revision=revision, index=index, shards=shards)
            for index in range(shards)
        ]
    else:
        spec = source["data_files"]
        candidates = [spec] if isinstance(spec, str) else list(spec)
        files = [str(f).format(revision=revision) for f in candidates]
    return sorted(files)


def iter_records(
    source: dict[str, Any], fixture_dir: str | None
) -> Iterator[dict[str, Any]]:
    """Stream a source's records, from a local fixture or the Hub."""
    if fixture_dir:
        path = os.path.join(fixture_dir, f"{source['key']}.jsonl")
        if not os.path.isfile(path):
            return
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "the datasets package is required to build from the Hub; run "
            "fidelity/bootstrap.sh or: uv pip install datasets"
        ) from exc
    if source.get("loader"):
        stream = load_dataset(
            source["loader"],
            data_files=_resolve_data_files(source),
            split=source.get("split", "train"),
            streaming=True,
        )
    else:
        stream = load_dataset(
            source["dataset"],
            source.get("config"),
            revision=source["revision"],
            split=source.get("split", "train"),
            streaming=True,
        )
    yield from stream


def harvest(
    source: dict[str, Any],
    tokenizer: Any,
    context_length: int,
    retain: int,
    dedup: dict[str, Any],
    fixture_dir: str | None,
) -> list[Document]:
    """Collect validated candidate contexts from one source.

    Validation is structural: the document must yield at least ``context_length``
    tokens, must not be mostly whitespace, and must come from a source unit not
    already represented. One context per coherent source unit, so a long article
    cannot dominate a stratum.
    """
    min_chars = int(source.get("min_chars", 4000))
    seen_clusters: set[str] = set()
    docs: list[Document] = []
    scanned = 0
    for record in iter_records(source, fixture_dir):
        scanned += 1
        if scanned > int(source.get("scan_limit", 20000)):
            break
        if len(docs) >= retain:
            break
        path = _text_field(record, source.get("path_field"))
        if not _suffix_eligible(path, source.get("path_suffix_any")):
            continue
        text = extract(record, source)
        if not text or len(text) < min_chars:
            continue
        if len(_WHITESPACE.sub("", text)) < min_chars // 2:
            continue
        cluster = _field_any(record, source.get("cluster_field"))
        cluster_id = str(cluster) if cluster not in (None, "") else _sha256_text(text)
        if cluster_id in seen_clusters:
            continue
        normalized = normalize_for_scan(text)
        if not content_eligible(source, path, normalized):
            continue

        encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
        if encoded and isinstance(encoded[0], list):
            encoded = encoded[0]
        if len(encoded) < context_length:
            continue

        content_sha = _sha256_text(text)
        span = len(encoded) - context_length + 1
        offset = _digest_int("token-offset", content_sha) % span
        window = list(encoded[offset: offset + context_length])
        seen_clusters.add(cluster_id)
        docs.append(
            Document(
                source_key=source["key"],
                cluster_id=cluster_id,
                representation=source.get("extractor", "plain"),
                content_sha256=content_sha,
                normalized_text=normalized,
                tokens=window,
                token_offset=offset,
                total_tokens=len(encoded),
                title=_text_field(record, source.get("title_field")),
                path=path,
                signature=minhash(
                    window,
                    dedup["minhash_permutations"],
                    dedup["shingle_tokens"],
                ),
            )
        )
    print(
        f"  {source['key']}: scanned {scanned}, retained {len(docs)} "
        f"candidate context(s)"
    )
    return docs


CACHE_FORMAT_VERSION = 1


def _cache_key(*parts: Any) -> str:
    payload = json.dumps(parts, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _cache_paths(cache_dir: str, name: str, key: str) -> tuple[str, str]:
    stem = os.path.join(cache_dir, f"{name}-{key}")
    return f"{stem}.jsonl", f"{stem}.meta.json"


def _write_cache(data_path: str, meta_path: str, rows: list[Any],
                 meta: dict[str, Any]) -> None:
    """Write a cache entry so an interrupted write cannot be mistaken for data."""
    os.makedirs(os.path.dirname(os.path.abspath(data_path)), exist_ok=True)
    tmp = f"{data_path}.tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    os.replace(tmp, data_path)
    _write_json(meta_path, meta)


def _read_cache(data_path: str, meta_path: str) -> tuple[list[Any], dict[str, Any]]:
    with open(meta_path, encoding="utf-8") as handle:
        meta = json.load(handle)
    rows = []
    with open(data_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows, meta


def _document_row(doc: Document) -> dict[str, Any]:
    return {
        "source_key": doc.source_key,
        "cluster_id": doc.cluster_id,
        "representation": doc.representation,
        "content_sha256": doc.content_sha256,
        "normalized_text": doc.normalized_text,
        "tokens": doc.tokens,
        "token_offset": doc.token_offset,
        "total_tokens": doc.total_tokens,
        "title": doc.title,
        "path": doc.path,
        "signature": doc.signature,
    }


def harvest_cached(
    source: dict[str, Any],
    tokenizer: Any,
    context_length: int,
    retain: int,
    dedup: dict[str, Any],
    fixture_dir: str | None,
    tok_identity: dict[str, Any],
    cache_dir: str | None,
    refresh: set[str],
) -> list[Document]:
    """Harvest a source, reusing a cached pool when the inputs are unchanged.

    Harvesting is the expensive, network-bound half of a build, and a build that
    fails late used to discard every completed source. The cache is keyed by
    everything that can change a pool, so a rerun after a fix re-harvests only what
    the fix touched.
    """
    if not cache_dir:
        return harvest(source, tokenizer, context_length, retain, dedup, fixture_dir)

    key = _cache_key(
        CACHE_FORMAT_VERSION, source, tok_identity, context_length, dedup, fixture_dir
    )
    data_path, meta_path = _cache_paths(cache_dir, f"pool-{source['key']}", key)
    if (
        source["key"] not in refresh
        and os.path.isfile(data_path)
        and os.path.isfile(meta_path)
    ):
        rows, meta = _read_cache(data_path, meta_path)
        if int(meta.get("retain", 0)) >= retain:
            docs = [Document(**row) for row in rows][:retain]
            print(
                f"  {source['key']}: reusing {len(docs)} cached candidate "
                f"context(s) from {os.path.basename(data_path)}"
            )
            return docs

    docs = harvest(source, tokenizer, context_length, retain, dedup, fixture_dir)
    _write_cache(
        data_path,
        meta_path,
        [_document_row(doc) for doc in docs],
        {
            "cache_format_version": CACHE_FORMAT_VERSION,
            "source_key": source["key"],
            "cache_key": key,
            "dataset": source["dataset"],
            "revision": source["revision"],
            "retain": retain,
            "documents": len(docs),
            "context_length": context_length,
            "written_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return docs


def _suffix_eligible(path: str | None, suffixes: list[str] | None) -> bool:
    if not suffixes:
        return True
    lowered = (path or "").lower()
    return any(lowered.endswith(str(s).lower()) for s in suffixes)


def content_eligible(
    spec: dict[str, Any], path: str | None, normalized_text: str
) -> bool:
    """Apply a content policy declared by either a source or a stratum.

    A source declares its policy to make a rare file type reachable: the filter
    runs while streaming, so the scan digs as deep as ``scan_limit`` allows rather
    than hoping the first few thousand records happen to contain one. A stratum
    declares the same policy to state what its slots are allowed to hold.
    """
    if not _suffix_eligible(path, spec.get("path_suffix_any")):
        return False
    required = spec.get("require_any")
    if required and not any(marker in normalized_text for marker in required):
        return False
    return True


def stratum_eligible(doc: Document, stratum: dict[str, Any]) -> bool:
    """Apply a stratum's content policy to a candidate."""
    return content_eligible(stratum, doc.path, doc.normalized_text)


def select(
    strata: list[dict[str, Any]],
    pools: dict[str, list[Document]],
    dedup: dict[str, Any],
    blocked: set[str],
    namespaces: dict[str, str] | None = None,
) -> tuple[dict[str, list[Document]], list[str]]:
    """Fill each stratum, rejecting duplicates and blocked documents.

    Sources are drawn round-robin so a stratum backed by several sources is not
    dominated by whichever one streamed first. Exact token duplicates and
    near-duplicates are rejected before allocation, as the recipe requires,
    which means a rejection here changes which document lands in a slot rather
    than leaving a slot empty.

    Harvesting keeps one context per source unit within a source. Some sources
    share an identifier space - three of them index the same GitHub repositories -
    so units are also deduplicated across sources that declare the same
    ``cluster_namespace``, which keeps one repository from contributing twice.
    """
    chosen: dict[str, list[Document]] = {}
    accepted_tokens: set[str] = set()
    accepted_clusters: set[tuple[str, str]] = set()
    accepted_sigs: list[list[int]] = []
    shortfalls: list[str] = []
    threshold = dedup["jaccard_threshold"]
    namespaces = namespaces or {}

    for stratum in strata:
        want = stratum["contexts"]
        keys = stratum["sources"]
        cursors = {key: 0 for key in keys}
        picked: list[Document] = []
        exhausted = False
        while len(picked) < want and not exhausted:
            exhausted = True
            for key in keys:
                if len(picked) >= want:
                    break
                pool = pools.get(key, [])
                while cursors[key] < len(pool):
                    doc = pool[cursors[key]]
                    cursors[key] += 1
                    exhausted = False
                    if doc.content_sha256 in blocked:
                        continue
                    if not stratum_eligible(doc, stratum):
                        continue
                    unit = (namespaces.get(key, key), doc.cluster_id)
                    if unit in accepted_clusters:
                        continue
                    token_hash = doc.token_sha256
                    if token_hash in accepted_tokens:
                        continue
                    if any(
                        jaccard(doc.signature, sig) >= threshold
                        for sig in accepted_sigs
                    ):
                        continue
                    accepted_tokens.add(token_hash)
                    accepted_clusters.add(unit)
                    accepted_sigs.append(doc.signature)
                    picked.append(doc)
                    break
        chosen[stratum["key"]] = picked
        if len(picked) < want:
            shortfalls.append(
                f"{stratum['key']}: {len(picked)} of {want} contexts "
                f"(sources: {', '.join(keys)})"
            )
    return chosen, shortfalls


def benchmark_items(
    spec: dict[str, Any], fixture_dir: str | None
) -> list[tuple[str, list[str]]]:
    """Normalized fragments per benchmark item.

    An item is a list of fragments plus a match rule. Under ``all``, every
    fragment must appear for the item to count as leaked: for MMLU that means the
    question and all of its answer choices, so a short general question occurring
    naturally in reference prose is not mistaken for benchmark contamination.
    """
    items: list[tuple[str, list[str]]] = []
    for index, record in enumerate(iter_records({**spec, "key": spec["name"]},
                                                fixture_dir)):
        fragments: list[str] = []
        for name in spec["fields"]:
            value = record.get(name)
            if isinstance(value, list):
                fragments += [normalize_for_scan(str(v)) for v in value]
            elif value:
                fragments.append(normalize_for_scan(str(value)))
        fragments = [f for f in fragments if len(f) >= 24]
        if fragments:
            items.append((f"{spec['name']}:{index}", fragments))
    return items


def benchmark_items_cached(
    spec: dict[str, Any],
    fixture_dir: str | None,
    cache_dir: str | None,
    refresh: set[str],
) -> list[tuple[str, list[str]]]:
    if not cache_dir:
        return benchmark_items(spec, fixture_dir)
    key = _cache_key(CACHE_FORMAT_VERSION, spec, fixture_dir)
    data_path, meta_path = _cache_paths(cache_dir, f"items-{spec['name']}", key)
    if (
        spec["name"] not in refresh
        and os.path.isfile(data_path)
        and os.path.isfile(meta_path)
    ):
        rows, _ = _read_cache(data_path, meta_path)
        return [(row[0], row[1]) for row in rows]
    items = benchmark_items(spec, fixture_dir)
    _write_cache(
        data_path,
        meta_path,
        [[item_id, fragments] for item_id, fragments in items],
        {
            "cache_format_version": CACHE_FORMAT_VERSION,
            "benchmark": spec["name"],
            "cache_key": key,
            "dataset": spec["dataset"],
            "revision": spec["revision"],
            "items": len(items),
            "written_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return items


_WORD = re.compile(r"\w+", re.UNICODE)


def _probe_words(fragment: str) -> list[str]:
    """Distinctive interior words of a normalized fragment.

    Interior only. A fragment's first and last words can be glued to neighbouring
    characters in a document, but an interior word is delimited by non-word
    characters inside the fragment itself, so those same delimiters appear in any
    document containing the fragment verbatim. That makes the index a necessary
    condition rather than a heuristic: no containment can be missed.
    """
    words = _WORD.findall(fragment)
    interior = words[1:-1]
    for floor in (8, 6, 4):
        chosen = sorted(
            {word for word in interior if len(word) >= floor},
            key=lambda word: (-len(word), word),
        )
        if chosen:
            return chosen[:2]
    return []


def _item_index(
    items: list[tuple[str, list[str]]], match_rule: str
) -> tuple[dict[str, set[int]], set[int]]:
    """Index items by probe word, listing those that must be checked directly.

    Under ``all`` every fragment is required, so probing the longest one is enough.
    Under ``any`` a hit on any fragment counts, so every fragment is probed.
    """
    index: dict[str, set[int]] = {}
    always: set[int] = set()
    for position, (_item_id, fragments) in enumerate(items):
        if not fragments:
            continue
        probe_sources = (
            [max(fragments, key=len)] if match_rule == "all" else fragments
        )
        probes: set[str] = set()
        for fragment in probe_sources:
            words = _probe_words(fragment)
            if not words:
                probes.clear()
                break
            probes.update(words)
        if probes:
            for word in probes:
                index.setdefault(word, set()).add(position)
        else:
            always.add(position)
    return index, always


def item_matches(text: str, fragments: list[str], match_rule: str) -> bool:
    if match_rule == "all":
        return all(fragment in text for fragment in fragments)
    return any(fragment in text for fragment in fragments)


def scan_items(
    docs: list[Document], items: list[tuple[str, list[str]]], match_rule: str
) -> dict[int, list[Document]]:
    """Which documents contain which items, keyed by item position.

    Documents are visited in order, so each item's hit list keeps corpus order.
    """
    index, always = _item_index(items, match_rule)
    hits: dict[int, list[Document]] = {}
    for doc in docs:
        words = set(_WORD.findall(doc.normalized_text))
        candidates = set(always)
        for word in words & index.keys():
            candidates |= index[word]
        for position in candidates:
            if item_matches(doc.normalized_text, items[position][1], match_rule):
                hits.setdefault(position, []).append(doc)
    return hits


def leakage_scan(
    recipe: dict[str, Any],
    pools: dict[str, list[Document]],
    fixture_dir: str | None,
    cache_dir: str | None = None,
    refresh: set[str] | None = None,
) -> tuple[dict[str, Any], set[str]]:
    """Scan candidates for complete benchmark items and block any that overlap.

    Comparing every item against every candidate is quadratic, and MMLU alone
    contributes over fourteen thousand items: the direct form does not finish. An
    inverted index over probe words reduces it to a containment check on the few
    items a document could possibly hold, without weakening the match rule.
    """
    blocked: set[str] = set()
    per_benchmark = []
    all_docs = [doc for pool in pools.values() for doc in pool]
    for spec in recipe.get("benchmarks", []):
        items = benchmark_items_cached(spec, fixture_dir, cache_dir, refresh or set())
        match_rule = spec.get("match", "any")
        hits_by_item = scan_items(all_docs, items, match_rule)
        hits = []
        for position in sorted(hits_by_item):
            for doc in hits_by_item[position]:
                hits.append({"item": items[position][0], "content_sha256":
                             doc.content_sha256, "source": doc.source_key})
                blocked.add(doc.content_sha256)
        per_benchmark.append(
            {
                "name": spec["name"],
                "dataset": spec["dataset"],
                "revision": spec["revision"],
                "config": spec.get("config"),
                "split": spec.get("split"),
                "match_rule": spec.get("match", "any"),
                "items_scanned": len(items),
                "overlaps": hits,
            }
        )
        print(
            f"  {spec['name']}: {len(items)} item(s) scanned, {len(hits)} "
            f"complete overlap(s)"
        )
    report = {
        "kind": "capability overlap scan",
        "normalization": "NFKC, casefold, whitespace-collapsed exact containment",
        "prefilter": (
            "inverted index over interior probe words of each item fragment, a "
            "necessary condition for containment, verified by exact containment"
        ),
        "candidates_scanned": len(all_docs),
        "benchmarks": per_benchmark,
        "blocked_documents": sorted(blocked),
    }
    return report, blocked


def assign_partitions(
    chosen: dict[str, list[Document]], recipe: dict[str, Any]
) -> dict[str, Any]:
    """Split each stratum into analysis and qualification, then mark sentinels.

    The split is stratified and deterministic: it is a function of document
    content, not of iteration order or a random seed, so a rebuilt suite assigns
    the same contexts to the same partition.
    """
    total = recipe["contexts"]
    qualification_share = recipe["partitions"]["qualification"] / total
    sentinel_share = recipe.get("sentinel_candidates", 0) / max(
        recipe["partitions"]["analysis"], 1
    )
    analysis: list[int] = []
    qualification: list[int] = []
    sentinels: list[int] = []
    by_stratum: dict[str, dict[str, list[int]]] = {}

    context_id = 0
    ids_by_stratum: dict[str, list[tuple[int, Document]]] = {}
    for stratum in recipe["strata"]:
        entries = []
        for doc in chosen[stratum["key"]]:
            entries.append((context_id, doc))
            context_id += 1
        ids_by_stratum[stratum["key"]] = entries

    for stratum in recipe["strata"]:
        entries = ids_by_stratum[stratum["key"]]
        ordered = sorted(
            entries, key=lambda e: _digest_int("partition", e[1].content_sha256)
        )
        want_qual = round(len(ordered) * qualification_share)
        qual = [cid for cid, _ in ordered[:want_qual]]
        ana = [cid for cid, _ in ordered[want_qual:]]
        ana_ordered = sorted(
            [e for e in ordered if e[0] in set(ana)],
            key=lambda e: _digest_int("sentinel", e[1].content_sha256),
        )
        want_sentinel = round(len(ana) * sentinel_share)
        strat_sentinels = [cid for cid, _ in ana_ordered[:want_sentinel]]
        analysis += ana
        qualification += qual
        sentinels += strat_sentinels
        by_stratum[stratum["key"]] = {
            "analysis": sorted(ana),
            "qualification": sorted(qual),
            "sentinel_candidates": sorted(strat_sentinels),
        }

    return {
        "kind": "deterministic partition assignment",
        "method": "per-stratum ordering by SHA-256 of domain-separated content hash",
        "analysis": sorted(analysis),
        "qualification": sorted(qualification),
        "sentinel_candidates": sorted(sentinels),
        "sentinel_policy": (
            "Sentinel repeat captures are not part of a normal campaign. Law 1 "
            "requires the reference to reproduce itself exactly, which sets the "
            "noise floor at zero. These contexts are the pre-agreed set to "
            "capture three times if a Law 1 override is ever approved."
        ),
        "by_stratum": by_stratum,
    }


def _write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)
        handle.write("\n")


def tokenizer_identity(tokenizer: Any, path: str) -> dict[str, Any]:
    return {
        "name_or_path": path,
        "class": type(tokenizer).__name__,
        "vocab_size": int(getattr(tokenizer, "vocab_size", 0)) or None,
        "unpadded_vocab_size": int(
            getattr(tokenizer, "actual_vocab_size", 0)
            or getattr(tokenizer, "vocab_size", 0)
        )
        or None,
    }


def write_suite(
    out_dir: str,
    recipe: dict[str, Any],
    chosen: dict[str, list[Document]],
    partitions: dict[str, Any],
    overlap: dict[str, Any],
    tok_identity: dict[str, Any],
    suite_id: str,
) -> dict[str, Any]:
    """Write tokens, provenance, partitions, and the suite manifest."""
    tokens_dir = os.path.join(out_dir, "tokens")
    os.makedirs(tokens_dir, exist_ok=True)

    contexts: list[dict[str, Any]] = []
    sources_detail: list[dict[str, Any]] = []
    by_source = {s["key"]: s for s in recipe["sources"]}
    context_id = 0
    all_tokens: list[int] = []

    for stratum in recipe["strata"]:
        for doc in chosen[stratum["key"]]:
            source = by_source[doc.source_key]
            name = f"context-{context_id:04d}.json"
            token_hash = doc.token_sha256
            _write_json(
                os.path.join(tokens_dir, name),
                {
                    "context_id": context_id,
                    "tokens": doc.tokens,
                    "token_sha256": token_hash,
                },
            )
            contexts.append(
                {
                    "context_id": context_id,
                    "file": f"tokens/{name}",
                    "stratum": stratum["key"],
                    "source_key": doc.source_key,
                    "source_namespace": (
                        source.get("cluster_namespace") or source["key"]
                    ),
                    "source_cluster_id": doc.cluster_id,
                    "token_count": len(doc.tokens),
                    "scored_positions": len(doc.tokens) - 1,
                    "token_sha256": token_hash,
                }
            )
            sources_detail.append(
                {
                    "context_id": context_id,
                    "stratum": stratum["key"],
                    "source_key": doc.source_key,
                    "dataset": source["dataset"],
                    "dataset_revision": source["revision"],
                    "dataset_config": source.get("config"),
                    "dataset_split": source.get("split"),
                    "license": source.get("license"),
                    "source_namespace": (
                        source.get("cluster_namespace") or source["key"]
                    ),
                    "source_cluster_id": doc.cluster_id,
                    "title": doc.title,
                    "path": doc.path,
                    "extraction_policy": doc.representation,
                    "content_sha256": doc.content_sha256,
                    "document_token_count": doc.total_tokens,
                    "deterministic_token_offset": doc.token_offset,
                    "token_count": len(doc.tokens),
                    "token_sha256": token_hash,
                }
            )
            all_tokens.extend(doc.tokens)
            context_id += 1

    order = {c["context_id"]: c["token_sha256"] for c in contexts}
    tokens_by_id = {
        c["context_id"]: json.load(
            open(os.path.join(out_dir, c["file"]), encoding="utf-8")
        )["tokens"]
        for c in contexts
    }

    def partition_hash(ids: list[int]) -> str:
        flat: list[int] = []
        for cid in sorted(ids):
            flat.extend(tokens_by_id[cid])
        return sha256_tokens(flat)

    suite_token_sha256 = sha256_tokens(all_tokens)
    manifest = {
        "kind": "Local Inference Lab distribution-fidelity token suite",
        "suite_id": suite_id,
        "recipe_id": recipe["recipe_id"],
        "format_version": recipe.get("format_version", 1),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "context_count": len(contexts),
        "context_length": recipe["context_length"],
        "scored_positions_per_context": recipe["context_length"] - 1,
        "total_scored_positions": len(contexts) * (recipe["context_length"] - 1),
        "tokenizer": tok_identity,
        "token_sha256": suite_token_sha256,
        "token_hash_method": (
            "sha256 of comma-joined token IDs, contexts concatenated in "
            "context_id order"
        ),
        "partition_token_sha256": {
            "all": suite_token_sha256,
            "analysis": partition_hash(partitions["analysis"]),
            "qualification": partition_hash(partitions["qualification"]),
        },
        "source_cluster_count": len(
            {(c["source_namespace"], c["source_cluster_id"]) for c in contexts}
        ),
        "source_cluster_identity": (
            "a source unit is the pair (source_namespace, source_cluster_id). The "
            "namespace matters because identifier spaces are per-source: article "
            "id 12 in the German Wikipedia is a different article from id 12 in "
            "the English one, while a repository name means the same repository in "
            "every source that indexes GitHub"
        ),
        "strata": [
            {
                "key": s["key"],
                "label": s["label"],
                "requested": s["contexts"],
                "contexts": len(chosen[s["key"]]),
            }
            for s in recipe["strata"]
        ],
        "contexts": contexts,
    }
    assert len(order) == len(contexts)

    _write_json(os.path.join(out_dir, "suite-manifest.json"), manifest)
    _write_json(os.path.join(out_dir, "sources.json"), sources_detail)
    _write_json(
        os.path.join(out_dir, "source-registry.json"),
        {
            "kind": "Local Inference Lab distribution-fidelity source registry",
            "dataset_fingerprint_method": (
                "SHA-256 over the domain-separated dataset repository, immutable "
                "revision, configuration, and split identity"
            ),
            "fingerprint_domain": FINGERPRINT_DOMAIN,
            "sources": [
                {
                    **{
                        k: v
                        for k, v in source.items()
                        if k
                        in (
                            "key", "dataset", "config", "revision", "split",
                            "scan_limit", "license", "extractor", "loader",
                            "data_files", "data_files_template",
                            "data_files_shards", "path_suffix_any", "min_chars",
                            "cluster_namespace", "upstream_dataset_fingerprint",
                        )
                    },
                    "dataset_fingerprint": dataset_fingerprint(source),
                }
                for source in recipe["sources"]
            ],
        },
    )
    _write_json(os.path.join(out_dir, "partitions.json"), partitions)
    _write_json(
        os.path.join(out_dir, "validation", "capability-overlap.json"), overlap
    )
    return manifest


def build(
    recipe_path: str,
    tokenizer_path: str,
    out_dir: str,
    fixture_dir: str | None,
    oversample: int,
    suite_id: str | None,
    allow_shortfall: bool,
    cache_dir: str | None = None,
    refresh: set[str] | None = None,
) -> int:
    with open(recipe_path, encoding="utf-8") as handle:
        recipe = json.load(handle)
    refresh = refresh or set()

    declared = sum(s["contexts"] for s in recipe["strata"])
    if declared != recipe["contexts"]:
        raise SystemExit(
            f"recipe is inconsistent: strata sum to {declared} but contexts is "
            f"{recipe['contexts']}"
        )

    if fixture_dir:
        tokenizer = _FixtureTokenizer()
        tok_identity = {"name_or_path": "fixture", "class": "_FixtureTokenizer",
                        "vocab_size": 32000, "unpadded_vocab_size": 32000}
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        tok_identity = tokenizer_identity(tokenizer, tokenizer_path)

    need: dict[str, int] = {}
    for stratum in recipe["strata"]:
        for key in stratum["sources"]:
            need[key] = need.get(key, 0) + stratum["contexts"]

    print(f"=== harvesting {len(recipe['sources'])} source(s)")
    if cache_dir:
        print(f"  cache: {cache_dir}")
    pools: dict[str, list[Document]] = {}
    for source in recipe["sources"]:
        want = need.get(source["key"], 0)
        if want == 0:
            continue
        pools[source["key"]] = harvest_cached(
            source,
            tokenizer,
            recipe["context_length"],
            want * int(source.get("oversample") or oversample),
            recipe["dedup"],
            fixture_dir,
            tok_identity,
            cache_dir,
            refresh,
        )

    print("=== scanning for benchmark leakage")
    overlap, blocked = leakage_scan(recipe, pools, fixture_dir, cache_dir, refresh)
    if blocked:
        print(f"  blocking {len(blocked)} candidate(s) with complete overlaps")

    print("=== allocating strata")
    chosen, shortfalls = select(
        recipe["strata"],
        pools,
        recipe["dedup"],
        blocked,
        {
            s["key"]: s.get("cluster_namespace") or s["key"]
            for s in recipe["sources"]
        },
    )
    for stratum in recipe["strata"]:
        print(
            f"  {stratum['key']}: {len(chosen[stratum['key']])} of "
            f"{stratum['contexts']}"
        )
    if shortfalls and not allow_shortfall:
        for line in shortfalls:
            print(f"SHORT  {line}", file=sys.stderr)
        raise SystemExit(
            "the suite is short of its allocation. Raise --oversample or a "
            "source's scan_limit, or pass --allow-shortfall to mint a smaller "
            "suite whose manifest records the true counts."
        )

    partitions = assign_partitions(chosen, recipe)
    resolved_id = suite_id or (
        f"{os.path.basename(tokenizer_path.rstrip('/\\')) or 'suite'}-"
        f"fidelity-{sum(len(v) for v in chosen.values())}x"
        f"{recipe['context_length']}-v1"
    ).lower()

    print("=== writing suite")
    manifest = write_suite(
        out_dir, recipe, chosen, partitions, overlap, tok_identity, resolved_id
    )
    print(
        f"\nsuite {manifest['suite_id']}: {manifest['context_count']} contexts, "
        f"{manifest['total_scored_positions']} scored positions, "
        f"{manifest['source_cluster_count']} source cluster(s)"
    )
    print(f"suite token sha256: {manifest['token_sha256']}")
    print(f"analysis: {len(partitions['analysis'])}  "
          f"qualification: {len(partitions['qualification'])}")
    print(f"written to {out_dir}")
    return 0


def _probe_source(
    source: dict[str, Any], fixture_dir: str | None, rows: int
) -> dict[str, Any]:
    """Stream a bounded prefix of one source and report what it actually holds."""
    limit = rows
    if source.get("path_suffix_any"):
        # A filtered source is looking for a rare file type, so a short prefix
        # says nothing about whether the filter can ever be satisfied.
        limit = min(int(source.get("scan_limit", 20000)), rows * 50)
    min_chars = int(source.get("min_chars", 4000))
    columns: set[str] = set()
    extensions: dict[str, int] = {}
    scanned = 0
    with_text = 0
    eligible = 0
    for record in iter_records(source, fixture_dir):
        scanned += 1
        if isinstance(record, dict):
            columns.update(record.keys())
        path = _text_field(record, source.get("path_field"))
        if path:
            suffix = os.path.splitext(path)[1].lower() or "(none)"
            extensions[suffix] = extensions.get(suffix, 0) + 1
        if _suffix_eligible(path, source.get("path_suffix_any")):
            text = extract(record, source)
            if text:
                with_text += 1
                if len(text) >= min_chars and content_eligible(
                    source, path, normalize_for_scan(text)
                ):
                    eligible += 1
        if scanned >= limit:
            break
    return {
        "scanned": scanned,
        "with_text": with_text,
        "eligible": eligible,
        "columns": sorted(columns),
        "extensions": sorted(extensions.items(), key=lambda kv: (-kv[1], kv[0])),
    }


def probe(recipe_path: str, fixture_dir: str | None, rows: int) -> int:
    """Check every source and benchmark is reachable and shaped as declared.

    A harvest costs minutes per source and a leakage scan needs every benchmark,
    so an unreachable repository, a renamed column, or a revision that no longer
    exists must surface here rather than after the expensive work.
    """
    with open(recipe_path, encoding="utf-8") as handle:
        recipe = json.load(handle)

    problems: list[str] = []
    print(f"=== probing {len(recipe['sources'])} source(s)")
    for source in recipe["sources"]:
        try:
            result = _probe_source(source, fixture_dir, rows)
        except Exception as exc:  # noqa: BLE001 - report, do not abort the sweep
            problems.append(
                f"{source['key']} ({source['dataset']}): {type(exc).__name__}: {exc}"
            )
            print(f"  {source['key']}: UNREACHABLE  {type(exc).__name__}: {exc}")
            continue
        print(
            f"  {source['key']}: scanned {result['scanned']}, "
            f"{result['with_text']} with text, {result['eligible']} eligible"
        )
        print(f"    columns: {', '.join(result['columns'][:12]) or 'none'}")
        if result["extensions"]:
            top = ", ".join(f"{ext} {n}" for ext, n in result["extensions"][:6])
            print(f"    extensions: {top}")
        if result["scanned"] == 0:
            problems.append(f"{source['key']}: yielded no records")
        elif result["with_text"] == 0:
            names = source["text_field"]
            names = names if isinstance(names, str) else " or ".join(names)
            problems.append(
                f"{source['key']}: no record carried text field {names}"
            )
        elif result["eligible"] == 0:
            problems.append(
                f"{source['key']}: no record in {result['scanned']} passed its own "
                f"filters, so its strata cannot fill"
            )

    print(f"=== probing {len(recipe.get('benchmarks', []))} benchmark(s)")
    for spec in recipe.get("benchmarks", []):
        source = {**spec, "key": spec["name"], "text_field": spec["fields"][0]}
        try:
            result = _probe_source({**source, "min_chars": 1}, fixture_dir, rows)
        except Exception as exc:  # noqa: BLE001 - report, do not abort the sweep
            problems.append(
                f"{spec['name']} ({spec['dataset']}): {type(exc).__name__}: {exc}"
            )
            print(f"  {spec['name']}: UNREACHABLE  {type(exc).__name__}: {exc}")
            continue
        missing = [f for f in spec["fields"] if f not in result["columns"]]
        print(
            f"  {spec['name']}: scanned {result['scanned']}, fields present: "
            f"{'yes' if not missing else 'missing ' + ', '.join(missing)}"
        )
        if result["scanned"] == 0:
            problems.append(f"{spec['name']}: yielded no records")
        if missing:
            problems.append(
                f"{spec['name']}: missing field(s) {', '.join(missing)}; the "
                f"leakage scan would silently find nothing"
            )

    for line in problems:
        print(f"PROBLEM  {line}", file=sys.stderr)
    if problems:
        print(f"{len(problems)} problem(s); build would fail", file=sys.stderr)
        return 1
    print("\nprobe passed: every source and benchmark is reachable and shaped")
    return 0


def verify(suite_dir: str) -> int:
    """Re-derive every hash in a suite from its own token files."""
    manifest_path = os.path.join(suite_dir, "suite-manifest.json")
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)

    problems: list[str] = []
    flat: list[int] = []
    per_id: dict[int, list[int]] = {}
    for entry in manifest["contexts"]:
        path = os.path.join(suite_dir, entry["file"].replace("/", os.sep))
        if not os.path.isfile(path):
            problems.append(f"missing {entry['file']}")
            continue
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        tokens = payload["tokens"]
        if len(tokens) != manifest["context_length"]:
            problems.append(
                f"{entry['file']}: {len(tokens)} tokens, expected "
                f"{manifest['context_length']}"
            )
        digest = sha256_tokens(tokens)
        if digest != entry["token_sha256"]:
            problems.append(f"{entry['file']}: token hash mismatch")
        if payload.get("token_sha256") != entry["token_sha256"]:
            problems.append(f"{entry['file']}: self-recorded hash disagrees")
        per_id[entry["context_id"]] = tokens
        flat.extend(tokens)

    if sha256_tokens(flat) != manifest["token_sha256"]:
        problems.append("suite token hash mismatch")

    units = [
        (entry.get("source_namespace") or entry["source_key"],
         entry["source_cluster_id"])
        for entry in manifest["contexts"]
    ]
    if len(set(units)) != len(units):
        repeated = len(units) - len(set(units))
        problems.append(
            f"{repeated} context(s) come from a source unit that already "
            f"contributed one"
        )
    recorded = manifest.get("source_cluster_count")
    if recorded is not None and recorded != len(set(units)):
        problems.append(
            f"source_cluster_count says {recorded} but the contexts hold "
            f"{len(set(units))} distinct units"
        )

    partitions_path = os.path.join(suite_dir, "partitions.json")
    if os.path.isfile(partitions_path):
        with open(partitions_path, encoding="utf-8") as handle:
            partitions = json.load(handle)
        for name in ("analysis", "qualification"):
            expected = manifest.get("partition_token_sha256", {}).get(name)
            ids = partitions.get(name, [])
            actual_tokens: list[int] = []
            for cid in sorted(ids):
                actual_tokens.extend(per_id.get(cid, []))
            if expected and sha256_tokens(actual_tokens) != expected:
                problems.append(f"{name} partition token hash mismatch")
        overlap = set(partitions.get("analysis", [])) & set(
            partitions.get("qualification", [])
        )
        if overlap:
            problems.append(f"{len(overlap)} context(s) in both partitions")

    for line in problems:
        print(f"PROBLEM  {line}", file=sys.stderr)
    if problems:
        print(f"{len(problems)} problem(s)", file=sys.stderr)
        return 1
    print(
        f"verified {manifest['suite_id']}: {manifest['context_count']} contexts, "
        f"token hash {manifest['token_sha256'][:16]}"
    )
    return 0


class _FixtureTokenizer:
    """Deterministic word-to-id tokenizer, for exercising the builder offline.

    Only used by ``selftest`` and ``--fixture``. It makes the allocation, dedup,
    leakage, partition, and hashing logic testable without a network or a real
    tokenizer, which is where the bugs actually live.
    """

    vocab_size = 32000

    def __call__(self, text: str, add_special_tokens: bool = False):
        ids = [
            _digest_int("fixture-token", word) % self.vocab_size
            for word in text.split()
        ]
        return {"input_ids": ids}


def _benchmark_probe(name: str, index: int) -> str:
    """The exact text a fixture benchmark item carries.

    Shared by the corpus and benchmark writers so the self-test can plant a
    genuine leak and prove the scan detects it.
    """
    return (
        f"benchmark {name} item {index} unique probe sequence alpha bravo "
        f"charlie delta echo foxtrot golf hotel india juliet"
    )


def _primary(spec: str | list[str]) -> str:
    """The field name a fixture writes when a source lists alternatives."""
    return spec if isinstance(spec, str) else spec[0]


def _set_field(record: dict[str, Any], spec: str | list[str], value: Any) -> None:
    """Write a possibly nested field, mirroring :func:`_field`."""
    parts = _primary(spec).split(".")
    target = record
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    target[parts[-1]] = value


def _fixture_corpus(root: str, recipe: dict[str, Any]) -> None:
    """Synthesize a corpus large enough to fill every stratum."""
    os.makedirs(root, exist_ok=True)
    need: dict[str, int] = {}
    by_key = {s["key"]: s for s in recipe["strata"]}
    for stratum in recipe["strata"]:
        for key in stratum["sources"]:
            need[key] = need.get(key, 0) + stratum["contexts"]

    math_markers = " ".join(by_key["worked_math_reasoning"]["require_any"])

    for source in recipe["sources"]:
        want = need.get(source["key"], 0)
        if not want:
            continue
        records = []
        suffixes = source.get("path_suffix_any") or by_key["structured_data_tools"][
            "path_suffix_any"
        ]
        for i in range(want * 3):
            body = " ".join(
                f"{source['key']}{i}w{j}" for j in range(recipe["context_length"] + 40)
            )
            if source["key"] == "libretexts":
                body = f"{math_markers} {body}"
            record: dict[str, Any] = {"id": f"{source['key']}-{i}"}
            if source.get("extractor") == "conversation":
                _set_field(record, source["text_field"], [
                    {"role": "user", "content": body[: len(body) // 2]},
                    {"role": "assistant", "content": body[len(body) // 2:]},
                ])
                record["conversation_hash"] = f"{source['key']}-{i}"
            else:
                _set_field(record, source["text_field"], body)
            if source.get("title_field"):
                _set_field(record, source["title_field"], f"Title {i}")
            if source.get("path_field"):
                suffix = suffixes[i % len(suffixes)] if i % 2 == 0 else ".py"
                _set_field(record, source["path_field"], f"pkg/file{i}{suffix}")
            cluster_field = source.get("cluster_field")
            if cluster_field and _field_any(record, cluster_field) in (None, ""):
                # One repository is shared by every source in the github_repo
                # namespace, so the cross-source unit dedup has a real collision
                # to reject.
                shared = i == 0 and source.get("cluster_namespace") == "github_repo"
                _set_field(
                    record,
                    cluster_field,
                    "org/shared-repo" if shared
                    else f"org/{source['key']}-unit{i}",
                )
            records.append(record)
        # Two exact duplicates and one near-duplicate, so dedup is exercised
        # rather than merely present.
        if len(records) >= 3:
            records.append(dict(records[0]))
            near = dict(records[1])
            near["id"] = f"{source['key']}-near"
            if source.get("extractor") != "conversation":
                text_key = _primary(source["text_field"])
                near[text_key] = str(near[text_key]) + " tailword"
            records.append(near)
        # A planted leak: one document that verbatim contains a benchmark item,
        # so the scan is proven to detect contamination rather than merely run.
        if source["key"] == "wikipedia_en":
            leaked = dict(records[0])
            leaked["id"] = "wikipedia_en-leaked"
            leaked[source["text_field"]] = (
                _benchmark_probe("humaneval", 0)
                + " "
                + str(records[0][source["text_field"]])
            )
            if source.get("title_field"):
                leaked[source["title_field"]] = "Leaked"
            records.insert(0, leaked)
        path = os.path.join(root, f"{source['key']}.jsonl")
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")

    for spec in recipe.get("benchmarks", []):
        path = os.path.join(root, f"{spec['name']}.jsonl")
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            for i in range(3):
                record = {}
                for name in spec["fields"]:
                    if name == "choices":
                        record[name] = [
                            f"choice {i} alpha bravo charlie delta echo foxtrot"
                        ]
                    else:
                        record[name] = _benchmark_probe(spec["name"], i)
                handle.write(json.dumps(record) + "\n")


def _field_alias_failures() -> list[str]:
    """Field addressing: nesting, alternatives, and their precedence."""
    cases = [
        ({"metadata": {"path": "a/b.json"}}, "metadata.path", "a/b.json"),
        ({"text": "second"}, ["content", "text"], "second"),
        ({"content": "first", "text": "second"}, ["content", "text"], "first"),
        ({"content": "", "text": "second"}, ["content", "text"], "second"),
        ({"other": 1}, ["content", "text"], None),
        ({"metadata": "not a dict"}, "metadata.repo_name", None),
    ]
    failures = []
    for record, spec, expected in cases:
        actual = _field_any(record, spec)
        if actual != expected:
            failures.append(
                f"field {spec} of {record} resolved to {actual!r}, "
                f"expected {expected!r}"
            )
    return failures


def _scan_equivalence_failures() -> list[str]:
    """Prove the indexed scan agrees with the direct comparison it replaced.

    The index is what makes the scan finish at all, so it has to be exactly as
    strict. The cases below are the ones that could plausibly slip through it:
    fragments whose only long words sit at an edge, fragments made entirely of
    short words, punctuation and digits at word boundaries, and non-Latin text.
    """
    passages = [
        "the quick brown fox jumps over the lazy dog near the riverbank",
        "def compute_total(values): return sum(values) / len(values) + 1",
        "a b c d e f g h i j k l m n o p",
        "unicode: \u0432\u043e\u043f\u0440\u043e\u0441 \u043e \u0442\u0435\u0440"
        "\u043c\u043e\u0434\u0438\u043d\u0430\u043c\u0438\u043a\u0435 \u0438 "
        "\u044d\u043d\u0442\u0440\u043e\u043f\u0438\u0438",
        "\u4e2d\u6587\u6d4b\u8bd5\u6587\u672c include ascii tokens too",
        "answer choices: (a) 12.5 (b) 13.75 (c) 14.0 (d) none of the above",
        "prefixmatch antidisestablishmentarianism suffixmatch",
    ]
    docs = [
        Document(
            source_key="synthetic",
            cluster_id=f"unit-{index}",
            representation="plain",
            content_sha256=_sha256_text(passage),
            normalized_text=normalize_for_scan(f"lead in text {passage} trailing text"),
            tokens=[index],
            token_offset=0,
            total_tokens=1,
        )
        for index, passage in enumerate(passages)
    ]
    fragments_by_item = [
        [normalize_for_scan(p)] for p in passages
    ] + [
        # Substrings that begin and end mid-word, absent items, and multi-fragment
        # items mixing a present fragment with a missing one.
        [normalize_for_scan("uick brown fox jumps ove")],
        [normalize_for_scan("no such passage exists anywhere in the corpus")],
        [normalize_for_scan(passages[0]), normalize_for_scan(passages[1])],
        [normalize_for_scan(passages[0]), normalize_for_scan("absent fragment here")],
        [normalize_for_scan("a b c d")],
        [normalize_for_scan("12.5 (b) 13.75")],
    ]
    items = [(f"synthetic:{i}", f) for i, f in enumerate(fragments_by_item)]

    failures = []
    for match_rule in ("any", "all"):
        expected = {}
        for position, (_item, fragments) in enumerate(items):
            matched = [
                doc
                for doc in docs
                if item_matches(doc.normalized_text, fragments, match_rule)
            ]
            if matched:
                expected[position] = matched
        actual = scan_items(docs, items, match_rule)
        if actual != expected:
            missed = {
                items[p][0]: [d.cluster_id for d in expected[p]]
                for p in expected
                if actual.get(p) != expected[p]
            }
            failures.append(
                f"indexed scan disagrees with direct comparison under "
                f"match={match_rule}: {missed}"
            )
    return failures


def selftest() -> int:
    """Build a small suite from synthetic sources and verify every invariant."""
    import shutil
    import tempfile

    recipe_path = os.path.join(HERE, "suites", "recipe-v1.json")
    with open(recipe_path, encoding="utf-8") as handle:
        recipe = json.load(handle)

    # Shrink the recipe so the test is fast but the structure is identical.
    scale = 8
    recipe["contexts"] = 0
    for stratum in recipe["strata"]:
        stratum["contexts"] = max(2, stratum["contexts"] // scale)
        recipe["contexts"] += stratum["contexts"]
    recipe["partitions"] = {
        "qualification": recipe["contexts"] // 4,
        "analysis": recipe["contexts"] - recipe["contexts"] // 4,
    }
    recipe["sentinel_candidates"] = max(1, recipe["partitions"]["analysis"] // 12)
    recipe["context_length"] = 64
    for source in recipe["sources"]:
        source["min_chars"] = 64
        source["scan_limit"] = 5000

    work = tempfile.mkdtemp(prefix="lil-suite-selftest-")
    try:
        fixture = os.path.join(work, "fixture")
        small_recipe = os.path.join(work, "recipe.json")
        _write_json(small_recipe, recipe)
        _fixture_corpus(fixture, recipe)
        if probe(small_recipe, fixture, 200) != 0:
            print("FAIL  probe rejected the fixture corpus", file=sys.stderr)
            return 1

        out = os.path.join(work, "suite")
        rc = build(small_recipe, "fixture", out, fixture, 6, "selftest-suite", False)
        if rc != 0:
            return rc
        if verify(out) != 0:
            return 1

        with open(os.path.join(out, "suite-manifest.json"), encoding="utf-8") as fh:
            manifest = json.load(fh)
        with open(os.path.join(out, "partitions.json"), encoding="utf-8") as fh:
            partitions = json.load(fh)
        with open(
            os.path.join(out, "validation", "capability-overlap.json"),
            encoding="utf-8",
        ) as fh:
            overlap = json.load(fh)

        failures = _field_alias_failures() + _scan_equivalence_failures()
        if manifest["context_count"] != recipe["contexts"]:
            failures.append(
                f"context count {manifest['context_count']} != "
                f"{recipe['contexts']}"
            )
        hashes = [c["token_sha256"] for c in manifest["contexts"]]
        if len(set(hashes)) != len(hashes):
            failures.append("duplicate contexts survived dedup")
        units = [
            (c["source_namespace"], c["source_cluster_id"])
            for c in manifest["contexts"]
        ]
        if len(set(units)) != len(units):
            failures.append("a source unit contributed more than one context")
        if manifest["source_cluster_count"] != len(set(units)):
            failures.append("source_cluster_count disagrees with the contexts")
        total = len(partitions["analysis"]) + len(partitions["qualification"])
        if total != manifest["context_count"]:
            failures.append(f"partitions cover {total} of {manifest['context_count']}")
        if not partitions["sentinel_candidates"]:
            failures.append("no sentinel candidates assigned")
        if not set(partitions["sentinel_candidates"]) <= set(partitions["analysis"]):
            failures.append("sentinel candidates leaked outside analysis")
        if overlap["candidates_scanned"] <= 0:
            failures.append("leakage scan examined nothing")
        if not overlap["blocked_documents"]:
            failures.append("the planted benchmark leak was not detected")
        with open(os.path.join(out, "sources.json"), encoding="utf-8") as fh:
            provenance = json.load(fh)
        published = {entry["content_sha256"] for entry in provenance}
        leaked_and_published = published & set(overlap["blocked_documents"])
        if leaked_and_published:
            failures.append(
                f"{len(leaked_and_published)} leaked document(s) reached the suite"
            )

        # Rebuilding from identical inputs must reproduce identical hashes.
        out2 = os.path.join(work, "suite2")
        build(small_recipe, "fixture", out2, fixture, 6, "selftest-suite", False)
        with open(os.path.join(out2, "suite-manifest.json"), encoding="utf-8") as fh:
            manifest2 = json.load(fh)
        if manifest2["token_sha256"] != manifest["token_sha256"]:
            failures.append("rebuild is not deterministic: suite token hash moved")
        if manifest2["partition_token_sha256"] != manifest["partition_token_sha256"]:
            failures.append("rebuild is not deterministic: partitions moved")

        # A cached build must reproduce the suite from the cache alone, which is
        # what makes a failed build resumable rather than merely faster.
        cache = os.path.join(work, "cache")
        out3 = os.path.join(work, "suite3")
        build(small_recipe, "fixture", out3, fixture, 6, "selftest-suite", False,
              cache, set())
        pooled = [n for n in os.listdir(cache) if n.startswith("pool-")]
        if not pooled:
            failures.append("no pools were cached")
        for name in os.listdir(fixture):
            os.remove(os.path.join(fixture, name))
        out4 = os.path.join(work, "suite4")
        build(small_recipe, "fixture", out4, fixture, 6, "selftest-suite", False,
              cache, set())
        with open(os.path.join(out4, "suite-manifest.json"), encoding="utf-8") as fh:
            manifest4 = json.load(fh)
        if manifest4["token_sha256"] != manifest["token_sha256"]:
            failures.append("cached rebuild did not reproduce the suite")
        with open(
            os.path.join(out4, "validation", "capability-overlap.json"),
            encoding="utf-8",
        ) as fh:
            overlap4 = json.load(fh)
        if overlap4["blocked_documents"] != overlap["blocked_documents"]:
            failures.append("cached rebuild lost the planted benchmark leak")

        for line in failures:
            print(f"FAIL  {line}", file=sys.stderr)
        if failures:
            return 1
        print(
            f"\nselftest passed: {manifest['context_count']} contexts, "
            f"{len(partitions['analysis'])} analysis / "
            f"{len(partitions['qualification'])} qualification, "
            f"{len(partitions['sentinel_candidates'])} sentinel candidate(s), "
            f"deterministic rebuild"
        )
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="mint a suite")
    b.add_argument("--recipe", default=os.path.join(HERE, "suites", "recipe-v1.json"))
    b.add_argument("--tokenizer", required=True, help="checkpoint or tokenizer path")
    b.add_argument("--out", required=True)
    b.add_argument("--suite-id")
    b.add_argument("--fixture", help="build from local JSONL fixtures, not the Hub")
    b.add_argument(
        "--oversample",
        type=int,
        default=6,
        help="candidates harvested per allocated slot; a source may override it",
    )
    b.add_argument("--allow-shortfall", action="store_true")
    b.add_argument(
        "--cache-dir",
        help="where harvested pools are cached so a failed build resumes. "
        "Defaults to <out>-cache, deliberately outside the published suite",
    )
    b.add_argument(
        "--no-cache", action="store_true", help="harvest every source from scratch"
    )
    b.add_argument(
        "--refresh",
        action="append",
        default=[],
        metavar="KEY",
        help="re-harvest this source or benchmark even if it is cached; repeatable",
    )
    b.set_defaults(
        func=lambda a: build(
            a.recipe, a.tokenizer, a.out, a.fixture, a.oversample, a.suite_id,
            a.allow_shortfall,
            None if a.no_cache else (
                a.cache_dir or f"{a.out.rstrip('/' + os.sep)}-cache"
            ),
            set(a.refresh),
        )
    )

    p = sub.add_parser(
        "probe", help="check every source and benchmark before a long build"
    )
    p.add_argument("--recipe", default=os.path.join(HERE, "suites", "recipe-v1.json"))
    p.add_argument("--fixture", help="probe local JSONL fixtures, not the Hub")
    p.add_argument("--rows", type=int, default=200)
    p.set_defaults(func=lambda a: probe(a.recipe, a.fixture, a.rows))

    v = sub.add_parser("verify", help="re-derive every hash from the token files")
    v.add_argument("--suite", required=True)
    v.set_defaults(func=lambda a: verify(a.suite))

    s = sub.add_parser("selftest", help="exercise the builder on synthetic sources")
    s.set_defaults(func=lambda a: selftest())

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
