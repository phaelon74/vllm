#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mint a frozen token-ID evaluation suite for distribution-fidelity work.

Replicates the Kimi K3 distribution-fidelity recipe — the same eighteen sources
at the same pinned revisions, the same ten allocation strata and counts, the same
dedup, leakage scan, and analysis/qualification split — but emits token IDs for
the tokenizer you name. Token IDs are not portable across tokenizers, so a suite
must be minted per tokenizer family; the methodology is what transfers.

The output is the evaluation input itself (Law 3): candidates consume the stored
IDs directly. Retokenizing source text does not reproduce the suite.

Usage:
    python fidelity/suite.py build \\
        --recipe fidelity/suites/recipe-v1.json \\
        --tokenizer /media/fmodels/Qwen/Qwen3.6-27B \\
        --out /mnt/kld/suites/qwen3.6-1024x2048-v1

    python fidelity/suite.py verify --suite /mnt/kld/suites/qwen3.6-1024x2048-v1
    python fidelity/suite.py selftest
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


def extract(record: dict[str, Any], source: dict[str, Any]) -> str:
    extractor = source.get("extractor", "plain")
    raw = record.get(source["text_field"])
    if extractor == "conversation":
        return _render_conversation(raw)
    return raw if isinstance(raw, str) else ""


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
        text = extract(record, source)
        if not text or len(text) < min_chars:
            continue
        if len(_WHITESPACE.sub("", text)) < min_chars // 2:
            continue
        cluster = record.get(source.get("cluster_field") or "", None)
        cluster_id = str(cluster) if cluster not in (None, "") else _sha256_text(text)
        if cluster_id in seen_clusters:
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
                normalized_text=normalize_for_scan(text),
                tokens=window,
                token_offset=offset,
                total_tokens=len(encoded),
                title=(
                    str(record.get(source["title_field"]))
                    if source.get("title_field") and record.get(source["title_field"])
                    else None
                ),
                path=(
                    str(record.get(source["path_field"]))
                    if source.get("path_field") and record.get(source["path_field"])
                    else None
                ),
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


def stratum_eligible(doc: Document, stratum: dict[str, Any]) -> bool:
    """Apply a stratum's content policy to a candidate."""
    suffixes = stratum.get("path_suffix_any")
    if suffixes:
        path = (doc.path or "").lower()
        if not any(path.endswith(s) for s in suffixes):
            return False
    required = stratum.get("require_any")
    if required and not any(marker in doc.normalized_text for marker in required):
        return False
    return True


def select(
    strata: list[dict[str, Any]],
    pools: dict[str, list[Document]],
    dedup: dict[str, Any],
    blocked: set[str],
) -> tuple[dict[str, list[Document]], list[str]]:
    """Fill each stratum, rejecting duplicates and blocked documents.

    Sources are drawn round-robin so a stratum backed by several sources is not
    dominated by whichever one streamed first. Exact token duplicates and
    near-duplicates are rejected before allocation, as the recipe requires,
    which means a rejection here changes which document lands in a slot rather
    than leaving a slot empty.
    """
    chosen: dict[str, list[Document]] = {}
    accepted_tokens: set[str] = set()
    accepted_sigs: list[list[int]] = []
    shortfalls: list[str] = []
    threshold = dedup["jaccard_threshold"]

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
                    token_hash = doc.token_sha256
                    if token_hash in accepted_tokens:
                        continue
                    if any(
                        jaccard(doc.signature, sig) >= threshold
                        for sig in accepted_sigs
                    ):
                        continue
                    accepted_tokens.add(token_hash)
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


def leakage_scan(
    recipe: dict[str, Any],
    pools: dict[str, list[Document]],
    fixture_dir: str | None,
) -> tuple[dict[str, Any], set[str]]:
    """Scan candidates for complete benchmark items and block any that overlap."""
    blocked: set[str] = set()
    per_benchmark = []
    all_docs = [doc for pool in pools.values() for doc in pool]
    for spec in recipe.get("benchmarks", []):
        items = benchmark_items(spec, fixture_dir)
        hits = []
        for item_id, fragments in items:
            for doc in all_docs:
                if spec.get("match") == "all":
                    matched = all(f in doc.normalized_text for f in fragments)
                else:
                    matched = any(f in doc.normalized_text for f in fragments)
                if matched:
                    hits.append({"item": item_id, "content_sha256":
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
        "source_cluster_count": len({c["source_cluster_id"] for c in contexts}),
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
                            "scan_limit", "license", "extractor",
                            "upstream_dataset_fingerprint",
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
) -> int:
    with open(recipe_path, encoding="utf-8") as handle:
        recipe = json.load(handle)

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
    pools: dict[str, list[Document]] = {}
    for source in recipe["sources"]:
        want = need.get(source["key"], 0)
        if want == 0:
            continue
        pools[source["key"]] = harvest(
            source,
            tokenizer,
            recipe["context_length"],
            want * oversample,
            recipe["dedup"],
            fixture_dir,
        )

    print("=== scanning for benchmark leakage")
    overlap, blocked = leakage_scan(recipe, pools, fixture_dir)
    if blocked:
        print(f"  blocking {len(blocked)} candidate(s) with complete overlaps")

    print("=== allocating strata")
    chosen, shortfalls = select(recipe["strata"], pools, recipe["dedup"], blocked)
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


def _fixture_corpus(root: str, recipe: dict[str, Any]) -> None:
    """Synthesize a corpus large enough to fill every stratum."""
    os.makedirs(root, exist_ok=True)
    need: dict[str, int] = {}
    for stratum in recipe["strata"]:
        for key in stratum["sources"]:
            need[key] = need.get(key, 0) + stratum["contexts"]

    math_markers = " ".join(recipe["strata"][6]["require_any"])
    suffixes = recipe["strata"][9]["path_suffix_any"]

    for source in recipe["sources"]:
        want = need.get(source["key"], 0)
        if not want:
            continue
        records = []
        for i in range(want * 3):
            body = " ".join(
                f"{source['key']}{i}w{j}" for j in range(recipe["context_length"] + 40)
            )
            if source["key"] == "libretexts":
                body = f"{math_markers} {body}"
            record: dict[str, Any] = {"id": f"{source['key']}-{i}"}
            if source.get("extractor") == "conversation":
                record[source["text_field"]] = [
                    {"role": "user", "content": body[: len(body) // 2]},
                    {"role": "assistant", "content": body[len(body) // 2:]},
                ]
                record["conversation_hash"] = f"{source['key']}-{i}"
            else:
                record[source["text_field"]] = body
            if source.get("title_field"):
                record[source["title_field"]] = f"Title {i}"
            if source.get("path_field"):
                suffix = suffixes[i % len(suffixes)] if i % 2 == 0 else ".py"
                record[source["path_field"]] = f"pkg/file{i}{suffix}"
                record[source["cluster_field"]] = f"org/repo{i}"
            records.append(record)
        # Two exact duplicates and one near-duplicate, so dedup is exercised
        # rather than merely present.
        if len(records) >= 3:
            records.append(dict(records[0]))
            near = dict(records[1])
            near["id"] = f"{source['key']}-near"
            if source.get("extractor") != "conversation":
                near[source["text_field"]] = (
                    str(near[source["text_field"]]) + " tailword"
                )
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

        failures = []
        if manifest["context_count"] != recipe["contexts"]:
            failures.append(
                f"context count {manifest['context_count']} != "
                f"{recipe['contexts']}"
            )
        hashes = [c["token_sha256"] for c in manifest["contexts"]]
        if len(set(hashes)) != len(hashes):
            failures.append("duplicate contexts survived dedup")
        clusters = [c["source_cluster_id"] for c in manifest["contexts"]]
        if len(set(clusters)) != len(clusters):
            failures.append("a source cluster contributed more than one context")
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
    b.add_argument("--oversample", type=int, default=6)
    b.add_argument("--allow-shortfall", action="store_true")
    b.set_defaults(
        func=lambda a: build(
            a.recipe, a.tokenizer, a.out, a.fixture, a.oversample, a.suite_id,
            a.allow_shortfall,
        )
    )

    v = sub.add_parser("verify", help="re-derive every hash from the token files")
    v.add_argument("--suite", required=True)
    v.set_defaults(func=lambda a: verify(a.suite))

    s = sub.add_parser("selftest", help="exercise the builder on synthetic sources")
    s.set_defaults(func=lambda a: selftest())

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
