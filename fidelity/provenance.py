#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Refuse a candidate whose architecture is not the reference's.

A popularity-ranked sweep will eventually pull in a fine-tune wearing a quant
label. Law 10 binds the *reference* identity, so a 0.4 mean KLD would then be
ambiguous between a bad quantization and a different model. Comparing the
architecture-defining config fields before any GPU time is spent closes that.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

# Fields that define the function being quantized, not how it is stored.
# Looked up on the config root first, then inside `text_config`, because Qwen
# multimodal checkpoints nest the language model there.
ARCHITECTURE_FIELDS = (
    "architectures",
    "hidden_size",
    "num_hidden_layers",
    "layer_types",
    "vocab_size",
    "intermediate_size",
    "head_dim",
)


def load_config(path: str) -> dict[str, Any]:
    with open(os.path.join(path, "config.json"), encoding="utf-8") as handle:
        return json.load(handle)


def architecture_identity(config: dict[str, Any]) -> dict[str, Any]:
    """The architecture fields, unwrapping `text_config` when the root omits them."""
    text = config.get("text_config")
    nested = text if isinstance(text, dict) else {}
    out: dict[str, Any] = {}
    for key in ARCHITECTURE_FIELDS:
        if key in config:
            out[key] = config[key]
        elif key in nested:
            out[key] = nested[key]
        else:
            out[key] = None
    return out


def tokenizer_fingerprint(path: str) -> str | None:
    """SHA-256 of tokenizer.json, or None when the file is absent."""
    target = os.path.join(path, "tokenizer.json")
    if not os.path.isfile(target):
        return None
    digest = hashlib.sha256()
    with open(target, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare(reference: str, candidate: str) -> dict[str, Any]:
    """Compare a candidate checkpoint against its claimed reference.

    Returns a JSON-serializable report. `ok` is True only when every architecture
    field matches. A tokenizer mismatch is reported but does not fail the guard
    on its own: some quantizations rewrite tokenizer.json without changing the
    vocabulary, and the suite already binds the tokens that were scored.
    """
    ref_cfg = load_config(reference)
    cand_cfg = load_config(candidate)
    ref_id = architecture_identity(ref_cfg)
    cand_id = architecture_identity(cand_cfg)
    differing = [
        {
            "field": key,
            "reference": ref_id[key],
            "candidate": cand_id[key],
        }
        for key in ARCHITECTURE_FIELDS
        if ref_id[key] != cand_id[key]
    ]
    ref_tok = tokenizer_fingerprint(reference)
    cand_tok = tokenizer_fingerprint(candidate)
    return {
        "reference": os.path.abspath(reference),
        "candidate": os.path.abspath(candidate),
        "ok": not differing,
        "identity": cand_id,
        "differing": differing,
        "tokenizer_sha256": {
            "reference": ref_tok,
            "candidate": cand_tok,
            "match": ref_tok is not None and ref_tok == cand_tok,
        },
    }


def write_report(report: dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")


def selftest() -> int:
    ref = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": {
            "hidden_size": 5120,
            "num_hidden_layers": 64,
            "layer_types": ["linear_attention", "full_attention"],
            "vocab_size": 248320,
            "intermediate_size": 17408,
            "head_dim": 256,
        },
    }
    ident = architecture_identity(ref)
    assert ident["architectures"] == ["Qwen3_5ForConditionalGeneration"]
    assert ident["hidden_size"] == 5120
    assert ident["vocab_size"] == 248320
    other = dict(ref)
    other["text_config"] = dict(ref["text_config"], hidden_size=2560)
    assert architecture_identity(other)["hidden_size"] == 2560

    import tempfile

    root = tempfile.mkdtemp()
    ref_dir = os.path.join(root, "ref")
    bad_dir = os.path.join(root, "bad")
    os.makedirs(ref_dir)
    os.makedirs(bad_dir)
    with open(os.path.join(ref_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(ref, handle)
    with open(os.path.join(bad_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(other, handle)
    mismatch = compare(ref_dir, bad_dir)
    assert mismatch["ok"] is False
    assert mismatch["differing"][0]["field"] == "hidden_size"
    match = compare(ref_dir, ref_dir)
    assert match["ok"] is True
    print("selftest passed")
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference")
    parser.add_argument("--candidate")
    parser.add_argument("--out")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        raise SystemExit(selftest())
    if not (args.reference and args.candidate):
        parser.error("--reference and --candidate are required")
    report = compare(args.reference, args.candidate)
    if args.out:
        write_report(report, args.out)
    status = "MATCH" if report["ok"] else "MISMATCH"
    print(f"{status}  {args.candidate}")
    for item in report["differing"]:
        print(f"  {item['field']}: {item['reference']!r} vs {item['candidate']!r}")
    raise SystemExit(0 if report["ok"] else 1)
