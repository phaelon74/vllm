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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from redaction import redact_env  # noqa: E402 - sibling module

LAWS_VERSION = 4
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


def _text_line(env_dir: str | None, name: str, needle: str) -> str:
    """One line out of a captured command's output, by substring."""
    if not env_dir:
        return "n/a"
    path = os.path.join(env_dir, name)
    if not os.path.isfile(path):
        return "n/a"
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if needle in line:
                return line.strip()
    return "n/a"


def _cudnn(value: Any) -> str:
    """torch reports cuDNN as a packed integer, which reads as a typo otherwise."""
    if not isinstance(value, int):
        return "n/a"
    return f"{value // 10000}.{(value % 10000) // 100}.{value % 100} ({value})"


def _environment(runtime_env: dict[str, Any], env_dir: str | None) -> list[str]:
    """The host, toolchain, and runtime the number is bound to, per Law 6.

    A fidelity number is a property of a stack, not of a checkpoint alone. What is
    summarized here is recorded in full under ``environment/``; this table is the
    part a reader needs before deciding whether their own stack is comparable.
    """
    torch_info = runtime_env.get("torch") or {}
    capture = runtime_env.get("capture_runtime_manifest") or {}
    vllm_info = runtime_env.get("vllm") or {}
    dirty = runtime_env.get("vllm_tree_dirty")
    rows = [
        ("Host", str(runtime_env.get("hostname") or "n/a")),
        ("Platform", str(runtime_env.get("platform") or "n/a")),
        ("Python", str(runtime_env.get("python") or "n/a")),
        ("vLLM", str(vllm_info.get("version") or "n/a")),
        (
            "vLLM commit",
            f"{runtime_env.get('vllm_commit') or 'n/a'}"
            f"{' (working tree dirty)' if dirty else ''}",
        ),
        ("torch", str(torch_info.get("version") or "n/a")),
        ("torch CUDA runtime", str(torch_info.get("cuda") or "n/a")),
        ("cuDNN", _cudnn(torch_info.get("cudnn"))),
        ("NCCL", str(torch_info.get("nccl") or "n/a")),
        ("CUDA arch list", ", ".join(torch_info.get("arch_list") or []) or "n/a"),
        ("nvcc", _text_line(env_dir, "toolchain-nvcc.txt", "release")),
        ("gcc", _text_line(env_dir, "toolchain-gcc.txt", "gcc (")),
        ("glibc / ldd", _text_line(env_dir, "toolchain-ldd.txt", "ldd (")),
        ("NVIDIA driver", str(capture.get("driver") or "n/a")),
        (
            "float32 matmul precision",
            str(torch_info.get("float32_matmul_precision") or "n/a"),
        ),
        ("TF32 (matmul / cuDNN)",
         f"{torch_info.get('allow_tf32_matmul')} / "
         f"{torch_info.get('allow_tf32_cudnn')}"),
        (
            "torch deterministic algorithms",
            str(torch_info.get("deterministic_algorithms")),
        ),
    ]
    out = ["## Environment", "", *_table(rows, ("Item", "Value")), ""]

    devices = runtime_env.get("devices") or []
    if devices:
        device_rows = [
            (
                str(d.get("index")),
                str(d.get("name")),
                str(d.get("capability")),
                f"{d.get('total_memory_gib')} GiB",
                str(d.get("multi_processor_count")),
            )
            for d in devices
        ]
        out += ["### GPUs", ""]
        out += _table(
            device_rows,
            ("Index", "Device", "Compute capability", "Memory", "SMs"),
        )
        out.append("")

    env_vars = runtime_env.get("env") or {}
    if env_vars:
        # Redacted again at render time, not only at capture time: an artifact
        # assembled from an older environment report must still be safe to publish.
        safe, hidden = redact_env(env_vars)
        out += [
            "### Captured environment variables",
            "",
            "Every set variable matching the prefixes the capture watches: "
            "`VLLM_`, `TORCH`, `PYTORCH`, `CUDA`, `CUBLAS`, `NCCL`, `TRITON`, "
            "`FLASHINFER`, `OMP_`, `MKL_`, `HF_`, `SAFETENSORS_`. Any of these can "
            "move a bitwise result, so they are part of the identity.",
            "",
        ]
        if hidden:
            out += [
                f"Credential values are never published. Set but redacted: "
                f"{', '.join('`' + name + '`' for name in hidden)}.",
                "",
            ]
        out += _table(
            [(f"`{k}`", f"`{v}`") for k, v in sorted(safe.items())],
            ("Variable", "Value"),
        )
        out.append("")
    return out


def _bytes(size: int) -> str:
    for limit, unit in ((2**30, "GiB"), (2**20, "MiB"), (2**10, "KiB")):
        if size >= limit:
            return f"{size / limit:.2f} {unit}"
    return f"{size} B"


def _size(path: str) -> str:
    if not os.path.isdir(path):
        return _bytes(os.path.getsize(path))
    total = 0
    count = 0
    for root, _, names in os.walk(path):
        count += len(names)
        total += sum(os.path.getsize(os.path.join(root, name)) for name in names)
    return f"{_bytes(total)} in {count} file{'' if count == 1 else 's'}"


def _files(artifact_dir: str | None, candidate: str | None) -> list[str]:
    """What every file in the artifact is for, and which ones to reuse.

    An artifact that a reader has to reverse-engineer is not reproducible in any
    useful sense. The reference distributions in particular are the expensive half
    of anyone else's comparison, so their reuse is spelled out rather than implied.
    """
    if not artifact_dir or not os.path.isdir(artifact_dir):
        return []
    prefix = f"{candidate}/" if candidate else ""
    entries: list[tuple[str, str]] = [
        (f"{prefix}report.md", "This document."),
        (f"{prefix}report.json",
         "Every statistic behind it, machine-readable: per-bucket means, "
         "percentiles, agreement rates, and the phase timings."),
        (f"{prefix}manifest.json",
         "The capture manifest this result is bound to (Law 5). Scoring refuses "
         "to run if the live configuration differs from it."),
        (f"{prefix}compliance.json",
         "The law-by-law receipt, including the comparability key."),
        ("baselines/self-kld.json",
         "The zero-baseline proof required by Law 1: the reference scored against "
         "a capture of itself."),
        ("suite/suite-manifest.json",
         "The frozen evaluation input's identity: token hashes per context and per "
         "partition, sources, strata, and the analysis/qualification split."),
        ("suite/tokens",
         "The token IDs themselves. These are the evaluation input, not a "
         "description of it; retokenizing source text does not reproduce them."),
        ("suite/sources.json",
         "Per-context provenance: dataset, revision, licence, source unit, and the "
         "deterministic token offset chosen within the document."),
        ("suite/validation/capability-overlap.json",
         "The benchmark-contamination scan and every document it blocked."),
        ("reference/manifest.json",
         "The reference capture's own manifest: geometry, vocabulary, storage "
         "mode, and a hash for every tensor file."),
        ("reference",
         "The reusable reference distributions. Pass this directory as "
         "`--reference-logits` to score a new candidate against the same "
         "reference without loading the reference checkpoint."),
        ("reference/lm_head.safetensors",
         "The reference language-model head, which turns the stored hidden states "
         "back into reference logits."),
        ("environment/runtime.json",
         "Machine-readable provenance: torch, CUDA, cuDNN, NCCL, driver, devices, "
         "and the captured environment variables."),
        ("environment/summary.md",
         "The same provenance as prose, plus an index of every captured file."),
        ("environment/toolchain-nvcc.txt",
         "`nvcc --version` verbatim; `toolchain-gcc.txt` and `toolchain-ldd.txt` "
         "sit beside it."),
        ("environment/gpu-smi-query.txt",
         "`nvidia-smi -q` verbatim: ECC state, persistence mode, clocks, and "
         "throttle reasons, any of which can move a bitwise result."),
        ("environment/pip-freeze.txt",
         "Every installed package version in the scoring environment."),
        ("environment/models",
         "Checkpoint fingerprints: file listing, sizes, config and tokenizer "
         "hashes, and `config.json` verbatim for each model scored."),
        ("checksums.txt",
         "`sha256sum --check` compatible over every other file here. This is "
         "authoritative for integrity (Law 12)."),
        ("LAWS.md",
         "The laws this artifact was produced under, including the override "
         "procedure."),
    ]
    rows = []
    for rel, purpose in entries:
        full = os.path.join(artifact_dir, rel.replace("/", os.sep))
        if not os.path.exists(full):
            # The one-pager is being written as this renders, so it is listed
            # regardless; anything else absent is simply not part of this artifact.
            if not rel.endswith("report.md"):
                continue
            rows.append((f"`{rel}`", "\u2014", purpose))
            continue
        rows.append((f"`{rel}`", _size(full), purpose))
    if not rows:
        return []
    return [
        "## Files in this artifact",
        "",
        "Paths are relative to the artifact root, the same paths `checksums.txt` "
        "uses. Verify the whole tree with `sha256sum --check checksums.txt` from "
        "that root.",
        "",
        *_table(rows, ("Path", "Size", "What it is")),
        "",
    ]


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


def _link(cell: dict[str, Any] | None) -> str:
    """Markdown links to a cell's published report and QDQ manifest.

    Only relative paths are linked. A cell still carrying its work-directory path
    was not published, and a link into a path the reader does not have is worse
    than no link.
    """
    if not isinstance(cell, dict):
        return ""
    links = []
    for key, label in (("report", "report"), ("qdq_manifest", "manifest")):
        path = cell.get(key)
        if isinstance(path, str) and not os.path.isabs(path):
            links.append(f"[{label}]({path})")
    return " · ".join(links)


def _deployed_quantization(deployed: dict[str, Any]) -> list[str]:
    """What the shipped checkpoint actually quantized, and how finely.

    Rendered before the cells because it decides which cells exist: a checkpoint
    that leaves its router in BF16 has no router weight cell to measure, and one
    that quantizes only part of its attention is not described by "FP8".
    """
    inspection = deployed.get("inspection")
    if not isinstance(inspection, dict):
        return []
    out = ["### What the deployed checkpoint quantizes", ""]
    facts = [
        ("Scheme", str(inspection.get("detected_scheme") or "n/a")),
        ("Block", str(inspection.get("detected_block") or "n/a")),
        ("Declared method", str(inspection.get("quant_method") or "n/a")),
        ("Activation scheme", str(inspection.get("activation_scheme") or "n/a")),
    ]
    out += _table(facts, ("Property", "Value"))
    out.append("")
    coverage = inspection.get("coverage") or {}
    rows = []
    for component, counts in coverage.items():
        total = (counts or {}).get("weights") or 0
        done = (counts or {}).get("quantized") or 0
        if not total and not done:
            continue
        verdict = "all" if done and done >= total else ("none" if not done else "some")
        rows.append((component, f"{done} / {total}", verdict))
    if rows:
        out += _table(rows, ("Component", "Quantized weights", "Coverage"))
        out.append("")
    partial = [name for name, _, verdict in rows if verdict == "some"]
    if partial:
        out += [
            f"Partial coverage in {', '.join(partial)}, so the format name alone "
            f"does not describe this checkpoint. A QDQ cell selects weights by "
            f"name pattern and rounds every one it matches, which for a partly "
            f"quantized component rounds more than the checkpoint does and makes "
            f"that cell an upper bound rather than a match.",
            "",
        ]
    return out


def _attribution(receipt: dict[str, Any]) -> list[str]:
    """Where a routed model's divergence came from, per Law 14."""
    attribution = receipt.get("attribution")
    if not isinstance(attribution, dict):
        return []
    deployed = receipt.get("mean_kld")
    expert_cell = attribution.get("expert_cell") or {}
    expert = expert_cell.get("mean_kld")
    router_cell = attribution.get("router_cell") or {}
    router_na = router_cell.get("status") == "not_applicable"
    router = None if router_na else router_cell.get("mean_kld")

    composite = attribution.get("composite_cell") or {}
    engine = attribution.get("engine_arithmetic")
    deployed_cell = attribution.get("deployed") or {}
    activations = (deployed_cell.get("inspection") or {}).get("activation_scheme")

    out = [
        "## Component attribution",
        "",
        "This reference routes tokens to experts, so the deployed mean is not a "
        "single effect. Each cell is the reference with one component rounded "
        "through the deployed scheme and run on BF16 kernels, scored on the same "
        "tokens against the same capture. The cells do not sum to the deployed "
        "mean and are not meant to: once a token is routed elsewhere, degrading "
        "the expert it no longer uses costs nothing.",
        "",
        "No cell here holds expert selection fixed. Rounding any weight changes "
        "the residual stream, and every router downstream of that change sees "
        "different inputs, so each cell carries some routing movement of its "
        "own. What routing costs is measured directly under Routing divergence "
        "below, not inferred from these cells.",
        "",
    ]
    out += _deployed_quantization(attribution.get("deployed") or {})
    out += ["### Component cells", ""]
    routing = receipt.get("routing") or {}
    rows = [
        (
            "Experts only",
            "expert weight precision",
            _kld(expert),
            _pct(expert_cell.get("selection_flip_rate")),
            _link(expert_cell),
        ),
        (
            "Router weights only",
            "router weight precision, not the routing term",
            "n/a — router ships unquantized" if router_na else _kld(router),
            "n/a" if router_na else _pct(router_cell.get("selection_flip_rate")),
            _link(router_cell),
        ),
    ]
    if composite:
        rows.append(
            (
                "Every quantized component",
                f"{', '.join(composite.get('components') or [])}, BF16 kernels",
                _kld(composite.get("mean_kld")),
                _pct(composite.get("selection_flip_rate")),
                _link(composite),
            )
        )
    rows.append(
        (
            "Deployed",
            "as shipped, quantized kernels",
            _kld(deployed),
            _pct(routing.get("selection_flip_rate")),
            "[report.json](report.json)",
        )
    )
    out += _table(
        rows, ("Cell", "What it isolates", "Mean KLD", "Selections changed", "Support")
    )
    out += [
        "",
        "The selections column is why none of these cells is routing-free: each "
        "one reroutes some tokens purely as a consequence of rounding weights "
        "upstream of a router it never touched.",
    ]
    out.append("")
    if isinstance(engine, (int, float)):
        share = (
            f" ({abs(engine) / float(deployed):.0%} of the deployed mean)"
            if isinstance(deployed, (int, float)) and deployed
            else ""
        )
        cause = (
            f"The checkpoint quantizes activations ({activations}), so this term "
            f"is activation quantization together with kernel arithmetic, not "
            f"kernel arithmetic alone."
            if activations in ("dynamic", "static")
            else "The checkpoint quantizes weights only, so this term is kernel "
            "arithmetic."
        )
        out += [
            f"**Beyond weight rounding: {engine:+.8f}**{share}. Every cell above "
            f"rounds weights and runs on BF16 kernels. The deployed checkpoint "
            f"differs from the composite cell by this much. {cause} A weight-only "
            f"analysis, which is what any QDQ cell is, cannot see it.",
            "",
        ]
    if router_na:
        out += [
            "The deployed checkpoint leaves the router in BF16, so router weight "
            "precision costs exactly nothing here. That is not the same as "
            "routing costing nothing: an identical router fed perturbed "
            "activations selects different experts, which is measured below.",
            "",
        ]
        evidence = router_cell.get("evidence")
        if evidence:
            out += [f"Evidence: {evidence}", ""]

    ladder = attribution.get("ladder")
    if isinstance(ladder, list) and ladder:
        out += [
            "### Scheme ladder",
            "",
            "The same expert weights rounded through each scheme, which is the "
            "cell that discriminates between them.",
            "",
        ]
        out += _table(
            [
                (
                    str(entry.get("scheme")),
                    _kld(entry.get("mean_kld")),
                    str(entry.get("variant") or ""),
                    _link(entry),
                )
                for entry in ladder
            ],
            ("Scheme", "Experts-only mean KLD", "Variant", "Support"),
        )
        out.append("")
    out += [
        "Full cells, digests, and the deployed checkpoint's inspection are in "
        "[attribution.json](attribution.json).",
        "",
    ]
    return out


def _routing(receipt: dict[str, Any]) -> list[str]:
    """What changing expert selection cost, measured rather than emulated."""
    routing = receipt.get("routing")
    if not isinstance(routing, dict) or not routing:
        return []
    held = routing.get("mean_kld_routing_held")
    flipped = routing.get("mean_kld_routing_flipped")
    excess = routing.get("routing_excess_mean")
    share = routing.get("flipped_share_of_total")
    deployed = receipt.get("mean_kld")

    out = [
        "## Routing divergence",
        "",
        f"Both runs saw the same tokens, and the candidate's routers are the "
        f"reference's routers to the bit where the checkpoint leaves them "
        f"unquantized. They still selected different experts, because the "
        f"activations arriving at them were already perturbed. This section "
        f"compares the two runs' recorded selections over "
        f"{routing.get('num_layers')} routed layers, "
        f"{routing.get('num_experts_per_tok')} experts per token.",
        "",
    ]
    out += _table(
        [
            (
                "Selections changed",
                _pct(routing.get("selection_flip_rate")),
                f"of {routing.get('layer_selections')} (token, layer) choices",
            ),
            (
                "Expert slots changed",
                _pct(routing.get("slot_disagreement_rate")),
                "of individual expert slots",
            ),
            (
                "Highest-weighted expert changed",
                _pct(routing.get("top1_expert_change_rate")),
                "includes reordering within an unchanged selection",
            ),
            (
                "Positions with any flip",
                _pct(routing.get("position_flip_rate")),
                f"{routing.get('positions_with_any_flip')} of "
                f"{routing.get('positions')} scored positions",
            ),
        ],
        ("Measure", "Rate", "Of what"),
    )
    out.append("")
    scored = routing.get("positions") or 0
    any_flip = routing.get("positions_with_any_flip") or 0
    out += _table(
        [
            (
                "Routing held",
                _kld(held),
                str(scored - any_flip),
            ),
            (
                "Routing flipped",
                _kld(flipped),
                str(routing.get("positions_with_any_flip")),
            ),
        ],
        ("Positions where", "Mean KLD", "Count"),
    )
    out.append("")
    if isinstance(excess, (int, float)):
        ratio = (
            f", {flipped / held:.1f}x the mean where routing held"
            if isinstance(held, (int, float))
            and isinstance(flipped, (int, float))
            and held
            else ""
        )
        pct_of_mean = (
            f" ({excess / float(deployed):.0%} of the deployed mean)"
            if isinstance(deployed, (int, float)) and deployed
            else ""
        )
        out += [
            f"**Ranking floor {_kld(excess)}**{pct_of_mean}. Positions whose "
            f"expert selection changed diverge at {_kld(flipped)}{ratio}, and "
            f"they carry "
            f"{'n/a' if share is None else f'{float(share) * 100:.1f}%'} of the "
            f"total. The floor is what the mean would give up if those "
            f"positions diverged no more than the ones where routing survived. "
            f"Two candidates whose deployed means differ by less than it are "
            f"not ranked by that difference; rank them on the experts-only "
            f"cell, which is a precision comparison.",
            "",
        ]
    buckets = [b for b in routing.get("buckets") or [] if b.get("positions")]
    if len(buckets) > 1:
        out += [
            "Divergence rises with how many layers rerouted, which is what "
            "distinguishes a single reroute a later layer can absorb from a "
            "token that took a different path through the model:",
            "",
        ]
        out += _table(
            [
                (
                    (
                        f"{b['flipped_layers_min']}"
                        if b.get("flipped_layers_max") == b["flipped_layers_min"]
                        else f"{b['flipped_layers_min']}+"
                        if b.get("flipped_layers_max") is None
                        else f"{b['flipped_layers_min']}-{b['flipped_layers_max']}"
                    ),
                    str(b["positions"]),
                    _kld(b.get("mean_kld")),
                )
                for b in buckets
            ],
            ("Layers rerouted", "Positions", "Mean KLD"),
        )
        out.append("")
    rates = routing.get("per_layer_flip_rate") or []
    if rates:
        worst = max(range(len(rates)), key=lambda i: rates[i])
        early = sum(rates[: max(1, len(rates) // 4)]) / max(1, len(rates) // 4)
        late = sum(rates[-max(1, len(rates) // 4):]) / max(1, len(rates) // 4)
        out += [
            f"Per-layer selection change runs from {min(rates) * 100:.2f}% to "
            f"{max(rates) * 100:.2f}% (worst at layer {worst}), averaging "
            f"{early * 100:.2f}% across the first quarter of routed layers and "
            f"{late * 100:.2f}% across the last. Rerouting that grows with "
            f"depth is accumulated perturbation, not router precision.",
            "",
        ]
    return out


def _domains(receipt: dict[str, Any]) -> list[str]:
    """Which kinds of text the divergence landed on, per Law 15."""
    strata = receipt.get("strata")
    if not isinstance(strata, dict):
        return []
    primary = strata.get("primary")
    rows = strata.get("stratum") or []
    overall = (strata.get("overall") or {}).get(primary) or {}
    if not rows or not overall:
        return []
    ranked = sorted(
        rows,
        key=lambda r: float(
            ((r.get("cells") or {}).get(primary) or {}).get("mean_kld") or 0.0
        ),
        reverse=True,
    )
    out = [
        "## Fidelity by domain",
        "",
        "The suite is stratified, so the mean above is an average over kinds of "
        "text that do not degrade equally. `x run` is a domain's mean divided by "
        "the run's, so 1.00 degrades exactly as much as the model overall. The "
        "reference's own top-1 probability is shown because a domain the "
        "reference finds harder will diverge more for that reason alone.",
        "",
    ]
    table = []
    for row in ranked:
        cell = (row.get("cells") or {}).get(primary) or {}
        mean = cell.get("mean_kld")
        base = overall.get("mean_kld")
        table.append(
            (
                str(row.get("label") or row.get("key")),
                str(cell.get("contexts") or ""),
                f"{float(cell.get('mean_ref_top1_prob') or 0.0) * 100:.1f}%",
                _kld(mean),
                f"{float(mean) / float(base):.2f}" if base else "n/a",
                _kld(cell.get("worst_context_kld")),
            )
        )
    out += _table(
        table,
        (
            "Domain",
            "Contexts",
            "Reference top-1",
            "Mean KLD",
            "x run",
            "Worst context",
        ),
    )
    out.append("")

    cells = [name for name in (strata.get("cells") or []) if name != primary]
    if cells:
        out += [
            "The same domains measured on the other attribution cells, which "
            "shows whether a domain's weakness is a property of the model or of "
            "the format:",
            "",
        ]
        out += _table(
            [
                (
                    str(row.get("label") or row.get("key")),
                    *[
                        _kld(((row.get("cells") or {}).get(name) or {}).get("mean_kld"))
                        for name in [primary, *cells]
                    ],
                )
                for row in ranked
            ],
            ("Domain", *[str(name) for name in [primary, *cells]]),
        )
        out.append("")

    worst, best = ranked[0], ranked[-1]
    worst_cell = (worst.get("cells") or {}).get(primary) or {}
    best_cell = (best.get("cells") or {}).get(primary) or {}
    if worst_cell.get("mean_kld") and best_cell.get("mean_kld"):
        spread = float(worst_cell["mean_kld"]) / float(best_cell["mean_kld"])
        out += [
            f"**{worst.get('label')}** is this candidate's weakest domain and "
            f"**{best.get('label')}** its strongest, a spread of "
            f"{spread:.1f}x. A deployment weighted toward the weakest domain "
            f"sees more divergence than the headline mean implies; the per-source "
            f"breakdown and the reading are in "
            f"[strata.md](strata.md) and [strata.json](strata.json).",
            "",
        ]
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
    env_dir: str | None = None,
    artifact_dir: str | None = None,
    candidate_dir: str | None = None,
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
    parts += _attribution(receipt)
    parts += _routing(receipt)
    parts += _domains(receipt)
    parts += _head_split(report, manifest)
    parts += _profiles(report)
    parts += _compliance(receipt, baseline)
    parts += _environment(runtime_env, env_dir)
    parts += _files(artifact_dir, candidate_dir)
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


def _weakest_domain(receipt: dict[str, Any]) -> tuple[str | None, float | None]:
    """The stratum with the highest mean, for the leaderboard's summary column."""
    strata = receipt.get("strata")
    if not isinstance(strata, dict):
        return None, None
    primary = strata.get("primary")
    rows = strata.get("stratum") or []
    best: tuple[str | None, float | None] = (None, None)
    for row in rows:
        mean = ((row.get("cells") or {}).get(primary) or {}).get("mean_kld")
        if isinstance(mean, (int, float)) and (
            best[1] is None or mean > best[1]
        ):
            best = (str(row.get("key")), float(mean))
    return best


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
            attribution = item["receipt"].get("attribution") or {}
            expert = (attribution.get("expert_cell") or {}).get("mean_kld")
            floor = item["receipt"].get("ranking_floor")
            weakest, weakest_kld = _weakest_domain(item["receipt"])
            rows.append(
                (
                    item["label"],
                    _kld(report.get("mean_kld")),
                    _kld(expert) if attribution else "n/a",
                    _kld(floor) if floor is not None else "n/a",
                    f"{weakest} {_kld(weakest_kld)}" if weakest else "n/a",
                    _kld(report.get("median_kld")),
                    _kld(report.get("p99_kld")),
                    _kld(report.get("max_kld")),
                    _pct(report.get("top1_agreement")),
                    str(report.get("num_positions")),
                    status,
                )
            )
            csv_rows.append([key[:12], item["label"], report.get("mean_kld"),
                             expert, floor, weakest, weakest_kld,
                             report.get("median_kld"), report.get("p99_kld"),
                             report.get("max_kld"), report.get("top1_agreement"),
                             report.get("num_positions"), compliant])
        lines += _table(
            rows,
            ("Candidate", "Mean KLD", "Experts only", "Routing term",
             "Weakest domain", "Median", "p99", "Max", "Top-1", "Positions",
             "Law-compliant"),
        )
        lines.append("")
        if any(_weakest_domain(m["receipt"])[0] for m in ranked):
            lines += [
                "The weakest-domain column is the stratum with the highest mean "
                "for that candidate (Law 15). Two candidates with the same mean "
                "are not interchangeable if they lose fidelity on different "
                "kinds of text; each candidate's `strata.md` has the full table.",
                "",
            ]
        if any(m["receipt"].get("ranking_floor") for m in ranked):
            lines += [
                "Rows for routed models are ranked on the deployed mean, but a "
                "routed model's deployed mean carries a saturating routing term "
                "(Law 14): tokens whose expert selection changed diverge far "
                "more than tokens whose selection survived, and that term does "
                "not scale with precision. The routing-term column is the floor "
                "below which two deployed means do not rank anything; compare "
                "the experts-only column to compare quantization schemes.",
                "",
            ]
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
    label = args.label or os.path.basename(
        str(report.get("student_model", "candidate"))
    )
    candidate_dir = args.candidate_dir
    if candidate_dir is None and args.artifact_dir:
        here = os.path.dirname(os.path.abspath(args.out))
        rel = os.path.relpath(here, os.path.abspath(args.artifact_dir))
        candidate_dir = rel.replace(os.sep, "/") if rel not in (".", "..") else None
    text = render_onepager(
        report, manifest, receipt, runtime_env, _load(args.self_report), label,
        args.env_dir, args.artifact_dir, candidate_dir,
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
            writer.writerow(["group", "candidate", "mean_kld",
                             "expert_cell_kld", "routing_term",
                             "weakest_domain", "weakest_domain_kld",
                             "median_kld", "p99_kld", "max_kld",
                             "top1_agreement", "positions", "law_compliant"])
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
    one.add_argument(
        "--artifact-dir",
        help="assembled artifact root, so the one-pager can index its own files",
    )
    one.add_argument(
        "--candidate-dir",
        help="this candidate's path within the artifact; inferred from --out",
    )
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
