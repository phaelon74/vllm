#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fail-closed enforcement of the Local Inference Lab fidelity laws.

Reads the artifacts a scoring campaign produces and emits a compliance receipt.
Exits non-zero when any law fails, so a pipeline cannot publish a
non-compliant artifact. See ``fidelity/LAWS.md`` for the laws themselves.

Usage:
    python fidelity/compliance.py \\
        --report results/candidate/report.json \\
        --manifest captures/candidate/manifest.json \\
        --self-report baselines/self-kld/report.json \\
        --env-dir environment \\
        --out compliance/receipt.json
"""

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

LAWS_VERSION = 4

PASS = "pass"
FAIL = "fail"
OVERRIDE = "override"
NOT_APPLICABLE = "not_applicable"

# Laws whose text permits a recorded deviation. Anything else is absolute and an
# approval entry cannot rescue it.
OVERRIDABLE = frozenset({1, 8, 11, 12, 14})


@dataclass
class Finding:
    """One law's evaluation against a campaign's artifacts."""

    law: int
    title: str
    status: str
    detail: str
    approval: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "law": self.law,
            "title": self.title,
            "status": self.status,
            "detail": self.detail,
        }
        if self.approval is not None:
            out["approval"] = self.approval
        return out


@dataclass
class Campaign:
    """Everything a compliance evaluation reads."""

    report: dict[str, Any]
    manifest: dict[str, Any]
    manifest_path: str
    self_report: dict[str, Any] | None = None
    env_dir: str | None = None
    env_runtime: dict[str, Any] | None = None
    suite: dict[str, Any] | None = None
    artifact_dir: str | None = None
    partition: str = "analysis"
    freeze_receipt: dict[str, Any] | None = None
    repeat_study: dict[str, Any] | None = None
    attribution: dict[str, Any] | None = None
    strata: dict[str, Any] | None = None
    approvals: dict[str, Any] = field(default_factory=dict)


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _missing(report: dict[str, Any], keys: tuple[str, ...]) -> list[str]:
    return [key for key in keys if report.get(key) is None]


def law_1_zero_baseline(c: Campaign) -> Finding:
    """Reference scored against itself must be exactly zero."""
    title = "Zero baseline"
    if c.self_report is None:
        return Finding(1, title, FAIL, "no self-comparison report supplied")
    mean = c.self_report.get("mean_kld")
    positions = c.self_report.get("num_positions") or 0
    if mean != 0.0:
        return Finding(1, title, FAIL, f"self-KLD is {mean!r}, must be exactly 0.0")
    if positions <= 0:
        return Finding(1, title, FAIL, "self-comparison scored no positions")
    baseline_runner = c.self_report.get("model_runner_v2")
    if baseline_runner != c.report.get("model_runner_v2"):
        return Finding(
            1,
            title,
            FAIL,
            f"baseline ran on model_runner_v2={baseline_runner!r} but the "
            f"candidate ran on {c.report.get('model_runner_v2')!r}",
        )
    return Finding(
        1, title, PASS, f"self-KLD exactly 0.0 over {positions} positions"
    )


def law_2_determinism(c: Campaign) -> Finding:
    """Eager execution, no autotuning, no prefix caching."""
    title = "Determinism"
    if not c.manifest.get("enforce_eager"):
        return Finding(2, title, FAIL, "capture manifest does not record eager mode")
    caching = c.manifest.get("enable_prefix_caching")
    if caching:
        return Finding(2, title, FAIL, "prefix caching was enabled")
    detail = "eager enforced"
    if caching is None:
        detail += "; manifest predates prefix-caching/max_num_seqs recording"
    else:
        detail += f"; prefix caching off, max_num_seqs={c.manifest.get('max_num_seqs')}"
    return Finding(2, title, PASS, detail)


def law_3_frozen_input(c: Campaign) -> Finding:
    """Scored tokens must come from a published, hash-matched suite."""
    title = "Frozen input"
    captured = c.manifest.get("token_sha256")
    if not captured:
        return Finding(3, title, FAIL, "capture manifest records no token hash")
    if c.suite is None:
        return Finding(
            3,
            title,
            FAIL,
            "no suite manifest supplied; tokens produced by run-time "
            "tokenization are not a frozen input",
        )
    # A run may score the whole suite or one partition of it, so the manifest's
    # hash must match the hash the suite publishes for whatever was scored.
    by_partition = c.suite.get("partition_token_sha256") or {}
    accepted = {
        "all": c.suite.get("token_sha256"),
        **{name: value for name, value in by_partition.items()},
    }
    matched = [name for name, value in accepted.items() if value == captured]
    if not matched:
        return Finding(
            3,
            title,
            FAIL,
            f"token hash {captured} matches no partition of suite "
            f"{c.suite.get('suite_id')}: {sorted(accepted)}",
        )
    return Finding(
        3,
        title,
        PASS,
        f"token hash matches suite {c.suite.get('suite_id')} "
        f"[{', '.join(sorted(matched))}]: {captured[:16]}",
    )


def law_4_real_vocabulary(c: Campaign) -> Finding:
    """KLD is scored over the tokenizer's real vocabulary."""
    title = "Real vocabulary"
    scored = c.report.get("kld_vocab_size")
    captured = c.manifest.get("kld_vocab_size")
    if not scored:
        return Finding(4, title, FAIL, "report records no kld_vocab_size")
    if scored != captured:
        return Finding(
            4,
            title,
            FAIL,
            f"report scored {scored} but capture bound {captured}",
        )
    declared = c.manifest.get("declared_vocab_size")
    if declared is not None and scored > declared:
        return Finding(
            4,
            title,
            FAIL,
            f"scored width {scored} exceeds the checkpoint's {declared}",
        )
    detail = f"scored {scored} real tokens"
    if declared:
        detail += f" of {declared} declared ({declared - scored} padding rows)"
    return Finding(4, title, PASS, detail)


def law_5_manifest_binding(c: Campaign) -> Finding:
    """The report must be tied to the exact capture manifest it scored."""
    title = "Manifest binding"
    bound = (
        "token_sha256",
        "tokenizer",
        "context_length",
        "rows",
        "score_from",
        "kld_vocab_size",
        "tensor_parallel_size",
        "enforce_eager",
        "runtime",
    )
    absent = [key for key in bound if key not in c.manifest]
    if absent:
        return Finding(
            5, title, FAIL, f"manifest is missing bound fields: {', '.join(absent)}"
        )
    recorded = c.report.get("capture_manifest_sha256")
    if not recorded:
        return Finding(5, title, FAIL, "report records no capture manifest hash")
    actual = _sha256_file(c.manifest_path)
    if recorded != actual:
        return Finding(
            5,
            title,
            FAIL,
            f"report cites manifest {recorded} but the file hashes to {actual}",
        )
    return Finding(5, title, PASS, f"all bound fields present; manifest {actual}")


def law_6_provenance(c: Campaign) -> Finding:
    """Every number ships with the stack that produced it."""
    title = "Provenance"
    if not c.env_dir or not os.path.isdir(c.env_dir):
        return Finding(6, title, FAIL, "no environment report directory")
    runtime = c.env_runtime or {}
    if runtime.get("torch_error") or runtime.get("vllm_error"):
        return Finding(6, title, FAIL, "environment probe failed to import torch/vLLM")
    torch_version = (runtime.get("torch") or {}).get("version")
    vllm_version = (runtime.get("vllm") or {}).get("version")
    devices = runtime.get("devices") or []
    driver = (runtime.get("capture_runtime_manifest") or {}).get("driver")
    gaps = []
    if not torch_version:
        gaps.append("torch version")
    if not vllm_version:
        gaps.append("vLLM version")
    if not devices:
        gaps.append("GPU inventory")
    if not driver:
        gaps.append("driver version")
    if not runtime.get("vllm_commit"):
        gaps.append("vLLM commit")
    required_files = ("gpu-smi-query.txt", "repo-git-status.txt", "pip-freeze.txt")
    for name in required_files:
        if not os.path.isfile(os.path.join(c.env_dir, name)):
            gaps.append(name)
    if gaps:
        return Finding(6, title, FAIL, f"environment report lacks: {', '.join(gaps)}")
    detail = (
        f"torch {torch_version}, vLLM {vllm_version} "
        f"@ {str(runtime.get('vllm_commit'))[:12]}, driver {driver}, "
        f"{len(devices)} GPU(s)"
    )
    if runtime.get("vllm_tree_dirty"):
        detail += "; WORKING TREE DIRTY at capture"
    return Finding(6, title, PASS, detail)


def law_7_storage_integrity(c: Campaign) -> Finding:
    """Hidden-state storage requires a bitwise-exact replay probe."""
    title = "Storage integrity"
    storage = c.manifest.get("storage")
    if storage != "hidden":
        return Finding(7, title, PASS, f"storage is {storage!r}; no replay dependency")
    probe = c.manifest.get("replay_probe") or {}
    if not probe.get("identical"):
        return Finding(
            7,
            title,
            FAIL,
            f"hidden storage without a bitwise-exact replay probe: {probe!r}",
        )
    return Finding(7, title, PASS, "hidden storage with bitwise-exact replay")


def law_8_head_transparency(c: Campaign) -> Finding:
    """Trunk and deployed KLD are both reported."""
    title = "Head transparency"
    absent = _missing(c.report, ("trunk_mean_kld", "deployed_mean_kld"))
    if absent:
        return Finding(
            8,
            title,
            FAIL,
            f"report lacks {', '.join(absent)}; rerun with --decompose-head",
        )
    delta = c.report.get("head_delta_kld")
    return Finding(
        8,
        title,
        PASS,
        f"trunk {c.report['trunk_mean_kld']:.8f}, "
        f"deployed {c.report['deployed_mean_kld']:.8f}, delta {delta!r}",
    )


def law_9_tail_and_depth(c: Campaign) -> Finding:
    """A mean never travels alone."""
    title = "Tail and depth disclosure"
    absent = _missing(
        c.report,
        (
            "mean_kld",
            "median_kld",
            "p90_kld",
            "p99_kld",
            "max_kld",
            "top1_agreement",
            "depth_buckets",
            "confidence_buckets",
        ),
    )
    if absent:
        return Finding(9, title, FAIL, f"report lacks {', '.join(absent)}")
    if not c.report["depth_buckets"]:
        return Finding(9, title, FAIL, "depth buckets are empty")
    return Finding(
        9,
        title,
        PASS,
        f"mean {c.report['mean_kld']:.8f}, median {c.report['median_kld']:.8f}, "
        f"max {c.report['max_kld']:.8f}, "
        f"{len(c.report['depth_buckets'])} depth buckets",
    )


def comparability_key(c: Campaign) -> dict[str, Any]:
    """The tuple within which results may be ranked against each other."""
    runtime = c.manifest.get("runtime") or {}
    # Suite identity comes from what the run recorded, never from the suite file
    # handed to the audit. Reading it from the latter lets a run that tokenized at
    # run time borrow the suite_id of a suite it never opened, and report a
    # complete comparability key for tokens that came from somewhere else.
    suite_used = c.manifest.get("token_suite") or {}
    return {
        "laws_version": LAWS_VERSION,
        "reference_config_sha256": c.manifest.get("reference_config_sha256"),
        "suite_id": suite_used.get("suite_id"),
        "token_sha256": c.manifest.get("token_sha256"),
        "context_length": c.manifest.get("context_length"),
        "rows": c.manifest.get("rows"),
        "score_from": c.manifest.get("score_from"),
        "stride": c.manifest.get("stride"),
        "kld_vocab_size": c.manifest.get("kld_vocab_size"),
        "tensor_parallel_size": c.manifest.get("tensor_parallel_size"),
        "model_runner_v2": c.report.get("model_runner_v2"),
        "torch": runtime.get("torch"),
        "driver": runtime.get("driver"),
        "gpu_names": runtime.get("gpu_names"),
    }


def law_10_comparability(c: Campaign) -> Finding:
    """Every result carries the identity that bounds its comparisons."""
    title = "Comparability"
    key = comparability_key(c)
    unresolved = [name for name, value in key.items() if value is None]
    if unresolved:
        return Finding(
            10,
            title,
            FAIL,
            f"comparability key is incomplete: {', '.join(unresolved)}",
        )
    return Finding(10, title, PASS, "comparability key fully resolved")


def law_11_freeze_before_qualification(c: Campaign) -> Finding:
    """Parameters are frozen on analysis before qualification is read."""
    title = "Freeze before qualification"
    if c.partition != "qualification":
        return Finding(
            11, title, NOT_APPLICABLE, f"partition is {c.partition!r}"
        )
    receipt = c.freeze_receipt or {}
    absent = [k for k in ("timestamp", "parameters_sha256") if not receipt.get(k)]
    if absent:
        return Finding(
            11, title, FAIL, f"freeze receipt lacks {', '.join(absent)}"
        )
    return Finding(
        11,
        title,
        PASS,
        f"frozen at {receipt['timestamp']} ({receipt['parameters_sha256'][:12]})",
    )


def law_12_reusable_reference(c: Campaign) -> Finding:
    """A third party can score a candidate without the reference checkpoint."""
    title = "Reusable reference"
    if not c.artifact_dir or not os.path.isdir(c.artifact_dir):
        return Finding(12, title, FAIL, "no assembled artifact directory to verify")
    required = (
        "checksums.txt",
        "LAWS.md",
        os.path.join("suite", "suite-manifest.json"),
        os.path.join("reference", "manifest.json"),
        os.path.join("reference", "lm_head.safetensors"),
    )
    gaps = [
        name.replace(os.sep, "/")
        for name in required
        if not os.path.exists(os.path.join(c.artifact_dir, name))
    ]
    if gaps:
        return Finding(12, title, FAIL, f"artifact lacks: {', '.join(gaps)}")
    return Finding(12, title, PASS, "suite, reference, head, and checksums present")


def _cell_comparable(cell: dict[str, Any], c: Campaign) -> str | None:
    """Why a component cell cannot be compared to the deployed cell, or None.

    A cell measured on different tokens, a different partition, or against a
    different reference capture is not a decomposition of this number. It would
    look like one on a one-pager, which is the failure worth preventing.
    """
    if not isinstance(cell, dict):
        return "is not an object"
    if not isinstance(cell.get("mean_kld"), (int, float)):
        return "carries no mean_kld"
    for field_name, expected in (
        ("partition", c.partition),
        ("token_sha256", c.manifest.get("token_sha256")),
        ("reference_config_sha256", c.manifest.get("reference_config_sha256")),
    ):
        actual = cell.get(field_name)
        if expected is not None and actual != expected:
            return f"{field_name} is {actual!r}, deployed is {expected!r}"
    return None


def law_14_component_attribution(c: Campaign) -> Finding:
    """A routed model's number is attributed to router and experts separately."""
    title = "Component attribution"
    if "reference_routing" not in c.manifest:
        return Finding(
            14,
            title,
            FAIL,
            "capture manifest predates routing detection, so whether this law "
            "applies is unknown; recapture the reference rather than let a "
            "routed model exempt itself by omission",
        )
    routing = c.manifest.get("reference_routing")
    if not routing or not routing.get("num_experts"):
        return Finding(
            14, title, NOT_APPLICABLE, "reference declares no experts"
        )
    experts = routing["num_experts"]
    if not c.attribution:
        return Finding(
            14,
            title,
            FAIL,
            f"reference routes over {experts} experts, so a single mean cannot "
            f"be published: no attribution supplied (build the cells with "
            f"fidelity/qdq.py)",
        )

    expert_cell = c.attribution.get("expert_cell")
    if not isinstance(expert_cell, dict):
        return Finding(14, title, FAIL, "attribution has no expert_cell")
    problem = _cell_comparable(expert_cell, c)
    if problem:
        return Finding(14, title, FAIL, f"expert_cell {problem}")

    # Routing divergence is a measurement, not a cell: the candidate's routers
    # can be bit-identical to the reference's and still select other experts,
    # because the activations reaching them were perturbed upstream. No
    # weight-rounding cell can produce this number, so it is required directly.
    routing = c.report.get("routing")
    if not routing or routing.get("routing_excess_mean") is None:
        return Finding(
            14,
            title,
            FAIL,
            f"reference routes over {experts} experts but the run measured no "
            f"routing divergence; rescore with --measure-routing. Router "
            f"weight precision is not a substitute: an unquantized router "
            f"still receives perturbed activations",
        )
    floor = float(routing["routing_excess_mean"])
    flip_rate = float(routing.get("selection_flip_rate") or 0.0)

    router_cell = c.attribution.get("router_cell")
    if not isinstance(router_cell, dict):
        return Finding(14, title, FAIL, "attribution has no router_cell")
    if router_cell.get("status") == NOT_APPLICABLE:
        if not router_cell.get("evidence"):
            return Finding(
                14,
                title,
                FAIL,
                "router_cell is not_applicable but cites no inspection "
                "evidence that the deployed checkpoint leaves the router "
                "unquantized",
            )
        router_detail = "router weights unquantized"
    else:
        problem = _cell_comparable(router_cell, c)
        if problem:
            return Finding(14, title, FAIL, f"router_cell {problem}")
        router_detail = f"router weight rounding {float(router_cell['mean_kld']):.8f}"

    return Finding(
        14,
        title,
        PASS,
        f"experts {float(expert_cell['mean_kld']):.8f}, {router_detail}; "
        f"selections changed at {flip_rate * 100:.3f}% of (token, layer) "
        f"choices; ranking floor {floor:.8f}",
    )


def _scored_strata(c: Campaign) -> tuple[set[str], int]:
    """Strata the scored contexts actually came from, and how many carry an id.

    Derived here rather than trusted from the strata report, so a report that
    silently drops a domain cannot pass by declaring a smaller suite.
    """
    contexts = {
        int(entry["context_id"]): entry
        for entry in (c.suite or {}).get("contexts", [])
    }
    found: set[str] = set()
    matched = 0
    for record in c.report.get("per_context") or []:
        context_id = record.get("context_id")
        context = None if context_id is None else contexts.get(int(context_id))
        if context is None:
            continue
        matched += 1
        found.add(str(context.get("stratum")))
    return found, matched


def law_15_domain_disclosure(c: Campaign) -> Finding:
    """A stratified suite is not reported as a single mean."""
    title = "Domain disclosure"
    strata = (c.suite or {}).get("strata") if c.suite else None
    if not strata or len(strata) < 2:
        return Finding(
            15, title, NOT_APPLICABLE, "the suite declares no domain strata"
        )
    records = c.report.get("per_context")
    if not records:
        return Finding(
            15,
            title,
            FAIL,
            f"the suite is stratified over {len(strata)} domains but the report "
            f"carries no per-context records, so the mean cannot be attributed "
            f"to any of them; rescore with a build that records them",
        )
    rows = c.report.get("num_rows")
    if rows is not None and len(records) != rows:
        return Finding(
            15,
            title,
            FAIL,
            f"{len(records)} per-context records for {rows} scored rows",
        )
    found, matched = _scored_strata(c)
    if matched != len(records):
        return Finding(
            15,
            title,
            FAIL,
            f"{len(records) - matched} scored context(s) are not in the suite "
            f"manifest, so the report and the suite disagree about what was "
            f"measured",
        )
    if not c.strata:
        return Finding(
            15,
            title,
            FAIL,
            f"no domain report supplied for {len(found)} scored domain(s); "
            f"build it with fidelity/strata.py",
        )
    primary = c.strata.get("primary")
    groups = (c.strata.get("groups") or {}).get("stratum") or []
    published = {str(row.get("key")) for row in groups}
    if published != found:
        missing = sorted(found - published)
        extra = sorted(published - found)
        return Finding(
            15,
            title,
            FAIL,
            f"the domain report does not cover what was scored: missing "
            f"{missing or 'none'}, unexpected {extra or 'none'}",
        )
    covered = sum(
        int(((row.get("cells") or {}).get(primary) or {}).get("contexts") or 0)
        for row in groups
    )
    if covered != len(records):
        return Finding(
            15,
            title,
            FAIL,
            f"the domain report accounts for {covered} contexts of "
            f"{len(records)} scored",
        )
    ranked = sorted(
        groups,
        key=lambda r: float(
            ((r.get("cells") or {}).get(primary) or {}).get("mean_kld") or 0.0
        ),
    )
    worst = (ranked[-1].get("cells") or {}).get(primary) or {}
    best = (ranked[0].get("cells") or {}).get(primary) or {}
    spread = (
        float(worst.get("mean_kld") or 0.0) / float(best["mean_kld"])
        if best.get("mean_kld")
        else float("nan")
    )
    return Finding(
        15,
        title,
        PASS,
        f"{len(found)} domains disclosed; weakest {ranked[-1].get('key')} at "
        f"{float(worst.get('mean_kld') or 0.0):.8f}, strongest "
        f"{ranked[0].get('key')} at {float(best.get('mean_kld') or 0.0):.8f}, "
        f"spread {spread:.1f}x",
    )


def routing_floor(report: dict[str, Any] | None) -> float | None:
    """The measured routing term, below which a difference ranks nothing.

    This is the excess the mean carries because flipped positions diverge more
    than positions whose expert selection survived. It replaces the router-weight
    cell as the floor: rounding a router's weights is not what changes routing in
    a deployment, and on a checkpoint that ships a BF16 router that cell is
    exactly zero while the routing term is not.
    """
    if not isinstance(report, dict):
        return None
    routing = report.get("routing")
    if not isinstance(routing, dict):
        return None
    excess = routing.get("routing_excess_mean")
    return float(excess) if isinstance(excess, (int, float)) else None


# Registered after every check is defined. Numbering is append-only, so Law 14
# sits at the end even though it is evaluated with the rest.
LAWS: tuple[tuple[int, Callable[[Campaign], Finding]], ...] = (
    (1, law_1_zero_baseline),
    (2, law_2_determinism),
    (3, law_3_frozen_input),
    (4, law_4_real_vocabulary),
    (5, law_5_manifest_binding),
    (6, law_6_provenance),
    (7, law_7_storage_integrity),
    (8, law_8_head_transparency),
    (9, law_9_tail_and_depth),
    (10, law_10_comparability),
    (11, law_11_freeze_before_qualification),
    (12, law_12_reusable_reference),
    (14, law_14_component_attribution),
    (15, law_15_domain_disclosure),
)


def _approval_for(c: Campaign, law: int) -> dict[str, Any] | None:
    """Return a complete approval entry for ``law``, or None.

    An approval missing any of approver, justification, or timestamp is not an
    approval; Law 13 requires all three, so it is ignored and the underlying
    failure stands.
    """
    entry = c.approvals.get(str(law)) or c.approvals.get(f"law_{law}")
    if not isinstance(entry, dict):
        return None
    if all(entry.get(k) for k in ("approver", "justification", "timestamp")):
        return entry
    return None


def repeat_spread(study: dict[str, Any] | None) -> float | None:
    """Widest pairwise mean KLD among repeat captures of the same reference.

    This is the nondeterminism floor a Law 1 override is carrying. Returns None
    when the study is absent or does not contain three complete pairwise
    comparisons.
    """
    if not isinstance(study, dict):
        return None
    pairs = study.get("pairwise")
    if not isinstance(pairs, list) or len(pairs) != 3:
        return None
    means = [p.get("mean_kld") for p in pairs if isinstance(p, dict)]
    if len(means) != 3 or any(not isinstance(m, (int, float)) for m in means):
        return None
    return float(max(means))


def evaluate(c: Campaign) -> list[Finding]:
    """Evaluate every law, applying recorded overrides where permitted."""
    findings: list[Finding] = []
    incomplete_approvals: list[int] = []
    for law, check in LAWS:
        finding = check(c)
        if finding.status == FAIL:
            approval = _approval_for(c, law)
            if approval is not None and law in OVERRIDABLE:
                spread = repeat_spread(c.repeat_study)
                if law == 1 and spread is None:
                    finding.detail += (
                        "; an approval was supplied but Law 1 also requires a "
                        "three-capture repeat study with all three pairwise "
                        "comparisons scored"
                    )
                else:
                    finding.status = OVERRIDE
                    finding.approval = approval
                    if law == 1:
                        finding.detail += (
                            f"; nondeterminism floor from repeat captures: "
                            f"{spread:.8f}"
                        )
            elif approval is not None:
                finding.detail += (
                    " (an approval was supplied but this law permits no override)"
                )
            elif c.approvals.get(str(law)) or c.approvals.get(f"law_{law}"):
                incomplete_approvals.append(law)
        findings.append(finding)

    if incomplete_approvals:
        laws = ", ".join(str(law) for law in incomplete_approvals)
        findings.append(
            Finding(
                13,
                "Recorded deviation",
                FAIL,
                f"approvals for law(s) {laws} lack approver, justification, "
                "or timestamp",
            )
        )
    else:
        overrides = [f for f in findings if f.status == OVERRIDE]
        findings.append(
            Finding(
                13,
                "Recorded deviation",
                PASS,
                f"{len(overrides)} override(s), each fully attributed"
                if overrides
                else "no overrides claimed",
            )
        )
    return findings


def strata_summary(strata: dict[str, Any] | None) -> dict[str, Any] | None:
    """The domain table, without the per-source detail the report keeps."""
    if not isinstance(strata, dict):
        return None
    return {
        "primary": strata.get("primary"),
        "cells": strata.get("cells"),
        "overall": strata.get("overall"),
        "stratum": (strata.get("groups") or {}).get("stratum"),
    }


def build_receipt(c: Campaign, findings: list[Finding]) -> dict[str, Any]:
    """Assemble the publishable compliance receipt."""
    failures = [f.law for f in findings if f.status == FAIL]
    overrides = [f.law for f in findings if f.status == OVERRIDE]
    baseline = (c.self_report or {}).get("mean_kld")
    return {
        "zero_baseline_kld": baseline,
        "nondeterminism_floor": (
            repeat_spread(c.repeat_study) if baseline not in (0.0, None) else 0.0
        ),
        "attribution": c.attribution,
        "ranking_floor": routing_floor(c.report),
        "routing": c.report.get("routing"),
        "strata": strata_summary(c.strata),
        "program": "Local Inference Lab — Distribution Fidelity",
        "laws_version": LAWS_VERSION,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "partition": c.partition,
        "compliant": not failures,
        "failed_laws": failures,
        "overridden_laws": overrides,
        "comparability_key": comparability_key(c),
        "candidate": c.report.get("student_model"),
        "mean_kld": c.report.get("mean_kld"),
        "findings": [f.as_dict() for f in findings],
    }


def _load(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, help="candidate report JSON")
    parser.add_argument("--manifest", required=True, help="capture manifest JSON")
    parser.add_argument("--self-report", help="self-comparison report JSON (Law 1)")
    parser.add_argument("--env-dir", help="kld_env_report.sh output directory (Law 6)")
    parser.add_argument("--suite", help="suite manifest JSON (Law 3)")
    parser.add_argument("--artifact-dir", help="assembled artifact root (Law 12)")
    parser.add_argument(
        "--partition",
        default="analysis",
        choices=("analysis", "qualification"),
        help="which partition this result belongs to (Law 11)",
    )
    parser.add_argument("--freeze-receipt", help="freeze receipt JSON (Law 11)")
    parser.add_argument(
        "--repeat-study",
        help="three-capture repeat study JSON, required to override Law 1",
    )
    parser.add_argument(
        "--attribution",
        help="component cells and scheme ladder JSON, required for a routed "
        "reference (Law 14)",
    )
    parser.add_argument(
        "--strata",
        help="per-domain report JSON from fidelity/strata.py, required for a "
        "stratified suite (Law 15)",
    )
    parser.add_argument("--approvals", help="recorded deviations JSON (Law 13)")
    parser.add_argument("--out", help="write the receipt here")
    args = parser.parse_args()

    env_runtime = None
    if args.env_dir:
        runtime_path = os.path.join(args.env_dir, "runtime.json")
        if os.path.isfile(runtime_path):
            env_runtime = _load(runtime_path)

    campaign = Campaign(
        report=_load(args.report) or {},
        manifest=_load(args.manifest) or {},
        manifest_path=args.manifest,
        self_report=_load(args.self_report),
        env_dir=args.env_dir,
        env_runtime=env_runtime,
        suite=_load(args.suite),
        artifact_dir=args.artifact_dir,
        partition=args.partition,
        freeze_receipt=_load(args.freeze_receipt),
        repeat_study=_load(args.repeat_study),
        attribution=_load(args.attribution),
        strata=_load(args.strata),
        approvals=_load(args.approvals) or {},
    )

    findings = evaluate(campaign)
    receipt = build_receipt(campaign, findings)

    width = max(len(f.title) for f in findings)
    for finding in findings:
        print(
            f"  Law {finding.law:>2}  {finding.status.upper():<14} "
            f"{finding.title:<{width}}  {finding.detail}"
        )
    print(
        f"\ncompliant={receipt['compliant']} "
        f"failed={receipt['failed_laws']} overridden={receipt['overridden_laws']}"
    )

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(receipt, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"receipt: {args.out}")

    return 0 if receipt["compliant"] else 1


if __name__ == "__main__":
    sys.exit(main())
