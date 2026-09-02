#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run a Local Inference Lab distribution-fidelity campaign end to end.

One config describes a campaign: a suite, and per reference model a set of
candidate checkpoints. The same code path then ingests, scores, and assembles
every model identically, which is the point — a comparison between two
candidates is only meaningful if nothing about the procedure differed.

Subcommands:
    download   fetch any checkpoint that is named by repo but absent locally
    score      capture the environment, gate on the zero baseline, score candidates
    assemble   build the library tree, checksums, receipts, one-pagers, leaderboard
    release    drop candidate weights this campaign leased and no longer needs
    all        score then assemble

A candidate that fails provenance, download, or scoring is skipped; the rest of
the sweep continues and the process exits non-zero. Law 1 still stops the model.

With ``fetch: lease`` in the campaign JSON, ``score`` downloads one candidate at
a time and deletes those weights after a successful score. The reference is never
leased. Only a directory this campaign fetched (a lease file under ``work/leases``)
is eligible for deletion.

Ordering is enforced, not suggested: the zero baseline gates every candidate
(Law 1), and checksums precede compliance (Law 12).
"""

import argparse
import glob
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, NamedTuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from redaction import redact_env  # noqa: E402 - sibling module
import provenance as _provenance  # noqa: E402 - sibling module

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
SCORER = os.path.join(REPO_ROOT, "examples", "offline_inference", "score_mode_kld.py")

# Last-resort cap so vLLM is never told it can claim the CUDA context.
UTIL_CEILING = 0.93
UTIL_FLOOR = 0.15
# CUDA context and cuBLAS workspaces reserved on every card.
CONTEXT_GIB = 2.0
# Residual activations inside the engine, not the scoring buffers.
ACTIVATION_GIB = 2.0
# vLLM's default cache block size (vllm.config.cache.CacheConfig.DEFAULT_BLOCK_SIZE).
BLOCK_SIZE = 16
# Floor so vLLM never receives a cache too small to hold one block.
KV_FLOOR_GIB = 0.25
# score_mode_kld.py sets max_model_len = context_length * 2. The KV
# reservation has to cover that engine length, not the scored window.
ENGINE_LEN_FACTOR = 2
# Two .float() copies plus their log_softmax outputs in compute_kld_chunk.
FP32_COPIES = 4
# Peak vs live tensors; expandable_segments keeps this from growing further.
ALLOCATOR_SLACK = 1.25
TP_CANDIDATES = (1, 2, 4, 8)


class CampaignError(Exception):
    """One candidate failed; the campaign may continue with the rest."""


@dataclass
class Candidate:
    name: str
    path: str
    hf_repo: str | None = None
    revision: str | None = None


@dataclass
class Model:
    name: str
    reference_path: str
    candidates: list[Candidate]
    reference_repo: str | None = None
    reference_revision: str | None = None


@dataclass
class Config:
    name: str
    library: str
    work: str
    models: list[Model]
    suite_dir: str | None = None
    suite_partition: str = "analysis"
    rows: int = 1024
    context_length: int = 2048
    score_from: int = 0
    runner_v2: bool = False
    storage: str = "hidden"
    max_num_seqs: int = 1
    partition: str = "analysis"
    dataset: str = "Salesforce/wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    # The expert cell is the one that discriminates between formats, so a routed
    # model carries it for every scheme on the ladder, not only the deployed one.
    ladder: list[str] = field(
        default_factory=lambda: ["fp8_block", "mxfp8", "nvfp4"]
    )
    prune_variants: bool = True
    approvals: str | None = None
    # "upfront" fetches every candidate during download. "lease" fetches one
    # candidate at a time during score and deletes those weights afterwards.
    fetch: str = "upfront"
    tensor_parallel_size: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def load_config(path: str) -> Config:
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    models = []
    for entry in raw["models"]:
        candidates = [
            Candidate(
                name=c["name"],
                path=c["path"],
                hf_repo=c.get("hf_repo"),
                revision=c.get("revision"),
            )
            for c in entry.get("candidates", [])
        ]
        models.append(
            Model(
                name=entry["name"],
                reference_path=entry["reference"]["path"],
                reference_repo=entry["reference"].get("hf_repo"),
                reference_revision=entry["reference"].get("revision"),
                candidates=candidates,
            )
        )
    known = {f for f in Config.__dataclass_fields__ if f not in {"models", "extra"}}
    kwargs = {k: v for k, v in raw.items() if k in known}
    kwargs["extra"] = {k: v for k, v in raw.items() if k not in known and k != "models"}
    return Config(models=models, **kwargs)


def resolve_python() -> str:
    """The interpreter vLLM is installed into. Never a system interpreter."""
    for candidate in (
        os.environ.get("KLD_PYTHON"),
        os.path.join(os.environ.get("VIRTUAL_ENV", ""), "bin", "python")
        if os.environ.get("VIRTUAL_ENV")
        else None,
        os.path.join(REPO_ROOT, ".venv", "bin", "python"),
        sys.executable,
    ):
        if candidate and os.path.exists(candidate):
            probe = subprocess.run(
                [candidate, "-c", "import vllm"],
                capture_output=True,
            )
            if probe.returncode == 0:
                return candidate
    raise SystemExit(
        "no interpreter can import vllm. Run fidelity/bootstrap.sh, or set "
        "KLD_PYTHON=/path/to/venv/bin/python"
    )


def checkpoint_gib(path: str) -> float:
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for name in filenames:
            if name.endswith((".safetensors", ".bin")):
                try:
                    total += os.path.getsize(os.path.join(dirpath, name))
                except OSError:
                    pass
    return total / (1 << 30)


def _parse_smi_rows(text: str) -> list[tuple[int, str, float, float]]:
    """Parse nvidia-smi csv: index, uuid, total MiB, free MiB."""
    rows: list[tuple[int, str, float, float]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            raise SystemExit(f"unreadable nvidia-smi row: {line}")
        try:
            rows.append((int(parts[0]), parts[1], float(parts[2]), float(parts[3])))
        except ValueError as exc:
            raise SystemExit(f"unreadable nvidia-smi row: {line}") from exc
    return rows


def _select_visible(
    rows: list[tuple[int, str, float, float]], visible: str | None
) -> list[tuple[int, str, float, float]]:
    """Map CUDA_VISIBLE_DEVICES onto nvidia-smi rows by index or UUID."""
    if not visible or not visible.strip():
        return rows
    by_index = {row[0]: row for row in rows}
    selected: list[tuple[int, str, float, float]] = []
    for tok in visible.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok.isdigit():
            row = by_index.get(int(tok))
        else:
            row = next((r for r in rows if r[1] == tok), None)
            if row is None:
                row = next((r for r in rows if r[1].endswith(tok)), None)
        if row is None:
            raise SystemExit(
                f"CUDA_VISIBLE_DEVICES token {tok!r} matches no GPU"
            )
        selected.append(row)
    return selected


def gpu_inventory() -> tuple[int, float, float]:
    """Visible GPU count, per-GPU total GiB, and minimum free GiB."""
    try:
        text = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"cannot query GPUs via nvidia-smi: {exc}") from exc
    rows = _select_visible(
        _parse_smi_rows(text), os.environ.get("CUDA_VISIBLE_DEVICES")
    )
    if not rows:
        raise SystemExit("nvidia-smi reported no GPUs")
    total = min(row[2] for row in rows) / 1024.0
    free = min(row[3] for row in rows) / 1024.0
    return len(rows), total, free


def engine_max_model_len(context_length: int) -> int:
    """The max_model_len score_mode_kld.py will pass to vLLM."""
    return context_length * ENGINE_LEN_FACTOR


def kv_cache_gib(
    geometry: dict[str, Any],
    context_length: int,
    max_num_seqs: int,
    tp: int,
) -> float:
    """KV reservation for the engine's max_model_len, replicated heads included."""
    tokens = math.ceil(context_length / BLOCK_SIZE) * BLOCK_SIZE
    heads = math.ceil(geometry["num_key_value_heads"] / tp)
    nbytes = (
        2
        * geometry["num_hidden_layers"]
        * heads
        * geometry["head_dim"]
        * tokens
        * max_num_seqs
        * geometry["elem_bytes"]
    )
    return max(nbytes / (1 << 30), KV_FLOOR_GIB)


def logits_headroom_gib(
    geometry: dict[str, Any],
    context_length: int,
    max_num_seqs: int,
) -> float:
    """Peak FP32 scoring buffers over the unpadded vocabulary."""
    positions = context_length * max_num_seqs
    nbytes = positions * geometry["vocab_size"] * (
        FP32_COPIES * 4 + 2 * geometry["elem_bytes"]
    )
    return ALLOCATOR_SLACK * nbytes / (1 << 30)


class GpuPlan(NamedTuple):
    tp: int
    util: float
    weight_gib: float
    kv_gib: float
    logits_gib: float
    pinned: bool


def _load_geometry(paths: list[str]) -> dict[str, Any]:
    for path in reversed(paths):
        if os.path.isfile(os.path.join(path, "config.json")):
            return _provenance.scoring_geometry(_provenance.load_config(path))
    raise SystemExit(
        "cannot plan GPUs: no config.json among " + ", ".join(paths)
    )


def plan_gpus(
    paths: list[str],
    config: Config,
    *,
    geometry: dict[str, Any] | None = None,
    inventory: tuple[int, float, float] | None = None,
    weight_gib: float | None = None,
) -> GpuPlan:
    """Smallest tensor-parallel size that leaves room for scoring buffers.

    TP is chosen so weights, KV, and logits fit in free memory. Utilization
    describes only what the vLLM engine claims: the FP32 copies are ordinary
    torch allocations, so that room has to be memory vLLM was never told it
    could take.
    """
    weights = (
        weight_gib
        if weight_gib is not None
        else max((checkpoint_gib(p) for p in paths), default=0.0)
    )
    geom = geometry if geometry is not None else _load_geometry(paths)
    count, total, free = inventory if inventory is not None else gpu_inventory()
    usable = free - CONTEXT_GIB
    kv_len = engine_max_model_len(config.context_length)
    logits = logits_headroom_gib(geom, config.context_length, config.max_num_seqs)
    pinned = config.tensor_parallel_size is not None
    if pinned:
        tp = int(config.tensor_parallel_size)
        if tp < 1:
            raise SystemExit(f"tensor_parallel_size must be >= 1, got {tp}")
        if tp > count:
            raise SystemExit(f"pinned TP={tp} but only {count} GPU(s) visible")
    else:
        tp = next(
            (
                candidate
                for candidate in TP_CANDIDATES
                if candidate <= count
                and weights / candidate
                + kv_cache_gib(
                    geom, kv_len, config.max_num_seqs, candidate
                )
                + logits
                <= usable
            ),
            count,
        )
    kv = kv_cache_gib(geom, kv_len, config.max_num_seqs, tp)
    util = (weights / tp + kv + ACTIVATION_GIB) / total
    util = min(max(util, UTIL_FLOOR), UTIL_CEILING)
    pin_note = " (pinned)" if pinned else ""
    print(
        f"plan: weights {weights:.2f} GiB, kv {kv:.2f} GiB "
        f"(max_model_len={kv_len}), logits {logits:.2f} GiB, "
        f"{count} x {total:.2f} free {free:.2f}"
        f"\n      -> TP={tp} util={util:.2f}{pin_note}"
    )
    return GpuPlan(tp, util, weights, kv, logits, pinned)


def _run(cmd: list[str], log_path: str | None = None, env: dict | None = None) -> int:
    """Run a command, teeing to a log so a failure leaves evidence."""
    print("$ " + " ".join(cmd))
    merged = {**os.environ, **(env or {})}
    if log_path is None:
        return subprocess.run(cmd, env=merged).returncode
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(
            cmd, env=merged, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, errors="replace", bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        return proc.wait()


def inspect_record(work: str, name: str) -> str:
    return os.path.join(work, "inspect", f"{name}.json")


def inspect_checkpoint(
    python: str, path: str, durable: str | None = None
) -> str | None:
    """Write inspect.json. Prefer a durable work-tree copy over the checkpoint.

    The beside-checkpoint file is a cache: releasing leased weights deletes it,
    so assembly and a re-download both need the work-tree record to survive.
    """
    cache = os.path.join(path, "inspect.json")
    dest = durable or cache
    if durable:
        os.makedirs(os.path.dirname(os.path.abspath(durable)), exist_ok=True)
        if os.path.isfile(durable):
            print(f"=== inspect already at {durable}")
            return durable
        if os.path.isfile(cache):
            shutil.copy2(cache, durable)
            print(f"=== inspect copied {cache} -> {durable}")
            return durable
    elif os.path.isfile(cache):
        print(f"=== inspect already at {cache}")
        return cache
    cmd = [
        python, os.path.join(HERE, "qdq.py"),
        "--inspect", path,
        "--json-out", dest,
    ]
    print(f"=== inspecting {path}")
    if _run(cmd) != 0:
        print(f"WARNING  inspect failed for {path}", file=sys.stderr)
        return None
    if durable and dest != cache and os.path.isdir(path):
        try:
            shutil.copy2(dest, cache)
        except OSError:
            pass
    return dest


def refuse_if_unloadable(path: str) -> None:
    """Raise CampaignError if config.json says vLLM will crash on load."""
    import qdq as _qdq
    reason = _qdq.unloadable_reason(path)
    if reason:
        raise CampaignError(f"will not load: {reason}")


def fetch_checkpoint(repo: str, dest: str, revision: str | None) -> int:
    cmd = ["hf", "download", repo, "--local-dir", dest]
    if revision:
        cmd += ["--revision", revision]
    print(f"=== downloading {repo} -> {dest}")
    return _run(cmd)


def lease_file(work: str, name: str) -> str:
    return os.path.join(work, "leases", f"{name}.json")


def write_lease(work: str, cand: Candidate) -> str:
    dest = lease_file(work, name=cand.name)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    payload = {
        "name": cand.name,
        "path": cand.path,
        "hf_repo": cand.hf_repo,
        "revision": cand.revision,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(dest, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return dest


def load_lease(work: str, name: str) -> dict[str, Any] | None:
    path = lease_file(work, name)
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def reference_paths(config: Config) -> set[str]:
    return {os.path.abspath(model.reference_path) for model in config.models}


def release_leased(
    work: str, cand: Candidate, *, protected: set[str]
) -> float | None:
    """Delete leased weights. GiB reclaimed, 0 if already gone, None if refused.

    No lease is a no-op. A path mismatch or a protected (reference) path is
    refused: the directory is left untouched and the lease stays.
    """
    lease = load_lease(work, cand.name)
    if lease is None:
        return 0.0
    dest = cand.path
    if os.path.abspath(dest) != os.path.abspath(lease.get("path") or ""):
        print(
            f"WARNING  lease for {cand.name} points at {lease.get('path')}, "
            f"not {dest}; refusing to delete",
            file=sys.stderr,
        )
        return None
    abs_dest = os.path.abspath(dest)
    if abs_dest in protected:
        print(
            f"WARNING  refusing to release reference path {dest}",
            file=sys.stderr,
        )
        return None
    reclaimed = checkpoint_gib(dest) if os.path.isdir(dest) else 0.0
    if os.path.isdir(dest):
        print(f"=== releasing {cand.name} ({reclaimed:.2f} GiB) at {dest}")
        shutil.rmtree(dest)
    try:
        os.remove(lease_file(work, cand.name))
    except OSError:
        pass
    return reclaimed


def cmd_download(config: Config, python: str | None = None) -> int:
    """Fetch checkpoints named by repo that are not already on disk.

    With fetch=lease the reference is fetched here and candidates wait for
    score, which downloads, scores, and releases them one at a time.
    """
    pending: list[tuple[str, str, str | None]] = []
    for model in config.models:
        if not os.path.isdir(model.reference_path) and model.reference_repo:
            pending.append(
                (model.reference_repo, model.reference_path,
                 model.reference_revision)
            )
        if config.fetch == "lease":
            continue
        for cand in model.candidates:
            if not os.path.isdir(cand.path) and cand.hf_repo:
                pending.append((cand.hf_repo, cand.path, cand.revision))

    required = [model.reference_path for model in config.models]
    if config.fetch != "lease":
        required.extend(
            cand.path for model in config.models for cand in model.candidates
        )
    missing = [p for p in required if not os.path.isdir(p)]
    missing_unfetchable = [
        p
        for p in missing
        if p not in {dest for _, dest, _ in pending}
    ]
    if missing_unfetchable:
        raise SystemExit(
            "these checkpoints are absent and have no hf_repo to fetch them "
            "from:\n  " + "\n  ".join(missing_unfetchable)
        )
    if config.fetch == "lease":
        print("fetch=lease: candidates will be downloaded during score")

    failed: list[str] = []
    for repo, dest, revision in pending:
        if fetch_checkpoint(repo, dest, revision) != 0:
            print(f"FAILED  download {repo}", file=sys.stderr)
            failed.append(repo)
    if not pending:
        print("every checkpoint this stage needs is already local")

    # Inspection is CPU-only and answers what each checkpoint actually
    # quantized, which is worth knowing before a GPU-hour is spent.
    if python:
        for model in config.models:
            for cand in model.candidates:
                config_path = os.path.join(cand.path, "config.json")
                if os.path.isdir(cand.path) and os.path.isfile(config_path):
                    inspect_checkpoint(
                        python, cand.path,
                        inspect_record(config.work, cand.name),
                    )
                    import qdq as _qdq
                    reason = _qdq.unloadable_reason(cand.path)
                    if reason:
                        print(
                            f"WARNING  {cand.name} will not load: {reason}",
                            file=sys.stderr,
                        )
    if failed:
        print(
            f"{len(failed)} download(s) failed: {', '.join(failed)}",
            file=sys.stderr,
        )
        return 1
    return 0


def _repo_head() -> str | None:
    probe = subprocess.run(
        ["git", "-C", REPO_ROOT, "rev-parse", "HEAD"],
        capture_output=True, text=True,
    )
    return probe.stdout.strip() or None if probe.returncode == 0 else None


def capture_environment(config: Config, python: str) -> str:
    """Capture the environment report, refreshing it when the tree moved (Law 6).

    Called when a scoring run is about to happen, never merely because a campaign
    was invoked. Refreshing it for a campaign whose every report is already cached
    would stamp the artifacts with a tree that scored nothing, which is the failure
    Law 6 exists to prevent rather than a fix for it.
    """
    env_dir = os.path.join(config.work, "environment")
    runtime_path = os.path.join(env_dir, "runtime.json")
    if os.path.isfile(runtime_path):
        head = _repo_head()
        recorded = None
        try:
            with open(runtime_path, encoding="utf-8") as handle:
                recorded = json.load(handle).get("vllm_commit")
        except (OSError, json.JSONDecodeError):
            pass
        if head and recorded and head == recorded:
            print(f"=== environment already captured at {head[:12]}")
            return env_dir
        print(
            f"=== recapturing environment: recorded {str(recorded)[:12]}, "
            f"tree is at {str(head)[:12]}"
        )
    script = os.path.join(REPO_ROOT, "scripts", "kld_env_report.sh")
    models = [
        p
        for model in config.models
        for p in [model.reference_path, *(c.path for c in model.candidates)]
    ]
    print("=== capturing environment")
    rc = _run(
        ["bash", script, env_dir, *models],
        log_path=os.path.join(config.work, "logs", "environment.log"),
        env={"KLD_PYTHON": python},
    )
    if rc != 0 or not os.path.isfile(os.path.join(env_dir, "runtime.json")):
        raise SystemExit(f"environment capture failed; see {env_dir}")
    return env_dir


def score_identity(
    config: Config,
    label: str,
    student: str,
    teacher: str,
    rows: int,
    suite_limit: int | None = None,
    plan_from: str | None = None,
) -> tuple[str, str, int, float, float]:
    """The tag, suffix, and GPU plan a scoring run would use.

    Separated from the run so a caller can find the report a run would produce
    without producing it. `plan_from` stands in for the student when planning: a
    QDQ variant has the reference's geometry, and once its weights are pruned the
    variant on disk no longer implies the tensor-parallel degree it was scored at.
    """
    plan = plan_gpus([plan_from or student, teacher], config)
    # Everything the capture manifest binds itself to belongs in the directory
    # name, or a reused capture becomes a confusing abort instead of a recapture.
    if config.suite_dir:
        scope = f"{config.suite_partition}{suite_limit or ''}"
    else:
        scope = f"r{rows}"
    suffix = (
        f"-v{int(config.runner_v2)}-tp{plan.tp}-{scope}"
        f"-c{config.context_length}-s{config.score_from}"
    )
    return f"{label}{suffix}", suffix, plan.tp, plan.util, plan.kv_gib


def score_report(
    config: Config,
    label: str,
    student: str,
    teacher: str,
    rows: int,
    suite_limit: int | None = None,
    plan_from: str | None = None,
) -> str:
    """Where a scoring run's report would land."""
    tag, *_ = score_identity(
        config, label, student, teacher, rows, suite_limit, plan_from
    )
    return os.path.join(config.work, "reports", f"{tag}.json")


def bind_weights(report_path: str, student: str) -> str | None:
    """Record on the report the identity of the weights that were scored.

    Returns the digest, or None when the weights are already gone. A report that
    cites a different digest than the directory now holds is refused rather than
    rewritten: the two disagree about what was measured, and only a rescore can
    settle it.
    """
    sys.path.insert(0, HERE)
    import qdq

    if not os.path.isdir(student):
        return None
    try:
        digest = qdq.weights_identity(student)
    except SystemExit as exc:
        # Law 16 reports the gap on the receipt. Discarding a finished score over
        # an unhashable checkpoint would cost a GPU-hour and prove nothing.
        print(f"WARNING  cannot bind {student}: {exc}", file=sys.stderr)
        return None
    with open(report_path, encoding="utf-8") as handle:
        report = json.load(handle)
    recorded = report.get("student_weights_sha256")
    if recorded and recorded != digest:
        raise CampaignError(
            f"{os.path.basename(report_path)} was scored against weights "
            f"{recorded[:16]} but {student} now holds {digest[:16]}. The report "
            f"and the checkpoint disagree about what was measured; rescore."
        )
    if recorded:
        return digest
    report["student_weights_sha256"] = digest
    with open(report_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return digest


def weight_collisions(
    scored: list[tuple[str, Any, str | None]],
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Split a family's candidates into impossible pairs and honest duplicates.

    Two candidates that scored the same mean from different weights is not a
    coincidence a suite of this size produces; it is one checkpoint measured
    twice, and both numbers are refused because nothing says which directory
    the scorer actually read. Two candidates that share a digest are the same
    weights under two names, which is a real finding about the upstream repos
    rather than a defect here, so they are reported and published once.

    Returns (impossible, duplicates), each a list of name pairs.
    """
    impossible: list[tuple[str, str]] = []
    duplicates: list[tuple[str, str]] = []
    for i, (name, mean, digest) in enumerate(scored):
        for other, other_mean, other_digest in scored[i + 1:]:
            same_weights = bool(digest) and digest == other_digest
            if same_weights:
                duplicates.append((name, other))
            elif (
                isinstance(mean, (int, float))
                and mean == other_mean
                and digest
                and other_digest
            ):
                impossible.append((name, other))
    return impossible, duplicates


def score_one(
    config: Config,
    python: str,
    label: str,
    student: str,
    teacher: str,
    rows: int,
    decompose: bool,
    suite_limit: int | None = None,
    capture_label: str | None = None,
    measure_routing: bool = False,
    plan_from: str | None = None,
) -> tuple[str, str]:
    """Score one pair. Returns (report_path, capture_dir).

    `capture_label` lets several candidates share one teacher capture. The
    manifest binds a capture to the tokens, geometry, and runtime rather than to
    the candidate, so recapturing the same reference per candidate would spend
    tens of gigabytes and a forward pass to produce identical tensors. It matters
    once a routed model adds a component cell and a scheme ladder.
    """
    tag, suffix, tp, util, kv = score_identity(
        config, label, student, teacher, rows, suite_limit, plan_from
    )
    capture = os.path.join(
        config.work, "captures", f"{capture_label or label}{suffix}"
    )
    report = os.path.join(config.work, "reports", f"{tag}.json")
    log = os.path.join(config.work, "logs", f"{tag}.log")
    os.makedirs(os.path.dirname(report), exist_ok=True)

    if os.path.isfile(report):
        print(f"=== {tag} already scored")
        bind_weights(report, student)
        return report, capture

    capture_environment(config, python)

    cmd = [
        python, SCORER,
        "--model", student,
        "--reference-model", teacher,
        "--reference-logits", capture,
        "--rows", str(rows),
        "--context-length", str(config.context_length),
        "--score-from", str(config.score_from),
        "--storage", config.storage,
        "--probe-replay",
        "--language-model-only",
        "--max-num-seqs", str(config.max_num_seqs),
        "--report-json", report,
        "--tensor-parallel-size", str(tp),
        "--gpu-memory-utilization", f"{util:.2f}",
        "--kv-cache-memory-gib", f"{kv:.3f}",
    ]
    if config.suite_dir:
        cmd += ["--token-suite", config.suite_dir,
                "--suite-partition", config.suite_partition]
        if suite_limit:
            cmd += ["--suite-limit", str(suite_limit)]
    else:
        cmd += ["--dataset", config.dataset,
                "--dataset-config", config.dataset_config]
    if decompose:
        cmd.append("--decompose-head")
    if measure_routing:
        # Reference selections live beside the shared capture, so the extra
        # reference pass is paid once per model rather than once per cell.
        cmd += [
            "--measure-routing",
            "--routing-dir",
            os.path.join(
                config.work, "captures", f"{capture_label or label}{suffix}-routing"
            ),
        ]

    print(f"=== {tag} (TP={tp} util={util:.2f} kv={kv:.2f} GiB)")
    rc = _run(cmd, log_path=log, env={
        "VLLM_USE_V2_MODEL_RUNNER": str(int(config.runner_v2)),
        # Scoring allocates and frees vocabulary-wide tensors thousands of
        # times; without expandable segments the pool fragments.
        "PYTORCH_CUDA_ALLOC_CONF": os.environ.get(
            "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
        ),
    })
    if rc != 0:
        raise CampaignError(f"scoring failed for {tag}; see {log}")
    bind_weights(report, student)
    return report, capture


def variant_path(
    config: Config,
    reference: str,
    components: str,
    scheme: str | None,
    match_digest: str | None = None,
) -> str:
    """Where a QDQ variant lives, whether or not its weights are still there.

    `match_digest` is the short per-tensor suffix. None keeps the path a
    fully-covered conversion already used, so those reports stay reusable.
    """
    slug = components.replace(",", "-")
    name = f"{os.path.basename(reference)}-qdq-{slug}-{scheme or 'matched'}"
    if match_digest:
        name += f"-m{match_digest}"
    return os.path.join(config.work, "variants", name)


def _manifest_block(path: str) -> int | None:
    """The grid a built variant recorded, or None if it did not record one."""
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle).get("block_size")
    except (OSError, json.JSONDecodeError):
        return None


def build_variant(
    config: Config,
    python: str,
    reference: str,
    match: str,
    components: str,
    scheme: str | None,
    *,
    per_tensor: bool = False,
    expect_digest: str | None = None,
    block_size: int | None = None,
) -> str:
    """Build a QDQ variant if absent, and return its path.

    The weights are deleted once its report exists (`prune_variants`), because a
    scheme ladder is several checkpoint copies and the report plus the QDQ
    manifest are what an artifact publishes.

    A variant path names its components and scheme but not its grid, so an
    existing one is reused only when its manifest records the same block size.
    Otherwise two widths would share a directory under one label.
    """
    out = variant_path(
        config, reference, components, scheme, match_digest=expect_digest
    )
    name = os.path.basename(out)
    manifest = os.path.join(out, "qdq-manifest.json")
    if os.path.isfile(manifest) and glob.glob(os.path.join(out, "*.safetensors")):
        if block_size is None or _manifest_block(manifest) == block_size:
            print(f"=== variant {name} already built")
            return out
        print(
            f"=== rebuilding {name}: it was rounded on a "
            f"{_manifest_block(manifest)}-wide grid, not {block_size}",
            file=sys.stderr,
        )
    cmd = [
        python, os.path.join(HERE, "qdq.py"),
        "--model", reference,
        "--out", out,
        "--components", components,
        "--match", match,
    ]
    if scheme:
        cmd += ["--scheme", scheme]
    if block_size is not None:
        cmd += ["--block-size", str(block_size)]
    if per_tensor:
        cmd += ["--per-tensor"]
    if expect_digest:
        cmd += ["--expect-match-digest", expect_digest]
    print(f"=== building variant {name}")
    rc = _run(cmd, log_path=os.path.join(config.work, "logs", f"{name}.log"))
    if rc != 0:
        raise CampaignError(f"variant build failed for {name}")
    return out


def _prune_variant(path: str) -> None:
    """Drop a variant's weights, keeping its QDQ manifest as provenance."""
    for name in sorted(os.listdir(path)):
        if name == "qdq-manifest.json":
            continue
        target = os.path.join(path, name)
        if os.path.isfile(target):
            os.remove(target)
    print(f"  pruned variant weights in {path}")


def _cell(
    report_path: str,
    capture_manifest: dict[str, Any],
    config: Config,
    variant: str,
    scheme: str,
    match_mode: str,
) -> dict[str, Any]:
    """One attribution cell, carrying the identity Law 14 compares against."""
    with open(report_path, encoding="utf-8") as handle:
        report = json.load(handle)
    routing = report.get("routing") or {}
    return {
        "mean_kld": report.get("mean_kld"),
        # Recorded per cell because it is the evidence that no weight-rounding
        # cell holds expert selection fixed.
        "selection_flip_rate": routing.get("selection_flip_rate"),
        "routing_excess_mean": routing.get("routing_excess_mean"),
        "partition": config.partition,
        "token_sha256": capture_manifest.get("token_sha256"),
        "reference_config_sha256": capture_manifest.get("reference_config_sha256"),
        "variant": os.path.basename(variant),
        "scheme": scheme,
        "match_mode": match_mode,
        "report": report_path,
        "qdq_manifest": os.path.join(variant, "qdq-manifest.json"),
    }


def _partial_components(
    inspection: dict[str, Any], components: list[str]
) -> list[str]:
    """Components the checkpoint quantizes only in part, of those given."""
    partial = []
    for component in components:
        counts = inspection.get("coverage", {}).get(component) or {}
        done = counts.get("quantized") or 0
        total = counts.get("weights") or 0
        if done and total and done < total:
            partial.append(component)
    return partial


def _append_deployed_ladder(
    ladder: list[dict[str, Any]],
    deployed_scheme: str | None,
    expert: dict[str, Any],
) -> None:
    """Put the deployed format on the ladder without another scoring run.

    Config.ladder is the shared FP8/NVFP4 rungs. The rung is the
    component-wide expert cell at the deployed scheme, not the per-tensor
    expert cell that decomposes the deployed mean.
    """
    if not deployed_scheme:
        return
    if any(entry.get("scheme") == deployed_scheme for entry in ladder):
        return
    rung = {
        "scheme": deployed_scheme,
        "mean_kld": expert.get("mean_kld"),
        "variant": expert.get("variant"),
        "report": expert.get("report"),
    }
    if expert.get("unavailable"):
        rung["unavailable"] = expert["unavailable"]
    ladder.append(rung)


def attribute_model(
    config: Config,
    python: str,
    model: Model,
    deployed: Candidate,
    capture_dir: str,
    deployed_report: str,
) -> str | None:
    """Score the component cells and the scheme ladder for a routed model.

    Returns the attribution path, or None when the reference is dense and Law 14
    does not apply.
    """
    sys.path.insert(0, HERE)
    import qdq

    routing = qdq_routing(model.reference_path)
    if not routing:
        return None

    inspection = qdq.inspect(deployed.path)
    public_inspection = qdq.inspect_for_disk(inspection)
    deployed_scheme = inspection["detected_scheme"]
    if deployed_scheme is None:
        raise CampaignError(
            f"{deployed.name} carries no weight scales, so there is no deployed "
            f"scheme to attribute. Law 14 needs one; is this candidate actually "
            f"quantized?"
        )
    router_quantized = bool(
        (inspection["coverage"].get("router") or {}).get("quantized")
    )
    print(
        f"Law 14: {model.name} routes over {routing['num_experts']} experts; "
        f"{deployed.name} is {deployed_scheme}, router "
        f"{'quantized' if router_quantized else 'left in BF16'}"
    )

    manifest_path = os.path.join(capture_dir, "manifest.json")
    capture_manifest: dict[str, Any] = {}
    if os.path.isfile(manifest_path):
        with open(manifest_path, encoding="utf-8") as handle:
            capture_manifest = json.load(handle)

    def score_variant(
        components: str, scheme: str | None, *, per_tensor: bool = False
    ) -> dict[str, Any]:
        parts = tuple(
            part.strip() for part in components.split(",") if part.strip()
        )
        # A cell decomposes this deployment, so it is rounded on the grid the
        # deployment used. A rung at any other scheme is rounded on that
        # scheme's own grid, never on the neighbouring candidate's.
        block = (
            inspection.get("detected_block")
            if scheme == deployed_scheme
            else None
        ) or (qdq.scheme_block_size(scheme, None) if scheme else None)
        digest = (
            qdq.match_digest(model.reference_path, inspection, parts)
            if per_tensor
            else None
        )
        label = (
            f"{model.name}-qdq-{components.replace(',', '-')}-"
            f"{scheme or 'matched'}"
        )
        if digest:
            label += f"-m{digest}"
        variant = variant_path(
            config,
            model.reference_path,
            components,
            scheme,
            match_digest=digest,
        )
        # A variant path names its components and scheme but not its grid, so a
        # report left by a different width would otherwise be reused under this
        # label. Refuse rather than guess which one the number describes.
        built = os.path.join(variant, "qdq-manifest.json")
        recorded = _manifest_block(built) if os.path.isfile(built) else None
        if block is not None and recorded is not None and recorded != block:
            raise CampaignError(
                f"{os.path.basename(variant)} was rounded on a {recorded}-wide "
                f"grid but this rung is {block}-wide. Its report describes a "
                f"format this label does not name; delete {variant} and the "
                f"report beside it, then rescore."
            )
        # `prune_variants` deletes a variant's weights once it is scored, so
        # rebuilding before checking for the report writes a full checkpoint
        # copy to disk only to discard it again.
        cached = score_report(
            config, label, variant, model.reference_path, config.rows,
            plan_from=model.reference_path,
        )
        if not os.path.isfile(cached):
            variant = build_variant(
                config, python, model.reference_path, deployed.path,
                components, scheme,
                per_tensor=per_tensor,
                expect_digest=digest,
                block_size=block,
            )
        report, _ = score_one(
            config, python, label, variant, model.reference_path, config.rows,
            decompose=False, capture_label=f"{model.name}-ref",
            # A cell that only rounds weights still reroutes tokens, and the
            # cheapest way to show that is to measure it in the cell itself.
            measure_routing=True,
            plan_from=model.reference_path,
        )
        cell = _cell(
            report, capture_manifest, config, variant,
            scheme or deployed_scheme,
            "per_tensor" if per_tensor else "per_component",
        )
        if config.prune_variants and glob.glob(
            os.path.join(variant, "*.safetensors")
        ):
            _prune_variant(variant)
        return cell

    with open(deployed_report, encoding="utf-8") as handle:
        deployed_mean = json.load(handle).get("mean_kld")
    attribution: dict[str, Any] = {
        "deployed": {
            "candidate": deployed.name,
            "scheme": deployed_scheme,
            "mean_kld": deployed_mean,
            "report": deployed_report,
            "inspection": public_inspection,
        },
        "expert_cell": score_variant(
            "experts", deployed_scheme, per_tensor=True
        ),
    }
    if router_quantized:
        attribution["router_cell"] = score_variant(
            "router", deployed_scheme, per_tensor=True
        )
    else:
        attribution["router_cell"] = {
            "status": "not_applicable",
            "evidence": (
                f"qdq.py --inspect {os.path.basename(deployed.path)}: router "
                f"0 of {(inspection['coverage']['router'] or {}).get('weights')} "
                f"weights quantized"
            ),
        }

    # Every component the deployed checkpoint quantizes, rounded through QDQ and
    # run on BF16 kernels. Subtracting it from the deployed mean leaves what the
    # quantized kernels themselves contribute, which no cell above can see.
    quantized = [
        component
        for component in qdq.COMPONENTS
        if (inspection["coverage"].get(component) or {}).get("quantized")
    ]
    if quantized == ["experts"]:
        composite = dict(attribution["expert_cell"])
    elif quantized:
        composite = score_variant(
            ",".join(quantized), deployed_scheme, per_tensor=True
        )
    else:
        composite = {}
    if composite:
        attribution["composite_cell"] = dict(
            composite,
            components=quantized,
            # Recorded so a per-component cell, or an attribution file that
            # predates match_mode, can refuse a calibration claim. A
            # per-tensor cell matched the names and ignores this list.
            partial_components=_partial_components(inspection, quantized),
        )
        if isinstance(deployed_mean, (int, float)) and isinstance(
            composite.get("mean_kld"), (int, float)
        ):
            attribution["engine_arithmetic"] = deployed_mean - composite["mean_kld"]
    algorithm = inspection.get("quant_algorithm")
    if algorithm:
        attribution["quant_algorithm"] = algorithm

    ladder = []
    for scheme in config.ladder:
        # A rung whose grid the reference's shapes cannot carry is reported
        # absent with its reason. Padding to fit would change the scales, and a
        # rung measured under a padding no deployment used is worse than a gap.
        reason = qdq.unexpressible_reason(
            model.reference_path, ("experts",), scheme
        )
        if reason:
            print(f"Law 14: ladder rung {scheme} unavailable - {reason}")
            ladder.append(
                {"scheme": scheme, "mean_kld": None, "unavailable": reason}
            )
            continue
        entry = score_variant("experts", scheme, per_tensor=False)
        ladder.append(
            {
                "scheme": scheme,
                "mean_kld": entry["mean_kld"],
                "variant": entry["variant"],
                "report": entry["report"],
            }
        )
    deployed_reason = (
        qdq.unexpressible_reason(
            model.reference_path,
            ("experts",),
            deployed_scheme,
            inspection.get("detected_block"),
        )
        if deployed_scheme
        else None
    )
    if deployed_reason:
        print(f"Law 14: deployed rung {deployed_scheme} unavailable - {deployed_reason}")
        _append_deployed_ladder(
            ladder, deployed_scheme, {"mean_kld": None, "unavailable": deployed_reason}
        )
    else:
        _append_deployed_ladder(
            ladder,
            deployed_scheme,
            score_variant("experts", deployed_scheme, per_tensor=False),
        )
    attribution["ladder"] = ladder

    out = os.path.join(config.work, "attribution", f"{deployed.name}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(attribution, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Law 14: wrote {out}")
    return out


def qdq_routing(reference_path: str) -> dict[str, Any] | None:
    """Expert counts declared by a reference checkpoint, or None if dense."""
    path = os.path.join(reference_path, "config.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as handle:
        config = json.load(handle)
    for section in (config, config.get("text_config") or {}):
        for key in (
            "num_experts",
            "n_routed_experts",
            "num_local_experts",
            "moe_num_experts",
        ):
            if isinstance(section.get(key), int) and section[key] > 1:
                return {"num_experts": section[key]}
    return None


def _report_exists(config: Config, cand: Candidate) -> bool:
    """True when a deployed report for this candidate is already on disk.

    Tags are `{name}-v{runner}-tp...`, so the prefix is `name + "-v"` rather
    than `name`, which would also match a longer sibling label.
    """
    root = os.path.join(config.work, "reports")
    if not os.path.isdir(root):
        return False
    prefix = cand.name + "-v"
    return any(
        name.startswith(prefix) and name.endswith(".json")
        for name in os.listdir(root)
    )


def _candidate_complete(config: Config, cand: Candidate, routed: bool) -> bool:
    if not _report_exists(config, cand):
        return False
    if not routed:
        return True
    return os.path.isfile(
        os.path.join(config.work, "attribution", f"{cand.name}.json")
    )


def _has_checkpoint_weights(path: str) -> bool:
    """True when config.json and at least one safetensors shard are present.

    A lease release can leave a directory with only config.json. Treating that
    as "weights present" skips the re-fetch Law 14 needs for inspect and QDQ.
    """
    if not os.path.isdir(path):
        return False
    if not os.path.isfile(os.path.join(path, "config.json")):
        return False
    try:
        names = os.listdir(path)
    except OSError:
        return False
    return any(name.endswith(".safetensors") for name in names)


def ensure_candidate_weights(config: Config, cand: Candidate) -> None:
    """Fetch a missing candidate. Raises CampaignError on a fetch fault.

    A successful fetch under fetch=lease writes a lease file so the directory
    is eligible for deletion after scoring. A directory that was already on
    disk is never leased.
    """
    if _has_checkpoint_weights(cand.path):
        return
    if not cand.hf_repo:
        raise CampaignError(
            f"checkpoint absent at {cand.path} and has no hf_repo"
        )
    if fetch_checkpoint(cand.hf_repo, cand.path, cand.revision) != 0:
        raise CampaignError(f"download failed for {cand.hf_repo}")
    if not _has_checkpoint_weights(cand.path):
        raise CampaignError(f"download produced no weights at {cand.path}")
    if config.fetch == "lease":
        write_lease(config.work, cand)


def maybe_release(config: Config, cand: Candidate) -> None:
    if config.fetch != "lease":
        return
    release_leased(config.work, cand, protected=reference_paths(config))


def _score_candidate(
    config: Config,
    python: str,
    model: Model,
    cand: Candidate,
    routed: bool,
) -> None:
    """Provenance, score, attribute. Raises CampaignError on a candidate fault."""
    prov_path = os.path.join(config.work, "provenance", f"{cand.name}.json")
    try:
        identity = _provenance.compare(model.reference_path, cand.path)
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        raise CampaignError(f"provenance unreadable: {exc}") from exc
    _provenance.write_report(identity, prov_path)
    if not identity["ok"]:
        fields = ", ".join(item["field"] for item in identity["differing"])
        raise CampaignError(
            f"architecture mismatch on {fields}; this is not a quantization of "
            f"{model.name}"
        )

    inspect_checkpoint(
        python, cand.path, inspect_record(config.work, cand.name)
    )
    refuse_if_unloadable(cand.path)

    report, capture = score_one(
        config, python, cand.name, cand.path, model.reference_path,
        config.rows, decompose=config.storage != "logits",
        capture_label=f"{model.name}-ref",
        measure_routing=routed,
    )
    try:
        with open(report, encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["candidate_provenance"] = identity
        with open(report, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignError(f"could not record provenance in {report}: {exc}") from exc

    attribute_model(config, python, model, cand, capture, report)


def preflight_suite(config: Config) -> None:
    """Refuse to score against a suite_dir that has not been minted.

    A missing suite used to surface as LAW 1 STOP because the baseline is the
    first scoring command. That is the wrong law: Law 3 requires frozen token
    IDs, and the model has not been loaded yet.
    """
    if not config.suite_dir:
        return
    manifest = os.path.join(config.suite_dir, "suite-manifest.json")
    if os.path.isfile(manifest):
        return
    raise SystemExit(
        f"token suite is not minted: {manifest}\n"
        "Law 3 requires frozen token IDs before any scoring run. Compare the "
        "tokenizer against an existing suite of the same family, then either "
        "point suite_dir at it or mint:\n"
        f"  python fidelity/suite.py build "
        f"--recipe fidelity/suites/recipe-v1.json "
        f"--tokenizer <reference> --out {config.suite_dir}"
    )


def cmd_score(config: Config, python: str) -> int:
    """Gate on the zero baseline, then score candidates.

    The environment is captured by the first scoring run that actually executes.
    With fetch=lease each candidate is downloaded just before it is scored and
    released afterwards; the reference stays on disk.
    """
    if config.fetch not in ("upfront", "lease"):
        raise SystemExit(
            f"unknown fetch={config.fetch!r}; want 'upfront' or 'lease'"
        )
    preflight_suite(config)
    os.makedirs(config.work, exist_ok=True)
    failed: list[str] = []

    for model in config.models:
        print(f"\n##### {model.name}")
        if not os.path.isdir(model.reference_path):
            if not model.reference_repo:
                raise SystemExit(
                    f"reference absent at {model.reference_path} and has no "
                    f"hf_repo to fetch it from"
                )
            if fetch_checkpoint(
                model.reference_repo, model.reference_path,
                model.reference_revision,
            ) != 0:
                raise SystemExit(
                    f"LAW 1 STOP: could not download reference {model.name}"
                )
        # Law 1: the reference must reproduce itself before anything else runs.
        # One row suffices, because the only acceptable answer is exact zero.
        try:
            baseline_report, _ = score_one(
                config, python, f"{model.name}-self", model.reference_path,
                model.reference_path, 1, decompose=False,
                suite_limit=1 if config.suite_dir else None,
            )
        except CampaignError as exc:
            raise SystemExit(
                f"LAW 1 STOP: {model.name} baseline scoring failed: {exc}"
            ) from exc
        with open(baseline_report, encoding="utf-8") as handle:
            baseline = json.load(handle)
        mean = baseline.get("mean_kld")
        if mean != 0.0:
            raise SystemExit(
                f"\nLAW 1 STOP: {model.name} scored {mean!r} against itself, not "
                f"exactly 0.0.\nThe campaign stops here. To proceed you must "
                f"record a named approval and run a three-capture repeat study; "
                f"see fidelity/LAWS.md Law 1.\nBaseline report: {baseline_report}"
            )
        print(f"Law 1 satisfied: {model.name} self-KLD is exactly 0.0")

        # Law 14's routing term is measured, not emulated, so a routed
        # reference needs its own expert selections recorded once.
        routed = bool(qdq_routing(model.reference_path))

        for cand in model.candidates:
            if _candidate_complete(config, cand, routed):
                print(f"=== {cand.name} already scored")
                maybe_release(config, cand)
                continue
            try:
                ensure_candidate_weights(config, cand)
                _score_candidate(config, python, model, cand, routed)
            except CampaignError as exc:
                print(f"FAILED  {cand.name}: {exc}", file=sys.stderr)
                failed.append(f"{model.name}/{cand.name}")
                continue
            maybe_release(config, cand)
    if failed:
        print(
            f"\n{len(failed)} candidate(s) failed: {', '.join(failed)}",
            file=sys.stderr,
        )
        return 1
    return 0


def _find(work: str, pattern: str) -> str | None:
    """Locate a produced artifact by tag prefix, newest first."""
    root = os.path.join(work, "reports")
    if not os.path.isdir(root):
        return None
    matches = sorted(
        (f for f in os.listdir(root) if f.startswith(pattern) and f.endswith(".json")),
        reverse=True,
    )
    return os.path.join(root, matches[0]) if matches else None


def _reference_identity(path: str) -> dict[str, Any] | None:
    """The manifest fields a capture is bound to, for comparing two captures.

    The same list `kld.manifest_mismatches` requires, kept literal here so
    assembly does not import torch to compare two JSON files.
    """
    try:
        with open(os.path.join(path, "manifest.json"), encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return {
        key: manifest.get(key)
        for key in (
            "token_sha256",
            "tokenizer",
            "context_length",
            "stride",
            "rows",
            "score_from",
            "kld_vocab_size",
            "tensor_parallel_size",
            "enforce_eager",
            "runtime",
        )
    }


def _copy_reference(capture: str, dest: str) -> None:
    """Publish the reusable reference: tensors, head, and its bound manifest.

    Captured tensors run to tens of gigabytes. When the capture and the library
    sit on one filesystem a hard link publishes them without a second copy; the
    files are never rewritten in place, so the two names cannot diverge. A copy is
    the fallback across filesystems.
    """
    os.makedirs(dest, exist_ok=True)
    if not os.path.isdir(capture):
        # Silence here would produce an artifact missing its reusable reference,
        # which Law 12 would then fail for a reason that points nowhere.
        print(f"WARNING  no capture directory at {capture}", file=sys.stderr)
        return
    # Files already at the destination are left alone, which is only safe while
    # they came from a capture bound to the same identity. When they did not, the
    # first capture ever published would win forever and the artifact would carry
    # a reference that its own manifest does not describe.
    source_id = _reference_identity(capture)
    published_id = _reference_identity(dest)
    if source_id and published_id and source_id != published_id:
        differing = sorted(k for k in source_id if source_id[k] != published_id[k])
        print(
            f"REPLACING reference in {dest}: the published reference is bound to a "
            f"different identity than {os.path.basename(capture)} "
            f"({', '.join(differing)})"
        )
        for name in sorted(os.listdir(dest)):
            target = os.path.join(dest, name)
            if os.path.isfile(target):
                os.remove(target)
    for name in sorted(os.listdir(capture)):
        src = os.path.join(capture, name)
        dst = os.path.join(dest, name)
        if not os.path.isfile(src) or os.path.exists(dst):
            continue
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def _publish_attribution(src: str, cand_dir: str) -> None:
    """Copy every cell's report and QDQ manifest next to the candidate.

    The scored attribution points at the work directory, which no reader of the
    artifact has. Each supporting file is published under `attribution/` and the
    paths are rewritten to be relative to the candidate directory, so the
    one-pager can link them and `checksums.txt` covers them.
    """
    with open(src, encoding="utf-8") as handle:
        attribution = json.load(handle)
    support = os.path.join(cand_dir, "attribution")
    os.makedirs(support, exist_ok=True)

    def publish(cell: dict[str, Any]) -> None:
        variant = cell.get("variant") or "cell"
        for key, suffix in (
            ("report", "report.json"),
            ("qdq_manifest", "qdq-manifest.json"),
        ):
            path = cell.get(key)
            if not path or not os.path.isfile(path):
                cell.pop(key, None)
                continue
            name = f"{variant}-{suffix}" if key == "qdq_manifest" else f"{variant}.json"
            shutil.copy2(path, os.path.join(support, name))
            cell[key] = f"attribution/{name}"

    for key in ("expert_cell", "router_cell", "composite_cell"):
        cell = attribution.get(key)
        if isinstance(cell, dict):
            publish(cell)
    for entry in attribution.get("ladder") or []:
        if isinstance(entry, dict):
            publish(entry)
    # The deployed report is already published as report.json beside this file.
    deployed = attribution.get("deployed")
    if isinstance(deployed, dict) and deployed.get("report"):
        deployed["report"] = "report.json"

    with open(
        os.path.join(cand_dir, "attribution.json"), "w",
        encoding="utf-8", newline="\n",
    ) as handle:
        json.dump(attribution, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _has_per_context(path: str) -> bool:
    """Whether a report can be attributed to a domain at all."""
    try:
        with open(path, encoding="utf-8") as handle:
            return bool(json.load(handle).get("per_context"))
    except (OSError, json.JSONDecodeError):
        return False


def _strata_cells(cand_dir: str) -> list[str]:
    """Name every published report worth attributing to a domain.

    The deployed result is the subject; the ladder rungs are what make the
    per-domain table answer whether a domain's weakness follows the format or
    the model. Paths are relative to ``cand_dir`` because that is how
    ``_publish_attribution`` rewrote them.

    A cell scored before per-context recording is left out rather than allowed to
    fail the whole table; the deployed report is always the first cell, so its
    own absence is reported by Law 15 instead of being hidden here.
    """
    cells = [f"deployed={os.path.join(cand_dir, 'report.json')}"]
    path = os.path.join(cand_dir, "attribution.json")
    if not os.path.isfile(path):
        return cells
    try:
        with open(path, encoding="utf-8") as handle:
            attribution = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return cells
    optional = [
        ((entry or {}).get("scheme"), (entry or {}).get("report"))
        for entry in attribution.get("ladder") or []
    ]
    optional.append(
        ("all_quantized", (attribution.get("composite_cell") or {}).get("report"))
    )
    for name, report in optional:
        if not name or not report or os.path.isabs(report):
            continue
        path = os.path.join(cand_dir, report)
        if os.path.isfile(path) and _has_per_context(path):
            cells.append(f"{name}={path}")
    return cells


def _build_strata(
    python: str, suite_manifest: str, cand_dir: str, label: str
) -> None:
    """Write the per-domain report Law 15 reads, or leave nothing behind.

    A failure here is not fatal to assembly: Law 15 reports the absence with a
    better message than a traceback would. Stale files are removed first so a
    previous campaign's domains cannot stand in for this one's.
    """
    out_json = os.path.join(cand_dir, "strata.json")
    out_md = os.path.join(cand_dir, "strata.md")
    for path in (out_json, out_md):
        if os.path.isfile(path):
            os.remove(path)
    cmd = [
        python, os.path.join(HERE, "strata.py"),
        "--suite", suite_manifest,
        "--label", label,
        "--out", out_md,
        "--json", out_json,
    ]
    for spec in _strata_cells(cand_dir):
        cmd += ["--cell", spec]
    if _run(cmd) != 0:
        print(
            f"no per-domain report for {os.path.basename(cand_dir)}; Law 15 "
            f"will record why",
            file=sys.stderr,
        )


PUBLISHED_FILES = (
    "report.json",
    "manifest.json",
    "compliance.json",
    "report.md",
    "attribution.json",
    "strata.json",
    "strata.md",
    "provenance.json",
    "inspect.json",
)


def _snapshot_candidate(cand_dir: str) -> dict[str, bytes] | None:
    """Hold a compliant published result in memory, or None if there isn't one.

    Assembly overwrites a candidate's files before compliance runs on the new
    ones, so a campaign pointed at the wrong config can replace a compliant
    result with a failing one and leave nothing to fall back to.
    """
    receipt = os.path.join(cand_dir, "compliance.json")
    if not os.path.isfile(receipt):
        return None
    try:
        with open(receipt, encoding="utf-8") as handle:
            if not json.load(handle).get("compliant"):
                return None
    except (OSError, json.JSONDecodeError):
        return None
    held: dict[str, bytes] = {}
    for name in PUBLISHED_FILES:
        path = os.path.join(cand_dir, name)
        if os.path.isfile(path):
            with open(path, "rb") as handle:
                held[name] = handle.read()
    return held


def _restore_candidate(cand_dir: str, held: dict[str, bytes]) -> None:
    for name, data in held.items():
        with open(os.path.join(cand_dir, name), "wb") as handle:
            handle.write(data)


def _scrub_environment(env_dir: str) -> None:
    """Strip credential values out of an assembled environment report."""
    path = os.path.join(env_dir, "runtime.json")
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as handle:
        runtime = json.load(handle)
    safe, hidden = redact_env(runtime.get("env") or {})
    if not hidden:
        return
    known = set(runtime.get("env_redacted") or [])
    runtime["env"] = safe
    runtime["env_redacted"] = sorted(known | set(hidden))
    runtime["env_redaction_policy"] = (
        "Values of variables whose names look like credentials are never "
        "recorded. The names are, so a reader can see what was set."
    )
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(runtime, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        f"redacted {len(hidden)} credential value(s) in {path}: "
        f"{', '.join(hidden)}"
    )


def cmd_assemble(config: Config, python: str, force: bool = False) -> int:
    """Build the library tree, then checksums, then receipts, then documents."""
    reverted: list[str] = []
    for model in config.models:
        model_root = os.path.join(config.library, model.name)
        os.makedirs(model_root, exist_ok=True)
        preserved: dict[str, dict[str, bytes]] = {}

        found = [
            (cand, report)
            for cand in model.candidates
            for report in [_find(config.work, cand.name)]
            if report
        ]
        baseline = _find(config.work, f"{model.name}-self")

        env_src = os.path.join(config.work, "environment")
        env_dst = os.path.join(model_root, "environment")
        # The source is scrubbed first, so a credential captured before the
        # redaction policy existed stops propagating into later model roots
        # instead of being cleaned up once per copy.
        _scrub_environment(env_src)
        # Refreshed whenever this work directory contributes a measurement, so a
        # model root cannot keep an environment from some earlier campaign. When
        # it contributes nothing the existing environment is the truthful record
        # of whatever did produce the published numbers, and overwriting it would
        # bind them to a stack that never ran them.
        if not (found or baseline):
            print(
                f"!!! {model.name}: no reports or baseline in {config.work}; "
                "leaving the published environment untouched"
            )
        elif os.path.isdir(env_src):
            shutil.copytree(env_src, env_dst, dirs_exist_ok=True)
        _scrub_environment(env_dst)
        # Law 12 reads the artifact directory, and each model root is published as
        # a self-contained repo, so the laws ship inside it as well as at the
        # library root. Both copies land before checksums.txt is written.
        laws = os.path.join(HERE, "LAWS.md")
        shutil.copy2(laws, os.path.join(config.library, "LAWS.md"))
        shutil.copy2(laws, os.path.join(model_root, "LAWS.md"))

        if config.suite_dir and os.path.isdir(config.suite_dir):
            suite_dst = os.path.join(model_root, "suite")
            if not os.path.isdir(suite_dst):
                shutil.copytree(config.suite_dir, suite_dst)

        if baseline:
            os.makedirs(os.path.join(model_root, "baselines"), exist_ok=True)
            shutil.copy2(
                baseline,
                os.path.join(model_root, "baselines", "self-kld.json"),
            )

        for cand in model.candidates:
            if not any(cand is c for c, _ in found):
                print(f"skipping {cand.name}: no report in {config.work}/reports")
        for cand, report in found:
            tag = os.path.basename(report)[: -len(".json")]
            # Candidates of one model share a teacher capture, so the capture
            # directory carries the reference's label with the candidate's
            # geometry suffix.
            suffix = tag[len(cand.name):]
            capture = os.path.join(
                config.work, "captures", f"{model.name}-ref{suffix}"
            )
            if not os.path.isdir(capture):
                capture = os.path.join(config.work, "captures", tag)
            _copy_reference(capture, os.path.join(model_root, "reference"))
            cand_dir = os.path.join(model_root, cand.name)
            os.makedirs(cand_dir, exist_ok=True)
            held = _snapshot_candidate(cand_dir)
            if held:
                preserved[cand.name] = held
            shutil.copy2(report, os.path.join(cand_dir, "report.json"))
            manifest_src = os.path.join(capture, "manifest.json")
            if os.path.isfile(manifest_src):
                shutil.copy2(manifest_src, os.path.join(cand_dir, "manifest.json"))
            inspect_src = inspect_record(config.work, cand.name)
            if not os.path.isfile(inspect_src):
                inspect_src = os.path.join(cand.path, "inspect.json")
            if os.path.isfile(inspect_src):
                shutil.copy2(inspect_src, os.path.join(cand_dir, "inspect.json"))
            prov_src = os.path.join(
                config.work, "provenance", f"{cand.name}.json"
            )
            if os.path.isfile(prov_src):
                shutil.copy2(prov_src, os.path.join(cand_dir, "provenance.json"))
            attribution = os.path.join(
                config.work, "attribution", f"{cand.name}.json"
            )
            if os.path.isfile(attribution):
                _publish_attribution(attribution, cand_dir)

        # Law 12 verifies the assembled tree, so checksums must exist before
        # compliance runs. Swapping these two steps produces a spurious failure.
        _run([python, os.path.join(HERE, "artifact.py"), "checksums",
              "--root", model_root])

        suite_manifest = os.path.join(model_root, "suite", "suite-manifest.json")
        failures = []
        for cand in model.candidates:
            cand_dir = os.path.join(model_root, cand.name)
            report = os.path.join(cand_dir, "report.json")
            manifest = os.path.join(cand_dir, "manifest.json")
            if not (os.path.isfile(report) and os.path.isfile(manifest)):
                continue
            receipt = os.path.join(cand_dir, "compliance.json")
            cmd = [
                python, os.path.join(HERE, "compliance.py"),
                "--report", report,
                "--manifest", manifest,
                "--self-report",
                os.path.join(model_root, "baselines", "self-kld.json"),
                "--env-dir", os.path.join(model_root, "environment"),
                "--artifact-dir", model_root,
                "--partition", config.partition,
                "--out", receipt,
            ]
            if os.path.isfile(suite_manifest):
                cmd += ["--suite", suite_manifest]
                _build_strata(
                    python,
                    suite_manifest,
                    cand_dir,
                    f"{model.name} / {cand.name}",
                )
            attribution = os.path.join(cand_dir, "attribution.json")
            if os.path.isfile(attribution):
                cmd += ["--attribution", attribution]
            strata = os.path.join(cand_dir, "strata.json")
            if os.path.isfile(strata):
                cmd += ["--strata", strata]
            inspection = os.path.join(cand_dir, "inspect.json")
            if os.path.isfile(inspection):
                cmd += ["--inspection", inspection]
            if config.approvals and os.path.isfile(config.approvals):
                cmd += ["--approvals", config.approvals]
            if _run(cmd) != 0:
                failures.append(cand.name)
                held = preserved.get(cand.name)
                if held and not force:
                    _restore_candidate(cand_dir, held)
                    reverted.append(f"{model.name}/{cand.name}")
                    print(
                        f"REVERTED {cand.name}: the published result was "
                        f"law-compliant and this one is not, so the previous "
                        f"report, receipt, and one-pager were restored. Fix the "
                        f"campaign, or pass --force to replace them.",
                        file=sys.stderr,
                    )
                    continue

            _run([
                python, os.path.join(HERE, "artifact.py"), "onepager",
                "--report", report,
                "--manifest", manifest,
                "--receipt", receipt,
                "--self-report",
                os.path.join(model_root, "baselines", "self-kld.json"),
                "--env-dir", os.path.join(model_root, "environment"),
                "--artifact-dir", model_root,
                "--candidate-dir", cand.name,
                "--label", f"{model.name} / {cand.name}",
                "--out", os.path.join(cand_dir, "report.md"),
            ])

        scored_weights: list[tuple[str, Any, str | None]] = []
        for cand in model.candidates:
            path = os.path.join(model_root, cand.name, "report.json")
            if not os.path.isfile(path):
                continue
            with open(path, encoding="utf-8") as handle:
                record = json.load(handle)
            scored_weights.append((
                cand.name,
                record.get("mean_kld"),
                record.get("student_weights_sha256"),
            ))
        impossible, duplicates = weight_collisions(scored_weights)
        for left, right in duplicates:
            print(
                f"DUPLICATE {left} and {right} are byte-identical weights under "
                f"two names; publish one and disclose the other as a re-upload."
            )
        for left, right in impossible:
            print(
                f"IMPOSSIBLE {left} and {right} carry different weights and the "
                f"same mean KLD, so at least one was scored against the other's "
                f"checkpoint. Both are refused; rescore them.",
                file=sys.stderr,
            )
            failures.extend([left, right])

        # Rewrite checksums so they cover the receipts and one-pagers too.
        _run([python, os.path.join(HERE, "artifact.py"), "checksums",
              "--root", model_root])
        if failures:
            print(f"NOT LAW-COMPLIANT: {', '.join(failures)}", file=sys.stderr)

    _run([
        python, os.path.join(HERE, "artifact.py"), "leaderboard",
        "--results-root", config.library,
        "--out", os.path.join(config.library, "leaderboard.md"),
        "--csv", os.path.join(config.library, "leaderboard.csv"),
    ])
    print(f"\nlibrary: {config.library}")
    if reverted:
        print(
            f"reverted to the previously compliant result for: "
            f"{', '.join(reverted)}",
            file=sys.stderr,
        )
        return 1
    return 0


def cmd_release(config: Config) -> int:
    """Drop candidate weights still under lease after an interrupted run."""
    protected = reference_paths(config)
    reclaimed = 0.0
    released = 0
    refused: list[str] = []
    for model in config.models:
        for cand in model.candidates:
            if load_lease(config.work, cand.name) is None:
                continue
            gib = release_leased(config.work, cand, protected=protected)
            if gib is None:
                refused.append(cand.name)
                continue
            released += 1
            reclaimed += gib
    print(
        f"released {released} leased checkpoint(s), {reclaimed:.2f} GiB reclaimed"
    )
    if refused:
        print(
            f"refused to release: {', '.join(refused)}",
            file=sys.stderr,
        )
        return 1
    return 0


def selftest() -> int:
    """Lease round-trip: leased dirs go, un-leased dirs and the reference stay."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        work = os.path.join(tmp, "work")
        org = os.path.join(tmp, "org")
        leased = os.path.join(org, "leased-model")
        kept = os.path.join(org, "kept-model")
        ref = os.path.join(org, "ref")
        for path in (leased, kept, ref):
            os.makedirs(path)
            with open(os.path.join(path, "w.safetensors"), "wb") as handle:
                handle.write(b"x" * 100)

        leased_cand = Candidate(
            "leased-model", leased, "org/leased-model", "abc"
        )
        kept_cand = Candidate("kept-model", kept, "org/kept-model", "def")
        ref_cand = Candidate("ref", ref, "org/ref", "aaa")
        write_lease(work, leased_cand)

        protected = {os.path.abspath(ref)}
        gone = release_leased(work, leased_cand, protected=protected)
        assert gone is not None and gone > 0
        assert not os.path.isdir(leased)
        assert load_lease(work, leased_cand.name) is None

        stayed = release_leased(work, kept_cand, protected=protected)
        assert stayed == 0.0
        assert os.path.isdir(kept)

        write_lease(work, ref_cand)
        refused = release_leased(work, ref_cand, protected=protected)
        assert refused is None
        assert os.path.isdir(ref)
        assert load_lease(work, ref_cand.name) is not None

        husk = os.path.join(tmp, "husk")
        os.makedirs(husk)
        with open(os.path.join(husk, "config.json"), "w", encoding="utf-8") as handle:
            handle.write("{}\n")
        assert not _has_checkpoint_weights(husk)
        with open(os.path.join(husk, "model.safetensors"), "wb") as handle:
            handle.write(b"x")
        assert _has_checkpoint_weights(husk)

        cache_dir = os.path.join(tmp, "ckpt")
        os.makedirs(cache_dir)
        cache = os.path.join(cache_dir, "inspect.json")
        with open(cache, "w", encoding="utf-8") as handle:
            handle.write('{"ok": true}\n')
        durable = inspect_record(work, "ckpt")
        copied = inspect_checkpoint("/no/such/python", cache_dir, durable)
        assert copied == durable
        assert os.path.isfile(durable)

    rows = _parse_smi_rows(
        "0, GPU-aaa, 97887, 97000\n1, GPU-bbb, 97887, 1000\n"
    )
    assert len(rows) == 2
    selected = _select_visible(rows, "1")
    assert selected[0][1] == "GPU-bbb"
    assert selected[0][3] == 1000.0
    by_uuid = _select_visible(rows, "GPU-aaa")
    assert by_uuid[0][0] == 0

    geom = {
        "num_hidden_layers": 60,
        "num_key_value_heads": 16,
        "head_dim": 256,
        "vocab_size": 262144,
        "elem_bytes": 2,
    }
    logits = logits_headroom_gib(geom, 2048, 1)
    assert abs(logits - 12.5) < 1e-6
    kv_window = kv_cache_gib(geom, 2048, 1, 1)
    assert 1.0 < kv_window < 2.5
    kv_engine = kv_cache_gib(geom, engine_max_model_len(2048), 1, 1)
    assert abs(kv_engine - 3.75) < 1e-6
    cfg = Config(name="t", library="l", work="w", models=[])
    plan = plan_gpus(
        [],
        cfg,
        geometry=geom,
        inventory=(4, 95.59, 94.83),
        weight_gib=65.10,
    )
    assert plan.tp == 1
    assert plan.pinned is False
    assert abs(plan.kv_gib - kv_engine) < 1e-9
    assert plan.util <= UTIL_CEILING
    cfg.tensor_parallel_size = 2
    pinned = plan_gpus(
        [],
        cfg,
        geometry=geom,
        inventory=(4, 95.59, 94.83),
        weight_gib=65.10,
    )
    assert pinned.tp == 2
    assert pinned.pinned is True

    rungs = [
        {"scheme": "fp8_block", "mean_kld": 0.1, "variant": "a", "report": "r"}
    ]
    expert = {
        "mean_kld": 0.02,
        "variant": "int4-cell",
        "report": "int4.json",
    }
    _append_deployed_ladder(rungs, "int4_g32_asym", expert)
    assert rungs[-1]["scheme"] == "int4_g32_asym"
    assert rungs[-1]["mean_kld"] == 0.02
    before = list(rungs)
    _append_deployed_ladder(rungs, "int4_g32_asym", expert)
    assert rungs == before
    already = [
        {"scheme": "fp8_block", "mean_kld": 0.05, "variant": "b", "report": "s"}
    ]
    _append_deployed_ladder(
        already,
        "fp8_block",
        {"mean_kld": 0.05, "variant": "b", "report": "s"},
    )
    assert len(already) == 1

    from artifact import (
        _beyond_rounding_what,
        _deployed_quantization,
        _derived_terms,
    )
    from compliance import (
        Campaign as _Campaign,
        law_16_weight_binding,
        routing_floor,
        routing_floor_state,
    )

    def _binding(scored: str | None, inspected: str | None) -> str:
        return law_16_weight_binding(
            _Campaign(
                report={"student_weights_sha256": scored} if scored else {},
                manifest={},
                manifest_path="",
                inspection={"weights_sha256": inspected} if inspected else None,
            )
        ).status

    assert _binding("a" * 64, "a" * 64) == "pass"
    assert _binding("a" * 64, "b" * 64) == "fail", "wrong weights went unseen"
    assert _binding(None, "a" * 64) == "fail", "an unbound report must not pass"
    assert _binding("a" * 64, None) == "fail", "nothing to check against"

    # A complete approval publishes a report that predates binding, and cannot
    # publish one whose digest contradicts the checkpoint beside it.
    from compliance import evaluate as _evaluate

    def _law16(scored: str | None, inspected: str) -> tuple[str, str]:
        camp = _Campaign(
            report={"student_weights_sha256": scored} if scored else {},
            manifest={},
            manifest_path="",
            inspection={"weights_sha256": inspected},
            approvals={
                "16": {
                    "approver": "selftest",
                    "justification": "weights released before Law 16 existed",
                    "timestamp": "2026-09-02T00:00:00Z",
                }
            },
        )
        found = next(f for f in _evaluate(camp) if f.law == 16)
        return found.status, found.detail

    status, _ = _law16(None, "a" * 64)
    assert status == "override", "an unbound legacy report must be excusable"
    status, detail = _law16("a" * 64, "b" * 64)
    assert status == "fail", "a contradicted digest must not be overridable"
    assert "permits no override" in detail, detail

    # Same mean from different weights is one checkpoint scored twice; a shared
    # digest is one upload under two names.
    impossible, duplicates = weight_collisions([
        ("mse", 0.0345, "a" * 64),
        ("plain", 0.0345, "b" * 64),
        ("rebrand", 0.09, "c" * 64),
        ("original", 0.09, "c" * 64),
    ])
    assert impossible == [("mse", "plain")], impossible
    assert duplicates == [("rebrand", "original")], duplicates
    # An unbound report cannot be accused of either.
    assert weight_collisions([("a", 0.1, None), ("b", 0.1, None)]) == ([], [])

    assert "calibration benefit" in _beyond_rounding_what("awq", -0.01)
    assert "beat round-to-nearest" in _beyond_rounding_what("awq", -0.01)
    assert "beat round-to-nearest" not in _beyond_rounding_what("awq", 0.01)

    # A composite that rounds more than the checkpoint does cannot support a
    # calibration claim, whatever the algorithm or the sign.
    confounded = _beyond_rounding_what("autoround", -0.01, ["attention"])
    assert "not attributable" in confounded
    assert "calibration benefit" not in confounded
    exact = _beyond_rounding_what(
        "autoround", -0.01, ["attention"], match_mode="per_tensor"
    )
    assert "calibration benefit" in exact
    assert "not attributable" not in exact
    coverage = {
        "coverage": {
            "experts": {"weights": 31488, "quantized": 31488},
            "shared_expert": {"weights": 123, "quantized": 3},
            "attention": {"weights": 44, "quantized": 4},
            "router": {"weights": 41, "quantized": 0},
        }
    }
    assert _partial_components(
        coverage, ["experts", "shared_expert", "attention"]
    ) == ["shared_expert", "attention"]
    assert _partial_components(coverage, ["experts"]) == []
    md = "\n".join(
        _deployed_quantization(
            {
                "inspection": {
                    "detected_scheme": "int4_g32_asym",
                    "detected_block": 32,
                    "quant_algorithm": "awq",
                    "coverage": {
                        "experts": {"weights": 41, "quantized": 41}
                    },
                }
            }
        )
    )
    assert "Algorithm" in md and "awq" in md
    assert "format (int4_g32_asym) only" in md
    partial_md = "\n".join(
        _deployed_quantization(
            {
                "inspection": {
                    "detected_scheme": "int4_g128_sym",
                    "quant_algorithm": "autoround",
                    "coverage": {
                        "experts": {"weights": 10, "quantized": 10},
                        "attention": {"weights": 44, "quantized": 4},
                    },
                }
            }
        )
    )
    assert "exactly the 14 weights" in partial_md
    assert "upper bound" not in partial_md
    derived = "\n".join(
        _derived_terms(0.02, -0.01, {}, algorithm="gptq")
    )
    assert "beat round-to-nearest" in derived
    guarded = "\n".join(
        _derived_terms(
            0.02, -0.01, {}, algorithm="autoround", partial=["attention"]
        )
    )
    assert "not attributable" in guarded
    assert "beat round-to-nearest" not in guarded
    exact_term = "\n".join(
        _derived_terms(
            0.02,
            -0.01,
            {},
            algorithm="autoround",
            partial=["attention"],
            match_mode="per_tensor",
        )
    )
    assert "calibration benefit" in exact_term
    assert "not attributable" not in exact_term

    # A null routing excess means opposite things at the two extremes, and
    # omitting the row would let saturation read as "routing cost nothing".
    saturated = "\n".join(
        _derived_terms(
            0.02,
            -0.01,
            {"selection_flip_rate": 0.56, "positions": 100},
            floor_state="saturated",
        )
    )
    assert "saturated" in saturated
    assert "no held-routing population" in saturated
    held = "\n".join(
        _derived_terms(
            0.02,
            -0.01,
            {"selection_flip_rate": 0.0, "positions": 100},
            floor_state="routing_held",
        )
    )
    assert "survived at every scored position" in held
    silent = "\n".join(
        _derived_terms(0.02, -0.01, {"selection_flip_rate": 0.5})
    )
    assert "Routing, measured" not in silent, (
        "without a floor state there is nothing to disclose, so the row must "
        "stay out rather than assert a value the report does not carry"
    )
    for report, expect in (
        ({"routing": {"positions": 10, "routing_excess_mean": 0.5}}, "measured"),
        ({"routing": {"positions": 10, "position_flip_rate": 1.0}}, "saturated"),
        ({"routing": {"positions": 10, "position_flip_rate": 0.0}},
         "routing_held"),
        ({"routing": {}}, "unmeasured"),
        ({}, "unmeasured"),
    ):
        assert routing_floor_state(report) == expect, report
    assert routing_floor(
        {"routing": {"positions": 10, "position_flip_rate": 0.0}}
    ) == 0.0, "routing that never flipped has a zero floor, not an unknown one"
    assert routing_floor(
        {"routing": {"positions": 10, "position_flip_rate": 1.0}}
    ) is None, "a saturated floor is undefined and must not read as zero"

    path = variant_path(cfg, "/models/Ref", "experts", "int4_g128_asym")
    assert path.endswith("Ref-qdq-experts-int4_g128_asym")
    digested = variant_path(
        cfg, "/models/Ref", "experts", "int4_g128_asym", match_digest="abc"
    )
    assert digested.endswith("Ref-qdq-experts-int4_g128_asym-mabc")

    print("selftest passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        nargs="?",
        choices=("download", "score", "assemble", "all", "release"),
    )
    parser.add_argument("--config")
    parser.add_argument(
        "--force",
        action="store_true",
        help="allow assembly to replace a compliant published result with one "
        "that fails a law",
    )
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        return selftest()
    if not args.stage or not args.config:
        parser.error("stage and --config are required")

    config = load_config(args.config)
    started = datetime.now(timezone.utc).isoformat()
    print(f"campaign {config.name} ({config.partition}) started {started}")

    if args.stage == "release":
        return cmd_release(config)

    python = None
    try:
        python = resolve_python()
        print(f"interpreter: {python}")
    except SystemExit:
        if args.stage != "download":
            raise
        print(
            "WARNING  no vLLM interpreter; download will skip inspect",
            file=sys.stderr,
        )

    if args.stage == "download":
        return cmd_download(config, python)

    assert python is not None
    score_rc = 0
    if args.stage in ("score", "all"):
        score_rc = cmd_score(config, python)
    if args.stage in ("assemble", "all"):
        assemble_rc = cmd_assemble(config, python, force=args.force)
        return assemble_rc or score_rc
    return score_rc


if __name__ == "__main__":
    sys.exit(main())
