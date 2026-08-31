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
    all        score then assemble

Ordering is enforced, not suggested: the zero baseline gates every candidate
(Law 1), and checksums precede compliance (Law 12).
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from redaction import redact_env  # noqa: E402 - sibling module

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
SCORER = os.path.join(REPO_ROOT, "examples", "offline_inference", "score_mode_kld.py")

# Scoring needs a full padded-vocabulary FP32 logits row per position, several
# GiB per request, which vLLM's memory profiler does not account for. These
# leave room for it rather than letting the KV cache claim the card.
WEIGHT_FRACTION = 0.60
# One sequence of at most a few thousand tokens needs a trivial KV cache, and
# every GiB reserved for it is a GiB the head-decomposition logits cannot use.
KV_CACHE_GIB = 2
# Room for a chunk of vocabulary-wide logits and their float copies, on top of
# the weights and the cache.
HEADROOM_GIB = 16
TP_CANDIDATES = (1, 2, 4, 8)


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


def gpu_inventory() -> tuple[int, float]:
    """Visible GPU count and per-GPU total memory in GiB."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        ).stdout.split()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"cannot query GPUs via nvidia-smi: {exc}") from exc
    count = len(visible.split(",")) if visible else len(out)
    per_gpu = float(out[0]) / 1024.0
    return max(count, 1), per_gpu


def plan_gpus(paths: list[str]) -> tuple[int, float, float]:
    """Smallest tensor-parallel size that leaves room for scoring buffers.

    Returns (tensor_parallel_size, gpu_memory_utilization, weight_gib).
    """
    weights = max((checkpoint_gib(p) for p in paths), default=0.0)
    count, per_gpu = gpu_inventory()
    budget = per_gpu * WEIGHT_FRACTION
    tp = next(
        (c for c in TP_CANDIDATES if c <= count and weights / c <= budget), count
    )
    util = (weights / tp + KV_CACHE_GIB + HEADROOM_GIB) / per_gpu
    util = min(max(util, 0.15), 0.95)
    print(
        f"plan: weights {weights:.2f} GiB, {count} x {per_gpu:.2f} GiB visible "
        f"-> TP={tp} util={util:.2f} kv={KV_CACHE_GIB} GiB"
    )
    return tp, util, weights


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


def cmd_download(config: Config) -> int:
    """Fetch checkpoints named by repo that are not already on disk."""
    pending: list[tuple[str, str, str | None]] = []
    for model in config.models:
        if not os.path.isdir(model.reference_path) and model.reference_repo:
            pending.append(
                (model.reference_repo, model.reference_path,
                 model.reference_revision)
            )
        for cand in model.candidates:
            if not os.path.isdir(cand.path) and cand.hf_repo:
                pending.append((cand.hf_repo, cand.path, cand.revision))

    missing = [
        p
        for model in config.models
        for p in [model.reference_path, *(c.path for c in model.candidates)]
        if not os.path.isdir(p)
    ]
    if not pending and missing:
        raise SystemExit(
            "these checkpoints are absent and have no hf_repo to fetch them "
            "from:\n  " + "\n  ".join(missing)
        )

    for repo, dest, revision in pending:
        cmd = ["hf", "download", repo, "--local-dir", dest]
        if revision:
            cmd += ["--revision", revision]
        print(f"=== downloading {repo} -> {dest}")
        if _run(cmd) != 0:
            raise SystemExit(f"download failed for {repo}")
    if not pending:
        print("every checkpoint is already local")
    return 0


def capture_environment(config: Config, python: str) -> str:
    """Run the environment report once per campaign (Law 6)."""
    env_dir = os.path.join(config.work, "environment")
    if os.path.isfile(os.path.join(env_dir, "runtime.json")):
        print(f"=== environment already captured in {env_dir}")
        return env_dir
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
) -> tuple[str, str]:
    """Score one pair. Returns (report_path, capture_dir).

    `capture_label` lets several candidates share one teacher capture. The
    manifest binds a capture to the tokens, geometry, and runtime rather than to
    the candidate, so recapturing the same reference per candidate would spend
    tens of gigabytes and a forward pass to produce identical tensors. It matters
    once a routed model adds a component cell and a scheme ladder.
    """
    tp, util, _ = plan_gpus([student, teacher])
    # Everything the capture manifest binds itself to belongs in the directory
    # name, or a reused capture becomes a confusing abort instead of a recapture.
    if config.suite_dir:
        scope = f"{config.suite_partition}{suite_limit or ''}"
    else:
        scope = f"r{rows}"
    suffix = (
        f"-v{int(config.runner_v2)}-tp{tp}-{scope}"
        f"-c{config.context_length}-s{config.score_from}"
    )
    tag = f"{label}{suffix}"
    capture = os.path.join(
        config.work, "captures", f"{capture_label or label}{suffix}"
    )
    report = os.path.join(config.work, "reports", f"{tag}.json")
    log = os.path.join(config.work, "logs", f"{tag}.log")
    os.makedirs(os.path.dirname(report), exist_ok=True)

    if os.path.isfile(report):
        print(f"=== {tag} already scored")
        return report, capture

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
        "--kv-cache-memory-gib", str(KV_CACHE_GIB),
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

    print(f"=== {tag} (TP={tp} util={util:.2f})")
    rc = _run(cmd, log_path=log, env={
        "VLLM_USE_V2_MODEL_RUNNER": str(int(config.runner_v2)),
        # Scoring allocates and frees vocabulary-wide tensors thousands of
        # times; without expandable segments the pool fragments.
        "PYTORCH_CUDA_ALLOC_CONF": os.environ.get(
            "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
        ),
    })
    if rc != 0:
        raise SystemExit(f"scoring failed for {tag}; see {log}")
    return report, capture


def build_variant(
    config: Config,
    python: str,
    reference: str,
    match: str,
    components: str,
    scheme: str | None,
) -> str:
    """Build a QDQ variant if absent, and return its path.

    The weights are deleted once its report exists (`prune_variants`), because a
    scheme ladder is several checkpoint copies and the report plus the QDQ
    manifest are what an artifact publishes.
    """
    slug = components.replace(",", "-")
    name = f"{os.path.basename(reference)}-qdq-{slug}-{scheme or 'matched'}"
    out = os.path.join(config.work, "variants", name)
    if os.path.isfile(os.path.join(out, "qdq-manifest.json")) and glob.glob(
        os.path.join(out, "*.safetensors")
    ):
        print(f"=== variant {name} already built")
        return out
    cmd = [
        python, os.path.join(HERE, "qdq.py"),
        "--model", reference,
        "--out", out,
        "--components", components,
        "--match", match,
    ]
    if scheme:
        cmd += ["--scheme", scheme]
    print(f"=== building variant {name}")
    rc = _run(cmd, log_path=os.path.join(config.work, "logs", f"{name}.log"))
    if rc != 0:
        raise SystemExit(f"variant build failed for {name}")
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


def _cell(report_path: str, capture_manifest: dict[str, Any], config: Config,
          variant: str, scheme: str) -> dict[str, Any]:
    """One attribution cell, carrying the identity Law 14 compares against."""
    with open(report_path, encoding="utf-8") as handle:
        report = json.load(handle)
    return {
        "mean_kld": report.get("mean_kld"),
        "partition": config.partition,
        "token_sha256": capture_manifest.get("token_sha256"),
        "reference_config_sha256": capture_manifest.get("reference_config_sha256"),
        "variant": os.path.basename(variant),
        "scheme": scheme,
        "report": report_path,
        "qdq_manifest": os.path.join(variant, "qdq-manifest.json"),
    }


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
    deployed_scheme = inspection["detected_scheme"]
    if deployed_scheme is None:
        raise SystemExit(
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

    def score_variant(components: str, scheme: str | None) -> dict[str, Any]:
        variant = build_variant(
            config, python, model.reference_path, deployed.path, components, scheme
        )
        label = f"{model.name}-qdq-{components.replace(',', '-')}-{scheme or 'matched'}"
        report, _ = score_one(
            config, python, label, variant, model.reference_path, config.rows,
            decompose=False, capture_label=f"{model.name}-ref",
        )
        cell = _cell(
            report, capture_manifest, config, variant, scheme or deployed_scheme
        )
        if config.prune_variants:
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
            "inspection": inspection,
        },
        "expert_cell": score_variant("experts", deployed_scheme),
    }
    if router_quantized:
        attribution["router_cell"] = score_variant("router", deployed_scheme)
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
        composite = score_variant(",".join(quantized), deployed_scheme)
    else:
        composite = {}
    if composite:
        attribution["composite_cell"] = dict(composite, components=quantized)
        if isinstance(deployed_mean, (int, float)) and isinstance(
            composite.get("mean_kld"), (int, float)
        ):
            attribution["engine_arithmetic"] = deployed_mean - composite["mean_kld"]

    ladder = []
    for scheme in config.ladder:
        if scheme == deployed_scheme:
            entry = dict(attribution["expert_cell"])
        else:
            entry = score_variant("experts", scheme)
        ladder.append(
            {
                "scheme": scheme,
                "mean_kld": entry["mean_kld"],
                "variant": entry["variant"],
                "report": entry["report"],
            }
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


def cmd_score(config: Config, python: str) -> int:
    """Capture the environment, gate on the zero baseline, score candidates."""
    os.makedirs(config.work, exist_ok=True)
    capture_environment(config, python)

    for model in config.models:
        print(f"\n##### {model.name}")
        # Law 1: the reference must reproduce itself before anything else runs.
        # One row suffices, because the only acceptable answer is exact zero.
        baseline_report, _ = score_one(
            config, python, f"{model.name}-self", model.reference_path,
            model.reference_path, 1, decompose=False,
            suite_limit=1 if config.suite_dir else None,
        )
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

        for cand in model.candidates:
            report, capture = score_one(
                config, python, cand.name, cand.path, model.reference_path,
                config.rows, decompose=config.storage != "logits",
                capture_label=f"{model.name}-ref",
            )
            # Law 14: a routed model's single mean is not publishable, so the
            # component cells are scored here rather than left to whoever reads
            # the artifact.
            attribute_model(config, python, model, cand, capture, report)
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


PUBLISHED_FILES = (
    "report.json",
    "manifest.json",
    "compliance.json",
    "report.md",
    "attribution.json",
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

        env_src = os.path.join(config.work, "environment")
        env_dst = os.path.join(model_root, "environment")
        # The source is scrubbed first, so a credential captured before the
        # redaction policy existed stops propagating into later model roots
        # instead of being cleaned up once per copy.
        _scrub_environment(env_src)
        if os.path.isdir(env_src) and not os.path.isdir(env_dst):
            shutil.copytree(env_src, env_dst)
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

        baseline = _find(config.work, f"{model.name}-self")
        if baseline:
            os.makedirs(os.path.join(model_root, "baselines"), exist_ok=True)
            shutil.copy2(
                baseline,
                os.path.join(model_root, "baselines", "self-kld.json"),
            )

        for cand in model.candidates:
            report = _find(config.work, cand.name)
            if not report:
                print(f"skipping {cand.name}: no report in {config.work}/reports")
                continue
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
            attribution = os.path.join(cand_dir, "attribution.json")
            if os.path.isfile(attribution):
                cmd += ["--attribution", attribution]
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("download", "score", "assemble", "all")
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--force",
        action="store_true",
        help="allow assembly to replace a compliant published result with one "
        "that fails a law",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    started = datetime.now(timezone.utc).isoformat()
    print(f"campaign {config.name} ({config.partition}) started {started}")

    if args.stage == "download":
        return cmd_download(config)

    python = resolve_python()
    print(f"interpreter: {python}")
    if args.stage in ("score", "all"):
        cmd_score(config, python)
    if args.stage in ("assemble", "all"):
        return cmd_assemble(config, python, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
