# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared KLD math used by both model runners and the score-mode example.

Per-position KL is ``KL(reference || candidate)`` in nats over the unpadded
vocabulary (``kld_vocab_size``). The reverse direction, reference top-1
probability, and ordered top-K agreement are computed from the same softmax.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import torch
import torch.nn.functional as F

CONFIDENCE_BINS: tuple[tuple[float, float], ...] = (
    (0.0, 0.25),
    (0.25, 0.5),
    (0.5, 0.75),
    (0.75, 0.95),
    (0.95, 1.01),
)
TOPK_MAX = 5


class KLDResult(NamedTuple):
    """Per-position KLD payload returned by the engine.

    Lists (not tensors) so the scheduler can serialize the result. Mean KLD
    is ``sum(kld_ref_to_model) / len(kld_ref_to_model)``.
    """

    kld_ref_to_model: list[float]
    kld_model_to_ref: list[float]
    ref_top1_prob: list[float]
    model_top1: list[int]
    ref_top1: list[int]
    topk_agree: list[list[int]]

    @property
    def kld_sum(self) -> float:
        return float(sum(self.kld_ref_to_model))

    @property
    def kld_count(self) -> int:
        return len(self.kld_ref_to_model)


def empty_kld_result() -> KLDResult:
    return KLDResult([], [], [], [], [], [])


def iter_eval_rows(
    tokens: Sequence[int],
    context_length: int,
    stride: int,
    rows: int,
) -> list[list[int]]:
    """Slice ``tokens`` into evaluation rows.

    Non-overlapping when ``stride == context_length`` (Turbo/EXL3 default).
    A smaller stride reproduces historical overlapping windows.
    """
    if context_length < 2:
        raise ValueError(f"context_length must be >= 2, got {context_length}")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if rows < 1:
        raise ValueError(f"rows must be >= 1, got {rows}")
    if len(tokens) < 2:
        raise ValueError("Not enough tokens after concatenation")
    if len(tokens) < context_length:
        raise ValueError(
            f"Need at least {context_length} tokens for a full evaluation row, "
            f"got {len(tokens)}"
        )
    windows: list[list[int]] = []
    stop = len(tokens) - context_length + 1
    for start in range(0, stop, stride):
        windows.append(list(tokens[start : start + context_length]))
        if len(windows) >= rows:
            break
    return windows


# Every tensor-parallel worker computes the trunk KLD from the same three files,
# so their answers cross-check that the replay head was installed identically on
# each rank. They are not required to be bit-identical: each rank multiplies its
# own vocab shard on its own device, and the gathered logits carry float32 noise
# in the last bits, which a sum over the vocabulary can amplify. These bound the
# noise that a correct run can produce; a disagreement above them means the ranks
# disagree about the model, not about rounding.
WORKER_KLD_ABS_TOLERANCE = 1e-5
WORKER_KLD_MEAN_REL_TOLERANCE = 1e-7


def nonfinite_summary(result: KLDResult) -> str:
    """Where a KLD payload is not a number, or "" when every value is finite.

    A NaN compares unequal to itself, so it masquerades as a disagreement
    between two workers that in fact computed the identical thing. Naming it
    directly is what separates a scoring fault from a sharding fault.
    """
    import math

    parts = []
    for field in ("kld_ref_to_model", "kld_model_to_ref", "ref_top1_prob"):
        values = getattr(result, field)
        bad = [i for i, v in enumerate(values) if not math.isfinite(float(v))]
        if bad:
            parts.append(
                f"{field}: {len(bad)} of {len(values)} not finite, first at "
                f"position {bad[0]} ({values[bad[0]]!r})"
            )
    return "; ".join(parts)


def worker_agreement(per_worker: Sequence[KLDResult]) -> dict[str, Any]:
    """How far tensor-parallel workers disagree about the same trunk KLD.

    Returns the largest divergence observed, whether it is within tolerance, and
    a description naming the field and position that diverged most. The
    description is the point: a bare inequality tells nobody whether two ranks
    differ in the last bit of a float or by a factor of two.
    """
    if not per_worker:
        raise ValueError("worker agreement needs at least one worker result")
    first = per_worker[0]
    # Checked before any comparison: a NaN is unequal to itself and would be
    # reported as workers disagreeing about a number neither of them produced.
    unfinite = nonfinite_summary(first)
    if unfinite:
        return {
            "agrees": False,
            "workers": len(per_worker),
            "max_abs_delta": 0.0,
            "mean_abs_delta": 0.0,
            "mean_rel_delta": 0.0,
            "top1_flips": 0,
            "nonfinite": unfinite,
            "detail": (
                f"the KLD is not a number, on every worker alike - {unfinite}. "
                f"This is a scoring fault, not a disagreement between ranks: "
                f"logits that overflow or a hidden state carrying NaN produce "
                f"it. Nothing here can be published."
            ),
        }
    detail = ""
    worst_abs = 0.0
    worst_field = ""
    worst_pos = -1
    worst_pair: tuple[float, float] = (0.0, 0.0)
    top1_flips = 0

    for rank, other in enumerate(per_worker[1:], start=1):
        for field in ("kld_ref_to_model", "kld_model_to_ref", "ref_top1_prob"):
            mine = getattr(first, field)
            theirs = getattr(other, field)
            if len(mine) != len(theirs):
                return {
                    "agrees": False,
                    "workers": len(per_worker),
                    "detail": (
                        f"rank 0 scored {len(mine)} positions but rank {rank} "
                        f"scored {len(theirs)} for {field}. The ranks did not "
                        f"score the same work, so no tolerance applies."
                    ),
                }
            for pos, (a, b) in enumerate(zip(mine, theirs)):
                delta = abs(float(a) - float(b))
                if delta > worst_abs:
                    worst_abs = delta
                    worst_field = f"{field} (rank {rank})"
                    worst_pos = pos
                    worst_pair = (float(a), float(b))
        # An argmax over near-tied logits can flip on last-bit noise, so a flip
        # is counted and disclosed rather than treated as a disagreement.
        top1_flips += sum(
            1 for a, b in zip(first.model_top1, other.model_top1) if a != b
        )

    means = [
        (sum(w.kld_ref_to_model) / len(w.kld_ref_to_model))
        if w.kld_ref_to_model
        else 0.0
        for w in per_worker
    ]
    base = means[0]
    mean_abs = max(abs(m - base) for m in means)
    mean_rel = mean_abs / abs(base) if base else mean_abs

    agrees = (
        worst_abs <= WORKER_KLD_ABS_TOLERANCE
        and mean_rel <= WORKER_KLD_MEAN_REL_TOLERANCE
    )
    if worst_abs:
        detail = (
            f"{len(per_worker)} workers agree to {worst_abs:.3e} absolute "
            f"(worst at {worst_field} position {worst_pos}: "
            f"{worst_pair[0]!r} vs {worst_pair[1]!r}); mean trunk KLD differs "
            f"by {mean_abs:.3e} ({mean_rel:.3e} relative); "
            f"{top1_flips} top-1 flip(s)"
        )
    else:
        detail = f"{len(per_worker)} workers agree exactly"
    if not agrees:
        detail += (
            f". Tolerance is {WORKER_KLD_ABS_TOLERANCE:.0e} absolute per "
            f"position and {WORKER_KLD_MEAN_REL_TOLERANCE:.0e} relative on the "
            f"mean. A gap this wide is a sharding or head-replay fault, not "
            f"float noise."
        )
    return {
        "agrees": agrees,
        "workers": len(per_worker),
        "max_abs_delta": worst_abs,
        "mean_abs_delta": mean_abs,
        "mean_rel_delta": mean_rel,
        "top1_flips": top1_flips,
        "detail": detail,
    }


def concat_kld_results(chunks: Sequence[KLDResult]) -> KLDResult:
    out = empty_kld_result()
    for chunk in chunks:
        out.kld_ref_to_model.extend(chunk.kld_ref_to_model)
        out.kld_model_to_ref.extend(chunk.kld_model_to_ref)
        out.ref_top1_prob.extend(chunk.ref_top1_prob)
        out.model_top1.extend(chunk.model_top1)
        out.ref_top1.extend(chunk.ref_top1)
        out.topk_agree.extend(chunk.topk_agree)
    return out


def resolve_vocab_size(
    model_width: int,
    ref_width: int,
    kld_vocab_size: int | None,
) -> int:
    """Unpadded vocab used for softmax; never larger than either logit width."""
    vs = min(model_width, ref_width)
    if kld_vocab_size is not None:
        if kld_vocab_size < 1:
            raise ValueError(
                f"kld_vocab_size must be >= 1, got {kld_vocab_size}"
            )
        vs = min(vs, kld_vocab_size)
    return vs


def compute_kld_chunk(
    model_logits: torch.Tensor,
    ref_logits: torch.Tensor,
    kld_vocab_size: int | None = None,
    topk: int = TOPK_MAX,
) -> KLDResult:
    """KL, reverse KL, reference confidence, and ordered top-K agreement.

    Args:
        model_logits: Candidate logits ``[T, V_model]``.
        ref_logits: Reference logits ``[T, V_ref]``.
        kld_vocab_size: Tokenizer unpadded size; truncation happens before
            softmax so padding rows cannot contribute.
        topk: Ordered top-K agreement depth (default 5).
    """
    if model_logits.ndim != 2 or ref_logits.ndim != 2:
        raise ValueError(
            "KLD expects 2D logits [positions, vocab]; "
            f"got {tuple(model_logits.shape)} and {tuple(ref_logits.shape)}"
        )
    if model_logits.shape[0] != ref_logits.shape[0]:
        raise ValueError(
            "Reference and model logits must have the same number of "
            f"positions; got {ref_logits.shape[0]} and {model_logits.shape[0]}"
        )
    vs = resolve_vocab_size(
        model_logits.shape[-1], ref_logits.shape[-1], kld_vocab_size
    )
    model = model_logits[..., :vs].float()
    ref = ref_logits[..., :vs].float()
    log_model = F.log_softmax(model, dim=-1)
    log_ref = F.log_softmax(ref, dim=-1)
    kld_rm = F.kl_div(log_model, log_ref, reduction="none", log_target=True).sum(-1)
    kld_mr = F.kl_div(log_ref, log_model, reduction="none", log_target=True).sum(-1)
    # Every path that produces a published mean passes through here, so this is
    # the one place that can guarantee the mean is a number. A candidate whose
    # forward pass emits NaN would otherwise be summarized, ranked, and
    # published, because a NaN propagates through a mean without complaint.
    if not bool(torch.isfinite(kld_rm).all() and torch.isfinite(kld_mr).all()):
        bad = int((~torch.isfinite(kld_rm)).sum())
        raise ValueError(
            f"KLD is not finite at {bad} of {kld_rm.numel()} positions. "
            f"Candidate logits reach {float(model.abs().max()):.4g} and the "
            f"reference's {float(ref.abs().max()):.4g}; a NaN or an infinity in "
            f"either makes the mean meaningless, so it is refused rather than "
            f"summarized."
        )
    ref_prob = log_ref.exp()
    ref_top1_prob, ref_top1 = ref_prob.max(dim=-1)
    model_top1 = log_model.argmax(dim=-1)
    k = min(topk, vs)
    model_topk = torch.topk(model, k, dim=-1).indices
    ref_topk = torch.topk(ref, k, dim=-1).indices
    agree_rows: list[list[int]] = []
    eq = model_topk == ref_topk
    for t in range(eq.shape[0]):
        row = [0] * topk
        prefix = True
        for i in range(k):
            prefix = prefix and bool(eq[t, i].item())
            row[i] = int(prefix)
        agree_rows.append(row)
    return KLDResult(
        kld_ref_to_model=kld_rm.tolist(),
        kld_model_to_ref=kld_mr.tolist(),
        ref_top1_prob=ref_top1_prob.tolist(),
        model_top1=model_top1.tolist(),
        ref_top1=ref_top1.tolist(),
        topk_agree=agree_rows,
    )


def load_reference_slice(
    path: str,
    key: str,
    start_pos: int,
    end_pos: int,
) -> torch.Tensor:
    """Load ``reference[start_pos:end_pos]`` from a safetensors file."""
    from safetensors.torch import safe_open

    with safe_open(path, framework="pt", device="cpu") as f:
        if key not in f.keys():
            raise ValueError(f"Safetensors {path!r} has no key {key!r}")
        sl = f.get_slice(key)
        shape = sl.get_shape()
        if len(shape) != 2 or shape[0] < end_pos:
            raise ValueError(
                f"Reference {key!r} has shape {shape}, but at least "
                f"{end_pos} positions are needed."
            )
        tensor = sl[start_pos:end_pos]
    if not torch.is_floating_point(tensor):
        raise ValueError(
            f"Reference tensor must be floating point, got {tensor.dtype}."
        )
    return tensor


def slice_kld_result(result: KLDResult, score_from: int) -> KLDResult:
    """Drop the first ``score_from`` positions (shallow-context prefix)."""
    if score_from < 0:
        raise ValueError(f"score_from must be >= 0, got {score_from}")
    if score_from == 0:
        return result
    return KLDResult(
        kld_ref_to_model=result.kld_ref_to_model[score_from:],
        kld_model_to_ref=result.kld_model_to_ref[score_from:],
        ref_top1_prob=result.ref_top1_prob[score_from:],
        model_top1=result.model_top1[score_from:],
        ref_top1=result.ref_top1[score_from:],
        topk_agree=result.topk_agree[score_from:],
    )


def _mean(xs: Sequence[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


def summarize_kld(
    result: KLDResult,
    *,
    score_from: int = 0,
    context_length: int | None = None,
) -> dict[str, Any]:
    """Mean/percentiles, confidence buckets, depth buckets, top-K agreement."""
    sliced = slice_kld_result(result, score_from)
    vals = sliced.kld_ref_to_model
    if not vals:
        raise ValueError("No scored positions after applying score_from")
    t = torch.tensor(vals, dtype=torch.float64)
    conf = torch.tensor(sliced.ref_top1_prob, dtype=torch.float64)
    report: dict[str, Any] = {
        "score_from": score_from,
        "num_positions": len(vals),
        "mean_kld": float(t.mean().item()),
        "mean_kld_reverse": _mean(sliced.kld_model_to_ref),
        "median_kld": float(t.median().item()),
        "p90_kld": float(t.quantile(0.90).item()),
        "p99_kld": float(t.quantile(0.99).item()),
        "max_kld": float(t.max().item()),
        "top1_agreement": _mean(
            [float(a == b) for a, b in zip(sliced.model_top1, sliced.ref_top1)]
        ),
        "topk_agreement": {},
        "confidence_buckets": [],
        "depth_buckets": [],
    }
    k_max = max((len(row) for row in sliced.topk_agree), default=0)
    for k in range(1, k_max + 1):
        hits = [row[k - 1] for row in sliced.topk_agree if len(row) >= k]
        report["topk_agreement"][k] = _mean([float(x) for x in hits])
    for lo, hi in CONFIDENCE_BINS:
        mask = (conf >= lo) & (conf < hi)
        n = int(mask.sum().item())
        bucket_mean = float(t[mask].mean().item()) if n else None
        report["confidence_buckets"].append(
            {
                "lo": lo,
                "hi": min(hi, 1.0),
                "n": n,
                "frac": n / len(vals),
                "mean_kld": bucket_mean,
            }
        )
    n_pos = len(vals)
    depth_start = score_from
    if context_length is None:
        context_length = depth_start + n_pos + 1
    edges = [0, context_length // 4, context_length // 2, 3 * context_length // 4]
    edges = sorted(
        {max(0, e - depth_start) for e in edges if e <= depth_start + n_pos}
    )
    edges.append(n_pos)
    for a, b in zip(edges, edges[1:]):
        if a >= b:
            continue
        report["depth_buckets"].append(
            {
                "depth_lo": depth_start + a,
                "depth_hi": depth_start + b - 1,
                "n": b - a,
                "mean_kld": float(t[a:b].mean().item()),
            }
        )
    return report


def summarize_kld_rows(
    rows: Sequence[KLDResult],
    *,
    score_from: int = 0,
    context_length: int,
) -> dict[str, Any]:
    """Summarize rows after dropping the shallow prefix from every row."""
    if not rows:
        raise ValueError("No KLD rows to summarize")
    scored_rows = [slice_kld_result(row, score_from) for row in rows]
    combined = concat_kld_results(scored_rows)
    report = summarize_kld(combined)
    report["score_from"] = score_from

    max_positions = max((row.kld_count for row in rows), default=0)
    quarter_edges = (
        0,
        context_length // 4,
        context_length // 2,
        3 * context_length // 4,
        context_length - 1,
    )
    edges = {score_from, max_positions}
    edges.update(
        edge for edge in quarter_edges if score_from < edge < max_positions
    )
    sorted_edges = sorted(edges)

    depth_buckets: list[dict[str, Any]] = []
    for lo, hi in zip(sorted_edges, sorted_edges[1:]):
        if lo >= hi:
            continue
        values = [
            value
            for row in rows
            for value in row.kld_ref_to_model[lo:min(hi, row.kld_count)]
        ]
        if values:
            depth_buckets.append(
                {
                    "depth_lo": lo,
                    "depth_hi": hi - 1,
                    "n": len(values),
                    "mean_kld": _mean(values),
                }
            )
    report["depth_buckets"] = depth_buckets
    return report


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tokens(tokens: Sequence[int]) -> str:
    payload = ",".join(str(t) for t in tokens).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def tokenizer_unpadded_vocab_size(tokenizer: Any) -> int:
    """Unpadded vocab (Turbo ``actual_vocab_size`` when present)."""
    vs = getattr(tokenizer, "actual_vocab_size", None)
    if vs is not None:
        return int(vs)
    vs = getattr(tokenizer, "vocab_size", None)
    if vs is None:
        try:
            vs = len(tokenizer)
        except TypeError:
            raise ValueError("Cannot determine tokenizer vocabulary size")
    return int(vs)


def capture_runtime_manifest() -> dict[str, Any]:
    """GPU / torch provenance fields for capture manifests."""
    import platform

    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "vllm_commit": os.environ.get("KLD_REPO_COMMIT") or _git_commit(),
        "vllm_tree_dirty": os.environ.get("KLD_REPO_DIRTY") == "1"
        or _git_dirty(),
        "vllm_dirty_digest": os.environ.get("KLD_REPO_DIRTY_DIGEST")
        or _git_dirty_digest(),
        "determinism": {
            "VLLM_BATCH_INVARIANT": os.environ.get("VLLM_BATCH_INVARIANT"),
            "VLLM_MOE_USE_DEEP_GEMM": os.environ.get("VLLM_MOE_USE_DEEP_GEMM"),
            "NCCL_DETERMINISTIC": os.environ.get("NCCL_DETERMINISTIC"),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
        "compiled_extensions": _compiled_extension_hashes(),
    }
    info["torch"] = torch.__version__
    if torch.cuda.is_available():
        info["cuda"] = torch.version.cuda
        info["gpu_names"] = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
        try:
            info["nccl"] = ".".join(str(v) for v in torch.cuda.nccl.version())
        except Exception:
            info["nccl"] = None
        try:
            import pynvml
        except ImportError:
            info["driver"] = None
        else:
            try:
                pynvml.nvmlInit()
                driver = pynvml.nvmlSystemGetDriverVersion()
                info["driver"] = (
                    driver.decode() if isinstance(driver, bytes) else driver
                )
                pynvml.nvmlShutdown()
            except pynvml.NVMLError:
                info["driver"] = None
    try:
        import flashinfer

        info["flashinfer"] = getattr(flashinfer, "__version__", str(flashinfer))
    except Exception:
        info["flashinfer"] = None
    info["compiled_extensions_sha256"] = _digest_mapping(
        info["compiled_extensions"]
    )
    return info


def _git_commit() -> str | None:
    probe = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        cwd=os.environ.get("KLD_REPO_ROOT") or None,
    )
    return probe.stdout.strip() or None if probe.returncode == 0 else None


def _git_dirty() -> bool:
    probe = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        cwd=os.environ.get("KLD_REPO_ROOT") or None,
    )
    return probe.returncode == 0 and bool(probe.stdout.strip())


def _git_dirty_digest() -> str | None:
    probe = subprocess.run(
        ["git", "diff", "HEAD"],
        capture_output=True,
        cwd=os.environ.get("KLD_REPO_ROOT") or None,
    )
    if probe.returncode != 0:
        return None
    return hashlib.sha256(probe.stdout).hexdigest()


def _compiled_extension_hashes() -> dict[str, str]:
    try:
        import vllm
    except Exception:
        return {}
    root = Path(vllm.__file__).resolve().parent
    hashes: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        name = path.name
        if not (
            name.startswith("_C")
            or name.startswith("_moe_C")
            or "marlin" in name.lower()
        ):
            continue
        if path.suffix not in {".so", ".pyd", ".dll"}:
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[str(path.relative_to(root)).replace("\\", "/")] = digest.hexdigest()
    return hashes


def _digest_mapping(values: dict[str, str]) -> str:
    payload = json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def load_lm_head_weight(
    head_path: str,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Load ``lm_head.safetensors`` weight tensor."""
    from safetensors.torch import load_file

    weight = load_file(head_path)["weight"]
    if device is not None:
        weight = weight.to(device)
    return weight


def _find_lm_head_owner(
    model: torch.nn.Module,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead,
    )

    for module in model.modules():
        candidate = getattr(module, "lm_head", None)
        if isinstance(candidate, ParallelLMHead):
            return module, candidate
    raise ValueError("Loaded model has no ParallelLMHead")


def build_replay_lm_head(
    model: torch.nn.Module,
    full_weight: torch.Tensor,
    device: torch.device | str,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Build a TP-aware unquantized head matching a loaded model."""
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead,
    )

    owner, original = _find_lm_head_owner(model)
    if getattr(original, "bias", None) is not None:
        raise ValueError("LM-head replay does not support a biased output head")

    with torch.device(device):
        replay = ParallelLMHead(
            num_embeddings=full_weight.shape[0],
            embedding_dim=full_weight.shape[1],
            bias=False,
            params_dtype=full_weight.dtype,
            org_num_embeddings=full_weight.shape[0],
            padding_size=original.padding_size,
            prefix="kld_reference_lm_head",
            disable_tp=original.disable_tp,
        )
    replay.weight_loader(replay.weight, full_weight)
    return owner, replay


def replay_lm_head_in_model(
    model: torch.nn.Module,
    replay_state: tuple[torch.nn.Module, torch.nn.Module],
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    """Replay via the model's exact ``compute_logits`` / logits-processor path."""
    owner, replay_head = replay_state
    original = owner.lm_head
    owner.lm_head = replay_head
    try:
        logits = model.compute_logits(hidden_states.to(replay_head.params_dtype))
    finally:
        owner.lm_head = original
    if logits is None:
        raise RuntimeError("Reference LM-head replay returned no gathered logits")
    return logits


def compute_kld_from_reference(
    model: torch.nn.Module,
    model_logits: torch.Tensor,
    reference_path: str,
    reference_key: str,
    start_pos: int,
    kld_vocab_size: int,
    replay_cache: dict[str, tuple[torch.nn.Module, torch.nn.Module]],
) -> KLDResult:
    """Load and score one reference chunk identically for V1 and V2."""
    end_pos = start_pos + model_logits.shape[0]
    reference = load_reference_slice(
        reference_path, reference_key, start_pos, end_pos
    )
    if reference_key == "hidden_states":
        head_path = os.path.join(
            os.path.dirname(reference_path), "lm_head.safetensors"
        )
        if not os.path.isfile(head_path):
            raise ValueError(
                "Hidden-state KLD requires lm_head.safetensors next to "
                f"the reference file, missing {head_path}"
            )
        replay_state = replay_cache.get(head_path)
        if replay_state is None:
            weight = load_lm_head_weight(head_path)
            replay_state = build_replay_lm_head(
                model, weight, model_logits.device
            )
            if len(replay_cache) >= 2:
                replay_cache.pop(next(iter(replay_cache)))
            replay_cache[head_path] = replay_state
        reference_logits = replay_lm_head_in_model(
            model, replay_state, reference.to(model_logits.device)
        )
    else:
        reference_logits = reference.to(model_logits.device)
    return compute_kld_chunk(
        model_logits, reference_logits, kld_vocab_size
    )


def probe_replay_exactness_in_model(
    model: torch.nn.Module,
    *,
    hidden_path: str,
    logits_path: str,
    head_path: str,
    kld_vocab_size: int | None,
) -> dict[str, Any]:
    """Probe serialized hidden/head replay through loaded model semantics."""
    from safetensors.torch import load_file

    hidden = load_file(hidden_path)["hidden_states"]
    live = load_file(logits_path)["logits"]
    weight = load_lm_head_weight(head_path)
    device = next(model.parameters()).device
    replay_state = build_replay_lm_head(model, weight, device)
    replayed = replay_lm_head_in_model(
        model, replay_state, hidden.to(device)
    ).float().cpu()
    vs = resolve_vocab_size(
        live.shape[-1], replayed.shape[-1], kld_vocab_size
    )
    live = live[..., :vs].float()
    replayed = replayed[..., :vs]
    delta = (replayed - live).abs()
    return {
        "identical": bool(torch.equal(replayed, live)),
        "max_abs": float(delta.max().item()) if delta.numel() else 0.0,
        "num_differing": int((delta > 0).sum().item()),
        "positions": int(live.shape[0]),
        "vocab": vs,
    }


def compute_trunk_kld_in_model(
    model: torch.nn.Module,
    *,
    student_hidden_path: str,
    teacher_hidden_path: str,
    head_path: str,
    kld_vocab_size: int | None,
    position_chunk: int = 256,
) -> KLDResult:
    """Compute shared-teacher-head KLD through model logits semantics.

    Positions are processed in chunks. A full row of logits over a large
    vocabulary is gigabytes, and two of them plus the float copies KLD needs do
    not fit beside a loaded model and its KV cache. Every quantity here is
    computed per position, so chunking changes nothing about the result.
    """
    from safetensors.torch import load_file

    student = load_file(student_hidden_path)["hidden_states"]
    teacher = load_file(teacher_hidden_path)["hidden_states"]
    if student.shape != teacher.shape:
        raise ValueError(
            "Student and teacher hidden-state shapes must match for trunk KLD; "
            f"got {tuple(student.shape)} and {tuple(teacher.shape)}"
        )
    if position_chunk < 1:
        raise ValueError(f"position_chunk must be positive, got {position_chunk}")
    weight = load_lm_head_weight(head_path)
    device = next(model.parameters()).device
    replay_state = build_replay_lm_head(model, weight, device)
    chunks: list[KLDResult] = []
    for start in range(0, student.shape[0], position_chunk):
        stop = min(start + position_chunk, student.shape[0])
        student_slice = student[start:stop].to(device)
        teacher_slice = teacher[start:stop].to(device)
        student_logits = replay_lm_head_in_model(model, replay_state, student_slice)
        teacher_logits = replay_lm_head_in_model(model, replay_state, teacher_slice)
        # Raised here, where the values that produced it are still in hand. A NaN
        # discovered later is only a NaN; discovered here it says which side went
        # bad and how large the hidden states were that took it there.
        for label, hidden, logits in (
            ("student", student_slice, student_logits),
            ("teacher", teacher_slice, teacher_logits),
        ):
            finite = torch.isfinite(logits)
            if not bool(finite.all()):
                bad = int((~finite).sum())
                raise ValueError(
                    f"{label} logits are not finite at positions "
                    f"{start}-{stop}: {bad} of {logits.numel()} values, from "
                    f"hidden states with max magnitude "
                    f"{float(hidden.abs().max()):.4g} in {hidden.dtype} through "
                    f"a {replay_state[1].params_dtype} head. A 4-bit or 8-bit "
                    f"trunk that leaves the hidden scale far from the "
                    f"reference's overflows the head's dtype here."
                )
        chunks.append(
            compute_kld_chunk(student_logits, teacher_logits, kld_vocab_size)
        )
        del student_logits, teacher_logits
    return concat_kld_results(chunks)


def tokenizer_identity(tokenizer: Any) -> dict[str, Any]:
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is None:
        try:
            vocab_size = len(tokenizer)
        except TypeError:
            vocab_size = None
    unpadded = getattr(tokenizer, "actual_vocab_size", None)
    vocab = tokenizer.get_vocab()
    vocab_payload = json.dumps(
        sorted(vocab.items()), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    special_tokens_map = json.loads(
        json.dumps(getattr(tokenizer, "special_tokens_map", {}), default=str)
    )
    return {
        "class": type(tokenizer).__name__,
        "vocab_size": int(vocab_size) if vocab_size is not None else None,
        "actual_vocab_size": int(unpadded) if unpadded is not None else None,
        "vocab_sha256": hashlib.sha256(vocab_payload).hexdigest(),
        "special_tokens_map": special_tokens_map,
    }


_PACKED_LM_HEAD_KEYS = (
    "qweight",
    "weight_packed",
    "weight_scale",
    "scales",
    "qzeros",
    "g_idx",
)
_FLOAT_SAFETENSORS_DTYPES = frozenset({"BF16", "F16", "F32", "F64"})


def _lm_head_tensor_info(model_path: str) -> tuple[dict[str, str], dict[str, str]]:
    """Return output-weight tensor-to-file and tensor-to-dtype mappings."""
    from safetensors import safe_open

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    weight_map: dict[str, str] = {}
    if os.path.isfile(index_path):
        with open(index_path, encoding="utf-8") as f:
            all_weights = json.load(f).get("weight_map") or {}
        weight_map = {
            key: filename
            for key, filename in all_weights.items()
            if "lm_head" in key.lower() or "embed_tokens" in key.lower()
        }
    else:
        single = os.path.join(model_path, "model.safetensors")
        if os.path.isfile(single):
            with safe_open(single, framework="pt", device="cpu") as f:
                weight_map = {
                    key: os.path.basename(single)
                    for key in f.keys()
                    if "lm_head" in key.lower()
                    or "embed_tokens" in key.lower()
                }

    dtypes: dict[str, str] = {}
    by_file: dict[str, list[str]] = {}
    for key, filename in weight_map.items():
        by_file.setdefault(filename, []).append(key)
    for filename, keys in by_file.items():
        path = os.path.join(model_path, filename)
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in keys:
                dtypes[key] = str(f.get_slice(key).get_dtype())
    return weight_map, dtypes


def detect_lm_head_quantization(model_path: str) -> dict[str, Any]:
    """Static detection from config.json and the safetensors index.

    Packed tensor names mean quantized. A lone bf16/fp16 ``lm_head.weight``
    (or a tied embedding) means not. ``unknown`` if neither signal is present.
    """
    config_path = os.path.join(model_path, "config.json")
    config: dict[str, Any] = {}
    if os.path.isfile(config_path):
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
    qcfg = config.get("quantization_config") or {}
    ignore = set(
        (qcfg.get("ignore") or [])
        + (qcfg.get("modules_to_not_convert") or [])
        + (config.get("quantization_config", {}).get("ignored_layers") or [])
    )
    tied = bool(config.get("tie_word_embeddings"))
    weight_map, dtypes = _lm_head_tensor_info(model_path)
    lm_keys = [k for k in weight_map if "lm_head" in k.lower()]
    output_keys = lm_keys
    if tied and not output_keys:
        output_keys = [
            k for k in weight_map if "embed_tokens" in k.lower()
        ]
    packed = [
        k
        for k in output_keys
        if any(p in k.lower() for p in _PACKED_LM_HEAD_KEYS)
    ]
    float_weights = [
        k
        for k in output_keys
        if k.endswith(("lm_head.weight", "embed_tokens.weight"))
        and dtypes.get(k) in _FLOAT_SAFETENSORS_DTYPES
    ]
    non_float_weights = [
        k
        for k in output_keys
        if k.endswith(("lm_head.weight", "embed_tokens.weight"))
        and dtypes.get(k) not in _FLOAT_SAFETENSORS_DTYPES
    ]
    ignored = any("lm_head" in str(name).lower() for name in ignore)
    if (packed or non_float_weights) and not ignored:
        state = "quantized"
    elif ignored or (float_weights and not packed):
        state = "unquantized"
    else:
        state = "unknown"
    return {
        "state": state,
        "tie_word_embeddings": tied,
        "ignored": ignored,
        "lm_head_keys": sorted(lm_keys),
        "output_weight_keys": sorted(output_keys),
        "lm_head_dtypes": dtypes,
        "packed_keys": sorted(packed),
        "quant_method": qcfg.get("quant_method"),
    }


def inspect_parallel_lm_head(lm_head: Any) -> dict[str, Any]:
    """Runtime detection from a loaded ``ParallelLMHead``."""
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        UnquantizedEmbeddingMethod,
    )

    quant_method = getattr(lm_head, "quant_method", None)
    unquantized = isinstance(quant_method, UnquantizedEmbeddingMethod)
    weight = getattr(lm_head, "weight", None)
    dtype = str(getattr(weight, "dtype", None))
    if quant_method is None:
        state = "unknown"
    else:
        state = "unquantized" if unquantized else "quantized"
    return {
        "state": state,
        "quant_method": type(quant_method).__name__ if quant_method else None,
        "weight_dtype": dtype,
        "org_vocab_size": getattr(lm_head, "org_vocab_size", None),
    }


def inspect_model_lm_heads(model: torch.nn.Module) -> dict[str, Any]:
    """Authoritative runtime LM-head inspection for ``LLM.apply_model``."""
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

    heads = []
    for name, module in model.named_modules():
        if isinstance(module, ParallelLMHead):
            heads.append({"name": name, **inspect_parallel_lm_head(module)})
    owner, primary = _find_lm_head_owner(model)
    primary_info = inspect_parallel_lm_head(primary)
    processor = getattr(owner, "logits_processor", None)
    processor_info = None
    if processor is not None:
        processor_info = {
            "type": type(processor).__name__,
            "scale": getattr(processor, "scale", None),
            "soft_cap": getattr(processor, "soft_cap", None),
            "vocab_size": getattr(processor, "vocab_size", None),
            "org_vocab_size": getattr(processor, "org_vocab_size", None),
            "head_dtype": str(getattr(processor, "head_dtype", None)),
        }
    return {
        "state": primary_info["state"],
        "primary": primary_info,
        "logits_processor": processor_info,
        "heads": heads,
    }


def inspect_model_moe_backends(model: torch.nn.Module) -> dict[str, Any]:
    """Record the loaded MoE router and expert implementation on one worker."""
    from vllm.model_executor.layers.fused_moe.layer import MoERunner

    layers = []
    for name, module in model.named_modules():
        if not isinstance(module, MoERunner):
            continue
        quant_method = module.routed_experts.quant_method
        kernel = getattr(quant_method, "moe_kernel", None)
        implementation = getattr(kernel, "impl", None)
        experts = getattr(implementation, "fused_experts", None)
        routing_method = getattr(
            experts,
            "routing_method_type",
            getattr(experts, "routing_method", None),
        )
        supports_batch_invariant = False
        if experts is not None:
            checker = getattr(type(experts), "_supports_batch_invariance", None)
            if callable(checker):
                supports_batch_invariant = bool(checker())
        expert_name = type(experts).__name__ if experts is not None else None
        moe_config = getattr(module, "moe_config", None)
        use_ep = bool(getattr(moe_config, "use_ep", False))
        certified = supports_batch_invariant and not use_ep
        layers.append(
            {
                "name": name,
                "layer_id": module.layer_id,
                "router": type(module.router).__name__,
                "quant_method": type(quant_method).__name__,
                "kernel": type(kernel).__name__ if kernel is not None else None,
                "experts": expert_name,
                "monolithic": bool(quant_method.is_monolithic),
                "routing_method": getattr(
                    routing_method, "name", str(routing_method)
                ),
                "renormalize": getattr(module.router, "renormalize", None),
                "scoring_func": getattr(module.router, "scoring_func", None),
                "use_ep": use_ep,
                "ep_size": getattr(moe_config, "ep_size", None),
                "tp_size": getattr(moe_config, "tp_size", None),
                "batch_invariant_supported": supports_batch_invariant,
                "certified_for_exact_repeat": certified,
            }
        )
    return {"layers": layers}


LOGITS_PROCESSOR_IDENTITY_KEYS = (
    "type",
    "scale",
    "soft_cap",
    "vocab_size",
    "org_vocab_size",
)


def logits_processor_identity(
    info: dict[str, Any] | None,
) -> dict[str, Any]:
    """The hidden-to-vocab map, excluding head storage dtype.

    AWQ exports commonly keep an unquantized FP16 lm_head on a BF16 teacher.
    That dtype is deployed behavior (Law 8), not a different vocabulary or
    logit transform.
    """
    if not info:
        return {}
    return {key: info.get(key) for key in LOGITS_PROCESSOR_IDENTITY_KEYS}


def copy_lm_head_from_checkpoint(model_path: str, dest_path: str) -> str:
    """Copy the unquantized ``lm_head.weight`` (or tied embed) to ``dest_path``.

    Returns the source tensor name. Raises if the head appears quantized.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    info = detect_lm_head_quantization(model_path)
    if info["state"] == "quantized":
        raise ValueError(
            f"Cannot copy a quantized LM head from {model_path} for replay"
        )
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    weight_map: dict[str, str] = {}
    if os.path.isfile(index_path):
        with open(index_path, encoding="utf-8") as f:
            weight_map = json.load(f).get("weight_map") or {}
    candidates = [k for k in weight_map if k.endswith("lm_head.weight")]
    if not candidates and info["tie_word_embeddings"]:
        candidates = [k for k in weight_map if k.endswith("embed_tokens.weight")]
    if not candidates:
        # Single-file checkpoint.
        single = os.path.join(model_path, "model.safetensors")
        if not os.path.isfile(single):
            raise FileNotFoundError(f"No safetensors weights under {model_path}")
        names = ["lm_head.weight"]
        if info["tie_word_embeddings"]:
            names.extend(("model.embed_tokens.weight", "embed_tokens.weight"))
        with safe_open(single, framework="pt", device="cpu") as f:
            for name in names:
                if name in f.keys():
                    weight = f.get_tensor(name)
                    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
                    save_file({"weight": weight}, dest_path)
                    return name
        raise FileNotFoundError(f"No lm_head.weight in {single}")
    # Prefer an explicit lm_head over a tied embedding.
    name = next((k for k in candidates if "lm_head" in k), candidates[0])
    shard = os.path.join(model_path, weight_map[name])
    with safe_open(shard, framework="pt", device="cpu") as f:
        if name not in f.keys():
            raise FileNotFoundError(f"{name} missing from {shard}")
        weight = f.get_tensor(name)
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    save_file({"weight": weight}, dest_path)
    return name


def write_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def read_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def manifest_mismatches(
    manifest: dict[str, Any],
    live: dict[str, Any],
    required: Iterable[str] = (
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
    ),
) -> list[str]:
    """Return human-readable mismatches between a capture manifest and live cfg."""
    errors: list[str] = []
    for key in required:
        if key not in manifest:
            errors.append(f"manifest missing {key!r}")
            continue
        if key not in live:
            errors.append(f"live config missing {key!r}")
            continue
        if manifest[key] != live[key]:
            errors.append(f"{key}: captured {manifest[key]!r} != live {live[key]!r}")
    return errors
