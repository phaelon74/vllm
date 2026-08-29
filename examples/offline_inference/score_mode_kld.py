#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
KLD (Kullback-Leibler Divergence) using vLLM score mode.

Compares a candidate (student) against a teacher capture via
``KL(teacher || student)`` over the unpadded vocabulary. Default windowing
is non-overlapping rows (Turbo/EXL3), not a sliding window. Use ``--stride``
only to regenerate historical overlapping numbers.

Usage:
    python examples/offline_inference/score_mode_kld.py \\
        --model /path/to/quantized_model \\
        --reference-model /path/to/reference_model \\
        --dataset wikitext --dataset-config wikitext-2-raw-v1
"""

import argparse
import gc
import glob
import json
import logging
import os
import tempfile
import time
from functools import partial
from typing import Any

import torch
from datasets import load_dataset
from safetensors.torch import save_file
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

logger = logging.getLogger(__name__)

# Best-effort compiled determinism config (--compiled only). Eager execution
# (the default) is the only mode proven bit-reproducible run-to-run on the
# current stack. This config disables known timing-based selectors but does
# not guarantee reproducibility; see docs/features/score_mode.md.
DETERMINISTIC_COMPILATION_CONFIG: dict[str, Any] = {
    "inductor_compile_config": {
        "combo_kernels": False,
        "benchmark_combo_kernel": False,
        "triton.autotune_pointwise": False,
        "max_autotune": False,
        "coordinate_descent_tuning": False,
        "benchmark_fusion": False,
    },
}


def apply_deterministic_env() -> None:
    """Best-effort: disable timing-based autotuners via environment variables.

    Used only with ``--compiled``. Does not make compiled scoring
    bit-reproducible; eager mode is required for that.
    """
    os.environ.setdefault("TORCHINDUCTOR_DETERMINISTIC", "1")
    os.environ.setdefault("VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE", "0")
    os.environ.setdefault("VLLM_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING", "0")
    try:
        # Cover the in-process path too (env var above only takes effect
        # at torch._inductor.config import time in fresh processes).
        import torch._inductor.config as inductor_config

        if hasattr(inductor_config, "deterministic"):
            inductor_config.deterministic = True
    except ImportError:
        pass


def apply_compiled_llm_kwargs(llm_kwargs: dict[str, Any]) -> None:
    """Apply best-effort compiled determinism settings (--compiled only)."""
    apply_deterministic_env()
    llm_kwargs["compilation_config"] = DETERMINISTIC_COMPILATION_CONFIG
    llm_kwargs["enable_flashinfer_autotune"] = False


def apply_eager_llm_kwargs(llm_kwargs: dict[str, Any]) -> None:
    """Apply guaranteed bit-reproducible eager execution (default)."""
    llm_kwargs["enforce_eager"] = True


def allow_apply_model_rpc() -> None:
    """Permit ``LLM.apply_model`` to reach an out-of-process engine core.

    LM-head inspection, hidden-capture verification, and the replay probe send
    local functions over the engine-core RPC, which msgspec cannot encode. Any
    configuration that moves the engine core into its own process (tensor
    parallelism, for one) therefore needs the pickle fallback. The pickled
    payloads are this script's own functions.

    Must run before the first ``LLM`` is constructed: the engine-core process
    inherits the environment at spawn time, and vLLM caches env lookups once
    initialization completes.
    """
    if os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1") != "1":
        print(
            "VLLM_ALLOW_INSECURE_SERIALIZATION is set to "
            f"{os.environ['VLLM_ALLOW_INSECURE_SERIALIZATION']!r}; LM-head "
            "inspection and the replay probe will fail unless the engine core "
            "runs in this process."
        )


def _load_local_parquet_split(
    dataset_dir: str,
    dataset_config: str | None,
    split: str,
):
    """Load a split from parquet files under a local Hub snapshot."""
    parquet_dir = (
        os.path.join(dataset_dir, dataset_config) if dataset_config else dataset_dir
    )
    files = sorted(glob.glob(os.path.join(parquet_dir, f"{split}-*.parquet")))
    if not files:
        return None
    return load_dataset("parquet", data_files={split: files}, split=split)


def load_dataset_texts(
    dataset_name: str,
    dataset_config: str | None = None,
    split: str | None = None,
) -> list[str]:
    """Load and extract text from a HuggingFace dataset or local snapshot."""
    if os.path.isdir(dataset_name):
        dataset_name = os.path.abspath(dataset_name)
        dataset_names = [dataset_name]
    else:
        # Hub moved script-based `wikitext` to `Salesforce/wikitext`.
        dataset_names = [dataset_name]
        if dataset_name == "wikitext":
            dataset_names.append("Salesforce/wikitext")

    load_kwargs: dict[str, Any] = {}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        load_kwargs["token"] = token

    loaded = None
    last_error: Exception | None = None
    if split is None:
        for name in dataset_names:
            for candidate_split in ["test", "train", "validation"]:
                try:
                    if os.path.isdir(name):
                        loaded = _load_local_parquet_split(
                            name, dataset_config, candidate_split
                        )
                        if loaded is None:
                            if dataset_config:
                                loaded = load_dataset(
                                    name,
                                    dataset_config,
                                    split=candidate_split,
                                    **load_kwargs,
                                )
                            else:
                                loaded = load_dataset(
                                    name, split=candidate_split, **load_kwargs
                                )
                    elif dataset_config:
                        loaded = load_dataset(
                            name, dataset_config, split=candidate_split, **load_kwargs
                        )
                    else:
                        loaded = load_dataset(
                            name, split=candidate_split, **load_kwargs
                        )
                    dataset_name = name
                    split = candidate_split
                    break
                except Exception as exc:
                    last_error = exc
                    continue
            if loaded is not None:
                break

        if loaded is None or split is None:
            raise ValueError(
                f"Could not load dataset {dataset_name} with any split "
                "(test/train/validation)"
                + (f": {last_error}" if last_error is not None else "")
            )
        dataset = loaded
    elif os.path.isdir(dataset_name):
        dataset = _load_local_parquet_split(dataset_name, dataset_config, split)
        if dataset is None:
            if dataset_config:
                dataset = load_dataset(
                    dataset_name, dataset_config, split=split, **load_kwargs
                )
            else:
                dataset = load_dataset(dataset_name, split=split, **load_kwargs)
    elif dataset_config:
        dataset = load_dataset(dataset_name, dataset_config, split=split, **load_kwargs)
    else:
        dataset = load_dataset(dataset_name, split=split, **load_kwargs)

    texts = []
    for example in dataset:
        if "text" in example:
            text = example["text"]
            if text and text.strip():
                texts.append(text)
        elif "messages" in example:
            messages = example["messages"]
            if isinstance(messages, list):
                text = "\n".join(
                    msg.get("content", "") for msg in messages if isinstance(msg, dict)
                )
                if text and text.strip():
                    texts.append(text)
        else:
            for key, value in example.items():
                if isinstance(value, str) and value.strip():
                    texts.append(value)
                    break

    if not texts:
        raise ValueError(f"No valid text found in dataset {dataset_name}")

    return texts


def tokenizer_vocab_size(tokenizer: Any) -> int:
    from vllm.v1.sample.kld import tokenizer_unpadded_vocab_size

    return tokenizer_unpadded_vocab_size(tokenizer)


def _window_filename(kind: str, idx: int) -> str:
    prefix = "hidden" if kind == "hidden" else "logits"
    return f"{prefix}_{idx}.safetensors"


def _runtime_lm_head_info(llm: LLM) -> dict[str, Any]:
    from vllm.v1.sample.kld import inspect_model_lm_heads

    per_worker = llm.apply_model(inspect_model_lm_heads)
    if not per_worker:
        raise RuntimeError("Runtime LM-head inspection returned no worker results")
    states = {info["state"] for info in per_worker}
    if len(states) != 1:
        raise RuntimeError(
            f"Runtime LM-head state differs across workers: {sorted(states)}"
        )
    return {
        "state": per_worker[0]["state"],
        "workers": per_worker,
    }


def calculate_kld(
    model_path: str,
    texts: list[str],
    context_length: int,
    stride: int,
    rows: int,
    score_from: int,
    reference_logits_path: str | None = None,
    reference_model_path: str | None = None,
    llm_kwargs: dict[str, Any] | None = None,
    num_samples: int | None = None,
    trust_remote_code: bool = False,
    capture_only: bool = False,
    storage: str = "logits",
    probe_replay: bool = False,
    run_gate: bool = True,
    decompose_head: bool = False,
) -> dict[str, Any]:
    """Two-phase KLD: capture teacher references, then score the student."""
    from vllm.v1.sample.kld import (
        KLDResult,
        capture_runtime_manifest,
        copy_lm_head_from_checkpoint,
        detect_lm_head_quantization,
        iter_eval_rows,
        manifest_mismatches,
        probe_replay_exactness_in_model,
        read_json,
        sha256_file,
        sha256_tokens,
        summarize_kld_rows,
        tokenizer_identity,
        tokenizer_unpadded_vocab_size,
        write_json,
    )

    if score_from < 0 or score_from >= context_length - 1:
        raise ValueError(
            f"score_from must be in [0, {context_length - 2}] for a full "
            f"{context_length}-token row, got {score_from}"
        )

    samples_to_process = texts[:num_samples] if num_samples else texts
    concatenated_text = "\n\n".join(samples_to_process)

    tokenizer_path = reference_model_path if reference_model_path else model_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=trust_remote_code
    )
    encoded = tokenizer(concatenated_text, add_special_tokens=False)
    tokens = encoded["input_ids"]
    if tokens and isinstance(tokens[0], list):
        tokens = tokens[0]
    windows = iter_eval_rows(tokens, context_length, stride, rows)
    if not windows:
        raise ValueError("Not enough tokens for any evaluation row")
    if any(len(window) - 1 <= score_from for window in windows):
        raise ValueError(
            "score_from leaves no scored positions in at least one row; "
            f"shortest row has {min(map(len, windows))} tokens"
        )
    unique_tokens = len(windows[0]) + sum(
        min(stride, len(window)) for window in windows[1:]
    )
    kld_vocab = tokenizer_vocab_size(tokenizer)
    print(
        f"Evaluation rows: {len(windows)}, unique tokens: {unique_tokens}, "
        f"kld_vocab_size: {kld_vocab}"
    )
    tok_id = tokenizer_identity(tokenizer)
    token_hash = sha256_tokens([t for w in windows for t in w])

    capture_kind = "logits"
    if storage in {"hidden", "auto"}:
        capture_kind = "hidden"
    if storage == "hidden" and not probe_replay:
        raise ValueError("--storage hidden requires --probe-replay")
    manifest: dict[str, Any] | None = None

    # Phase 1: Generate reference capture if a teacher model is provided.
    if reference_model_path is not None:
        model_name = os.path.basename(reference_model_path.rstrip("/\\"))
        ref_dir = reference_logits_path or os.path.join(
            os.getcwd(),
            f"ref_{model_name}_rows{len(windows)}_ctx{context_length}_s{stride}",
        )
        reference_logits_path = ref_dir
        os.makedirs(ref_dir, exist_ok=True)
        existing = sorted(
            glob.glob(os.path.join(ref_dir, "logits_*.safetensors"))
            + glob.glob(os.path.join(ref_dir, "hidden_*.safetensors"))
        )
        if not existing:
            if run_gate:
                _run_determinism_gate(
                    reference_model_path,
                    llm_kwargs or {},
                    windows[0],
                )
            print(f"Phase 1: capturing reference from {reference_model_path}")
            teacher_head_static = detect_lm_head_quantization(reference_model_path)
            print(f"  Teacher LM head (static): {teacher_head_static['state']}")
            ref_llm = LLM(model=reference_model_path, **(llm_kwargs or {}))
            teacher_head_runtime = _runtime_lm_head_info(ref_llm)
            print(f"  Teacher LM head (runtime): {teacher_head_runtime['state']}")
            if teacher_head_runtime["state"] != "unquantized":
                if storage == "hidden":
                    raise ValueError(
                        "Hidden-state capture requires an unquantized runtime "
                        "teacher LM head"
                    )
                capture_kind = "logits"
                print(
                    "  Runtime teacher head is not unquantized; "
                    "forcing --storage logits"
                )
            teacher_quant = teacher_head_runtime["state"] != "unquantized"
            want_hidden = not teacher_quant and (
                capture_kind == "hidden" or probe_replay or storage == "auto"
            )
            want_logits = (
                capture_kind == "logits" or probe_replay or storage == "auto"
            )
            for idx, window_tokens in enumerate(windows):
                sampling_params = SamplingParams(
                    max_tokens=1,
                    return_prompt_logits=want_logits,
                    return_prompt_hidden_states=want_hidden,
                )
                prompt: TokensPrompt = {"prompt_token_ids": window_tokens}
                out = ref_llm.generate([prompt], sampling_params=sampling_params)[0]
                if want_logits:
                    if out.prompt_logits is None:
                        raise RuntimeError(
                            "prompt_logits is None; return_prompt_logits plumbing "
                            "is broken in this build"
                        )
                    save_file(
                        {"logits": out.prompt_logits.cpu()},
                        os.path.join(ref_dir, _window_filename("logits", idx)),
                    )
                if want_hidden:
                    hidden = getattr(out, "prompt_hidden_states", None)
                    if hidden is None:
                        raise RuntimeError(
                            "prompt_hidden_states is None; "
                            "return_prompt_hidden_states plumbing is broken"
                        )
                    save_file(
                        {"hidden_states": hidden.cpu()},
                        os.path.join(ref_dir, _window_filename("hidden", idx)),
                    )
            head_path = os.path.join(ref_dir, "lm_head.safetensors")
            if want_hidden:
                copy_lm_head_from_checkpoint(reference_model_path, head_path)
            probe_report = None
            if probe_replay or storage == "auto":
                probe_report = _probe_first_window(
                    ref_llm,
                    ref_dir,
                    kld_vocab,
                    probe_replay_exactness_in_model,
                )
                print(f"  Replay probe: {probe_report}")
                if storage == "auto":
                    if probe_report.get("identical"):
                        capture_kind = "hidden"
                        print("  Storage: hidden (replay is bitwise exact)")
                    else:
                        capture_kind = "logits"
                        print(
                            "  Storage: logits (replay is NOT bitwise exact; "
                            "hidden-state scoring stays disabled)"
                        )
            capture_model_runner_v2 = bool(
                ref_llm.llm_engine.vllm_config.use_v2_model_runner
            )
            del ref_llm
            gc.collect()
            torch.accelerator.empty_cache()
            if capture_kind == "hidden":
                if not probe_report or not probe_report.get("identical"):
                    raise RuntimeError(
                        "Hidden-state storage requires a bitwise-exact replay "
                        "probe. Re-run with --probe-replay, or use "
                        "--storage logits."
                    )
            capture_files = [
                _window_filename(capture_kind, i) for i in range(len(windows))
            ]
            if capture_kind == "hidden":
                capture_files.append("lm_head.safetensors")
            reference_config = os.path.join(reference_model_path, "config.json")
            manifest = {
                "token_sha256": token_hash,
                "tokenizer": tok_id,
                "context_length": context_length,
                "stride": stride,
                "rows": len(windows),
                "score_from": score_from,
                "kld_vocab_size": kld_vocab,
                "tensor_parallel_size": (llm_kwargs or {}).get(
                    "tensor_parallel_size", 1
                ),
                "enforce_eager": bool((llm_kwargs or {}).get("enforce_eager")),
                "reference_model": os.path.abspath(reference_model_path),
                "reference_config_sha256": (
                    sha256_file(reference_config)
                    if os.path.isfile(reference_config)
                    else None
                ),
                "reference_engine_config": {
                    key: (llm_kwargs or {}).get(key)
                    for key in (
                        "dtype",
                        "enforce_eager",
                        "moe_backend",
                        "tensor_parallel_size",
                    )
                },
                "model_runner_v2": capture_model_runner_v2,
                "lm_head": {
                    "static": teacher_head_static,
                    "runtime": teacher_head_runtime,
                },
                "storage": capture_kind,
                "runtime": capture_runtime_manifest(),
                "file_hashes": {
                    name: sha256_file(os.path.join(ref_dir, name))
                    for name in capture_files
                },
                "replay_probe": probe_report,
            }
            write_json(os.path.join(ref_dir, "manifest.json"), manifest)
            print(f"Saved {len(windows)} reference windows to {ref_dir}/")
        else:
            man_path = os.path.join(ref_dir, "manifest.json")
            print(
                f"Phase 1 skipped: reusing {len(existing)} existing capture "
                f"files in {ref_dir} "
                f"(manifest.json {'present' if os.path.isfile(man_path) else 'MISSING'})"
            )
            if os.path.isfile(man_path):
                capture_kind = read_json(man_path).get("storage", capture_kind)

    if reference_logits_path is None:
        raise ValueError(
            "Either --reference-logits or --reference-model must be provided"
        )
    if not os.path.exists(reference_logits_path):
        raise FileNotFoundError(
            f"Reference logits path not found: {reference_logits_path}"
        )
    ref_is_directory = os.path.isdir(reference_logits_path)
    if ref_is_directory:
        man_path = os.path.join(reference_logits_path, "manifest.json")
        if not os.path.isfile(man_path):
            raise FileNotFoundError(
                "Reference directory has no manifest.json. Recapture with "
                "this script so tokenizer, token hash, and storage are "
                "recorded. A single legacy safetensors file is still allowed."
            )
        manifest = read_json(man_path)
        live = {
            "token_sha256": token_hash,
            "tokenizer": tok_id,
            "context_length": context_length,
            "stride": stride,
            "rows": len(windows),
            "score_from": score_from,
            "kld_vocab_size": kld_vocab,
            "tensor_parallel_size": (llm_kwargs or {}).get(
                "tensor_parallel_size", 1
            ),
            "enforce_eager": bool((llm_kwargs or {}).get("enforce_eager")),
            "runtime": capture_runtime_manifest(),
        }
        mismatches = manifest_mismatches(manifest, live)
        if mismatches:
            raise ValueError(
                "Capture manifest does not match this scoring run:\n  "
                + "\n  ".join(mismatches)
            )
        capture_kind = manifest.get("storage", capture_kind)
        if capture_kind == "hidden":
            probe = manifest.get("replay_probe") or {}
            if not probe.get("identical"):
                raise ValueError(
                    "Refusing hidden-state scoring: the capture replay probe "
                    "was not bitwise exact. Recapture with --storage logits."
                )
        expected = [
            os.path.join(
                reference_logits_path, _window_filename(capture_kind, i)
            )
            for i in range(len(windows))
        ]
        missing = [p for p in expected if not os.path.isfile(p)]
        if missing:
            raise FileNotFoundError(
                "Reference capture is incomplete; missing "
                f"{len(missing)} files, first: {missing[0]}"
            )
        from safetensors import safe_open

        ref_key = "hidden_states" if capture_kind == "hidden" else "logits"
        for path, window in zip(expected, windows):
            with safe_open(path, framework="pt", device="cpu") as f:
                if ref_key not in f.keys():
                    raise ValueError(
                        f"Reference file {path} has no {ref_key!r} tensor"
                    )
                shape = f.get_slice(ref_key).get_shape()
            expected_positions = len(window) - 1
            if len(shape) != 2 or shape[0] != expected_positions:
                raise ValueError(
                    f"Reference {path} has shape {shape}; expected "
                    f"[{expected_positions}, width]"
                )
        file_hashes = manifest.get("file_hashes")
        if not isinstance(file_hashes, dict):
            raise ValueError(
                "Capture manifest has no file_hashes; recapture references "
                "with this version before scoring"
            )
        hash_mismatches = [
            path
            for path in expected
            if file_hashes.get(os.path.basename(path)) != sha256_file(path)
        ]
        if capture_kind == "hidden":
            head_path = os.path.join(
                reference_logits_path, "lm_head.safetensors"
            )
            if not os.path.isfile(head_path):
                raise FileNotFoundError(
                    f"Hidden-state capture is missing {head_path}"
                )
            if file_hashes.get("lm_head.safetensors") != sha256_file(head_path):
                hash_mismatches.append(head_path)
        if hash_mismatches:
            raise ValueError(
                "Reference capture file hash mismatch; first: "
                f"{hash_mismatches[0]}"
            )
    else:
        if not os.path.exists(reference_logits_path):
            raise FileNotFoundError(
                f"Reference capture does not exist: {reference_logits_path}"
            )
        raise ValueError(
            "The KLD workflow requires a manifest-backed capture directory; "
            f"legacy single-file references are not accepted: "
            f"{reference_logits_path}"
        )

    if capture_kind == "hidden" and not (llm_kwargs or {}).get(
        "enforce_eager", False
    ):
        raise ValueError(
            "Hidden-state LM-head replay requires enforce_eager=True"
        )

    if capture_only:
        print(f"Capture-only: skipping Phase 2. References at {reference_logits_path}")
        return {"mean_kld": 0.0, "num_positions": 0, "capture_only": True}

    student_head_static = detect_lm_head_quantization(model_path)
    print(f"Student LM head (static): {student_head_static['state']}")
    print("Phase 2: Computing KLD...")
    print(f"Loading test model: {model_path}")
    llm = LLM(model=model_path, **(llm_kwargs or {}))
    student_uses_v2 = bool(
        llm.llm_engine.vllm_config.use_v2_model_runner
    )
    if manifest is not None:
        captured_uses_v2 = manifest.get("model_runner_v2")
        if captured_uses_v2 is None:
            raise ValueError(
                "Capture manifest does not record model_runner_v2; recapture "
                "with this version"
            )
        if bool(captured_uses_v2) != student_uses_v2:
            raise ValueError(
                "Capture model runner does not match scoring model runner: "
                f"captured V2={captured_uses_v2}, live V2={student_uses_v2}"
            )
    student_head_runtime = _runtime_lm_head_info(llm)
    print(f"Student LM head (runtime): {student_head_runtime['state']}")
    student_head = {
        "state": student_head_runtime["state"],
        "static": student_head_static,
        "runtime": student_head_runtime,
    }
    if capture_kind == "hidden" and manifest is not None:
        teacher_workers = (
            manifest.get("lm_head", {})
            .get("runtime", {})
            .get("workers", [])
        )
        student_workers = student_head_runtime.get("workers", [])
        if not teacher_workers or not student_workers:
            raise ValueError(
                "Hidden-state scoring requires recorded teacher and student "
                "logits-processor metadata"
            )
        teacher_semantics = teacher_workers[0].get("logits_processor")
        student_semantics = student_workers[0].get("logits_processor")
        if teacher_semantics != student_semantics:
            raise ValueError(
                "Teacher and student logits-processor semantics differ: "
                f"teacher={teacher_semantics!r}, student={student_semantics!r}"
            )
    if capture_kind == "hidden" and student_head["state"] == "quantized":
        print(
            "  Note: teacher-head replay measures trunk+student-head via "
            "in-engine student logits; a quantized student head is visible."
        )
    if (
        decompose_head
        and student_head["state"] == "quantized"
        and capture_kind != "hidden"
    ):
        raise ValueError(
            "--decompose-head with a quantized student head requires "
            "hidden-state captures (shared teacher head for trunk KLD)"
        )

    chunks: list[KLDResult] = []
    for idx, window_tokens in enumerate(windows):
        if ref_is_directory:
            ref_file = os.path.join(
                reference_logits_path, _window_filename(capture_kind, idx)
            )
            ref_key = "hidden_states" if capture_kind == "hidden" else "logits"
        else:
            ref_file = reference_logits_path
            ref_key = f"logits_{idx}"
        prompt: TokensPrompt = {
            "prompt_token_ids": window_tokens,
            "reference_logits_path": ref_file,
            "reference_logits_key": ref_key,
            "kld_vocab_size": kld_vocab,
        }
        sampling_params = SamplingParams(
            max_tokens=1,
            kld_mode=True,
        )
        out = llm.generate([prompt], sampling_params=sampling_params)[0]
        if out.kld_result is None:
            raise RuntimeError("kld_result is None; KLD plumbing is broken")
        chunks.append(out.kld_result)

    report = summarize_kld_rows(
        chunks, score_from=score_from, context_length=context_length
    )
    report["unique_tokens"] = unique_tokens
    report["num_rows"] = len(windows)
    report["kld_vocab_size"] = kld_vocab
    report["model_runner_v2"] = student_uses_v2
    report["student_lm_head"] = student_head
    report["student_model"] = os.path.abspath(model_path)
    if manifest is not None:
        report["teacher_lm_head"] = manifest.get("lm_head")
        report["capture_manifest_sha256"] = sha256_file(
            os.path.join(reference_logits_path, "manifest.json")
        )
    report["storage"] = capture_kind
    if decompose_head:
        extra = _decompose_head_kld(
            llm,
            windows,
            reference_logits_path,
            kld_vocab,
            score_from,
            context_length,
            capture_kind,
            ref_is_directory,
        )
        extra["deployed_mean_kld"] = report["mean_kld"]
        extra["head_delta_kld"] = (
            report["mean_kld"] - extra["trunk_mean_kld"]
        )
        report.update(extra)
    return report


def _decompose_head_kld(
    llm: LLM,
    windows: list[list[int]],
    reference_logits_path: str,
    kld_vocab: int,
    score_from: int,
    context_length: int,
    capture_kind: str,
    ref_is_directory: bool,
) -> dict[str, Any]:
    """Trunk KLD through the shared teacher head vs deployed student logits."""
    from vllm.v1.sample.kld import (
        compute_trunk_kld_in_model,
        summarize_kld_rows,
    )

    if capture_kind != "hidden":
        raise ValueError("--decompose-head requires hidden-state storage")
    if not ref_is_directory:
        raise ValueError("--decompose-head requires a reference directory")
    head_path = os.path.join(reference_logits_path, "lm_head.safetensors")
    if not os.path.isfile(head_path):
        raise FileNotFoundError(f"Missing teacher head at {head_path}")
    print("Phase 2b: trunk KLD (shared teacher head)...")
    trunk_chunks = []
    hidden_params = SamplingParams(
        prompt_logprobs=1,
        max_tokens=1,
        return_prompt_hidden_states=True,
    )
    with tempfile.TemporaryDirectory(prefix="vllm-kld-trunk-") as temp_dir:
        for idx, window_tokens in enumerate(windows):
            hidden_out = llm.generate(
                [{"prompt_token_ids": window_tokens}],
                sampling_params=hidden_params,
            )[0]
            student_h = hidden_out.prompt_hidden_states
            if student_h is None:
                raise RuntimeError("student prompt_hidden_states is None")
            student_path = os.path.join(temp_dir, f"student_{idx}.safetensors")
            save_file({"hidden_states": student_h.cpu()}, student_path)
            teacher_path = os.path.join(
                reference_logits_path, _window_filename("hidden", idx)
            )
            per_worker = llm.apply_model(
                partial(
                    compute_trunk_kld_in_model,
                    student_hidden_path=student_path,
                    teacher_hidden_path=teacher_path,
                    head_path=head_path,
                    kld_vocab_size=kld_vocab,
                )
            )
            if not per_worker:
                raise RuntimeError("Trunk KLD returned no worker results")
            if any(result != per_worker[0] for result in per_worker[1:]):
                raise RuntimeError("Trunk KLD differs across workers")
            trunk_chunks.append(per_worker[0])
    trunk_report = summarize_kld_rows(
        trunk_chunks,
        score_from=score_from,
        context_length=context_length,
    )
    return {
        "trunk_mean_kld": trunk_report["mean_kld"],
        "trunk_report": trunk_report,
    }


def _probe_first_window(
    llm: LLM,
    ref_dir: str,
    kld_vocab: int,
    probe_fn,
) -> dict[str, Any]:
    logits_path = os.path.join(ref_dir, "logits_0.safetensors")
    hidden_path = os.path.join(ref_dir, "hidden_0.safetensors")
    head_path = os.path.join(ref_dir, "lm_head.safetensors")
    if not (
        os.path.isfile(logits_path)
        and os.path.isfile(hidden_path)
        and os.path.isfile(head_path)
    ):
        return {"identical": False, "reason": "missing probe files"}
    per_worker = llm.apply_model(
        partial(
            probe_fn,
            hidden_path=hidden_path,
            logits_path=logits_path,
            head_path=head_path,
            kld_vocab_size=kld_vocab,
        )
    )
    reports = [report for report in per_worker if report is not None]
    if not reports:
        return {"identical": False, "reason": "no probe result"}
    if any(report != reports[0] for report in reports[1:]):
        return {
            "identical": False,
            "reason": "probe results differ across workers",
            "workers": reports,
        }
    return reports[0]


def _is_glm53_checkpoint(model_path: str) -> bool:
    config_path = os.path.join(model_path, "config.json")
    if os.path.isfile(config_path):
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        architectures = config.get("architectures") or []
        if "GlmMoeDsaForCausalLM" in architectures:
            return True
    lowered = model_path.lower()
    return "glm" in lowered and "5.3" in lowered


def _run_determinism_gate(
    model_path: str,
    llm_kwargs: dict[str, Any],
    token_ids: list[int],
) -> None:
    """Refuse capture unless the GLM-5.3 probe reports bitwise self-identity.

    Other models skip the gate when their checkpoint architecture and path do
    not identify GLM-5.3.
    """
    probe = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "scripts",
        "glm53_determinism_probe.py",
    )
    probe = os.path.abspath(probe)
    if not _is_glm53_checkpoint(model_path):
        print("Determinism gate: skipped (not a GLM-5.3 architecture)")
        return
    if not os.path.isfile(probe):
        raise FileNotFoundError(
            f"Determinism gate requires {probe} for GLM-5.3 captures"
        )
    print("Determinism gate: running glm53_determinism_probe.py --gate")
    import subprocess
    import sys

    env = os.environ.copy()
    if env.get("VLLM_SPARSE_MLA_SORT_TOPK") != "1":
        raise RuntimeError(
            "GLM-5.3 capture requires VLLM_SPARSE_MLA_SORT_TOPK=1; "
            "refusing to enable it implicitly"
        )
    if len(token_ids) < 130:
        raise ValueError(
            "GLM-5.3 determinism gate requires context_length >= 130"
        )
    with tempfile.TemporaryDirectory(prefix="vllm-glm53-gate-") as temp_dir:
        token_path = os.path.join(temp_dir, "tokens.json")
        with open(token_path, "w", encoding="utf-8") as f:
            json.dump(token_ids, f)
        lengths = [
            length
            for length in (129, 130, 192, 256, 512, len(token_ids))
            if length <= len(token_ids)
        ]
        cmd = [
            sys.executable,
            probe,
            "--model",
            model_path,
            "--token-ids-file",
            token_path,
            "--out-dir",
            temp_dir,
            "--ctx",
            str(len(token_ids)),
            "--stride",
            str(len(token_ids)),
            "--lengths",
            ",".join(str(length) for length in sorted(set(lengths))),
            "--gate",
            "--gate-max-self-kld",
            "0.0",
            "--tp",
            str(llm_kwargs.get("tensor_parallel_size", 1)),
            "--gpu-memory-utilization",
            str(llm_kwargs.get("gpu_memory_utilization", 0.9)),
        ]
        if llm_kwargs.get("moe_backend"):
            cmd.extend(["--moe-backend", str(llm_kwargs["moe_backend"])])
        if llm_kwargs.get("max_num_batched_tokens") is not None:
            cmd.extend(
                [
                    "--max-num-batched-tokens",
                    str(llm_kwargs["max_num_batched_tokens"]),
                ]
            )
        if not llm_kwargs.get("enforce_eager", False):
            cmd.append("--no-enforce-eager")
        result = subprocess.run(cmd, env=env, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            "Determinism gate failed; refusing to capture a noisy reference. "
            "Set VLLM_SPARSE_MLA_SORT_TOPK=1 and re-run."
        )


def _print_kld_report(report: dict[str, Any]) -> None:
    print("\nResults:")
    print(f"  Mean KLD (ref || student): {report['mean_kld']:.8f}")
    print(f"  Mean KLD (student || ref): {report['mean_kld_reverse']:.8f}")
    print("  Median / p90 / p99 / max:")
    print(
        f"    {report['median_kld']:.8f} / {report['p90_kld']:.8f} / "
        f"{report['p99_kld']:.8f} / {report['max_kld']:.8f}"
    )
    print(f"  Positions: {report['num_positions']}")
    print(
        f"  Rows: {report.get('num_rows')}  "
        f"unique tokens: {report.get('unique_tokens')}"
    )
    print(f"  kld_vocab_size: {report.get('kld_vocab_size')}")
    print(f"  Model runner: {'V2' if report.get('model_runner_v2') else 'V1'}")
    print(f"  Top-1 agreement: {report['top1_agreement']:.6f}")
    print("  Top-K agreement:")
    for k, v in sorted(report["topk_agreement"].items()):
        print(f"    K={k}: {v:.6f}")
    print("  By reference confidence:")
    for b in report["confidence_buckets"]:
        mean = "n/a" if b["mean_kld"] is None else f"{b['mean_kld']:.6f}"
        print(
            f"    [{b['lo']:.2f}, {b['hi']:.2f}): {100 * b['frac']:5.1f}% "
            f"mean {mean}"
        )
    print("  By context depth:")
    for b in report["depth_buckets"]:
        print(
            f"    depth [{b['depth_lo']}, {b['depth_hi']}]: n={b['n']} "
            f"mean {b['mean_kld']:.6f}"
        )
    head = report.get("student_lm_head") or {}
    print(f"  Student LM head: {head.get('state', 'unknown')}")
    print(f"  Storage: {report.get('storage')}")
    if report.get("trunk_mean_kld") is not None:
        print(
            "  Trunk mean KLD (shared teacher head): "
            f"{report['trunk_mean_kld']:.8f}"
        )
        print(
            "  Deployed mean KLD (student logits): "
            f"{report['deployed_mean_kld']:.8f}"
        )
        print(
            "  Head-associated KLD delta (deployed - trunk; not additive): "
            f"{report['head_delta_kld']:.8f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Calculate KLD using vLLM's score mode"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to test model"
    )
    parser.add_argument(
        "--reference-model",
        type=str,
        default=None,
        help="Path to reference model (generates ref logits if needed)",
    )
    parser.add_argument(
        "--reference-logits",
        type=str,
        default=None,
        help="Path to reference logits directory (per-window safetensors) "
        "or a single legacy safetensors file",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        help="Quantization method (e.g., 'awq', 'gptq')",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Hub dataset id or a local dataset directory",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default=None,
        help="Dataset configuration (e.g., 'wikitext-2-raw-v1')",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of dataset rows to concatenate (default: all). Unlike "
        "the old 99*stride cap, this actually changes coverage.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=100,
        help="Number of evaluation rows (default: 100, matching EXL3).",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=2048,
        help="Tokens per row (default: 2048)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="DEPRECATED. Overlapping stride for regenerating historical "
        "numbers. Default is context-length (non-overlapping).",
    )
    parser.add_argument(
        "--score-from",
        type=int,
        default=0,
        help="Skip this many leading positions per row (default: 0). "
        "Use context_length//2 for llama.cpp/Turbo deep-context parity.",
    )
    parser.add_argument(
        "--storage",
        type=str,
        choices=("logits", "hidden", "auto"),
        default="logits",
        help="Reference storage. 'logits' is always valid. 'hidden' stores "
        "teacher hidden states plus lm_head (small). 'auto' uses hidden "
        "only if the bitwise replay probe is exact, else logits.",
    )
    parser.add_argument(
        "--probe-replay",
        action="store_true",
        help="During capture, compare live teacher logits to hidden-state "
        "LM-head replay. Required for --storage auto/hidden to be trusted.",
    )
    parser.add_argument(
        "--decompose-head",
        action="store_true",
        help="Report shared-teacher-head trunk KLD and the non-additive "
        "deployed-minus-trunk delta. Requires hidden-state storage.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallelism size (default: 1)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.35,
        help="GPU memory utilization (default: 0.35). vLLM reserves this fraction "
        "of each GPU for model+KV cache. 0.7 on 95GB GPUs = 66GB/GPU, which "
        "is excessive for 8B models (~8GB). Use 0.35 or lower.",
    )
    parser.add_argument(
        "--kv-cache-memory-gib",
        type=float,
        default=None,
        help="Pin KV cache size per GPU in GiB, overriding the "
        "--gpu-memory-utilization sizing. Scoring needs only "
        "max_model_len * max_num_seqs of KV but a large per-position logits "
        "buffer, so letting vLLM claim every spare byte for KV cache is what "
        "causes OOM on wide-vocabulary models.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading model",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=128,
        help="Maximum number of sequences per batch (default: 128). Hybrid "
        "Mamba models require this to be <= available Mamba cache blocks. "
        "For KLD evaluation (one window at a time), 128 is more than enough.",
    )
    parser.add_argument(
        "--language-model-only",
        action="store_true",
        help="Disable multimodal modules for text-only models (e.g., Qwen-3.5) "
        "to save GPU memory",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help="Cap the dummy profile / max batch tokens. On B200 the default "
        "is 16384; for one-window KLD capture use 4096 (or context-length).",
    )
    parser.add_argument(
        "--capture-only",
        action="store_true",
        help="Run Phase 1 only: write reference logits and exit. Does not "
        "load --model for KLD. Requires --reference-model.",
    )
    parser.add_argument(
        "--report-json",
        type=str,
        default=None,
        help="Optional path for the complete score report and head provenance.",
    )
    parser.add_argument(
        "--compiled",
        action="store_true",
        help="Use torch.compile with best-effort determinism settings. "
        "Faster but NOT bit-reproducible run-to-run on the current stack; "
        "for speed experiments only. Eager mode is the default.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    if args.reference_model is None and args.reference_logits is None:
        parser.error("Either --reference-model or --reference-logits is required")
    if args.capture_only and args.reference_model is None:
        parser.error("--capture-only requires --reference-model")
    if args.storage == "hidden" and not args.probe_replay:
        parser.error("--storage hidden requires --probe-replay")
    if args.compiled and args.storage != "logits":
        parser.error(
            "Hidden-state LM-head replay requires eager mode; "
            "use --storage logits with --compiled"
        )

    allow_apply_model_rpc()

    print(f"Loading dataset: {args.dataset}")
    texts = load_dataset_texts(args.dataset, args.dataset_config)
    print(f"Loaded {len(texts)} text samples")

    llm_kwargs: dict[str, Any] = {
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": args.trust_remote_code,
        "enable_prefix_caching": False,
        "max_model_len": args.context_length * 2,
        "max_num_seqs": args.max_num_seqs,
    }
    if args.kv_cache_memory_gib is not None:
        llm_kwargs["kv_cache_memory_bytes"] = int(
            args.kv_cache_memory_gib * 1024**3
        )
    if args.quantization:
        llm_kwargs["quantization"] = args.quantization
    if args.language_model_only:
        llm_kwargs["language_model_only"] = True
    if args.max_num_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    if args.compiled:
        apply_compiled_llm_kwargs(llm_kwargs)
        print(
            "WARNING: --compiled mode is NOT bit-reproducible run-to-run. "
            "Use default eager mode for authoritative KLD scoring."
        )
    else:
        apply_eager_llm_kwargs(llm_kwargs)
        print("Deterministic (eager) mode: bit-reproducible scoring")

    moe_backend = os.environ.get("VLLM_MOE_BACKEND")
    if moe_backend:
        llm_kwargs["moe_backend"] = moe_backend
        print(f"MoE backend override (VLLM_MOE_BACKEND): {moe_backend}")

    if args.stride is None:
        stride = args.context_length
    else:
        stride = args.stride
        print(
            "WARNING: --stride is deprecated overlapping windowing. "
            "Omit it for non-overlapping rows (EXL3/Turbo)."
        )
    print("\nCalculating KLD...")
    print(f"  Context length: {args.context_length}")
    print(f"  Stride: {stride}")
    print(f"  Rows: {args.rows}")
    print(f"  Score-from: {args.score_from}")
    print(f"  Storage: {args.storage}")
    print(f"  Samples: {args.num_samples or len(texts)}")

    start_time = time.time()
    report = calculate_kld(
        args.model,
        texts,
        args.context_length,
        stride,
        args.rows,
        args.score_from,
        reference_logits_path=args.reference_logits,
        reference_model_path=args.reference_model,
        llm_kwargs=llm_kwargs,
        num_samples=args.num_samples,
        trust_remote_code=args.trust_remote_code,
        capture_only=args.capture_only,
        storage=args.storage,
        probe_replay=args.probe_replay,
        run_gate=True,
        decompose_head=args.decompose_head,
    )
    elapsed_time = time.time() - start_time

    if args.capture_only:
        print("\nCapture complete (Phase 1 only).")
        print(f"  Time elapsed: {elapsed_time:.2f} seconds")
        return

    if args.report_json:
        report_path = os.path.abspath(args.report_json)
        os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"Report JSON: {report_path}")

    _print_kld_report(report)
    print(f"  Time elapsed: {elapsed_time:.2f} seconds")
    npos = report["num_positions"]
    if elapsed_time > 0:
        print(f"  Positions/second: {npos / elapsed_time:.2f}")


if __name__ == "__main__":
    main()
