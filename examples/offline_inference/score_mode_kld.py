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
import contextlib
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
# (the default) avoids graph-level timing choices, but custom GPU kernels may
# still be numerically nondeterministic. Paired routing measures that floor.
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
    """Avoid graph-level autotuning and compilation choices (default)."""
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


@contextlib.contextmanager
def _phase(timings: dict[str, float], name: str):
    """Accumulate wall time per run phase.

    Total run time is dominated by loading weights twice — once for the teacher
    capture, once for the student — which is a fixed cost that does not grow
    with row count. Reporting it separately keeps the per-position rate from
    being read as scoring throughput.
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        timings[name] = timings.get(name, 0.0) + time.perf_counter() - start


def load_token_suite(
    suite_dir: str, partition: str, tokenizer: Any, limit: int | None = None
) -> tuple[list[list[int]], dict[str, Any]]:
    """Load a frozen token suite as evaluation rows.

    The stored IDs are the evaluation input (Law 3); nothing is retokenized here.
    Every context's recorded hash is re-derived from the file it was read from, so
    a corrupted or edited suite fails before a model loads.

    Raises:
        ValueError: if the suite is inconsistent, or was minted for a tokenizer
            whose vocabulary differs from this model's.
    """
    from vllm.v1.sample.kld import sha256_tokens

    manifest_path = os.path.join(suite_dir, "suite-manifest.json")
    if not os.path.isfile(manifest_path):
        raise ValueError(f"no suite-manifest.json in {suite_dir}")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    suite_vocab = (manifest.get("tokenizer") or {}).get("unpadded_vocab_size")
    live_vocab = tokenizer_vocab_size(tokenizer)
    if suite_vocab and int(suite_vocab) != int(live_vocab):
        raise ValueError(
            f"suite {manifest.get('suite_id')} was minted for a tokenizer with "
            f"{suite_vocab} tokens but this model's tokenizer has {live_vocab}. "
            f"Token IDs are not portable across tokenizers; mint a suite for "
            f"this tokenizer family."
        )

    selected = [entry["context_id"] for entry in manifest["contexts"]]
    if partition != "all":
        partitions_path = os.path.join(suite_dir, "partitions.json")
        if not os.path.isfile(partitions_path):
            raise ValueError(f"no partitions.json in {suite_dir}")
        with open(partitions_path, encoding="utf-8") as f:
            partitions = json.load(f)
        if partition not in partitions:
            raise ValueError(
                f"partition {partition!r} is not in {partitions_path}"
            )
        selected = sorted(partitions[partition])

    # A bounded prefix is for the zero baseline, where the only acceptable answer
    # is exact zero and one context proves it. A limited run cannot match the
    # suite's published partition hash, so Law 3 will reject it as a candidate
    # measurement; that is intended.
    if limit is not None and limit > 0:
        selected = selected[:limit]

    by_id = {entry["context_id"]: entry for entry in manifest["contexts"]}
    windows: list[list[int]] = []
    for context_id in selected:
        entry = by_id.get(context_id)
        if entry is None:
            raise ValueError(f"context {context_id} is not in the suite manifest")
        path = os.path.join(suite_dir, entry["file"].replace("/", os.sep))
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        tokens = [int(t) for t in payload["tokens"]]
        if sha256_tokens(tokens) != entry["token_sha256"]:
            raise ValueError(f"{entry['file']}: token hash does not match the suite")
        windows.append(tokens)

    lengths = {len(w) for w in windows}
    if len(lengths) != 1:
        raise ValueError(f"suite rows have differing lengths: {sorted(lengths)}")
    expected = manifest.get("partition_token_sha256", {}).get(partition)
    actual = sha256_tokens([t for w in windows for t in w])
    if limit:
        expected = None
    if expected and expected != actual:
        raise ValueError(
            f"{partition} partition hashes to {actual} but the suite records "
            f"{expected}"
        )

    identity = {
        "suite_id": manifest.get("suite_id"),
        "recipe_id": manifest.get("recipe_id"),
        "partition": partition,
        "limit": limit or None,
        "context_length": lengths.pop(),
        "contexts": len(windows),
        "context_ids": selected,
        "suite_token_sha256": manifest.get("token_sha256"),
        "partition_token_sha256": actual,
        "tokenizer": manifest.get("tokenizer"),
    }
    print(
        f"Token suite {identity['suite_id']} [{partition}]: "
        f"{identity['contexts']} contexts x {identity['context_length']} tokens"
    )
    return windows, identity


def _dump_positions(chunks: Any, score_from: int, path: str) -> None:
    """Write the per-position arrays the summary reduces away.

    A mean over two million positions hides whether it describes all of them or
    a few hundred outliers, and those are different findings: broad precision
    loss versus a small number of positions where something structural changed.
    The arrays are what a tail analysis needs and are far too small to justify
    recomputing a scoring run to get them.
    """
    from vllm.v1.sample.kld import slice_kld_result

    kld: list[float] = []
    reverse: list[float] = []
    confidence: list[float] = []
    agree: list[int] = []
    row_index: list[int] = []
    depth: list[int] = []
    for index, chunk in enumerate(chunks):
        sliced = slice_kld_result(chunk, score_from)
        kld += list(sliced.kld_ref_to_model)
        reverse += list(sliced.kld_model_to_ref)
        confidence += list(sliced.ref_top1_prob)
        agree += [
            int(a == b) for a, b in zip(sliced.model_top1, sliced.ref_top1)
        ]
        row_index += [index] * len(sliced.kld_ref_to_model)
        depth += [
            score_from + offset for offset in range(len(sliced.kld_ref_to_model))
        ]
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    save_file(
        {
            "kld": torch.tensor(kld, dtype=torch.float32),
            "kld_reverse": torch.tensor(reverse, dtype=torch.float32),
            "ref_top1_prob": torch.tensor(confidence, dtype=torch.float32),
            "top1_agree": torch.tensor(agree, dtype=torch.uint8),
            "row": torch.tensor(row_index, dtype=torch.int32),
            "depth": torch.tensor(depth, dtype=torch.int32),
        },
        path,
    )
    print(f"Wrote {len(kld)} per-position values to {path}")


ROUTING_MANIFEST = "routing-manifest.json"
ROUTING_TRACE_PROTOCOL_VERSION = 2
PAIRED_ROUTED_SCORE_PROTOCOL_VERSION = 3
CONTROL_POSITION_BASE_TOLERANCE = 1e-5
CONTROL_MEAN_BASE_TOLERANCE = 1e-7
CONTROL_REPEATABILITY_MULTIPLIER = 2.0


def _routing_filename(idx: int) -> str:
    return f"routing_{idx:05d}.safetensors"


def _validate_routing_trace(
    routing_dir: str,
    manifest: dict[str, Any],
    token_hash: str,
    rows: int,
) -> None:
    """Refuse incomplete, stale, or internally inconsistent routing traces."""
    import numpy as np
    from safetensors.numpy import load_file as load_numpy
    from vllm.v1.sample.kld import sha256_file

    if manifest.get("protocol_version") != ROUTING_TRACE_PROTOCOL_VERSION:
        raise ValueError("routing trace predates the BxQ binding protocol")
    for field in ("reference_weights_sha256", "capture_manifest_sha256"):
        digest = manifest.get(field)
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"routing trace has no complete {field}")
    if manifest.get("token_sha256") != token_hash or manifest.get("rows") != rows:
        raise ValueError("routing trace is bound to different tokens or row count")
    num_layers = int(manifest["num_layers"])
    topk = int(manifest["num_experts_per_tok"])
    num_experts = int(manifest["num_experts"])
    layer_map = manifest.get("layer_map")
    if (
        not isinstance(layer_map, list)
        or not layer_map
        or layer_map != sorted(set(layer_map))
        or min(layer_map) < 0
        or max(layer_map) >= num_layers
    ):
        raise ValueError("routing trace layer map is incomplete or invalid")
    shapes = manifest.get("tensor_shapes")
    dtypes = manifest.get("tensor_dtypes")
    hashes = manifest.get("file_hashes")
    if (
        not isinstance(hashes, dict)
        or len(hashes) != rows
        or not isinstance(shapes, dict)
        or len(shapes) != rows
        or not isinstance(dtypes, dict)
        or len(dtypes) != rows
    ):
        raise ValueError("routing trace has incomplete tensor metadata or file hashes")
    for idx in range(rows):
        name = _routing_filename(idx)
        path = os.path.join(routing_dir, name)
        if hashes.get(name) != sha256_file(path):
            raise ValueError(f"routing trace hash mismatch for {name}")
        routed = load_numpy(path)["routed_experts"]
        if str(routed.dtype) != dtypes.get(name):
            raise ValueError(f"routing trace {name} dtype does not match manifest")
        if list(routed.shape) != shapes.get(name):
            raise ValueError(
                f"routing trace {name} shape {list(routed.shape)} does not "
                f"match {shapes.get(name)}"
            )
        selected = routed[:, layer_map, :]
        if selected.min() < 0 or selected.max() >= num_experts:
            raise ValueError(f"routing trace {name} contains out-of-range expert IDs")
        ordered = np.sort(selected, axis=-1)
        if topk > 1 and np.any(ordered[..., 1:] == ordered[..., :-1]):
            raise ValueError(
                f"routing trace {name} repeats an expert within a top-k row"
            )


def _capture_reference_routing(
    reference_model_path: str,
    windows: list[list[int]],
    routing_dir: str,
    llm_kwargs: dict[str, Any],
    token_hash: str,
    reference_weights_sha256: str,
    capture_manifest_sha256: str,
) -> dict[str, Any]:
    """Record which experts the reference selected, per token and per layer.

    This is a separate pass over the reference rather than part of the
    hidden-state capture. Both artifacts are bound to the same reference weight
    digest; a current Phase 1 capture can therefore be reused safely.

    Returns:
        The routing manifest, whether it was just written or already present.
    """
    from safetensors.numpy import save_file as save_numpy
    from vllm.v1.sample.kld import (
        capture_runtime_manifest,
        inspect_model_moe_backends,
        read_json,
        sha256_file,
        write_json,
    )

    os.makedirs(routing_dir, exist_ok=True)
    manifest_path = os.path.join(routing_dir, ROUTING_MANIFEST)
    reference_config = os.path.join(reference_model_path, "config.json")
    if os.path.isfile(manifest_path):
        manifest = read_json(manifest_path)
        if manifest.get("protocol_version") == ROUTING_TRACE_PROTOCOL_VERSION:
            if (
                manifest.get("reference_weights_sha256")
                != reference_weights_sha256
                or manifest.get("capture_manifest_sha256")
                != capture_manifest_sha256
            ):
                print(
                    "Routing capture names different teacher weights or a "
                    "different reference capture and will be replaced."
                )
                os.remove(manifest_path)
                return _capture_reference_routing(
                    reference_model_path,
                    windows,
                    routing_dir,
                    llm_kwargs,
                    token_hash,
                    reference_weights_sha256,
                    capture_manifest_sha256,
                )
            if manifest.get("reference_config_sha256") != sha256_file(
                reference_config
            ):
                print(
                    "Routing capture names a different reference config and "
                    "will be replaced."
                )
                os.remove(manifest_path)
                return _capture_reference_routing(
                    reference_model_path,
                    windows,
                    routing_dir,
                    llm_kwargs,
                    token_hash,
                    reference_weights_sha256,
                    capture_manifest_sha256,
                )
            try:
                _validate_routing_trace(
                    routing_dir, manifest, token_hash, len(windows)
                )
            except (OSError, ValueError) as exc:
                print(f"Routing capture is invalid and will be replaced: {exc}")
                os.remove(manifest_path)
                return _capture_reference_routing(
                    reference_model_path,
                    windows,
                    routing_dir,
                    llm_kwargs,
                    token_hash,
                    reference_weights_sha256,
                    capture_manifest_sha256,
                )
            print(
                f"Routing capture reused: {len(windows)} windows in "
                f"{routing_dir}"
            )
            return manifest
        print(f"Routing capture is historical and will be replaced: {manifest_path}")

    print(f"Routing capture: reading reference selections from {routing_dir}")
    llm = LLM(
        model=reference_model_path,
        enable_return_routed_experts=True,
        **llm_kwargs,
    )
    shape: tuple[int, int] | None = None
    worker_backends = llm.apply_model(inspect_model_moe_backends)
    worker_layer_maps = [
        sorted(
            {
                int(layer["layer_id"])
                for layer in worker.get("layers", [])
                if isinstance(layer.get("layer_id"), int)
            }
        )
        for worker in worker_backends
    ]
    if not worker_layer_maps or not worker_layer_maps[0]:
        raise ValueError("routing capture found no MoE layers in the reference")
    if any(layer_map != worker_layer_maps[0] for layer_map in worker_layer_maps[1:]):
        raise ValueError("reference MoE layer IDs differ across workers")
    layer_map = worker_layer_maps[0]
    tensor_shapes: dict[str, list[int]] = {}
    tensor_dtypes: dict[str, str] = {}
    try:
        for idx, window_tokens in enumerate(windows):
            prompt: TokensPrompt = {"prompt_token_ids": window_tokens}
            out = llm.generate(
                [prompt], sampling_params=SamplingParams(max_tokens=1)
            )[0]
            routing = out.outputs[0].routed_experts
            if routing is None:
                raise RuntimeError(
                    "routed_experts is None; this build returned no routing "
                    "for a prompt, so the routing term cannot be measured"
                )
            if routing.shape[0] != len(window_tokens):
                raise RuntimeError(
                    f"routing has {routing.shape[0]} rows for a "
                    f"{len(window_tokens)}-token window"
                )
            shape = (int(routing.shape[1]), int(routing.shape[2]))
            name = _routing_filename(idx)
            tensor_shapes[name] = list(routing.shape)
            tensor_dtypes[name] = str(routing.dtype)
            save_numpy(
                {"routed_experts": routing},
                os.path.join(routing_dir, name),
            )
    finally:
        del llm
        gc.collect()
        torch.accelerator.empty_cache()

    assert shape is not None
    names = [_routing_filename(i) for i in range(len(windows))]
    routing_info = _routing_info(reference_model_path)
    if routing_info is None:
        raise ValueError("reference routing was captured but config declares no experts")
    manifest = {
        "protocol_version": ROUTING_TRACE_PROTOCOL_VERSION,
        "token_sha256": token_hash,
        "rows": len(windows),
        "num_layers": shape[0],
        "layer_map": layer_map,
        "num_experts": routing_info["num_experts"],
        "num_experts_per_tok": shape[1],
        "routing_method": "ordered_logical_topk_before_eplb",
        "tensor_key": "routed_experts",
        "tensor_dtypes": tensor_dtypes,
        "tensor_shapes": tensor_shapes,
        "reference_model": os.path.abspath(reference_model_path),
        "reference_config_sha256": sha256_file(reference_config),
        "reference_weights_sha256": reference_weights_sha256,
        "capture_manifest_sha256": capture_manifest_sha256,
        "runtime": capture_runtime_manifest(),
        "file_hashes": {
            name: sha256_file(os.path.join(routing_dir, name)) for name in names
        },
    }
    write_json(manifest_path, manifest)
    _validate_routing_trace(routing_dir, manifest, token_hash, len(windows))
    print(
        f"Saved routing for {len(windows)} windows "
        f"({shape[0]} layers x {shape[1]} experts per token)"
    )
    return manifest


class _RoutingDivergence:
    """Accumulate how often the candidate chose different experts, and its cost.

    The question this answers is the one a router-weight cell cannot: the
    candidate's routers are bit-identical to the reference's, yet the
    activations reaching them have already been perturbed, so selections change.
    Nothing here perturbs anything; it compares two runs over the same frozen
    tokens.
    """

    BUCKETS = ((0, 0), (1, 1), (2, 3), (4, 7), (8, 1 << 30))

    def __init__(self, num_layers: int, num_experts_per_tok: int):
        import numpy as np

        self.np = np
        self.num_layers = num_layers
        self.topk = num_experts_per_tok
        self.positions = 0
        self.pairs = 0
        self.flipped_pairs = 0
        self.top1_changed = 0
        self.slot_disagreements = 0
        self.per_layer_flipped = np.zeros(num_layers, dtype=np.int64)
        self.bucket_positions = [0] * len(self.BUCKETS)
        self.bucket_kld = [0.0] * len(self.BUCKETS)
        self.kld_total = 0.0

    def update(self, reference, candidate, kld: list[float], score_from: int):
        """Fold in one window, aligning selections with the positions scored.

        Row ``p`` of the routing arrays is the routing used while predicting
        token ``p + 1``, which is the position KLD index ``p - score_from``
        scores, so the two are aligned by construction.
        """
        np = self.np
        end = score_from + len(kld)
        ref = reference[score_from:end]
        cand = candidate[score_from:end]
        if ref.shape != cand.shape:
            raise ValueError(
                f"routing shapes differ: reference {ref.shape}, candidate "
                f"{cand.shape}"
            )
        values = np.asarray(kld, dtype=np.float64)

        # Selection is a set, not a ranking: two runs that pick the same experts
        # in a different order compute the same function.
        ref_sorted = np.sort(ref.astype(np.int32), axis=-1)
        cand_sorted = np.sort(cand.astype(np.int32), axis=-1)
        same_slots = ref_sorted == cand_sorted
        layer_flipped = ~same_slots.all(axis=-1)

        self.positions += ref.shape[0]
        self.pairs += ref.shape[0] * self.num_layers
        self.flipped_pairs += int(layer_flipped.sum())
        self.slot_disagreements += int((~same_slots).sum())
        self.per_layer_flipped += layer_flipped.sum(axis=0).astype(np.int64)
        # Rank 0 is the highest-weighted expert. This also moves when the same
        # experts are chosen in a different order, which is not a selection
        # change but does change the weight each expert's output carries.
        self.top1_changed += int((ref[:, :, 0] != cand[:, :, 0]).sum())

        flipped_layers = layer_flipped.sum(axis=-1)
        self.kld_total += float(values.sum())
        for index, (low, high) in enumerate(self.BUCKETS):
            mask = (flipped_layers >= low) & (flipped_layers <= high)
            self.bucket_positions[index] += int(mask.sum())
            self.bucket_kld[index] += float(values[mask].sum())

    def finalize(self) -> dict[str, Any]:
        if not self.positions:
            return {}
        held = self.bucket_positions[0]
        flipped = self.positions - held
        mean_held = self.bucket_kld[0] / held if held else None
        mean_flipped = (
            (self.kld_total - self.bucket_kld[0]) / flipped if flipped else None
        )
        # What the mean would lose if flipped positions diverged no more than
        # positions whose routing survived. This is the routing term.
        excess = (
            (flipped / self.positions) * (mean_flipped - mean_held)
            if mean_held is not None and mean_flipped is not None
            else None
        )
        return {
            "num_layers": self.num_layers,
            "num_experts_per_tok": self.topk,
            "positions": self.positions,
            "layer_selections": self.pairs,
            "selection_flip_rate": self.flipped_pairs / self.pairs,
            "slot_disagreement_rate": self.slot_disagreements
            / (self.pairs * self.topk),
            "top1_expert_change_rate": self.top1_changed / self.pairs,
            "positions_with_any_flip": flipped,
            "position_flip_rate": flipped / self.positions,
            "mean_kld_routing_held": mean_held,
            "mean_kld_routing_flipped": mean_flipped,
            "flipped_share_of_total": (
                (self.kld_total - self.bucket_kld[0]) / self.kld_total
                if self.kld_total
                else None
            ),
            "routing_excess_mean": excess,
            "per_layer_flip_rate": [
                float(count) / self.positions
                for count in self.per_layer_flipped.tolist()
            ],
            "buckets": [
                {
                    "flipped_layers_min": low,
                    "flipped_layers_max": None if high > self.num_layers else high,
                    "positions": self.bucket_positions[index],
                    "mean_kld": (
                        self.bucket_kld[index] / self.bucket_positions[index]
                        if self.bucket_positions[index]
                        else None
                    ),
                }
                for index, (low, high) in enumerate(self.BUCKETS)
            ],
        }


def _compare_window_routing(
    divergence: "_RoutingDivergence",
    routing_dir: str,
    idx: int,
    out: Any,
    kld_result: Any,
    score_from: int,
    layer_map: list[int] | None = None,
) -> None:
    """Fold one window's candidate selections against the captured reference."""
    from safetensors.numpy import load_file as load_numpy

    candidate = out.outputs[0].routed_experts if out.outputs else None
    if candidate is None:
        raise RuntimeError(
            "the candidate returned no routed_experts; routing divergence "
            "cannot be measured without them"
        )
    reference = load_numpy(os.path.join(routing_dir, _routing_filename(idx)))[
        "routed_experts"
    ]
    if layer_map is not None:
        reference = reference[:, layer_map, :]
        candidate = candidate[:, layer_map, :]
    divergence.update(
        reference, candidate, kld_result.kld_ref_to_model[score_from:], score_from
    )


def _assert_forced_window_routing(routing_dir: str, idx: int, out: Any) -> None:
    """Prove the BxQ pass dispatched the captured ordered logical IDs."""
    import numpy as np
    from safetensors.numpy import load_file as load_numpy

    actual = out.outputs[0].routed_experts if out.outputs else None
    if actual is None:
        raise RuntimeError("BxQ pass returned no routed experts to verify")
    expected = load_numpy(os.path.join(routing_dir, _routing_filename(idx)))[
        "routed_experts"
    ]
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        raise RuntimeError(
            f"BxQ routing replay mismatch in window {idx}: expected "
            f"{expected.shape}, got {actual.shape}"
        )


def _per_context_summary(
    chunks: Any, score_from: int, context_ids: list[int] | None
) -> list[dict[str, Any]]:
    """Reduce each row to a publishable record, keyed to its suite context.

    The suite is stratified by domain, so a single mean answers "how far apart
    are these two models" but not "on what kind of text". One record per context
    is what Law 15 joins against the suite manifest, and 768 records cost less
    than a kilobyte each while the per-position arrays cost tens of megabytes.
    """
    from vllm.v1.sample.kld import slice_kld_result

    records: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks):
        sliced = slice_kld_result(chunk, score_from)
        vals = sorted(sliced.kld_ref_to_model)
        count = len(vals)
        if not count:
            continue
        agree = sum(
            1 for a, b in zip(sliced.model_top1, sliced.ref_top1) if a == b
        )
        records.append(
            {
                "row": index,
                "context_id": (
                    context_ids[index]
                    if context_ids and index < len(context_ids)
                    else None
                ),
                "positions": count,
                "mean_kld": float(sum(vals) / count),
                "median_kld": float(vals[count // 2]),
                "p99_kld": float(vals[min(count - 1, int(0.99 * count))]),
                "max_kld": float(vals[-1]),
                "mean_ref_top1_prob": float(
                    sum(sliced.ref_top1_prob) / count
                ),
                "top1_agreement": agree / count,
            }
        )
    return records


def _routing_info(model_path: str) -> dict[str, Any] | None:
    """Expert counts for a routed checkpoint, or None if it is dense.

    Recorded at capture time because Law 14 applies to routed models only, and
    the decision of whether it applies must come from the checkpoint rather than
    from whoever configured the campaign.
    """
    config_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_path):
        return None
    try:
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    count_keys = (
        "num_experts",
        "n_routed_experts",
        "num_local_experts",
        "moe_num_experts",
    )
    topk_keys = (
        "num_experts_per_tok",
        "moe_topk",
        "num_experts_per_token",
        # Gemma 4's name for it. vLLM already reads this; recording None here
        # would report a declared top-k of unknown for a model that declares 8.
        "top_k_experts",
        "moe_top_k",
    )
    for section in (config, config.get("text_config") or {}):
        experts = next(
            (section[key] for key in count_keys if isinstance(section.get(key), int)),
            None,
        )
        if not experts:
            continue
        topk = next(
            (section[key] for key in topk_keys if isinstance(section.get(key), int)),
            None,
        )
        return {"num_experts": experts, "num_experts_per_tok": topk}
    return None


def _declared_vocab_size(model_path: str) -> int | None:
    """Read the checkpoint's declared output width, padding included.

    Recorded so an artifact can show how many alignment-padding rows were
    excluded from scoring rather than asserting it.
    """
    config_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_path):
        return None
    try:
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    for section in (config, config.get("text_config") or {}):
        size = section.get("vocab_size")
        if isinstance(size, int):
            return size
    return None


def _prune_fallback_logits(ref_dir: str, num_windows: int) -> None:
    """Drop per-row teacher logits that `--storage auto` kept as a fallback.

    Hidden-state scoring reads only `hidden_*` and `lm_head`. Row 0's logits
    stay so the capture can be re-probed. At roughly 2 GiB per row on a
    wide-vocabulary model these files otherwise dominate the capture size.
    """
    freed = 0
    for idx in range(1, num_windows):
        path = os.path.join(ref_dir, _window_filename("logits", idx))
        if os.path.isfile(path):
            freed += os.path.getsize(path)
            os.remove(path)
    if freed:
        print(f"  Pruned {freed / 1024**3:.1f} GiB of fallback teacher logits")


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
    token_suite: str | None = None,
    suite_partition: str = "all",
    suite_limit: int | None = None,
    dump_positions: str | None = None,
    measure_routing: bool = False,
    routing_dir: str | None = None,
    paired_routing: bool = False,
    reference_weights_sha256: str | None = None,
) -> dict[str, Any]:
    """Two-phase KLD: capture teacher references, then score the student."""
    from vllm.v1.sample.kld import (
        KLDResult,
        capture_runtime_manifest,
        copy_lm_head_from_checkpoint,
        detect_lm_head_quantization,
        iter_eval_rows,
        logits_processor_identity,
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

    tokenizer_path = reference_model_path if reference_model_path else model_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=trust_remote_code
    )
    suite_identity: dict[str, Any] | None = None
    if token_suite:
        windows, suite_identity = load_token_suite(
            token_suite, suite_partition, tokenizer, suite_limit
        )
        context_length = suite_identity["context_length"]
        stride = context_length
        rows = len(windows)
    else:
        samples_to_process = texts[:num_samples] if num_samples else texts
        concatenated_text = "\n\n".join(samples_to_process)
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

    timings: dict[str, float] = {}
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
        existing_manifest_path = os.path.join(ref_dir, "manifest.json")
        if existing and reference_weights_sha256 is not None:
            existing_manifest = (
                read_json(existing_manifest_path)
                if os.path.isfile(existing_manifest_path)
                else {}
            )
            if (
                existing_manifest.get("reference_weights_sha256")
                != reference_weights_sha256
            ):
                print(
                    "Reference capture predates content binding or names "
                    "different teacher weights; recapturing it."
                )
                for path in [
                    *existing,
                    os.path.join(ref_dir, "lm_head.safetensors"),
                    existing_manifest_path,
                ]:
                    if os.path.isfile(path):
                        os.remove(path)
                existing = []
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
            with _phase(timings, "teacher_load"):
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
                # Only row 0 feeds the replay probe, so an explicit hidden
                # capture does not need a full-vocabulary logits file per row.
                # `auto` keeps them all until the probe decides, then prunes.
                write_logits = want_logits and (storage != "hidden" or idx == 0)
                sampling_params = SamplingParams(
                    max_tokens=1,
                    return_prompt_logits=write_logits,
                    return_prompt_hidden_states=want_hidden,
                )
                prompt: TokensPrompt = {"prompt_token_ids": window_tokens}
                with _phase(timings, "capture_forward"):
                    out = ref_llm.generate(
                        [prompt], sampling_params=sampling_params
                    )[0]
                if write_logits:
                    if out.prompt_logits is None:
                        raise RuntimeError(
                            "prompt_logits is None; return_prompt_logits plumbing "
                            "is broken in this build"
                        )
                    with _phase(timings, "capture_write"):
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
                    with _phase(timings, "capture_write"):
                        save_file(
                            {"hidden_states": hidden.cpu()},
                            os.path.join(ref_dir, _window_filename("hidden", idx)),
                        )
            head_path = os.path.join(ref_dir, "lm_head.safetensors")
            if want_hidden:
                copy_lm_head_from_checkpoint(reference_model_path, head_path)
            probe_report = None
            if probe_replay or storage == "auto":
                with _phase(timings, "replay_probe"):
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
                        _prune_fallback_logits(ref_dir, len(windows))
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
                "enable_prefix_caching": bool(
                    (llm_kwargs or {}).get("enable_prefix_caching")
                ),
                "max_num_seqs": (llm_kwargs or {}).get("max_num_seqs"),
                "declared_vocab_size": _declared_vocab_size(reference_model_path),
                "reference_routing": _routing_info(reference_model_path),
                "token_suite": suite_identity,
                "reference_model": os.path.abspath(reference_model_path),
                "reference_config_sha256": (
                    sha256_file(reference_config)
                    if os.path.isfile(reference_config)
                    else None
                ),
                "reference_weights_sha256": reference_weights_sha256,
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

    if paired_routing and not measure_routing:
        raise ValueError("--paired-routing requires --measure-routing")
    routing_manifest: dict[str, Any] | None = None
    if measure_routing:
        if reference_model_path is None:
            raise ValueError(
                "--measure-routing needs --reference-model: expert selections "
                "have to be read from the reference itself"
            )
        routing_dir = routing_dir or os.path.join(reference_logits_path, "routing")
        if (
            not isinstance(reference_weights_sha256, str)
            or len(reference_weights_sha256) != 64
        ):
            raise ValueError(
                "paired routing requires a complete reference weight digest"
            )
        routing_manifest = _capture_reference_routing(
            reference_model_path,
            windows,
            routing_dir,
            llm_kwargs or {},
            token_hash,
            reference_weights_sha256,
            sha256_file(os.path.join(reference_logits_path, "manifest.json")),
        )

    if capture_only:
        print(f"Capture-only: skipping Phase 2. References at {reference_logits_path}")
        return {
            "mean_kld": 0.0,
            "num_positions": 0,
            "capture_only": True,
            "timings": timings,
        }

    student_head_static = detect_lm_head_quantization(model_path)
    print(f"Student LM head (static): {student_head_static['state']}")
    print("Phase 2: Computing KLD...")
    print(f"Loading test model: {model_path}")
    student_kwargs = dict(llm_kwargs or {})
    if routing_manifest is not None:
        student_kwargs["enable_return_routed_experts"] = True
    with _phase(timings, "student_load"):
        llm = LLM(model=model_path, **student_kwargs)
    moe_backends: list[dict[str, Any]] = []
    if routing_manifest is not None:
        from vllm.v1.sample.kld import inspect_model_moe_backends

        moe_backends = llm.apply_model(inspect_model_moe_backends)
        if not moe_backends or not all(item.get("layers") for item in moe_backends):
            raise ValueError("routed scoring found no replay-capable MoE layers")
        candidate_layer_maps = [
            sorted(
                {
                    int(layer["layer_id"])
                    for layer in worker.get("layers", [])
                    if isinstance(layer.get("layer_id"), int)
                }
            )
            for worker in moe_backends
        ]
        if any(
            layer_map != routing_manifest["layer_map"]
            for layer_map in candidate_layer_maps
        ):
            raise ValueError(
                "candidate MoE layer IDs do not match the BF16 routing trace"
            )
        backend_profiles = {
            json.dumps(
                {
                    key: layer.get(key)
                    for key in (
                        "router",
                        "quant_method",
                        "kernel",
                        "experts",
                        "monolithic",
                        "routing_method",
                        "renormalize",
                        "scoring_func",
                    )
                },
                sort_keys=True,
            )
            for worker in moe_backends
            for layer in worker["layers"]
        }
        print("Student MoE backend profiles:")
        for profile in sorted(backend_profiles):
            print(f"  {profile}")
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
        teacher_id = logits_processor_identity(teacher_semantics)
        student_id = logits_processor_identity(student_semantics)
        if teacher_id != student_id:
            raise ValueError(
                "Teacher and student logits-processor semantics differ: "
                f"teacher={teacher_semantics!r}, student={student_semantics!r}"
            )
        if teacher_semantics != student_semantics:
            print(
                "  Note: logits-processor storage differs "
                f"(teacher={teacher_semantics!r}, "
                f"student={student_semantics!r}); "
                "identity fields match, scoring the deployed student head."
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

    divergence: _RoutingDivergence | None = None
    if routing_manifest is not None:
        divergence = _RoutingDivergence(
            len(routing_manifest["layer_map"]),
            int(routing_manifest["num_experts_per_tok"]),
        )

    chunks: list[KLDResult] = []
    natural_repeat_chunks: list[KLDResult] = []
    control_chunks: list[KLDResult] = []
    control_repeat_chunks: list[KLDResult] = []
    natural_repeat_route_mismatches = 0
    natural_repeat_route_values = 0
    control_temp = (
        tempfile.TemporaryDirectory(prefix="vllm-kld-qxq-control-")
        if paired_routing
        else None
    )
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
        with _phase(timings, "score_forward"):
            out = llm.generate([prompt], sampling_params=sampling_params)[0]
        if out.kld_result is None:
            raise RuntimeError("kld_result is None; KLD plumbing is broken")
        chunks.append(out.kld_result)
        if divergence is not None:
            with _phase(timings, "routing_compare"):
                _compare_window_routing(
                    divergence,
                    routing_dir,
                    idx,
                    out,
                    out.kld_result,
                    score_from,
                    routing_manifest["layer_map"],
                )
        if control_temp is not None:
            import numpy as np
            from safetensors.numpy import save_file as save_numpy

            natural_ids = out.outputs[0].routed_experts if out.outputs else None
            if natural_ids is None:
                raise RuntimeError("QxQ pass returned no routes for control parity")
            with _phase(timings, "qxq_natural_repeat_forward"):
                repeat_out = llm.generate(
                    [prompt],
                    sampling_params=SamplingParams(max_tokens=1, kld_mode=True),
                )[0]
            repeat_ids = (
                repeat_out.outputs[0].routed_experts
                if repeat_out.outputs
                else None
            )
            if repeat_out.kld_result is None or repeat_ids is None:
                raise RuntimeError("QxQ natural repeat returned incomplete evidence")
            if natural_ids.shape != repeat_ids.shape:
                raise RuntimeError(
                    "QxQ natural repeat returned different routing geometry: "
                    f"{natural_ids.shape} != {repeat_ids.shape}"
                )
            natural_repeat_route_mismatches += int(
                np.count_nonzero(natural_ids != repeat_ids)
            )
            natural_repeat_route_values += int(natural_ids.size)
            natural_repeat_chunks.append(repeat_out.kld_result)
            control_path = os.path.join(
                control_temp.name, _routing_filename(idx)
            )
            save_numpy({"routed_experts": natural_ids}, control_path)
            control_prompt: TokensPrompt = {
                **prompt,
                "reference_routing_path": control_path,
                "reference_routing_sha256": sha256_file(control_path),
            }
            with _phase(timings, "qxq_control_forward"):
                control_out = llm.generate(
                    [control_prompt],
                    sampling_params=SamplingParams(max_tokens=1, kld_mode=True),
                )[0]
            if control_out.kld_result is None:
                raise RuntimeError("QxQ control returned no KLD result")
            _assert_forced_window_routing(control_temp.name, idx, control_out)
            control_chunks.append(control_out.kld_result)
            with _phase(timings, "qxq_control_repeat_forward"):
                control_repeat_out = llm.generate(
                    [control_prompt],
                    sampling_params=SamplingParams(max_tokens=1, kld_mode=True),
                )[0]
            if control_repeat_out.kld_result is None:
                raise RuntimeError("QxQ control repeat returned no KLD result")
            _assert_forced_window_routing(
                control_temp.name,
                idx,
                control_repeat_out,
            )
            control_repeat_chunks.append(control_repeat_out.kld_result)

    bxq_chunks: list[KLDResult] = []
    if paired_routing:
        assert routing_manifest is not None
        assert routing_dir is not None
        print("Phase 2b: BxQ with BF16 expert IDs and student gating weights...")
        for idx, window_tokens in enumerate(windows):
            ref_file = os.path.join(
                reference_logits_path, _window_filename(capture_kind, idx)
            )
            routing_name = _routing_filename(idx)
            prompt = {
                "prompt_token_ids": window_tokens,
                "reference_logits_path": ref_file,
                "reference_logits_key": (
                    "hidden_states" if capture_kind == "hidden" else "logits"
                ),
                "kld_vocab_size": kld_vocab,
                "reference_routing_path": os.path.join(
                    routing_dir, routing_name
                ),
                "reference_routing_sha256": routing_manifest["file_hashes"][
                    routing_name
                ],
            }
            with _phase(timings, "bxq_score_forward"):
                out = llm.generate(
                    [prompt],
                    sampling_params=SamplingParams(max_tokens=1, kld_mode=True),
                )[0]
            if out.kld_result is None:
                raise RuntimeError("BxQ kld_result is None; KLD plumbing is broken")
            _assert_forced_window_routing(routing_dir, idx, out)
            bxq_chunks.append(out.kld_result)

    if dump_positions:
        _dump_positions(chunks, score_from, dump_positions)

    report = summarize_kld_rows(
        chunks, score_from=score_from, context_length=context_length
    )
    report["timings"] = timings
    report["unique_tokens"] = unique_tokens
    report["num_rows"] = len(windows)
    report["kld_vocab_size"] = kld_vocab
    report["model_runner_v2"] = student_uses_v2
    report["student_lm_head"] = student_head
    report["student_model"] = os.path.abspath(model_path)
    if manifest is not None:
        report["teacher_lm_head"] = manifest.get("lm_head")
        report["reference_weights_sha256"] = manifest.get(
            "reference_weights_sha256"
        )
        report["capture_manifest_sha256"] = sha256_file(
            os.path.join(reference_logits_path, "manifest.json")
        )
    report["storage"] = capture_kind
    report["per_context"] = _per_context_summary(
        chunks, score_from, (suite_identity or {}).get("context_ids")
    )
    if divergence is not None:
        report["routing"] = divergence.finalize()
    control_evidence: dict[str, Any] | None = None
    if control_chunks:
        import numpy as np

        natural_values = np.concatenate(
            [
                np.asarray(chunk.kld_ref_to_model[score_from:], dtype=np.float64)
                for chunk in chunks
            ]
        )
        control_values = np.concatenate(
            [
                np.asarray(chunk.kld_ref_to_model[score_from:], dtype=np.float64)
                for chunk in control_chunks
            ]
        )
        natural_repeat_values = np.concatenate(
            [
                np.asarray(chunk.kld_ref_to_model[score_from:], dtype=np.float64)
                for chunk in natural_repeat_chunks
            ]
        )
        control_repeat_values = np.concatenate(
            [
                np.asarray(chunk.kld_ref_to_model[score_from:], dtype=np.float64)
                for chunk in control_repeat_chunks
            ]
        )
        for name, values in (
            ("natural", natural_values),
            ("natural repeat", natural_repeat_values),
            ("forced control", control_values),
            ("forced control repeat", control_repeat_values),
        ):
            if values.shape != natural_values.shape:
                raise RuntimeError(
                    f"{name} KLD shape {values.shape} does not match "
                    f"natural KLD shape {natural_values.shape}"
                )
            if not np.all(np.isfinite(values)):
                raise RuntimeError(
                    f"non-finite KLD in {name} during repeatability control"
                )
        repeat_max_abs = float(
            np.max(
                np.abs(natural_values - natural_repeat_values),
                initial=0.0,
            )
        )
        repeat_mean_delta = float(
            natural_repeat_values.mean() - natural_values.mean()
        )
        repeat_mean_abs = float(
            np.mean(np.abs(natural_values - natural_repeat_values))
        )
        control_repeat_max_abs = float(
            np.max(
                np.abs(control_values - control_repeat_values),
                initial=0.0,
            )
        )
        control_repeat_mean_abs = float(
            np.mean(np.abs(control_values - control_repeat_values))
        )
        control_max_abs = max(
            float(
                np.max(
                    np.abs(natural_values - control_values),
                    initial=0.0,
                )
            ),
            float(
                np.max(
                    np.abs(natural_values - control_repeat_values),
                    initial=0.0,
                )
            ),
        )
        control_mean_delta = max(
            abs(float(control_values.mean() - natural_values.mean())),
            abs(float(control_repeat_values.mean() - natural_values.mean())),
        )
        position_tolerance = max(
            CONTROL_POSITION_BASE_TOLERANCE,
            CONTROL_REPEATABILITY_MULTIPLIER
            * max(repeat_max_abs, control_repeat_max_abs),
        )
        mean_tolerance = max(
            CONTROL_MEAN_BASE_TOLERANCE,
            CONTROL_REPEATABILITY_MULTIPLIER
            * max(repeat_mean_abs, control_repeat_mean_abs),
        )
        control_evidence = {
            "protocol": "natural_repeatability_envelope_v1",
            "passed": (
                control_max_abs <= position_tolerance
                and control_mean_delta <= mean_tolerance
            ),
            "natural_samples": 2,
            "control_samples": 2,
            "natural_repeat_route_mismatches": (
                natural_repeat_route_mismatches
            ),
            "natural_repeat_route_values": natural_repeat_route_values,
            "natural_repeat_route_flip_rate": (
                natural_repeat_route_mismatches
                / max(natural_repeat_route_values, 1)
            ),
            "natural_repeat_max_absolute_position_delta": repeat_max_abs,
            "natural_repeat_absolute_mean_delta": abs(repeat_mean_delta),
            "natural_repeat_mean_absolute_position_delta": repeat_mean_abs,
            "control_repeat_max_absolute_position_delta": (
                control_repeat_max_abs
            ),
            "control_repeat_mean_absolute_position_delta": (
                control_repeat_mean_abs
            ),
            "max_absolute_position_delta": control_max_abs,
            "absolute_mean_delta": control_mean_delta,
            "position_absolute_tolerance": position_tolerance,
            "mean_absolute_tolerance": mean_tolerance,
            "repeatability_multiplier": CONTROL_REPEATABILITY_MULTIPLIER,
            "deterministic": (
                repeat_max_abs <= CONTROL_POSITION_BASE_TOLERANCE
                and repeat_mean_abs <= CONTROL_MEAN_BASE_TOLERANCE
                and control_repeat_max_abs <= CONTROL_POSITION_BASE_TOLERANCE
                and control_repeat_mean_abs <= CONTROL_MEAN_BASE_TOLERANCE
            ),
        }
        if not control_evidence["passed"]:
            raise RuntimeError(
                "QxQ forced-route control exceeds deployed natural "
                f"repeatability: max position delta {control_max_abs:.3e} "
                f"(limit {position_tolerance:.3e}), mean delta "
                f"{control_mean_delta:.3e} (limit {mean_tolerance:.3e})"
            )
    if bxq_chunks:
        bxq = summarize_kld_rows(
            bxq_chunks, score_from=score_from, context_length=context_length
        )
        bxq["per_context"] = _per_context_summary(
            bxq_chunks, score_from, (suite_identity or {}).get("context_ids")
        )
        routing_manifest_path = os.path.join(routing_dir, ROUTING_MANIFEST)
        backend_names = sorted(
            {
                layer.get("experts") or layer.get("quant_method") or "unknown"
                for worker in moe_backends
                for layer in worker.get("layers", [])
            }
        )
        binding = {
            "protocol_version": PAIRED_ROUTED_SCORE_PROTOCOL_VERSION,
            "routing_trace_protocol_version": ROUTING_TRACE_PROTOCOL_VERSION,
            "routing_trace_sha256": sha256_file(routing_manifest_path),
            "routing_trace_manifest": routing_manifest_path,
            "reference_weights_sha256": routing_manifest.get(
                "reference_weights_sha256"
            ),
            "candidate_weights_unchanged": None,
            "backend_identity": moe_backends,
            "backend_evidence": {
                "backend": ", ".join(backend_names),
                "replay_supported": True,
                "workers": len(moe_backends),
            },
            "natural_control_parity": control_evidence,
        }
        report["paired_routing_protocol_version"] = (
            PAIRED_ROUTED_SCORE_PROTOCOL_VERSION
        )
        report["qxq_cell"] = {
            **summarize_kld_rows(
                chunks, score_from=score_from, context_length=context_length
            ),
            "routing_mode": "student_natural",
            "timings": {
                "score_forward": timings.get("score_forward", 0.0),
                "control_forward": timings.get("qxq_control_forward", 0.0),
            },
            **binding,
        }
        report["bxq_cell"] = {
            **bxq,
            "routing_mode": "teacher_ids_student_weights",
            "timings": {
                "score_forward": timings.get("bxq_score_forward", 0.0)
            },
            **binding,
        }
        report["routing_intervention_delta"] = (
            report["qxq_cell"]["mean_kld"] - report["bxq_cell"]["mean_kld"]
        )
    if control_temp is not None:
        control_temp.cleanup()
    if decompose_head:
        with _phase(timings, "decompose_head"):
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
        worker_agreement,
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
    worst_agreement: dict[str, Any] = {}
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
            agreement = worker_agreement(per_worker)
            if not agreement["agrees"]:
                fault = (
                    "Trunk KLD is not a number"
                    if agreement.get("nonfinite")
                    else "Trunk KLD differs across workers"
                )
                raise RuntimeError(
                    f"{fault} on window {idx}: {agreement['detail']}"
                )
            if agreement["max_abs_delta"] > worst_agreement.get(
                "max_abs_delta", 0.0
            ):
                worst_agreement = agreement
            trunk_chunks.append(per_worker[0])
            # A window's hidden states and the logits derived from them are
            # gigabytes. Held across a thousand windows, the freed-but-reserved
            # blocks fragment the pool until an allocation that fits nowhere
            # fails on a GPU that reports free memory.
            del student_h, hidden_out
            torch.cuda.empty_cache()
    trunk_report = summarize_kld_rows(
        trunk_chunks,
        score_from=score_from,
        context_length=context_length,
    )
    out: dict[str, Any] = {
        "trunk_mean_kld": trunk_report["mean_kld"],
        "trunk_report": trunk_report,
    }
    if worst_agreement:
        # Carried into the report so the artifact can say how far the ranks
        # diverged rather than implying they were bit-identical.
        out["trunk_worker_agreement"] = {
            key: worst_agreement[key]
            for key in (
                "workers",
                "max_abs_delta",
                "mean_abs_delta",
                "mean_rel_delta",
                "top1_flips",
            )
        }
    return out


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
    routing = report.get("routing")
    if routing:
        held = routing.get("mean_kld_routing_held")
        flipped = routing.get("mean_kld_routing_flipped")
        print("  Routing divergence (identical routers, perturbed inputs):")
        print(
            f"    selections changed: "
            f"{100 * routing['selection_flip_rate']:.3f}% of "
            f"{routing['layer_selections']} (token, layer) choices"
        )
        print(
            f"    positions with any flip: "
            f"{100 * routing['position_flip_rate']:.2f}%, carrying "
            f"{100 * (routing.get('flipped_share_of_total') or 0):.1f}% of the "
            f"total KLD"
        )
        print(
            f"    mean where routing held: "
            f"{'n/a' if held is None else f'{held:.8f}'}; where it flipped: "
            f"{'n/a' if flipped is None else f'{flipped:.8f}'}"
        )
        excess = routing.get("routing_excess_mean")
        print(
            f"    routing term: "
            f"{'n/a' if excess is None else f'{excess:+.8f}'}"
        )
    bxq = report.get("bxq_cell")
    if bxq:
        print(
            "  Paired routing intervention: "
            f"QxQ={report['qxq_cell']['mean_kld']:.8f}, "
            f"BxQ={bxq['mean_kld']:.8f}, "
            f"QxQ-BxQ={report['routing_intervention_delta']:+.8f}"
        )
        control = bxq["natural_control_parity"]
        print(
            "  Natural control: "
            + (
                "deterministic"
                if control["deterministic"]
                else "repeatability-calibrated"
            )
            + f", max={control['max_absolute_position_delta']:.3e}/"
            f"{control['position_absolute_tolerance']:.3e}, "
            f"mean={control['absolute_mean_delta']:.3e}/"
            f"{control['mean_absolute_tolerance']:.3e}, "
            "natural route-repeat flips="
            f"{control['natural_repeat_route_flip_rate']:.3%}"
        )
    head = report.get("student_lm_head") or {}
    print(f"  Student LM head: {head.get('state', 'unknown')}")
    print(f"  Storage: {report.get('storage')}")
    if report.get("trunk_mean_kld") is not None:
        print(
            "  Trunk mean KLD (shared teacher head): "
            f"{report['trunk_mean_kld']:.8f}"
        )
        pact = report.get("trunk_worker_agreement")
        if pact and pact.get("workers", 1) > 1:
            print(
                f"  Trunk cross-worker agreement: {pact['workers']} workers "
                f"within {pact['max_abs_delta']:.3e} per position, "
                f"{pact['mean_rel_delta']:.3e} relative on the mean"
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
        "--reference-weights-sha256",
        help="Content digest of the reference safetensors, required for paired "
        "routing so the BF16 IDs and teacher capture cannot come from "
        "different weights.",
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
        default=None,
        help="Hub dataset id or a local dataset directory. Ignored, and not "
        "required, when --token-suite is given",
    )
    parser.add_argument(
        "--token-suite",
        type=str,
        default=None,
        help="Directory of a frozen token suite built by fidelity/suite.py. "
        "Its stored token IDs become the evaluation input, and --rows, "
        "--context-length, and --stride are taken from the suite",
    )
    parser.add_argument(
        "--suite-partition",
        type=str,
        default="all",
        choices=("all", "analysis", "qualification"),
        help="Which suite partition to score. Freeze parameters on 'analysis' "
        "before reading 'qualification'",
    )
    parser.add_argument(
        "--suite-limit",
        type=int,
        default=None,
        help="Score only the first N contexts of the partition. For the zero "
        "baseline only; a limited run cannot match the suite's partition hash",
    )
    parser.add_argument(
        "--dump-positions",
        type=str,
        default=None,
        help="Write per-position KLD, reference confidence, top-1 agreement, "
        "row, and depth to this safetensors file, for tail analysis by "
        "fidelity/tails.py",
    )
    parser.add_argument(
        "--measure-routing",
        action="store_true",
        help="For a routed reference: record its expert selections in a "
        "separate pass, then report how often the candidate selected "
        "different experts and what that cost. Router weight precision is not "
        "the mechanism; perturbed activations reaching an identical router are",
    )
    parser.add_argument(
        "--routing-dir",
        type=str,
        default=None,
        help="Where reference expert selections live (default: "
        "<reference capture>/routing)",
    )
    parser.add_argument(
        "--paired-routing",
        action="store_true",
        help="After natural QxQ, score BxQ by forcing the BF16 trace's ordered "
        "expert IDs while retaining the student's own gating weights.",
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

    if not args.token_suite and not args.dataset:
        parser.error("one of --token-suite or --dataset is required")

    allow_apply_model_rpc()

    texts: list[str] = []
    if args.token_suite:
        print(f"Token suite: {args.token_suite} [{args.suite_partition}]")
        # The suite's geometry, not the flag default, has to reach the engine:
        # max_model_len is sized from context length before scoring begins.
        suite_manifest = os.path.join(args.token_suite, "suite-manifest.json")
        if not os.path.isfile(suite_manifest):
            parser.error(f"no suite-manifest.json in {args.token_suite}")
        with open(suite_manifest, encoding="utf-8") as f:
            args.context_length = int(json.load(f)["context_length"])
        print(f"Suite context length: {args.context_length}")
    else:
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
    if args.token_suite:
        print(f"  Rows, context length, and stride: from {args.suite_partition}")
    else:
        print(f"  Context length: {args.context_length}")
        print(f"  Stride: {stride}")
        print(f"  Rows: {args.rows}")
        print(f"  Samples: {args.num_samples or len(texts)}")
    print(f"  Score-from: {args.score_from}")
    print(f"  Storage: {args.storage}")

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
        token_suite=args.token_suite,
        suite_partition=args.suite_partition,
        suite_limit=args.suite_limit,
        dump_positions=args.dump_positions,
        measure_routing=args.measure_routing,
        routing_dir=args.routing_dir,
        paired_routing=args.paired_routing,
        reference_weights_sha256=args.reference_weights_sha256,
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
    timings = report.get("timings") or {}
    if timings:
        print("  Phase breakdown (seconds):")
        for name in (
            "teacher_load",
            "capture_forward",
            "capture_write",
            "replay_probe",
            "student_load",
            "score_forward",
            "qxq_natural_repeat_forward",
            "qxq_control_forward",
            "qxq_control_repeat_forward",
            "bxq_score_forward",
            "decompose_head",
        ):
            if name in timings:
                print(f"    {name}: {timings[name]:.2f}")
        fixed = timings.get("teacher_load", 0.0) + timings.get("student_load", 0.0)
        per_row = (
            timings.get("capture_forward", 0.0)
            + timings.get("capture_write", 0.0)
            + timings.get("score_forward", 0.0)
            + timings.get("qxq_natural_repeat_forward", 0.0)
            + timings.get("qxq_control_forward", 0.0)
            + timings.get("qxq_control_repeat_forward", 0.0)
            + timings.get("bxq_score_forward", 0.0)
        ) / max(report["num_rows"], 1)
        print(f"    weight loading (fixed, independent of rows): {fixed:.2f}")
        print(f"    marginal cost per row: {per_row:.2f}")
    npos = report["num_positions"]
    scoring_time = sum(
        timings.get(name, 0.0)
        for name in (
            "score_forward",
            "qxq_natural_repeat_forward",
            "qxq_control_forward",
            "qxq_control_repeat_forward",
            "bxq_score_forward",
        )
    )
    if scoring_time:
        print(f"  Scoring throughput: {npos / scoring_time:.0f} positions/second")


if __name__ == "__main__":
    main()
