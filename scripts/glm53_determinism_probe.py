#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash prompt-logits determinism probe.

Findings this probe is built to chase down (4x180GB, TP=4, BF16, eager):

* Two back-to-back runs of an identical 2048-token prompt in one engine do not
  produce identical prompt logits. ``max|dlogit|`` reaches 8-10 and 30-50 of
  2047 positions flip argmax.
* Positions 0..127 are bit-identical in every run; positions 128..2046 differ
  in every run. The boundary is exactly 128 and stable across runs.
* No fixed point: run2 != run3 != run4, so this is nondeterminism, not state
  left behind by an earlier request.
* In-process self-KLD against logits captured moments earlier is 6.06e-3,
  matching the divergence. The KLD path is exact; the forward pass varies.

The 128 boundary points at the GLM KV-pooling path (``_kpool_tail_seed_kernel``,
``_kpool_softmax_rotate_write_cache_kernel``): the first 128 tokens attend only
to raw KV, and from 128 on attention starts reading the pooled representation of
the first completed block.

Each configuration must run as its own process, since several switches only take
effect before vLLM is imported.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

_ENV_KEYS = (
    "VLLM_BATCH_INVARIANT",
    "VLLM_WORKER_MULTIPROC_METHOD",
    "VLLM_USE_V2_MODEL_RUNNER",
    "VLLM_ATTENTION_BACKEND",
    "VLLM_ALLREDUCE_USE_FLASHINFER",
    "VLLM_USE_DEEP_GEMM",
    "VLLM_MOE_USE_DEEP_GEMM",
    "VLLM_DEEP_GEMM_WARMUP",
    "VLLM_USE_FLASHINFER_SAMPLER",
    "VLLM_FLOAT32_MATMUL_PRECISION",
    "CUBLAS_WORKSPACE_CONFIG",
    "CUDA_VISIBLE_DEVICES",
    "GLM53_TORCH_DETERMINISTIC",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config-name",
        default="baseline",
        help="Label for this run; names the JSON result file.",
    )
    p.add_argument("--model", required=True)
    p.add_argument(
        "--dataset-dir",
        default=None,
        help="Directory holding <dataset-config>/test-*.parquet.",
    )
    p.add_argument(
        "--token-ids-file",
        default=None,
        help="JSON list of token IDs. Preferred for an embedded capture gate.",
    )
    p.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    p.add_argument("--out-dir", default="determinism_probe")
    p.add_argument("--ctx", type=int, default=2048)
    p.add_argument("--stride", type=int, default=512)
    p.add_argument("--tp", type=int, default=4)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    p.add_argument("--moe-backend", default=None)
    p.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help="Set below --ctx to force chunked prefill.",
    )
    p.add_argument(
        "--repeats",
        type=int,
        default=4,
        help="Full-context captures to compare pairwise.",
    )
    p.add_argument(
        "--lengths",
        default="129,130,192,256,512",
        help="Prompt lengths for the boundary sweep. Index 128 only "
        "exists once length >= 130.",
    )
    p.add_argument(
        "--reference",
        default=None,
        help="safetensors of previously captured window-0 logits.",
    )
    p.add_argument("--reference-key", default="logits")
    p.add_argument("--skip-kld-selftest", action="store_true")
    p.add_argument(
        "--disable-flashinfer-autotune",
        action="store_true",
        help="Restore the pre-Jul-14 guard that score_mode_kld.py "
        "now applies only under --compiled. FlashInfer "
        "autotune picks kernel tactics by timing them.",
    )
    p.add_argument("--enforce-eager", action="store_true", default=True)
    p.add_argument("--no-enforce-eager", dest="enforce_eager", action="store_false")
    p.add_argument(
        "--gate",
        action="store_true",
        help="Fail with exit code 3 unless this configuration is "
        "bit-reproducible. Intended as a precondition for a "
        "reference-logits capture: a non-zero noise floor "
        "makes every KLD measured against those logits "
        "meaningless.",
    )
    p.add_argument(
        "--gate-max-self-kld",
        type=float,
        default=0.0,
        help="Largest mean self-KLD the gate accepts. The default "
        "of 0.0 is the only defensible value; raise it only "
        "to record a known floor deliberately.",
    )
    p.add_argument(
        "--gate-max-abs-dlogit",
        type=float,
        default=0.0,
        help="Largest max|dlogit| between repeat captures the gate accepts.",
    )
    return p.parse_args()


def env_report() -> dict[str, Any]:
    report: dict[str, Any] = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "env": {k: os.environ.get(k) for k in _ENV_KEYS},
    }
    if torch.cuda.is_available():
        report["gpu"] = torch.cuda.get_device_name(0)
        report["gpu_count"] = torch.accelerator.device_count()
    return report


def math_self_kl() -> float:
    """KL of a distribution against itself must be exactly zero."""
    logits = torch.randn(64, 8192, dtype=torch.float32)
    logp = F.log_softmax(logits, dim=-1)
    kl = F.kl_div(logp, logp, reduction="none", log_target=True).sum(-1)
    return float(kl.abs().max())


def build_tokens(args: argparse.Namespace) -> list[int]:
    if args.token_ids_file:
        with open(args.token_ids_file, encoding="utf-8") as f:
            tokens = json.load(f)
        if not isinstance(tokens, list) or not all(
            isinstance(token, int) for token in tokens
        ):
            raise SystemExit("--token-ids-file must contain a JSON integer list")
        if len(tokens) < args.ctx:
            raise SystemExit(
                f"token file has {len(tokens)} tokens, need {args.ctx}"
            )
        return tokens
    if not args.dataset_dir:
        raise SystemExit("one of --dataset-dir or --token-ids-file is required")

    from datasets import load_dataset
    from transformers import AutoTokenizer

    pattern = os.path.join(args.dataset_dir, args.dataset_config, "test-*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no parquet files matched {pattern}")
    ds = load_dataset("parquet", data_files={"test": files}, split="test")
    col = "text" if "text" in ds.column_names else ds.column_names[0]

    cap = args.ctx + 99 * args.stride
    text = "\n\n".join(t for t in ds[col] if t and t.strip())[: cap * 5]
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokens = tok(text, add_special_tokens=False, truncation=True, max_length=cap)[
        "input_ids"
    ]
    if tokens and isinstance(tokens[0], list):
        tokens = tokens[0]
    if len(tokens) < args.ctx:
        raise SystemExit(f"only tokenized {len(tokens)} tokens, need {args.ctx}")
    return list(tokens)


def build_llm(args: argparse.Namespace):
    from vllm import LLM

    kwargs: dict[str, Any] = dict(
        model=args.model,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False,
        max_model_len=args.ctx * 2,
        max_num_seqs=1,
        enforce_eager=args.enforce_eager,
        language_model_only=True,
    )
    if args.moe_backend:
        kwargs["moe_backend"] = args.moe_backend
    if args.max_num_batched_tokens is not None:
        kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    if args.disable_flashinfer_autotune:
        kwargs["enable_flashinfer_autotune"] = False
    return LLM(**kwargs)


def capture(llm, token_ids: list[int]) -> torch.Tensor:
    """Return prompt logits as fp32 CPU, shape (len(token_ids) - 1, vocab)."""
    from vllm import SamplingParams

    out = llm.generate(
        # No target_token_ids: supplying them selects the score-mode fast path,
        # which asserts on the extra column that prompt_logprobs=1 returns. We
        # want the full logits, not per-target logprobs.
        [{"prompt_token_ids": token_ids}],
        sampling_params=SamplingParams(
            max_tokens=1, return_prompt_logits=True
        ),
    )[0]
    if out.prompt_logits is None:
        raise SystemExit(
            "prompt_logits is None; return_prompt_logits plumbing "
            "is broken in this build"
        )
    return out.prompt_logits.float().contiguous()


def score_kld(
    llm, token_ids: list[int], ref_path: str, ref_key: str
) -> tuple[float, int]:
    from vllm import SamplingParams
    from vllm.v1.sample.kld import tokenizer_unpadded_vocab_size

    tokenizer = llm.llm_engine.tokenizer

    out = llm.generate(
        [
            {
                "prompt_token_ids": token_ids,
                "reference_logits_path": ref_path,
                "reference_logits_key": ref_key,
                "kld_vocab_size": tokenizer_unpadded_vocab_size(tokenizer),
            }
        ],
        sampling_params=SamplingParams(max_tokens=1, kld_mode=True),
    )[0]
    if out.kld_result is None:
        raise SystemExit("kld_result is None; KLD plumbing is broken in this build")
    kld_sum, count = out.kld_result.kld_sum, out.kld_result.kld_count
    return float(kld_sum), int(count)


def compare(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    """Summarize how two captures of the same prompt differ."""
    rows = min(a.shape[0], b.shape[0])
    a, b = a[:rows], b[:rows]

    diff = (a - b).abs()
    per_pos = diff.amax(dim=-1)
    differing = (per_pos > 0).nonzero().flatten()
    flips = (a.argmax(-1) != b.argmax(-1)).nonzero().flatten()

    logp_a = F.log_softmax(a, dim=-1)
    logp_b = F.log_softmax(b, dim=-1)
    kl = F.kl_div(logp_a, logp_b, reduction="none", log_target=True).sum(-1)

    return {
        "rows": rows,
        "bitwise_identical": bool(torch.equal(a, b)),
        "first_differing_pos": int(differing[0]) if differing.numel() else None,
        "identical_prefix_len": int(differing[0]) if differing.numel() else rows,
        "num_differing_pos": int(differing.numel()),
        "num_pos_gt_1e_3": int((per_pos > 1e-3).sum()),
        "max_abs_dlogit": float(diff.max()),
        "mean_abs_dlogit": float(diff.mean()),
        "argmax_flips": int(flips.numel()),
        "flip_positions": flips[:64].tolist(),
        "kl_mean": float(kl.mean()),
        "kl_median": float(kl.median()),
        "kl_p99": float(kl.quantile(0.99)),
        "kl_max": float(kl.max()),
    }


def fingerprint(logits: torch.Tensor, probe_rows: tuple[int, ...]) -> dict:
    """Small, saveable summary so runs can be compared across processes."""
    rows = logits.shape[0]
    picked = [r for r in probe_rows if r < rows]
    out = {
        "argmax": logits.argmax(-1).to(torch.int32),
        "row_max": logits.amax(dim=-1),
        "row_logsumexp": torch.logsumexp(logits, dim=-1),
    }
    if picked:
        out["probe_row_index"] = torch.tensor(picked, dtype=torch.int32)
        out["probe_rows"] = logits[picked].clone()
    return out


def boundary_sweep(llm, tokens: list[int], lengths: list[int]) -> list[dict]:
    """Two captures per length, to find where determinism breaks.

    Row index 128 only exists once the prompt reaches 130 tokens, so a length
    of 129 that comes back identical while 130 diverges localizes the cause to
    absolute position 128 rather than to a fraction of the sequence.
    """
    results = []
    for length in lengths:
        window = tokens[:length]
        first = capture(llm, window)
        second = capture(llm, window)
        entry = {"length": length, **compare(first, second)}
        results.append(entry)
        print(
            f"[sweep len={length:>5}] identical={entry['bitwise_identical']} "
            f"rows={entry['rows']} first_diff={entry['first_differing_pos']} "
            f"differing={entry['num_differing_pos']} "
            f"max|d|={entry['max_abs_dlogit']:.4e} "
            f"flips={entry['argmax_flips']}",
            flush=True,
        )
        del first, second
    return results


def repeat_matrix(llm, window: list[int], repeats: int) -> tuple[list, list]:
    captures = []
    for i in range(repeats):
        start = time.time()
        captures.append(capture(llm, window))
        print(
            f"[capture {i}] shape={tuple(captures[-1].shape)} "
            f"{time.time() - start:.1f}s",
            flush=True,
        )

    pairs = []
    for i in range(len(captures)):
        for j in range(i + 1, len(captures)):
            entry = {"a": i, "b": j, **compare(captures[i], captures[j])}
            pairs.append(entry)
            print(
                f"[{i}v{j}] identical={entry['bitwise_identical']} "
                f"first_diff={entry['first_differing_pos']} "
                f"differing={entry['num_differing_pos']}/{entry['rows']} "
                f"max|d|={entry['max_abs_dlogit']:.4e} "
                f"flips={entry['argmax_flips']} "
                f"kl_mean={entry['kl_mean']:.4e}",
                flush=True,
            )
    return captures, pairs


def evaluate_gate(args: argparse.Namespace, results: dict[str, Any]) -> dict[str, Any]:
    """Decide whether this configuration is fit to produce reference logits.

    Every check is a hard requirement. A reference-logits capture inherits
    whatever noise floor the producing configuration has, and that floor is
    indistinguishable from real quantization error in the KLD that follows.
    """
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    math_kl = results.get("math_self_kl_max")
    add(
        "math_self_kl",
        math_kl == 0.0,
        f"max self-KL over the reference implementation = {math_kl}",
    )

    for entry in results.get("boundary_sweep", []):
        add(
            f"boundary_len_{entry['length']}",
            entry["bitwise_identical"],
            f"first_diff={entry['first_differing_pos']} "
            f"differing={entry['num_differing_pos']}/{entry['rows']} "
            f"max|d|={entry['max_abs_dlogit']:.4e}",
        )

    for entry in results.get("repeat_pairs", []):
        ok = (
            entry["bitwise_identical"]
            or entry["max_abs_dlogit"] <= args.gate_max_abs_dlogit
        )
        add(
            f"repeat_{entry['a']}v{entry['b']}",
            ok,
            f"first_diff={entry['first_differing_pos']} "
            f"differing={entry['num_differing_pos']}/{entry['rows']} "
            f"max|d|={entry['max_abs_dlogit']:.4e} "
            f"flips={entry['argmax_flips']}",
        )

    selftest = results.get("kld_selftest")
    if selftest is not None:
        mean = selftest["mean"]
        add(
            "self_kld",
            mean is not None and mean <= args.gate_max_self_kld,
            f"mean={mean} limit={args.gate_max_self_kld}",
        )

    vs_ref = results.get("vs_reference")
    if vs_ref is not None:
        add(
            "vs_reference",
            vs_ref["bitwise_identical"],
            f"first_diff={vs_ref['first_differing_pos']} "
            f"differing={vs_ref['num_differing_pos']}/{vs_ref['rows']} "
            f"max|d|={vs_ref['max_abs_dlogit']:.4e} "
            f"flips={vs_ref['argmax_flips']}",
        )

    passed = all(c["passed"] for c in checks)
    print("\n=== determinism gate ===", flush=True)
    for check in checks:
        print(
            f"  [{'PASS' if check['passed'] else 'FAIL'}] "
            f"{check['check']:<24} {check['detail']}",
            flush=True,
        )
    print(
        f"gate: {'PASS' if passed else 'FAIL'} "
        f"({sum(not c['passed'] for c in checks)} of {len(checks)} checks "
        f"failed)",
        flush=True,
    )
    if not passed:
        print(
            "Do not capture reference logits from this configuration. Run "
            "scripts/glm53_layer_bisect.py to localize the divergence.",
            flush=True,
        )
    return {"passed": passed, "checks": checks}


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {
        "config_name": args.config_name,
        "args": vars(args),
        "environment": env_report(),
    }
    print(json.dumps(results["environment"], indent=2), flush=True)

    results["math_self_kl_max"] = math_self_kl()
    print(
        f"[math] self-KL max = {results['math_self_kl_max']:.3e} (must be 0.0)",
        flush=True,
    )

    tokens = build_tokens(args)
    window = tokens[: args.ctx]
    llm = build_llm(args)

    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    results["boundary_sweep"] = boundary_sweep(llm, tokens, lengths)

    captures, results["repeat_pairs"] = repeat_matrix(llm, window, args.repeats)

    if not args.skip_kld_selftest:
        from safetensors.torch import save_file

        tmp = out_dir / f"{args.config_name}_selfref.safetensors"
        save_file({"logits": captures[0]}, str(tmp))
        kld_sum, count = score_kld(llm, window, str(tmp), "logits")
        results["kld_selftest"] = {
            "kld_sum": kld_sum,
            "positions": count,
            "mean": kld_sum / count if count else None,
        }
        print(
            f"[kld] sum={kld_sum:.6e} positions={count} "
            f"mean={kld_sum / max(count, 1):.6e} (target 0.0)",
            flush=True,
        )
        tmp.unlink(missing_ok=True)

    if args.reference:
        from safetensors.torch import safe_open

        with safe_open(args.reference, framework="pt", device="cpu") as f:
            ref = f.get_slice(args.reference_key)[: captures[0].shape[0]]
        entry = compare(captures[0], ref.float())
        results["vs_reference"] = {"path": args.reference, **entry}
        print(
            f"[vs-ref] identical={entry['bitwise_identical']} "
            f"first_diff={entry['first_differing_pos']} "
            f"differing={entry['num_differing_pos']}/{entry['rows']} "
            f"max|d|={entry['max_abs_dlogit']:.4e} "
            f"flips={entry['argmax_flips']} "
            f"kl_mean={entry['kl_mean']:.4e}",
            flush=True,
        )

    from safetensors.torch import save_file

    probe_rows = (0, 1, 126, 127, 128, 129, 130, 255, 256, 511, 1023)
    for idx, cap in enumerate(captures):
        save_file(
            fingerprint(cap, probe_rows),
            str(out_dir / f"{args.config_name}_fingerprint_{idx}.safetensors"),
        )

    if args.gate:
        results["gate"] = evaluate_gate(args, results)

    json_path = out_dir / f"{args.config_name}.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {json_path}", flush=True)

    if args.gate and not results["gate"]["passed"]:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
