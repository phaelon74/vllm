# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Report which NVFP4 activation scales a checkpoint left for the loader to fill.

Loads a checkpoint and walks both the routed experts and the dense NVFP4
projections, printing the substitutions a scoring run would disclose under
Law 17. Use it to decide whether a published result predates a disclosure it
should have carried, without paying for a rescore to find out.

    python3 scripts/scan_nvfp4_scale_substitutions.py \\
        --model /media/fmodels2/unsloth/gemma-4-26B-A4B-it-NVFP4 \\
        --moe-backend cutlass

Pass the same ``--moe-backend`` the scoring run pinned, or the routed half of
the answer describes a kernel that never scored anything.
"""

import argparse
import json
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--moe-backend")
    parser.add_argument(
        "--batch-invariant",
        default="1",
        help="match the scoring run; 0 only to compare against it",
    )
    args = parser.parse_args()

    os.environ["VLLM_BATCH_INVARIANT"] = args.batch_invariant
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    from vllm import LLM
    from vllm.v1.sample.kld import (
        inspect_model_moe_backends,
        inspect_model_nvfp4_dense_scales,
    )

    findings = 0
    for model in args.model:
        print(f"\n=== {model}")
        kwargs = {
            "tensor_parallel_size": args.tp,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "enforce_eager": True,
            # The vision tower has its own token budget and would refuse to
            # load beside a small batch limit; nothing here inspects it.
            "language_model_only": True,
        }
        if args.moe_backend:
            kwargs["moe_backend"] = args.moe_backend
        llm = LLM(model=model, **kwargs)

        dense = llm.apply_model(inspect_model_nvfp4_dense_scales)
        scanned = sum(int(w.get("layers_scanned") or 0) for w in dense)
        filled = sum(int(w.get("layers_filled") or 0) for w in dense)
        print(f"  dense NVFP4 layers scanned : {scanned}")
        print(f"  dense NVFP4 layers filled  : {filled}")
        for worker in dense:
            for record in worker.get("substitutions") or ():
                findings += 1
                print(f"    {json.dumps(record, sort_keys=True)}")

        try:
            routed = llm.apply_model(inspect_model_moe_backends)
        except Exception as error:  # noqa: BLE001 - dense-only models have none
            print(f"  routed experts             : unavailable ({error})")
            routed = []
        experts = sorted(
            {
                layer.get("experts")
                for worker in routed
                for layer in worker.get("layers") or ()
                if layer.get("experts")
            }
        )
        if experts:
            print(f"  routed experts             : {', '.join(experts)}")
        for worker in routed:
            for record in worker.get("substitutions") or ():
                findings += 1
                print(f"    {json.dumps(record, sort_keys=True)}")
        del llm

    print(f"\nsubstitution records found: {findings}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
