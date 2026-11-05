# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the ricursive (Ripple-65B-Preview) architecture.

Compares the native vLLM implementation against the HF remote-code model on a
tiny random checkpoint, covering every architectural quirk: embed scale,
value residual lerp, NoPE global layer, sliding-window layer, gated attention,
per-head Q/K norms, and FFN clamping.
"""

import pytest

from ....models.utils import check_logprobs_close
from .ricursive_fixtures.build import build_tiny_ricursive_checkpoint

pytestmark = pytest.mark.cpu_test


@pytest.fixture(scope="module")
def ricursive_tiny_random() -> str:
    return str(build_tiny_ricursive_checkpoint())


@pytest.mark.skipif(
    not __import__(
        "vllm.platforms", fromlist=["current_platform"]
    ).current_platform.is_cpu(),
    reason="CPU-only numerical parity test",
)
def test_ricursive_hf_parity(
    hf_runner,
    vllm_runner,
    example_prompts,
    ricursive_tiny_random,
) -> None:
    max_tokens, num_logprobs = 8, 5
    with hf_runner(
        ricursive_tiny_random, dtype="float32", trust_remote_code=True
    ) as hf_model:
        hf_outputs = hf_model.generate_greedy_logprobs_limit(
            example_prompts, max_tokens, num_logprobs
        )

    with vllm_runner(
        ricursive_tiny_random,
        dtype="float32",
        max_model_len=256,
        kv_cache_memory_bytes=64 * 1024 * 1024,
    ) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy_logprobs(
            example_prompts, max_tokens, num_logprobs
        )

    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )
