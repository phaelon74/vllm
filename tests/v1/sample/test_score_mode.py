# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for score mode: SamplingParams validation, gather_target_logprobs,
the fast-path prompt logprobs processor, shared KLD math, windowing, and
LM-head detection."""

import json

import pytest
import torch

from vllm import SamplingParams
from vllm.logprobs import create_prompt_logprobs
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.sample.sampler import Sampler

# ---------------------------------------------------------------------------
# 1. SamplingParams validation
# ---------------------------------------------------------------------------


class TestSamplingParamsValidation:
    def test_score_mode_requires_prompt_logprobs(self):
        with pytest.raises(ValueError, match="score_mode requires prompt_logprobs"):
            SamplingParams(score_mode=True, max_tokens=1)

    def test_score_mode_valid(self):
        params = SamplingParams(score_mode=True, prompt_logprobs=1, max_tokens=1)
        assert params.score_mode is True
        assert params.prompt_logprobs == 1

    def test_kld_mode_mutually_exclusive_with_return_prompt_logits(self):
        with pytest.raises(
            ValueError,
            match="return_prompt_logits and kld_mode are mutually exclusive",
        ):
            SamplingParams(
                return_prompt_logits=True,
                kld_mode=True,
                prompt_logprobs=1,
                max_tokens=1,
            )

    def test_kld_mode_mutually_exclusive_with_return_prompt_hidden_states(self):
        with pytest.raises(
            ValueError,
            match="return_prompt_hidden_states and kld_mode are mutually",
        ):
            SamplingParams(
                return_prompt_hidden_states=True,
                kld_mode=True,
                prompt_logprobs=1,
                max_tokens=1,
            )

    def test_return_prompt_hidden_states_valid(self):
        params = SamplingParams(
            return_prompt_hidden_states=True,
            prompt_logprobs=1,
            max_tokens=1,
        )
        assert params.return_prompt_hidden_states is True

    def test_return_prompt_logits_valid(self):
        params = SamplingParams(
            return_prompt_logits=True, prompt_logprobs=1, max_tokens=1
        )
        assert params.return_prompt_logits is True

    def test_kld_mode_valid(self):
        params = SamplingParams(kld_mode=True, prompt_logprobs=1, max_tokens=1)
        assert params.kld_mode is True


# ---------------------------------------------------------------------------
# 2. Sampler.gather_target_logprobs
# ---------------------------------------------------------------------------


def _make_logprobs_tensor(num_tokens: int, vocab_size: int) -> torch.Tensor:
    """Create a logprobs tensor from random logits."""
    logits = torch.randn(num_tokens, vocab_size)
    return Sampler.compute_logprobs(logits)


class TestGatherTargetLogprobs:
    def test_basic(self):
        vocab_size = 100
        num_tokens = 4
        logprobs = _make_logprobs_tensor(num_tokens, vocab_size)
        target_ids = torch.tensor([3, 42, 0, 99], dtype=torch.int64)

        result = Sampler.gather_target_logprobs(logprobs, target_ids)
        indices, lps, ranks, *_ = result

        assert indices.shape == (num_tokens, 1)
        assert lps.shape == (num_tokens, 1)

        for i, tid in enumerate(target_ids.tolist()):
            assert indices[i, 0].item() == tid
            torch.testing.assert_close(
                lps[i, 0], logprobs[i, tid], atol=1e-6, rtol=1e-5
            )

    @pytest.mark.parametrize("num_tokens", [1, 8, 64])
    def test_output_shapes(self, num_tokens: int):
        vocab_size = 256
        logprobs = _make_logprobs_tensor(num_tokens, vocab_size)
        target_ids = torch.randint(0, vocab_size, (num_tokens,), dtype=torch.int64)

        indices, lps, ranks, *_ = Sampler.gather_target_logprobs(logprobs, target_ids)

        assert indices.shape == (num_tokens, 1)
        assert lps.shape == (num_tokens, 1)
        assert ranks.shape == (num_tokens,)

    def test_output_dtypes(self):
        logprobs = _make_logprobs_tensor(4, 50)
        target_ids = torch.tensor([1, 2, 3, 4], dtype=torch.int64)

        indices, lps, ranks, *_ = Sampler.gather_target_logprobs(logprobs, target_ids)

        assert indices.dtype == torch.int32
        assert lps.dtype == torch.float32

    def test_rank_correctness(self):
        """The rank of the highest-logprob token should be 0."""
        logprobs = _make_logprobs_tensor(1, 50)
        best_token = logprobs[0].argmax().item()
        target_ids = torch.tensor([best_token], dtype=torch.int64)

        _, _, ranks, *_ = Sampler.gather_target_logprobs(logprobs, target_ids)
        assert ranks[0].item() == 0


# ---------------------------------------------------------------------------
# 3. LogprobsProcessor._update_prompt_logprobs_fast_path
# ---------------------------------------------------------------------------


class TestFastPathLogprobs:
    @staticmethod
    def _make_processor(
        target_token_ids: list[int],
    ):
        """Build a minimal LogprobsProcessor for fast-path testing."""
        from vllm.v1.engine.logprobs import LogprobsProcessor

        return LogprobsProcessor(
            tokenizer=None,
            logprobs=None,
            prompt_logprobs=create_prompt_logprobs(flat_logprobs=False),
            cumulative_logprob=None,
            num_logprobs=None,
            num_prompt_logprobs=1,
            target_token_ids=target_token_ids,
        )

    def test_produces_correct_logprobs(self):
        target_ids = [10, 20, 30]
        processor = self._make_processor(target_ids)

        token_ids_tensor = torch.tensor([[10], [20], [30]], dtype=torch.int32)
        logprobs_tensor = torch.tensor([[-1.5], [-2.0], [-0.5]], dtype=torch.float32)
        ranks_tensor = torch.tensor([3, 7, 0], dtype=torch.int64)

        tensors = LogprobsTensors(token_ids_tensor, logprobs_tensor, ranks_tensor)
        processor._update_prompt_logprobs_fast_path(tensors, target_ids)

        prompt_lps = processor.prompt_logprobs
        # First entry is always None (position 0 has no logprobs).
        assert prompt_lps[0] is None
        # Subsequent entries should have one logprob each.
        assert len(prompt_lps) == 4  # None + 3 positions
        for i, tid in enumerate(target_ids):
            entry = prompt_lps[i + 1]
            assert tid in entry
            assert entry[tid].logprob == logprobs_tensor[i, 0].item()

    def test_token_id_mismatch_raises(self):
        target_ids = [10, 20, 30]
        processor = self._make_processor(target_ids)

        wrong_ids_tensor = torch.tensor([[10], [99], [30]], dtype=torch.int32)
        logprobs_tensor = torch.tensor([[-1.0], [-1.0], [-1.0]], dtype=torch.float32)
        ranks_tensor = torch.tensor([0, 0, 0], dtype=torch.int64)

        tensors = LogprobsTensors(wrong_ids_tensor, logprobs_tensor, ranks_tensor)

        with pytest.raises(ValueError, match="Token ID mismatch at position 1"):
            processor._update_prompt_logprobs_fast_path(tensors, target_ids)


# ---------------------------------------------------------------------------
# 4. Shared KLD math, windowing, manifest, LM-head detection
# ---------------------------------------------------------------------------


class TestComputeKldChunk:
    def test_vocab_truncation_excludes_padding(self):
        from vllm.v1.sample.kld import compute_kld_chunk

        torch.manual_seed(0)
        real = torch.randn(4, 32)
        pad_model = torch.full((4, 16), 50.0)
        pad_ref = torch.zeros(4, 16)
        model = torch.cat([real, pad_model], dim=-1)
        ref = torch.cat([real.clone(), pad_ref], dim=-1)
        untruncated = compute_kld_chunk(model, ref)
        truncated = compute_kld_chunk(model, ref, kld_vocab_size=32)
        assert any(v != 0.0 for v in untruncated.kld_ref_to_model)
        assert truncated.kld_count == 4
        for v in truncated.kld_ref_to_model:
            assert v == 0.0

    def test_self_kld_is_exactly_zero(self):
        from vllm.v1.sample.kld import compute_kld_chunk

        torch.manual_seed(1)
        logits = torch.randn(8, 64)
        result = compute_kld_chunk(logits, logits.clone())
        assert result.kld_count == 8
        for v in result.kld_ref_to_model:
            assert v == 0.0
        for v in result.kld_model_to_ref:
            assert v == 0.0
        assert result.model_top1 == result.ref_top1

    def test_per_position_length_matches_rows(self):
        from vllm.v1.sample.kld import compute_kld_chunk, summarize_kld

        torch.manual_seed(2)
        model = torch.randn(11, 40)
        ref = torch.randn(11, 40)
        result = compute_kld_chunk(model, ref, kld_vocab_size=32)
        assert result.kld_count == 11
        assert len(result.ref_top1_prob) == 11
        assert len(result.topk_agree) == 11
        report = summarize_kld(result, score_from=3, context_length=12)
        assert report["num_positions"] == 8
        assert report["score_from"] == 3


class TestEvalRows:
    def test_short_corpus_fails_instead_of_emitting_partial_row(self):
        from vllm.v1.sample.kld import iter_eval_rows

        with pytest.raises(ValueError, match="full evaluation row"):
            iter_eval_rows(list(range(127)), 128, 128, 1)

    def test_non_overlapping_row_and_token_counts(self):
        from vllm.v1.sample.kld import iter_eval_rows

        tokens = list(range(2048 * 100))
        rows = iter_eval_rows(tokens, 2048, 2048, 100)
        assert len(rows) == 100
        assert all(len(r) == 2048 for r in rows)
        unique = (len(rows) - 1) * 2048 + len(rows[-1])
        assert unique == 204_800
        scored_positions = sum(len(r) - 1 for r in rows)
        assert scored_positions == 204_700

    def test_overlapping_stride_is_opt_in(self):
        from vllm.v1.sample.kld import iter_eval_rows

        tokens = list(range(2048 + 99 * 512))
        rows = iter_eval_rows(tokens, 2048, 512, 100)
        assert len(rows) == 100
        unique = (len(rows) - 1) * 512 + len(rows[-1])
        assert unique == 2048 + 99 * 512


class TestLmHeadDetection:
    def test_runtime_unquantized_method_is_authoritative(self):
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            UnquantizedEmbeddingMethod,
        )
        from vllm.v1.sample.kld import inspect_parallel_lm_head

        class Head:
            quant_method = UnquantizedEmbeddingMethod()
            weight = torch.zeros(4, 4, dtype=torch.bfloat16)
            org_vocab_size = 4

        info = inspect_parallel_lm_head(Head())
        assert info["state"] == "unquantized"
        assert info["weight_dtype"] == "torch.bfloat16"

    def test_unquantized_float_weight(self, tmp_path):
        from safetensors.torch import save_file

        from vllm.v1.sample.kld import detect_lm_head_quantization

        (tmp_path / "config.json").write_text("{}", encoding="utf-8")
        save_file(
            {"lm_head.weight": torch.zeros(4, 4, dtype=torch.float16)},
            tmp_path / "model-00001-of-00001.safetensors",
        )
        index = {
            "weight_map": {
                "lm_head.weight": "model-00001-of-00001.safetensors",
            }
        }
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps(index), encoding="utf-8"
        )
        info = detect_lm_head_quantization(str(tmp_path))
        assert info["state"] == "unquantized"

    def test_quantized_packed_weight(self, tmp_path):
        from safetensors.torch import save_file

        from vllm.v1.sample.kld import detect_lm_head_quantization

        config = {"quantization_config": {"quant_method": "awq"}}
        (tmp_path / "config.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        save_file(
            {
                "lm_head.qweight": torch.zeros(4, 4, dtype=torch.int32),
                "lm_head.scales": torch.ones(4, dtype=torch.float16),
            },
            tmp_path / "model-00001-of-00001.safetensors",
        )
        index = {
            "weight_map": {
                "lm_head.qweight": "model-00001-of-00001.safetensors",
                "lm_head.scales": "model-00001-of-00001.safetensors",
            }
        }
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps(index), encoding="utf-8"
        )
        info = detect_lm_head_quantization(str(tmp_path))
        assert info["state"] == "quantized"
        assert info["packed_keys"]

    def test_ignored_lm_head_is_unquantized(self, tmp_path):
        from safetensors.torch import save_file

        from vllm.v1.sample.kld import detect_lm_head_quantization

        config = {
            "quantization_config": {
                "quant_method": "compressed-tensors",
                "ignore": ["lm_head"],
            }
        }
        (tmp_path / "config.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        save_file(
            {"lm_head.weight": torch.zeros(4, 4, dtype=torch.float16)},
            tmp_path / "model-00001-of-00001.safetensors",
        )
        index = {
            "weight_map": {
                "lm_head.weight": "model-00001-of-00001.safetensors",
            }
        }
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps(index), encoding="utf-8"
        )
        info = detect_lm_head_quantization(str(tmp_path))
        assert info["state"] == "unquantized"
        assert info["ignored"] is True

    def test_tied_quantized_embedding_is_quantized(self, tmp_path):
        from safetensors.torch import save_file

        from vllm.v1.sample.kld import detect_lm_head_quantization

        config = {
            "tie_word_embeddings": True,
            "quantization_config": {"quant_method": "awq"},
        }
        (tmp_path / "config.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        save_file(
            {
                "model.embed_tokens.qweight": torch.zeros(
                    4, 4, dtype=torch.int32
                ),
                "model.embed_tokens.scales": torch.ones(
                    4, dtype=torch.float16
                ),
            },
            tmp_path / "model-00001-of-00001.safetensors",
        )
        index = {
            "weight_map": {
                "model.embed_tokens.qweight": (
                    "model-00001-of-00001.safetensors"
                ),
                "model.embed_tokens.scales": (
                    "model-00001-of-00001.safetensors"
                ),
            }
        }
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps(index), encoding="utf-8"
        )
        info = detect_lm_head_quantization(str(tmp_path))
        assert info["state"] == "quantized"

    def test_fp16_head_dtype_is_not_a_semantics_mismatch(self):
        from vllm.v1.sample.kld import logits_processor_identity

        teacher = {
            "type": "LogitsProcessor",
            "scale": 1.0,
            "soft_cap": None,
            "vocab_size": 248320,
            "org_vocab_size": 248320,
            "head_dtype": "torch.bfloat16",
        }
        student = dict(teacher, head_dtype="torch.float16")
        assert logits_processor_identity(teacher) == logits_processor_identity(
            student
        )
        student_cap = dict(student, soft_cap=30.0)
        assert logits_processor_identity(teacher) != logits_processor_identity(
            student_cap
        )


class TestTokenizerVocab:
    def test_unpadded_prefers_actual_vocab_size(self):
        from vllm.v1.sample.kld import tokenizer_unpadded_vocab_size

        class Tok:
            actual_vocab_size = 32000
            vocab_size = 128256

        assert tokenizer_unpadded_vocab_size(Tok()) == 32000


class TestManifestMismatches:
    def test_tokenizer_mismatch_is_reported(self):
        from vllm.v1.sample.kld import manifest_mismatches

        captured = {
            "token_sha256": "aaa",
            "tokenizer": {"name_or_path": "teacher", "vocab_size": 100},
            "context_length": 2048,
            "stride": 2048,
            "rows": 100,
            "score_from": 0,
            "kld_vocab_size": 100,
            "tensor_parallel_size": 1,
            "enforce_eager": True,
        }
        live = dict(captured)
        live["tokenizer"] = {"name_or_path": "student", "vocab_size": 100}
        errors = manifest_mismatches(captured, live)
        assert any("tokenizer" in e for e in errors)

    def test_token_hash_mismatch_is_reported(self):
        from vllm.v1.sample.kld import manifest_mismatches

        captured = {
            "token_sha256": "aaa",
            "tokenizer": {"name_or_path": "same"},
            "context_length": 2048,
            "stride": 2048,
            "rows": 100,
            "score_from": 0,
            "kld_vocab_size": 100,
            "tensor_parallel_size": 1,
            "enforce_eager": True,
        }
        live = dict(captured)
        live["token_sha256"] = "bbb"
        errors = manifest_mismatches(captured, live)
        assert any("token_sha256" in e for e in errors)


def test_score_from_is_applied_to_every_row():
    from vllm.v1.sample.kld import KLDResult, summarize_kld_rows

    def row(values: list[float]) -> KLDResult:
        n = len(values)
        return KLDResult(
            values,
            values,
            [0.5] * n,
            [1] * n,
            [1] * n,
            [[1] * 5 for _ in range(n)],
        )

    report = summarize_kld_rows(
        [row([100.0, 1.0, 2.0]), row([200.0, 3.0, 4.0])],
        score_from=1,
        context_length=4,
    )
    assert report["num_positions"] == 4
    assert report["mean_kld"] == 2.5
    assert report["depth_buckets"] == [
        {
            "depth_lo": 1,
            "depth_hi": 1,
            "n": 2,
            "mean_kld": 2.0,
        },
        {
            "depth_lo": 2,
            "depth_hi": 2,
            "n": 2,
            "mean_kld": 3.0,
        },
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_v1_v2_engine_kld_parity(monkeypatch, tmp_path):
    """Both model runners must reproduce the same zero self-KLD payload."""
    from safetensors.torch import save_file

    from vllm import LLM, SamplingParams
    from vllm.v1.sample.kld import tokenizer_unpadded_vocab_size

    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    capture_llm = LLM(
        model="hmellor/tiny-random-LlamaForCausalLM",
        enforce_eager=True,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.1,
        max_model_len=64,
        max_num_batched_tokens=31,
        max_num_seqs=1,
    )
    tokens = list(range(8, 40))
    out = capture_llm.generate(
        [{"prompt_token_ids": tokens}],
        sampling_params=SamplingParams(
            max_tokens=1,
            return_prompt_logits=True,
        ),
    )[0]
    assert out.prompt_logits is not None
    ref_path = tmp_path / "reference.safetensors"
    save_file({"logits": out.prompt_logits}, ref_path)
    vocab_size = tokenizer_unpadded_vocab_size(
        capture_llm.llm_engine.tokenizer
    )
    capture_llm.shutdown()

    results = []
    for use_v2 in (False, True):
        monkeypatch.setenv(
            "VLLM_USE_V2_MODEL_RUNNER", "1" if use_v2 else "0"
        )
        llm = LLM(
            model="hmellor/tiny-random-LlamaForCausalLM",
            enforce_eager=True,
            enable_chunked_prefill=True,
            enable_prefix_caching=False,
            gpu_memory_utilization=0.1,
            max_model_len=64,
            max_num_batched_tokens=31,
            max_num_seqs=1,
        )
        scored = llm.generate(
            [
                {
                    "prompt_token_ids": tokens,
                    "reference_logits_path": str(ref_path),
                    "reference_logits_key": "logits",
                    "kld_vocab_size": vocab_size,
                }
            ],
            sampling_params=SamplingParams(max_tokens=1, kld_mode=True),
        )[0]
        assert scored.kld_result is not None
        results.append(scored.kld_result)
        llm.shutdown()

    assert results[0] == results[1]
    assert results[0].kld_count == len(tokens) - 1
    assert all(value == 0.0 for value in results[0].kld_ref_to_model)


def test_nonfinite_logits_are_refused_not_summarized():
    """A NaN mean must never reach a report; it propagates silently otherwise."""
    from vllm.v1.sample.kld import compute_kld_chunk

    torch.manual_seed(4)
    ref = torch.randn(4, 32)
    model = ref.clone()
    model[2, 5] = float("nan")
    with pytest.raises(ValueError, match="not finite at"):
        compute_kld_chunk(model, ref)


def test_shared_helper_is_deterministic():
    """V1 and V2 both call compute_kld_chunk; identical inputs must match."""
    from vllm.v1.sample.kld import compute_kld_chunk

    torch.manual_seed(3)
    model = torch.randn(16, 48)
    ref = torch.randn(16, 48)
    a = compute_kld_chunk(model, ref, kld_vocab_size=40)
    b = compute_kld_chunk(model.clone(), ref.clone(), kld_vocab_size=40)
    assert a.kld_ref_to_model == b.kld_ref_to_model
    assert a.ref_top1 == b.ref_top1


class TestWorkerAgreement:
    """Tensor-parallel workers scoring the same files must agree, not match bits.

    Each rank multiplies its own vocab shard on its own device, so demanding
    bit-identical floats fails a correct run; accepting any gap hides a sharding
    fault. These pin the boundary and the disclosure.
    """

    @staticmethod
    def _result(klds: list[float], top1: list[int] | None = None):
        from vllm.v1.sample.kld import KLDResult

        n = len(klds)
        return KLDResult(
            list(klds),
            list(klds),
            [0.5] * n,
            list(top1) if top1 is not None else [1] * n,
            [1] * n,
            [[1] * 5 for _ in range(n)],
        )

    def test_last_bit_noise_is_accepted_and_reported(self):
        from vllm.v1.sample.kld import worker_agreement

        base = [1.25, 0.5, 2.0]
        noisy = [1.25 + 1e-8, 0.5, 2.0 - 1e-8]
        found = worker_agreement([self._result(base), self._result(noisy)])
        assert found["agrees"], found["detail"]
        assert 0.0 < found["max_abs_delta"] < 1e-5
        assert "agree to" in found["detail"]

    def test_a_real_disagreement_is_refused_with_the_field_and_values(self):
        from vllm.v1.sample.kld import worker_agreement

        found = worker_agreement(
            [self._result([1.0, 1.0]), self._result([1.0, 2.0])]
        )
        assert not found["agrees"]
        # The message must name where and by how much, or the failure is
        # undiagnosable from a log.
        assert "position 1" in found["detail"]
        assert "kld_ref_to_model" in found["detail"]
        assert "sharding" in found["detail"]

    def test_workers_scoring_different_position_counts_never_pass(self):
        from vllm.v1.sample.kld import worker_agreement

        found = worker_agreement([self._result([1.0, 1.0]), self._result([1.0])])
        assert not found["agrees"]
        assert "did not score the same work" in found["detail"]

    def test_a_top1_flip_is_disclosed_not_refused(self):
        from vllm.v1.sample.kld import worker_agreement

        found = worker_agreement(
            [
                self._result([1.0, 1.0], top1=[7, 7]),
                self._result([1.0, 1.0], top1=[7, 8]),
            ]
        )
        assert found["agrees"], found["detail"]
        assert found["top1_flips"] == 1

    def test_a_nan_is_named_not_read_as_a_disagreement(self):
        """A NaN is unequal to itself, so it impersonates a worker disagreement."""
        from vllm.v1.sample.kld import worker_agreement

        nan = float("nan")
        found = worker_agreement(
            [self._result([1.0, nan]), self._result([1.0, nan])]
        )
        assert not found["agrees"]
        assert found["nonfinite"], found
        assert "not a number" in found["detail"]
        assert "position 1" in found["detail"]
        # Never blamed on the ranks, which computed the identical value.
        assert "disagreement between ranks" not in found["detail"].split("not a")[0]

    def test_a_single_worker_agrees_exactly(self):
        from vllm.v1.sample.kld import worker_agreement

        found = worker_agreement([self._result([1.0, 2.0])])
        assert found["agrees"]
        assert found["max_abs_delta"] == 0.0
