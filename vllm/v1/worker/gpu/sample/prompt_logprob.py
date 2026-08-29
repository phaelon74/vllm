# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import numpy as np
import torch

from vllm.config.model import LogprobsMode
from vllm.sampling_params import SamplingParams
from vllm.triton_utils import tl, triton
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.sample.logprob import compute_topk_scores


class PromptLogprobsWorker:
    def __init__(self, max_num_reqs: int, logprobs_mode: LogprobsMode = "raw_logprobs"):
        self.max_num_reqs = max_num_reqs
        self.logprobs_mode = logprobs_mode

        self.uses_prompt_logprobs = np.zeros(self.max_num_reqs, dtype=bool)
        self.num_prompt_logprobs = np.zeros(self.max_num_reqs, dtype=np.int32)
        # req_idx -> list of in-progress LogprobsTensors
        self.in_progress_prompt_logprobs: dict[str, list[LogprobsTensors]] = {}

        # Raw prompt logits and KLD are computed from the same logits as prompt
        # logprobs but bypass the top-k reduction, so they are tracked here too.
        self.uses_prompt_logits = np.zeros(self.max_num_reqs, dtype=bool)
        self.uses_kld = np.zeros(self.max_num_reqs, dtype=bool)
        self.in_progress_prompt_logits: dict[str, list[torch.Tensor]] = {}
        self.in_progress_kld: dict[str, tuple[float, int]] = {}
        self.reference_logits: dict[str, tuple[str, str]] = {}

    def add_request(
        self,
        req_id: str,
        req_idx: int,
        sampling_params: SamplingParams,
        reference_logits_path: str | None = None,
        reference_logits_key: str | None = None,
    ):
        uses_prompt_logprobs = sampling_params.prompt_logprobs is not None
        self.uses_prompt_logprobs[req_idx] = uses_prompt_logprobs
        self.num_prompt_logprobs[req_idx] = sampling_params.prompt_logprobs or 0
        if uses_prompt_logprobs:
            self.in_progress_prompt_logprobs[req_id] = []

        uses_kld = bool(
            sampling_params.kld_mode
            and reference_logits_path is not None
            and reference_logits_key is not None
        )
        self.uses_prompt_logits[req_idx] = sampling_params.return_prompt_logits
        self.uses_kld[req_idx] = uses_kld
        if sampling_params.return_prompt_logits:
            self.in_progress_prompt_logits[req_id] = []
        if uses_kld:
            assert reference_logits_path is not None
            assert reference_logits_key is not None
            self.in_progress_kld[req_id] = (0.0, 0)
            self.reference_logits[req_id] = (
                reference_logits_path,
                reference_logits_key,
            )

    def remove_request(self, req_id: str) -> None:
        self.in_progress_prompt_logprobs.pop(req_id, None)
        self.in_progress_prompt_logits.pop(req_id, None)
        self.in_progress_kld.pop(req_id, None)
        self.reference_logits.pop(req_id, None)

    def compute_prompt_logprobs(
        self,
        logits_fn: Callable[[torch.Tensor], torch.Tensor],
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
        # [max_num_reqs, max_model_len]
        all_token_ids: torch.Tensor,
        # [max_num_reqs]
        num_computed_tokens: torch.Tensor,
        # [max_num_reqs]
        prompt_lens: np.ndarray,
    ) -> dict[str, LogprobsTensors]:
        idx_mapping_np = input_batch.idx_mapping_np
        needs_prompt_logprobs = self.uses_prompt_logprobs[idx_mapping_np]
        if not np.any(needs_prompt_logprobs):
            # Common case: No request asks for prompt logprobs.
            return {}

        num_prompt_logprobs = self.num_prompt_logprobs[idx_mapping_np]
        prompt_lens = prompt_lens[idx_mapping_np]
        computed_prefill = input_batch.num_computed_prefill_tokens_np
        includes_prompt = computed_prefill < prompt_lens
        # NOTE(woosuk): If the request was resumed after preemption, its prompt
        # logprobs must have been computed before preemption. Skip.
        resumed_after_prompt = prompt_lens < input_batch.prefill_len_np
        needs_prompt_logprobs &= includes_prompt & ~resumed_after_prompt
        if not np.any(needs_prompt_logprobs):
            return {}

        # get the maximum number in this batch
        requested_num_prompt_logprobs = num_prompt_logprobs[needs_prompt_logprobs]
        max_num_prompt_logprobs = (
            -1
            if np.any(requested_num_prompt_logprobs == -1)
            else int(requested_num_prompt_logprobs.max())
        )

        # Get the prompt logprobs token_ids.
        prompt_logprobs_token_ids = get_prompt_logprobs_token_ids(
            input_batch.num_tokens,
            input_batch.query_start_loc,
            input_batch.idx_mapping,
            num_computed_tokens,
            all_token_ids,
        )
        prompt_token_ids, prompt_logprobs, prompt_ranks = (
            compute_prompt_logprobs_with_chunking(
                prompt_logprobs_token_ids,
                hidden_states[: input_batch.num_tokens],
                logits_fn,
                max_num_prompt_logprobs,
                self.logprobs_mode,
            )
        )

        pos_after_step = computed_prefill + input_batch.num_scheduled_tokens
        is_prompt_chunked = pos_after_step < prompt_lens

        query_start_loc_np = input_batch.query_start_loc_np
        prompt_logprobs_dict: dict[str, LogprobsTensors] = {}
        for i, req_id in enumerate(input_batch.req_ids):
            if not needs_prompt_logprobs[i]:
                continue

            req_is_prompt_chunked = is_prompt_chunked[i]
            req_num_prompt_logprobs = int(num_prompt_logprobs[i])
            start_idx = query_start_loc_np[i]
            end_idx = query_start_loc_np[i + 1]
            assert start_idx < end_idx, (
                f"start_idx ({start_idx}) >= end_idx ({end_idx})"
            )
            if not req_is_prompt_chunked:
                end_idx -= 1

            width = (
                prompt_logprobs.shape[1]
                if req_num_prompt_logprobs == -1
                else req_num_prompt_logprobs + 1
            )
            # no logprobs if start_idx >= end_idx
            logprobs = (
                None
                if start_idx >= end_idx
                else LogprobsTensors(
                    logprob_token_ids=prompt_token_ids[start_idx:end_idx, :width],
                    logprobs=prompt_logprobs[start_idx:end_idx, :width],
                    selected_token_ranks=prompt_ranks[start_idx:end_idx],
                )
            )

            prompt_logprobs_list = self.in_progress_prompt_logprobs[req_id]
            if logprobs is not None and (req_is_prompt_chunked or prompt_logprobs_list):
                prompt_logprobs_list.append(logprobs)
            if req_is_prompt_chunked:
                # Prompt is chunked. Do not return the logprobs yet.
                continue

            if prompt_logprobs_list:
                # Merge the in-progress logprobs.
                logprobs = LogprobsTensors.cat(prompt_logprobs_list)
                prompt_logprobs_list.clear()

            if logprobs is None:
                continue

            prompt_logprobs_dict[req_id] = logprobs
        return prompt_logprobs_dict

    def compute_prompt_logits(
        self,
        logits_fn: Callable[[torch.Tensor], torch.Tensor],
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
        # [max_num_reqs]
        prompt_lens: np.ndarray,
    ) -> tuple[dict[str, torch.Tensor], dict[str, tuple[float, int]]]:
        """Raw prompt logits and GPU-side KLD against reference logits.

        Position p of the returned logits predicts prompt token p + 1, matching
        the prompt logprobs convention. Both results accumulate across prefill
        chunks and are emitted only once the prompt is fully processed.
        """
        idx_mapping_np = input_batch.idx_mapping_np
        uses_logits = self.uses_prompt_logits[idx_mapping_np]
        uses_kld = self.uses_kld[idx_mapping_np]
        needs_logits = uses_logits | uses_kld
        if not np.any(needs_logits):
            return {}, {}

        prompt_lens = prompt_lens[idx_mapping_np]
        computed_prefill = input_batch.num_computed_prefill_tokens_np
        resumed_after_prompt = prompt_lens < input_batch.prefill_len_np
        needs_logits &= (computed_prefill < prompt_lens) & ~resumed_after_prompt
        if not np.any(needs_logits):
            return {}, {}

        pos_after_step = computed_prefill + input_batch.num_scheduled_tokens
        is_prompt_chunked = pos_after_step < prompt_lens
        query_start_loc_np = input_batch.query_start_loc_np

        prompt_logits_dict: dict[str, torch.Tensor] = {}
        kld_result_dict: dict[str, tuple[float, int]] = {}
        for i, req_id in enumerate(input_batch.req_ids):
            if not needs_logits[i]:
                continue

            start_idx = query_start_loc_np[i]
            end_idx = query_start_loc_np[i + 1]
            if not is_prompt_chunked[i]:
                # The final prompt position predicts the first output token,
                # which is not part of the prompt.
                end_idx -= 1
            if start_idx >= end_idx:
                continue

            logits = logits_fn(hidden_states[start_idx:end_idx])
            if uses_kld[i]:
                self._accumulate_kld(req_id, logits, int(computed_prefill[i]))
                if not is_prompt_chunked[i]:
                    kld_result_dict[req_id] = self.in_progress_kld[req_id]
                    self.in_progress_kld[req_id] = (0.0, 0)
                continue

            chunks = self.in_progress_prompt_logits[req_id]
            chunks.append(logits.float().cpu())
            if not is_prompt_chunked[i]:
                prompt_logits_dict[req_id] = torch.cat(chunks, dim=0)
                chunks.clear()
        return prompt_logits_dict, kld_result_dict

    def _accumulate_kld(
        self, req_id: str, logits: torch.Tensor, start_pos: int
    ) -> None:
        from safetensors.torch import safe_open

        path, key = self.reference_logits[req_id]
        end_pos = start_pos + logits.shape[0]
        with safe_open(path, framework="pt", device="cpu") as f:
            ref_slice = f.get_slice(key)
            ref_shape = ref_slice.get_shape()
            if len(ref_shape) != 2 or ref_shape[0] < end_pos:
                raise ValueError(
                    f"Reference logits {key!r} have shape {ref_shape}, but at "
                    f"least {end_pos} positions are needed."
                )
            if ref_shape[1] != logits.shape[-1]:
                raise ValueError(
                    "Reference and model logits must have identical vocabulary "
                    f"sizes; got {ref_shape[1]} and {logits.shape[-1]}."
                )
            ref_logits_cpu = ref_slice[start_pos:end_pos]
        if not torch.is_floating_point(ref_logits_cpu):
            raise ValueError(
                f"Reference logits must be floating point, got {ref_logits_cpu.dtype}."
            )
        ref_logits = ref_logits_cpu.to(logits.device)
        kld_per_pos = torch.nn.functional.kl_div(
            torch.log_softmax(logits.float(), dim=-1),
            torch.log_softmax(ref_logits.float(), dim=-1),
            reduction="none",
            log_target=True,
        ).sum(dim=-1)
        kld_sum, kld_count = self.in_progress_kld[req_id]
        self.in_progress_kld[req_id] = (
            kld_sum + kld_per_pos.sum().item(),
            kld_count + kld_per_pos.numel(),
        )


@triton.jit
def _prompt_logprobs_token_ids_kernel(
    prompt_logprobs_token_ids_ptr,
    query_start_loc_ptr,
    idx_mapping_ptr,
    num_computed_tokens_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)

    query_start = tl.load(query_start_loc_ptr + batch_idx)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    query_len = query_end - query_start

    num_computed_tokens = tl.load(num_computed_tokens_ptr + req_state_idx)
    for i in range(0, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        # NOTE(woosuk): We should shift the pos by one
        # because the logprob is computed for the next token.
        target_pos = num_computed_tokens + 1 + block
        token_ids = tl.load(
            all_token_ids_ptr + req_state_idx * all_token_ids_stride + target_pos,
            mask=mask,
        )
        tl.store(
            prompt_logprobs_token_ids_ptr + query_start + block, token_ids, mask=mask
        )


def get_prompt_logprobs_token_ids(
    num_tokens: int,
    query_start_loc: torch.Tensor,
    idx_mapping: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    all_token_ids: torch.Tensor,
) -> torch.Tensor:
    token_ids = torch.empty(num_tokens, dtype=torch.int64, device=idx_mapping.device)
    num_reqs = idx_mapping.shape[0]
    _prompt_logprobs_token_ids_kernel[(num_reqs,)](
        token_ids,
        query_start_loc,
        idx_mapping,
        num_computed_tokens,
        all_token_ids,
        all_token_ids.stride(0),
        BLOCK_SIZE=1024,
    )
    return token_ids


def compute_prompt_logprobs_with_chunking(
    prompt_token_ids: torch.Tensor,
    prompt_hidden_states: torch.Tensor,
    logits_fn: Callable[[torch.Tensor], torch.Tensor],
    num_prompt_logprobs: int,
    logprobs_mode: LogprobsMode = "raw_logprobs",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Since materializing the full prompt logits can take too much memory,
    # we compute it in chunks.
    CHUNK_SIZE = 1024
    token_ids = []
    scores = []
    ranks = []
    logits_mode = logprobs_mode in ("raw_logits", "processed_logits")
    prompt_token_ids = prompt_token_ids.to(torch.int64)
    for start_idx in range(0, prompt_token_ids.shape[0], CHUNK_SIZE):
        end_idx = start_idx + CHUNK_SIZE
        # NOTE(woosuk): logits_fn can be slow because it involves all-gather.
        prompt_logits = logits_fn(prompt_hidden_states[start_idx:end_idx])
        requested_num = (
            prompt_logits.shape[-1]
            if num_prompt_logprobs == -1
            else num_prompt_logprobs
        )
        result = compute_topk_scores(
            prompt_logits,
            requested_num,
            prompt_token_ids[start_idx:end_idx],
            logits_mode=logits_mode,
        )
        token_ids.append(result.logprob_token_ids)
        scores.append(result.logprobs)
        ranks.append(result.selected_token_ranks)

    token_ids = torch.cat(token_ids, dim=0) if len(token_ids) > 1 else token_ids[0]
    scores = torch.cat(scores, dim=0) if len(scores) > 1 else scores[0]
    ranks = torch.cat(ranks, dim=0) if len(ranks) > 1 else ranks[0]
    return token_ids, scores, ranks
