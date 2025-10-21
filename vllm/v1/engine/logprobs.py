# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
from collections.abc import Iterable
from dataclasses import dataclass

from vllm.logger import init_logger
from vllm.logprobs import Logprob, PromptLogprobs, SampleLogprobs
from vllm.transformers_utils.detokenizer_utils import (
    AnyTokenizer,
    convert_ids_list_to_tokens,
)
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest
from vllm.v1.outputs import LogprobsLists, LogprobsTensors

logger = init_logger(__name__)

NONES = itertools.repeat(None)


@dataclass
class LogprobsProcessor:
    # Tokenizer for this request,
    # None if detokenization is disabled.
    tokenizer: AnyTokenizer | None

    # Logprobs for this request
    logprobs: SampleLogprobs | None
    prompt_logprobs: PromptLogprobs | None
    cumulative_logprob: float | None
    num_logprobs: int | None
    num_prompt_logprobs: int | None
    
    # Target token IDs for efficient extraction (score_mode optimization)
    target_token_ids: list[int] | None = None

    @classmethod
    def from_new_request(
        cls,
        tokenizer: AnyTokenizer | None,
        request: EngineCoreRequest,
    ) -> "LogprobsProcessor":
        assert request.sampling_params is not None
        num_logprobs = request.sampling_params.logprobs
        num_prompt_logprobs = request.sampling_params.prompt_logprobs
        
        # Extract target_token_ids if provided (for score_mode optimization)
        target_token_ids = request.target_token_ids
        
        return cls(
            tokenizer=tokenizer,
            cumulative_logprob=(None if num_logprobs is None else 0.0),
            logprobs=(None if num_logprobs is None else []),
            # NOTE: logprob of first prompt token is None.
            prompt_logprobs=(None if num_prompt_logprobs is None else [None]),
            num_prompt_logprobs=num_prompt_logprobs,
            num_logprobs=num_logprobs,
            target_token_ids=target_token_ids,
        )

    def _update_sample_logprobs(self, logprobs_lists: LogprobsLists) -> None:
        """Update with sample logprobs from EngineCore.

        Outer lists are only of len > 1 if EngineCore made
        >1 tokens in prior step (e.g. in spec decoding).

        Args:
          logprobs_lists: the lists of logprob tokens, logprobs, and ranks.

        """

        assert self.num_logprobs is not None
        assert self.logprobs is not None
        assert self.cumulative_logprob is not None

        token_ids_lst, logprobs_lst, ranks_lst = logprobs_lists

        for rank, logprobs, token_ids in zip(ranks_lst, logprobs_lst, token_ids_lst):
            # Detokenize (non-incrementally).
            decoded_tokens = (
                NONES
                if self.tokenizer is None
                else (convert_ids_list_to_tokens(self.tokenizer, token_ids))
            )

            # Sampler puts the sampled logprob in first.
            sampled_token_logprob = logprobs[0]
            self.cumulative_logprob += sampled_token_logprob

            # Update with the Logprob dictionary for this pos.
            self.logprobs.append(
                self._make_logprob_dict(
                    logprobs,
                    token_ids,
                    decoded_tokens,
                    rank,
                    self.num_logprobs,
                )
            )

    def _update_prompt_logprobs(
        self,
        prompt_logprobs_tensors: LogprobsTensors,
    ) -> None:
        """Update with prompt logprobs from EngineCore.

        Args:
          prompt_logprobs_tensors: tuple containing the prompt logprobs
                                   tensors.

        """

        # Prompt logprobs are enabled.
        assert self.num_prompt_logprobs is not None
        assert self.prompt_logprobs is not None

        token_ids, logprobs, ranks = prompt_logprobs_tensors

        # FAST PATH: If target_token_ids provided, extract only those
        # (avoids creating 262M Logprob objects for full vocab)
        if self.target_token_ids is not None:
            self._update_prompt_logprobs_fast_path(
                prompt_logprobs_tensors, self.target_token_ids
            )
            return

        # STANDARD PATH: Extract top-K logprobs (create many Logprob objects)
        # Detokenize non-incrementally.
        # Output is flat: [num_tok, num_lps] -> [num_tok * num_lps]
        decoded_tokens = (
            None
            if self.tokenizer is None
            else (
                convert_ids_list_to_tokens(self.tokenizer, token_ids.flatten().tolist())
            )
        )

        # Recover shapes.
        num_prompt_tokens, num_logprobs = logprobs.shape

        # Pythonize the torch tensors.
        prompt_token_ranks = ranks.tolist()
        prompt_logprobs = logprobs.tolist()
        token_ids = token_ids.tolist()

        # Make Logprob for each position.
        for pos in range(num_prompt_tokens):
            # Handle flattening.
            offset = pos * num_logprobs
            offset_end = offset + num_logprobs
            decoded_tokens_for_pos = (
                NONES if decoded_tokens is None else decoded_tokens[offset:offset_end]
            )

            # Update with the Logprob dictionary for this pos.
            self.prompt_logprobs.append(
                self._make_logprob_dict(
                    prompt_logprobs[pos],
                    token_ids[pos],
                    decoded_tokens_for_pos,
                    prompt_token_ranks[pos],
                    self.num_prompt_logprobs,
                )
            )

    def _update_prompt_logprobs_fast_path(
        self,
        prompt_logprobs_tensors: LogprobsTensors,
        target_token_ids: list[int],
    ) -> None:
        """Fast path for score_mode: extract only target token logprobs.
        
        This optimization avoids creating 262M Logprob objects when full vocab is requested.
        Instead, it extracts only the target tokens (e.g., ground-truth for perplexity).
        
        Key optimization: Create only ~2048 Logprob objects instead of 262M
        
        Args:
          prompt_logprobs_tensors: tuple containing (token_ids, logprobs, ranks)
                                   where tensors contain FULL vocabulary data
          target_token_ids: list of ground-truth token IDs to extract
                           Should have length = num_positions - 1 (skipping position 0)
        """
        import torch
        
        token_ids_tensor, logprobs_tensor, ranks_tensor = prompt_logprobs_tensors
        
        # When using target_token_ids optimization, Sampler already extracted the targets:
        # token_ids_tensor.shape = [num_positions, 1] (target tokens already extracted!)
        # logprobs_tensor.shape = [num_positions, 1] (target logprobs already extracted!)
        # ranks_tensor.shape = [num_positions] (target ranks already computed!)
        
        num_positions = logprobs_tensor.shape[0]
        
        # target_token_ids should match num_positions exactly
        # vLLM already excludes position 0 (no context) from logprobs_tensor
        # So logprobs_tensor contains positions 1 through N of the original window
        if len(target_token_ids) != num_positions:
            raise ValueError(
                f"target_token_ids length ({len(target_token_ids)}) doesn't match "
                f"num_positions ({num_positions}). Note: vLLM already excluded position 0."
            )
        
        # Data is already extracted by Sampler - just squeeze and transfer to CPU!
        target_logprobs_cpu = logprobs_tensor.squeeze(-1).cpu().tolist()  # [num_positions, 1] -> [num_positions]
        target_ranks_cpu = ranks_tensor.cpu().tolist()  # [num_positions]
        target_token_ids_cpu = token_ids_tensor.squeeze(-1).cpu().tolist()  # [num_positions, 1] -> [num_positions]
        
        # Validate that extracted tokens match what we requested
        if target_token_ids_cpu != target_token_ids:
            raise ValueError(
                f"Extracted tokens don't match requested targets!\n"
                f"Requested: {target_token_ids[:10]}...\n"
                f"Got: {target_token_ids_cpu[:10]}..."
            )
        
        # Optionally detokenize (only target tokens, very fast)
        decoded_tokens = (
            NONES
            if self.tokenizer is None
            else convert_ids_list_to_tokens(self.tokenizer, target_token_ids_cpu)
        )
        
        # Build minimal dict: only 1 Logprob object per position (starting from position 1)
        # Note: self.prompt_logprobs already has [None] at position 0 from __init__
        for pos, (token_id, logprob, rank, token) in enumerate(
            zip(target_token_ids_cpu, target_logprobs_cpu, target_ranks_cpu, decoded_tokens)
        ):
            # Create dict with single entry (target token only)
            self.prompt_logprobs.append({
                token_id: Logprob(
                    logprob=logprob,
                    rank=rank,
                    decoded_token=token,
                )
            })

    def pop_prompt_logprobs(self) -> PromptLogprobs | None:
        """Pop and return all request prompt logprobs

        The logprobs processor aggregates prompt chunk logprobs
        over one or more prefill chunks. This method returns
        all prompt logprobs at once and then forgets them.
        Ensures correct RequestOutputKind.DELTA semantics
        wherein all prompt logprobs are returned at once at
        the end of prefill.

        Returns:
          None if prompt logprobs are disabled for this request.
          List of all prompt logprobs, otherwise.
        """
        plp = self.prompt_logprobs
        if plp:
            self.prompt_logprobs = []
        return plp

    @staticmethod
    def _make_logprob_dict(
        logprobs: list[float],
        logprob_token_ids: list[int],
        decoded_tokens: Iterable[str | None],
        rank: int,
        num_logprobs: int,
    ) -> dict[int, Logprob]:
        """Make a Logprob dictionary for a position.

        Args:
          logprobs: list of log probabilities
          logprob_token_ids: list of top token ids
          decoded_tokens: list of decoded top tokens
          rank: rank of the sampled token
          num_logprobs: number of logprobs requested
            by the user (in addition to sampled logprob)

        Returns:
          dict[token id, Logprob]
        """
        if num_logprobs == -1:
            num_logprobs = len(logprobs)
        # We do not need a special case for the sampled token
        # being in the topk, since inserting duplicated data
        # into a dictionary twice is the same as doing it once.
        topk_ranks = range(1, num_logprobs + 1)
        ranks = itertools.chain((rank,), topk_ranks)

        return {
            token_id: Logprob(
                logprob=logprob,
                rank=rank,
                decoded_token=token,
            )
            for token_id, logprob, rank, token in zip(
                logprob_token_ids, logprobs, ranks, decoded_tokens
            )
        }

    def update_from_output(self, output: EngineCoreOutput) -> None:
        if output.new_logprobs is not None:
            self._update_sample_logprobs(output.new_logprobs)
        if output.new_prompt_logprobs_tensors is not None:
            self._update_prompt_logprobs(output.new_prompt_logprobs_tensors)
