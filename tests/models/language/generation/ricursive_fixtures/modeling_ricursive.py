# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""HuggingFace modeling implementation for the Ricursive transformer architecture.

Designed for inference with vLLM (via --trust-remote-code). Key differences from
standard Llama/Mistral:
- PeriNorm blocks: 4 layer norms per block (pre/post attention +
- pre/post FFN)
- Gated attention: elementwise sigmoid gate on attention output
- Mixed local/global layers: local = sliding window + RoPE,
- global = full attention + no RoPE
- Per-head Q/K RMSNorm
"""

from typing import ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

from .configuration_ricursive import RicursiveConfig


class RicursiveRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)


class RicursiveRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        # NOTE: inv_freq is computed on-the-fly in forward() rather than stored
        # as a buffer, because non-persistent buffers get corrupted during
        # from_pretrained() loading.

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        inv_freq = 1.0 / (
            self.theta
            ** (torch.arange(0, self.dim, 2, device=x.device).float() / self.dim)
        )
        inv_freq_expanded = (
            inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)  # (B, 1, T, dim)
    sin = sin.unsqueeze(1)  # (B, 1, T, dim)
    # OLMo-core applies RoPE in full precision (float32) for numerical stability
    orig_dtype = q.dtype
    q_float, k_float = q.float(), k.float()
    cos_float, sin_float = cos.float(), sin.float()
    q_embed = (q_float * cos_float) + (rotate_half(q_float) * sin_float)
    k_embed = (k_float * cos_float) + (rotate_half(k_float) * sin_float)
    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


class RicursiveAttention(nn.Module):
    def __init__(self, config: RicursiveConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )
        if config.use_valres:
            self.value_res_lambda = nn.Parameter(torch.zeros(1))

        # Per-head Q/K norms (RMSNorm on head_dim)
        self.q_norm = RicursiveRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RicursiveRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Gated attention
        if config.use_gated_attention:
            self.g_proj = nn.Linear(
                self.hidden_size, self.num_heads * self.head_dim, bias=False
            )

        # RoPE only for local (sliding_attention) layers
        if self.layer_type == "sliding_attention":
            self.rotary_emb = RicursiveRotaryEmbedding(
                self.head_dim, theta=config.rope_theta
            )

    def _apply_sliding_window_mask(
        self,
        attention_mask: torch.Tensor | None,
        q_len: int,
        kv_len: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.LongTensor | None,
    ) -> torch.Tensor | None:
        if self.layer_type != "sliding_attention":
            return attention_mask

        key_positions = torch.arange(kv_len, device=device)
        if cache_position is not None:
            # transformers >= 5 may pass a stale/grown cache_position for
            # remote-code models; trust the real KV length instead.
            query_positions = torch.arange(
                max(kv_len - q_len, 0), kv_len, device=device
            )
        else:
            query_positions = torch.arange(kv_len - q_len, kv_len, device=device)

        window = self.config.sliding_window
        allowed = (
            (key_positions[None, :] <= query_positions[:, None])
            & (key_positions[None, :] >= query_positions[:, None] - (window - 1))
        )[None, None, :, :]

        if attention_mask is None:
            return allowed
        if attention_mask.dtype == torch.bool:
            return attention_mask & allowed
        return attention_mask.masked_fill(~allowed, torch.finfo(dtype).min)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_value: Cache | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.LongTensor | None = None,
        value_residual: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        bsz, q_len, _ = hidden_states.shape

        # Project Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        value_states = hidden_states
        if value_residual is not None:
            if not self.config.use_valres:
                raise RuntimeError(
                    "value_residual was supplied but config.use_valres is false"
                )
            blend = torch.sigmoid(self.value_res_lambda[0]).to(
                dtype=hidden_states.dtype
            )
            value_states = torch.lerp(
                value_residual.to(dtype=hidden_states.dtype), hidden_states, blend
            )
        v = self.v_proj(value_states)

        # Reshape to (B, T, n_heads, head_dim) then transpose to
        # (B, n_heads, T, head_dim)
        q = q.view(bsz, q_len, self.num_heads, self.head_dim)
        k = k.view(bsz, q_len, self.num_kv_heads, self.head_dim)
        v = v.view(bsz, q_len, self.num_kv_heads, self.head_dim)

        # Per-head Q/K norm (applied before transpose, on head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Transpose to (B, n_heads, T, head_dim) for attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # RoPE only for local layers
        if self.layer_type == "sliding_attention":
            cos, sin = self.rotary_emb(v, position_ids)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # KV cache
        if past_key_value is not None:
            k, v = past_key_value.update(k, v, self.layer_idx)

        # GQA: expand KV heads
        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)

        attention_mask = self._apply_sliding_window_mask(
            attention_mask=attention_mask,
            q_len=q.shape[2],
            kv_len=k.shape[2],
            dtype=q.dtype,
            device=q.device,
            cache_position=cache_position,
        )

        # Attention (use is_causal when no explicit mask for correct causal behavior)
        # Cast to plain bool to avoid SymBool issues with torch.compile
        is_causal = bool(attention_mask is None and q.shape[2] > 1)
        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=0.0, is_causal=is_causal
        )

        # (B, n_heads, T, head_dim) -> (B, T, n_heads * head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()

        # Gated attention: output * sigmoid(g_proj(input))
        if self.config.use_gated_attention:
            gate = self.g_proj(hidden_states)
            if self.config.gate_full_precision:
                gate = gate.float()
            gate = torch.sigmoid(gate).to(attn_output.dtype)
            attn_output = attn_output.view(bsz, q_len, -1) * gate
        else:
            attn_output = attn_output.view(bsz, q_len, -1)

        # Output projection
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


class RicursiveMLP(nn.Module):
    def __init__(self, config: RicursiveConfig):
        super().__init__()
        self.clamp_threshold = config.ffn_clamp_threshold
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        if self.clamp_threshold:
            gate = gate.clamp(min=None, max=self.clamp_threshold)
            up = up.clamp(min=-self.clamp_threshold, max=self.clamp_threshold)
        return self.down_proj(F.silu(gate) * up)


class RicursiveDecoderLayer(nn.Module):
    """PeriNorm decoder layer with 4 layer norms."""

    def __init__(self, config: RicursiveConfig, layer_idx: int):
        super().__init__()
        self.self_attn = RicursiveAttention(config, layer_idx)
        self.mlp = RicursiveMLP(config)

        # PeriNorm: 4 norms
        self.input_layernorm = RicursiveRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = RicursiveRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = RicursiveRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = RicursiveRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_value: Cache | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.LongTensor | None = None,
        value_residual: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, ...]:
        # Attention sub-block (PeriNorm: pre-norm -> attn -> post-norm -> residual)
        normed = self.input_layernorm(hidden_states)
        attn_output, attn_weights, present_key_value = self.self_attn(
            normed,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            value_residual=value_residual,
            **kwargs,
        )
        hidden_states = hidden_states + self.post_attention_layernorm(attn_output)

        # FFN sub-block (PeriNorm: pre-norm -> ffn -> post-norm -> residual)
        normed = self.pre_feedforward_layernorm(hidden_states)
        ff_output = self.mlp(normed)
        hidden_states = hidden_states + self.post_feedforward_layernorm(ff_output)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


class RicursivePreTrainedModel(PreTrainedModel):
    config_class = RicursiveConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["RicursiveDecoderLayer"]
    _supports_sdpa = True
    _supports_cache_class = True


class RicursiveModel(RicursivePreTrainedModel):
    def __init__(self, config: RicursiveConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [RicursiveDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RicursiveRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple | BaseModelOutputWithPast:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # embed_scale: non-folded checkpoints carry the scale
        # (e.g. sqrt(hidden_size)) in the config and apply it here at runtime;
        # folded checkpoints set embed_scale=None and bake the scale into
        # embed_tokens.weight, so this is a no-op. This mirrors the SGLang
        # native port (sglang_ricursive) so served and trained forwards agree.
        # It must run BEFORE the value-residual seed below, whose source is
        # the (scaled) layer-0 embeddings.
        embed_scale = getattr(self.config, "embed_scale", None)
        if embed_scale is not None:
            inputs_embeds = inputs_embeds * embed_scale

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # Build causal mask
        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values
        )

        hidden_states = inputs_embeds
        value_residual = hidden_states if self.config.use_valres else None
        all_hidden_states: tuple = () if output_hidden_states else None
        all_self_attns: tuple = () if output_attentions else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                value_residual=value_residual,
                **kwargs,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = past_key_values if use_cache else None

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor | None,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache | None,
    ) -> torch.Tensor | None:
        # When there's no explicit attention_mask (no padding) and we're doing
        # standard prefill or cached decoding, return None and let SDPA use
        # its efficient is_causal path. This also avoids numerical issues
        # from float min mask values with large activations.
        if attention_mask is None:
            return None
        if torch.all(attention_mask == 1):
            return None

        # Only build an explicit mask when we have a padding mask
        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1]

        if past_key_values is not None:
            target_length = past_key_values.get_seq_length() + sequence_length
        else:
            target_length = sequence_length

        causal_mask = torch.full(
            (sequence_length, target_length),
            fill_value=torch.finfo(dtype).min,
            dtype=dtype,
            device=device,
        )
        if sequence_length != 1:
            causal_mask = torch.triu(
                causal_mask, diagonal=target_length - sequence_length + 1
            )

        causal_mask = causal_mask[None, None, :, :].expand(
            input_tensor.shape[0], 1, -1, -1
        )

        if attention_mask.dim() == 2:
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[
                :, None, None, :
            ].to(causal_mask.dtype).masked_fill(
                attention_mask[:, None, None, :] == 0, torch.finfo(dtype).min
            )
            causal_mask = (
                torch.cat([padding_mask, causal_mask[:, :, :, mask_length:]], dim=-1)
                if mask_length < target_length
                else padding_mask
            )

        return causal_mask


class RicursiveForCausalLM(RicursivePreTrainedModel):
    _tied_weights_keys: ClassVar[list[str]] = []

    def __init__(self, config: RicursiveConfig):
        super().__init__(config)
        self.model = RicursiveModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple | CausalLMOutputWithPast:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        next_sequence_length: int | None = None,
        **kwargs,
    ):
        if next_sequence_length is not None:
            input_ids = input_ids[:, -next_sequence_length:]
        if cache_position is None:
            past_seen = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen,
                past_seen + input_ids.shape[1],
                device=input_ids.device,
            )
        if past_key_values is not None and cache_position.shape[0] < input_ids.shape[1]:
            input_ids = input_ids[:, cache_position]

        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": cache_position.unsqueeze(0)
                if cache_position is not None
                else None,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
            }
        )
        model_inputs.update(
            {
                k: v
                for k, v in kwargs.items()
                if k in ("output_hidden_states", "output_attentions")
            }
        )
        return model_inputs
