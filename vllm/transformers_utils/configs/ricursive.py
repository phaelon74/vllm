# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Config class for the ricursive model architecture."""

from transformers.configuration_utils import PreTrainedConfig


class RicursiveConfig(PreTrainedConfig):
    model_type = "ricursive"

    def __init__(
        self,
        vocab_size=100352,
        hidden_size=3072,
        intermediate_size=24576,
        num_hidden_layers=50,
        num_attention_heads=48,
        num_key_value_heads=8,
        head_dim=128,
        max_position_embeddings=2048,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        sliding_window=1024,
        layer_types=None,
        use_gated_attention=True,
        gate_full_precision=True,
        use_valres=False,
        ffn_clamp_threshold=None,
        embed_scale=None,
        tie_word_embeddings=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.sliding_window = sliding_window
        self.use_gated_attention = use_gated_attention
        self.gate_full_precision = gate_full_precision
        self.use_valres = use_valres
        self.ffn_clamp_threshold = ffn_clamp_threshold
        self.embed_scale = embed_scale

        # Serialized hybrid-SWA metadata must not override the architecture
        # encoded by layer_types.
        kwargs.pop("is_hybrid_swa", None)
        kwargs.pop("hybrid_layer_pattern", None)

        if layer_types is not None:
            self.layer_types = layer_types
        else:
            self.layer_types = [
                "full_attention" if i % 5 == 4 else "sliding_attention"
                for i in range(num_hidden_layers)
            ]
        if len(self.layer_types) != num_hidden_layers:
            raise ValueError(
                f"layer_types has {len(self.layer_types)} entries, "
                f"expected {num_hidden_layers}"
            )
        invalid = set(self.layer_types) - {
            "sliding_attention",
            "full_attention",
        }
        if invalid:
            raise ValueError(f"unsupported Ricursive layer_types: {sorted(invalid)}")

        self.use_cache = kwargs.pop("use_cache", True)

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

        self.architectures = ["RicursiveForCausalLM"]
