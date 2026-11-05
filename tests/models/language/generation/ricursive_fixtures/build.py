# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Builds a tiny random Ricursive checkpoint for tests.

Vendored from the ricursive remote-code repo (modeling_ricursive.py /
configuration_ricursive.py) so the fixture needs no network access.
"""

import tempfile
from pathlib import Path

import torch

from .modeling_ricursive import RicursiveForCausalLM as HfRicursiveForCausalLM

VENDOR_DIR = Path(__file__).parent


def build_tiny_ricursive_checkpoint(
    num_hidden_layers: int = 2,
    hidden_size: int = 64,
    intermediate_size: int = 128,
    num_attention_heads: int = 4,
    num_key_value_heads: int = 2,
    head_dim: int = 32,
    sliding_window: int = 8,
    max_position_embeddings: int = 256,
    use_valres: bool = True,
    vocab_size: int = 50272,
    ffn_clamp_threshold: float | None = 20.0,
    embed_scale: float | None = None,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 0,
) -> Path:
    """Returns the path of a saved tiny random ricursive checkpoint."""
    from .configuration_ricursive import RicursiveConfig

    if embed_scale is None:
        embed_scale = hidden_size**0.5

    config = RicursiveConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        max_position_embeddings=max_position_embeddings,
        sliding_window=sliding_window,
        layer_types=["sliding_attention", "full_attention"] * (num_hidden_layers // 2),
        use_valres=use_valres,
        ffn_clamp_threshold=ffn_clamp_threshold,
        embed_scale=embed_scale,
        tie_word_embeddings=False,
    )
    torch.manual_seed(seed)
    model = HfRicursiveForCausalLM(config).to(dtype)
    path = Path(tempfile.mkdtemp(prefix="ricursive_tiny_"))
    from transformers import AutoTokenizer

    AutoTokenizer.from_pretrained("gpt2").save_pretrained(path)
    model.save_pretrained(path, safe_serialization=True)
    # HF requires the remote-code files next to the weights for
    # trust_remote_code loading from a local path.
    import shutil

    for f in ("modeling_ricursive.py", "configuration_ricursive.py"):
        shutil.copy(VENDOR_DIR / f, path / f)
    return path
