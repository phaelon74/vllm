# Devstral-2 AWQ Quantized Model Support Implementation

## Overview

This document details the implementation of vLLM support for AWQ-quantized Devstral-2 models using the `compressed_tensors` format. Devstral-2 uses the `Ministral3ForCausalLM` architecture (model_type: `ministral3`), which required creating a new text-only model executor in vLLM.

## Problem Statement

When attempting to load AWQ-quantized Devstral-2 models, vLLM encountered the following error:

```
KeyError: 'layers.0.mlp.down_proj.activation_scale'
```

This error occurred because:

1. **Architecture Gap**: vLLM only had a multimodal `Mistral3ForConditionalGeneration` executor, but Devstral-2 requires a text-only `Ministral3ForCausalLM` executor.

2. **Quantization Parameter Handling**: AWQ-quantized models using `compressed_tensors` format store quantization parameters (like `activation_scale`, `weight_scale`, `weight_zero_point`) as buffers in the state dict. These parameters need to be properly declared and loaded by the model executor.

3. **Missing Model Registration**: The `Ministral3ForCausalLM` architecture was not registered in vLLM's model registry, causing vLLM to fall back to generic handlers that don't understand the quantization format.

## Solution Architecture

The solution involved three main components:

1. **Creating a new Ministral3 text-only executor** (`ministral3_text.py`)
2. **Registering the model** in vLLM's model registry
3. **Handling quantization parameters** correctly during weight loading

## Detailed Code Changes

### 1. Creation of `vllm/model_executor/models/ministral3_text.py`

This new file implements a complete Ministral3 text-only model executor, mirroring the structure of Llama/Mistral models but adapted for Ministral3 architecture.

#### 1.1 Model Structure Components

The implementation follows vLLM's standard model executor pattern with the following components:

**`Ministral3MLP` Class** (lines 72-112)
```python
class Ministral3MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        ...
    ):
        self.gate_up_proj = MergedColumnParallelLinear(...)
        self.down_proj = RowParallelLinear(...)
```

**Key Features:**
- Uses `MergedColumnParallelLinear` for gate and up projections (fused for efficiency)
- Uses `RowParallelLinear` for down projection
- Accepts `quant_config` parameter which enables quantization support
- The Linear layers automatically handle quantization parameters when `quant_config` is provided

**`Ministral3Attention` Class** (lines 115-249)
```python
class Ministral3Attention(nn.Module):
    def __init__(
        self,
        config: Mistral3Config,
        ...
        quant_config: QuantizationConfig | None = None,
        ...
    ):
        self.qkv_proj = QKVParallelLinear(...)
        self.o_proj = RowParallelLinear(...)
```

**Key Features:**
- Uses `QKVParallelLinear` for fused QKV projections
- Uses `RowParallelLinear` for output projection
- Supports sliding window attention (if configured)
- Integrates with vLLM's rotary embedding system
- All Linear layers accept `quant_config` for quantization support

**`Ministral3DecoderLayer` Class** (lines 252-353)
```python
class Ministral3DecoderLayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, ...):
        self.self_attn = Ministral3Attention(...)
        self.mlp = Ministral3MLP(...)
        self.input_layernorm = RMSNorm(...)
        self.post_attention_layernorm = RMSNorm(...)
```

**Key Features:**
- Combines attention and MLP layers
- Uses RMSNorm for layer normalization (consistent with Mistral architecture)
- Supports pipeline parallelism through `get_quant_config` method
- Implements residual connections in the forward pass

**`Ministral3Model` Class** (lines 365-462)
```python
class Ministral3Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, ...):
        self.embed_tokens = VocabParallelEmbedding(...)
        self.layers = make_layers(...)
        self.norm = RMSNorm(...)
```

**Key Features:**
- Manages the full model structure (embeddings, layers, normalization)
- Supports pipeline parallelism with `PPMissingLayer` placeholders
- Implements `load_weights` method for custom weight loading logic
- Handles stacked parameters (QKV, gate_up) for efficient loading

**`Ministral3ForCausalLM` Class** (lines 465-568)
```python
class Ministral3ForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    def __init__(self, *, vllm_config: VllmConfig, ...):
        self.model = Ministral3Model(...)
        self.lm_head = ParallelLMHead(...)
```

**Key Features:**
- Main model class that wraps `Ministral3Model`
- Implements `SupportsLoRA` and `SupportsPP` interfaces
- Defines `packed_modules_mapping` for LoRA support
- Uses `AutoWeightsLoader` for weight loading (handles quantization automatically)

#### 1.2 Critical Weight Loading Implementation

The most important part of the implementation is the `load_weights` method in `Ministral3Model` (lines 399-462). This method handles:

**Stacked Parameter Mapping:**
```python
stacked_params_mapping = [
    (".qkv_proj", ".q_proj", "q"),
    (".qkv_proj", ".k_proj", "k"),
    (".qkv_proj", ".v_proj", "v"),
    (".gate_up_proj", ".gate_proj", 0),
    (".gate_up_proj", ".up_proj", 1),
]
```

This maps HuggingFace's separate Q/K/V and gate/up projections to vLLM's fused implementations.

**Quantization Parameter Handling:**
```python
# Check if it's a quantization parameter (buffer) - these are handled
# by the Linear layer's weight_loader automatically
if name not in params_dict:
    # Quantization parameters like activation_scale are buffers,
    # not parameters. They're handled by AutoWeightsLoader or
    # the Linear layer's weight_loader, so we skip them here.
    continue
```

**Key Fix:** The critical fix was adding a check to skip parameters that aren't in `params_dict`. Quantization parameters like `activation_scale` are registered as buffers by the quantized Linear layers, not as regular parameters. By skipping them in the custom `load_weights` method, we allow `AutoWeightsLoader` to handle them correctly when it processes the Linear layers directly.

**Why This Works:**
1. When `AutoWeightsLoader` encounters a weight like `layers.0.mlp.down_proj.activation_scale`, it first checks if the parent module (`layers.0.mlp.down_proj`) has a custom `load_weights` method
2. Since `RowParallelLinear` (the down_proj layer) doesn't override `load_weights`, `AutoWeightsLoader` processes it directly
3. The quantized Linear layer's `weight_loader` function knows how to handle `activation_scale` buffers because they were registered during layer initialization when `quant_config` was provided
4. By skipping quantization parameters in `Ministral3Model.load_weights`, we prevent the `KeyError` and let the proper handlers process them

### 2. Model Registry Registration

**File:** `vllm/model_executor/models/registry.py`

**Change:** Added entry to `_TEXT_GENERATION_MODELS` dictionary (line 149)

```python
"Ministral3ForCausalLM": ("ministral3_text", "Ministral3ForCausalLM"),
```

**Explanation:**
- Maps the HuggingFace architecture name `Ministral3ForCausalLM` to vLLM's implementation
- `"ministral3_text"` is the module name (corresponds to `ministral3_text.py`)
- `"Ministral3ForCausalLM"` is the class name within that module
- This registration allows vLLM to automatically detect and use the correct executor when loading Devstral-2 models

**How It Works:**
1. When vLLM loads a model, it reads the `architectures` field from `config.json`
2. It looks up the architecture name in `_TEXT_GENERATION_MODELS`
3. If found, it imports the corresponding module and class
4. The model executor is then used to load and run the model

### 3. Quantization Parameter Handling Details

#### 3.1 How Quantization Parameters Are Registered

When a quantized Linear layer is created with `quant_config`, the quantization method (e.g., `CompressedTensorsLinearMethod`) registers buffers during `create_weights`:

```python
# Inside CompressedTensorsLinearMethod.create_weights()
if activation_quantization_needed:
    layer.register_buffer('activation_scale', ...)
    layer.register_buffer('weight_scale', ...)
```

These buffers are not in `named_parameters()` but are in `named_buffers()`.

#### 3.2 Weight Loading Flow

The weight loading process follows this flow:

1. **`Ministral3ForCausalLM.load_weights`** (line 563)
   - Creates `AutoWeightsLoader` instance
   - Delegates to `AutoWeightsLoader.load_weights`

2. **`AutoWeightsLoader.load_weights`** (in `utils.py`)
   - Iterates through all weights from the checkpoint
   - For each weight, calls `_load_module` which recursively processes modules

3. **`AutoWeightsLoader._load_module`** (in `utils.py`)
   - Checks if module has custom `load_weights` method
   - If yes (like `Ministral3Model`), calls it
   - If no, processes parameters/buffers directly

4. **`Ministral3Model.load_weights`** (line 399)
   - Handles stacked parameters (QKV, gate_up)
   - Skips quantization parameters (lets AutoWeightsLoader handle them)
   - Processes regular parameters

5. **Quantized Linear Layer Weight Loading**
   - When `AutoWeightsLoader` processes a Linear layer directly
   - It uses the layer's `weight_loader` function
   - The `weight_loader` knows how to handle quantization buffers
   - Buffers like `activation_scale` are loaded correctly

#### 3.3 The Fix Explained

The original error occurred because:

1. `Ministral3Model.load_weights` tried to access `params_dict[name]` for `activation_scale`
2. `activation_scale` is a buffer, not a parameter, so it wasn't in `params_dict`
3. This caused a `KeyError`

The fix:

1. Added check: `if name not in params_dict: continue`
2. This skips quantization parameters in the custom loader
3. `AutoWeightsLoader` then processes them when it reaches the Linear layers
4. The Linear layers' `weight_loader` functions handle buffers correctly

## Architecture Similarities

Ministral3 is architecturally very similar to Mistral/Llama, which is why the implementation closely mirrors `llama.py`:

- **Same layer structure**: Attention + MLP with RMSNorm
- **Same parameter fusing**: QKV fused, gate/up fused
- **Same quantization support**: Uses the same quantized Linear layers
- **Same parallelism support**: Tensor parallelism and pipeline parallelism

The main differences are:
- Config class: `Mistral3Config` vs `LlamaConfig`
- Some config attributes may differ (e.g., sliding window settings)

## Testing Considerations

To verify the implementation works correctly:

1. **Model Loading Test:**
   ```python
   from vllm import LLM
   llm = LLM(model="mistralai/Devstral-2-123B-Instruct-2512", quantization="awq")
   ```

2. **Quantization Parameter Verification:**
   ```python
   # Check that quantization parameters are loaded
   assert hasattr(model.layers[0].mlp.down_proj, 'activation_scale')
   assert model.layers[0].mlp.down_proj.activation_scale is not None
   ```

3. **Inference Test:**
   ```python
   outputs = llm.generate(["Hello, how are you?"])
   ```

## Dependencies

The implementation relies on:

- **Transformers >= 4.56.0**: For `Mistral3Config` support
- **compressed-tensors == 0.12.2**: For AWQ quantization format support
- **vLLM's quantization infrastructure**: The existing quantized Linear layer implementations

## Files Modified

1. **New File**: `vllm/model_executor/models/ministral3_text.py`
   - Complete Ministral3 text-only executor implementation
   - ~570 lines of code

2. **Modified**: `vllm/model_executor/models/registry.py`
   - Added one line to register `Ministral3ForCausalLM`

## Summary

The implementation successfully enables vLLM to load and run AWQ-quantized Devstral-2 models by:

1. Creating a complete Ministral3 text-only executor that mirrors Llama/Mistral architecture
2. Registering the model in vLLM's registry
3. Properly handling quantization parameters by letting the quantized Linear layers process them

The key insight was understanding that quantization parameters are buffers (not parameters) and should be handled by the Linear layers' weight loaders, not by the model-level weight loader. This allows the existing quantization infrastructure to work seamlessly with the new model architecture.

