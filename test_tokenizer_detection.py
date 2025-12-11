#!/usr/bin/env python3
"""
Test script to verify which tokenizer vLLM is using for the model.
This script checks the vLLM server logs or uses a simpler approach.
"""

import subprocess
import sys

# Instead of importing vLLM directly (which has circular import issues),
# let's check what tokenizer files exist and what vLLM would detect

model_path = "/media/fmodels/TheHouseOfTheDude/Devstral-2-123B-Instruct-2512_Compressed-Tensors/W4A16"

print("=" * 60)
print("Tokenizer Detection Test")
print("=" * 60)
print(f"\nModel path: {model_path}\n")

# Check for tokenizer files
import os
if os.path.exists(model_path):
    files = os.listdir(model_path)
    tokenizer_files = [f for f in files if 'tokenizer' in f.lower() or 'tekken' in f.lower() or f.endswith('.model')]
    print("Tokenizer-related files found:")
    for f in tokenizer_files:
        print(f"  ✓ {f}")
    print()
    
    # Check for tekken.json specifically
    if 'tekken.json' in files:
        print("✓ tekken.json found - vLLM should use MistralTokenizer")
        print()
        print("To verify which tokenizer vLLM is actually using:")
        print("1. Check vLLM startup logs for 'tokenizer_mode' or 'MistralTokenizer'")
        print("2. Or add logging to vllm/v1/engine/detokenizer.py to see which")
        print("   detokenizer class is being instantiated")
    else:
        print("✗ tekken.json not found - vLLM will use HF tokenizer")
else:
    print(f"✗ Model path does not exist: {model_path}")

print("\n" + "=" * 60)
print("Next Steps:")
print("=" * 60)
print("1. Check vLLM server startup logs for tokenizer type")
print("2. Look for lines containing 'tokenizer_mode' or 'MistralTokenizer'")
print("3. If MistralTokenizer is not being used, check if mistral_common is installed")
print("4. Run: pip list | grep mistral")
print()
