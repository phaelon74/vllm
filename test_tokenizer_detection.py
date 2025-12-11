#!/usr/bin/env python3
"""
Test script to verify which tokenizer vLLM is using for the model.
"""

import sys
sys.path.insert(0, 'vllm')

from vllm.tokenizers import get_tokenizer

model_path = "/media/fmodels/TheHouseOfTheDude/Devstral-2-123B-Instruct-2512_Compressed-Tensors/W4A16"

print("Testing tokenizer detection...")
print(f"Model path: {model_path}")
print()

try:
    tokenizer = get_tokenizer(
        model_path,
        tokenizer_mode="auto",
        trust_remote_code=False,
    )
    
    print(f"Tokenizer type: {type(tokenizer).__name__}")
    print(f"Tokenizer class: {type(tokenizer)}")
    print(f"Is MistralTokenizer: {type(tokenizer).__name__ == 'MistralTokenizer'}")
    print()
    
    # Test encoding
    test_text = "Hello, how are you?"
    print(f"Test encoding: '{test_text}'")
    encoded = tokenizer.encode(test_text, add_special_tokens=False)
    print(f"Encoded: {encoded}")
    print()
    
    # Test decoding
    print(f"Test decoding: {encoded}")
    decoded = tokenizer.decode(encoded, skip_special_tokens=False)
    print(f"Decoded: '{decoded}'")
    print()
    
    # Test single token decode
    if encoded:
        single_token = encoded[0]
        print(f"Test single token decode: [{single_token}]")
        single_decoded = tokenizer.decode([single_token], skip_special_tokens=False)
        print(f"Decoded: '{single_decoded}'")
        print()
    
    print("✓ Tokenizer loaded successfully")
    
except Exception as e:
    print(f"✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

