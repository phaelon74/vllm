#!/usr/bin/env python3
"""
Tokenizer Test Script
Tests if the tokenizer can properly encode and decode tokens for the Devstral-2 model.
"""

import sys
import os
from pathlib import Path

# Add vllm to path if needed
sys.path.insert(0, str(Path(__file__).parent / "vllm"))

try:
    from transformers import AutoTokenizer
    import torch
except ImportError as e:
    print(f"ERROR: Missing required package: {e}")
    print("Please install: pip install transformers torch")
    sys.exit(1)

def test_tokenizer(model_path: str):
    """Test tokenizer encoding and decoding."""
    print("=" * 60)
    print("Tokenizer Test for Devstral-2 Model")
    print("=" * 60)
    print(f"\nModel path: {model_path}\n")
    
    # Check if model path exists
    if not os.path.exists(model_path):
        print(f"ERROR: Model path does not exist: {model_path}")
        return False
    
    # Check for tokenizer files
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json"
    ]
    
    print("Checking for tokenizer files:")
    for file in tokenizer_files:
        file_path = os.path.join(model_path, file)
        exists = os.path.exists(file_path)
        status = "✓" if exists else "✗"
        print(f"  {status} {file}")
    print()
    
    # Load tokenizer
    print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=False
        )
        print("✓ Tokenizer loaded successfully")
        print(f"  Tokenizer type: {type(tokenizer).__name__}")
        print(f"  Vocab size: {tokenizer.vocab_size}")
        print()
    except Exception as e:
        print(f"✗ ERROR: Failed to load tokenizer: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test encoding
    print("=" * 60)
    print("Test 1: Encoding")
    print("=" * 60)
    test_strings = [
        "Hello, how are you?",
        "Say hello.",
        "The quick brown fox",
        "测试"  # Test with non-ASCII
    ]
    
    for test_str in test_strings:
        try:
            encoded = tokenizer.encode(test_str, add_special_tokens=False)
            print(f"Input: '{test_str}'")
            print(f"  Encoded: {encoded}")
            print(f"  Token count: {len(encoded)}")
            print()
        except Exception as e:
            print(f"✗ ERROR encoding '{test_str}': {e}")
            return False
    
    # Test decoding
    print("=" * 60)
    print("Test 2: Decoding")
    print("=" * 60)
    test_token_ids = [
        [1, 2, 3],  # Simple test
        [9906, 11, 527, 499, 30],  # "Hello, how are you?" tokens (approximate)
        [9906],  # Single token
        [1],  # Special token
    ]
    
    for token_ids in test_token_ids:
        try:
            decoded = tokenizer.decode(token_ids, skip_special_tokens=False)
            print(f"Token IDs: {token_ids}")
            print(f"  Decoded: '{decoded}'")
            print()
        except Exception as e:
            print(f"✗ ERROR decoding {token_ids}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Test round-trip
    print("=" * 60)
    print("Test 3: Round-trip (encode -> decode)")
    print("=" * 60)
    test_round_trip = "Hello, this is a test!"
    try:
        encoded = tokenizer.encode(test_round_trip, add_special_tokens=False)
        decoded = tokenizer.decode(encoded, skip_special_tokens=False)
        print(f"Original: '{test_round_trip}'")
        print(f"Encoded: {encoded}")
        print(f"Decoded: '{decoded}'")
        if decoded.strip() == test_round_trip.strip():
            print("✓ Round-trip successful")
        else:
            print(f"⚠ WARNING: Round-trip mismatch")
            print(f"  Original: '{test_round_trip}'")
            print(f"  Decoded:  '{decoded}'")
        print()
    except Exception as e:
        print(f"✗ ERROR in round-trip test: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test special tokens
    print("=" * 60)
    print("Test 4: Special Tokens")
    print("=" * 60)
    try:
        if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
            print(f"BOS token ID: {tokenizer.bos_token_id}")
        if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
            print(f"EOS token ID: {tokenizer.eos_token_id}")
        if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
            print(f"PAD token ID: {tokenizer.pad_token_id}")
        print()
    except Exception as e:
        print(f"⚠ WARNING: Could not check special tokens: {e}")
        print()
    
    print("=" * 60)
    print("Tokenizer Test Complete")
    print("=" * 60)
    return True

if __name__ == "__main__":
    model_path = "/media/fmodels/TheHouseOfTheDude/Devstral-2-123B-Instruct-2512_Compressed-Tensors/W4A16"
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    success = test_tokenizer(model_path)
    sys.exit(0 if success else 1)

