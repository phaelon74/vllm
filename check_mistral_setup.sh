#!/bin/bash
# Quick check script to verify Mistral tokenizer setup

echo "============================================================"
echo "Mistral Tokenizer Setup Check"
echo "============================================================"
echo ""

echo "1. Checking if mistral_common is installed:"
pip list | grep -i mistral || echo "  ✗ mistral_common not found in pip list"
echo ""

echo "2. Checking Python import:"
python3 -c "import mistral_common; print('  ✓ mistral_common imported successfully')" 2>&1 || echo "  ✗ Failed to import mistral_common"
echo ""

echo "3. Checking if vLLM can import MistralTokenizer:"
python3 -c "
import sys
sys.path.insert(0, 'vllm')
try:
    from vllm.tokenizers.mistral import MistralTokenizer
    print('  ✓ MistralTokenizer can be imported')
    print(f'  MistralTokenizer class: {MistralTokenizer}')
except Exception as e:
    print(f'  ✗ Failed to import MistralTokenizer: {e}')
" 2>&1
echo ""

echo "4. What to check in vLLM logs:"
echo "  - Look for: 'Using MistralIncrementalDetokenizer'"
echo "  - Look for: 'tokenizer_mode=mistral'"
echo "  - Look for: 'MistralTokenizer'"
echo ""
echo "============================================================"

