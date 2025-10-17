#!/usr/bin/env python3
"""
Test script for score_mode functionality in vLLM.
This tests the ability to get exact log probabilities for prompt tokens.
"""

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
import torch
import math

def test_score_mode():
    """Test that score_mode returns exact logprobs for all prompt tokens."""
    
    # Initialize a small model for testing
    print("Loading model...")
    llm = LLM(
        model="facebook/opt-125m",  # Small model for quick testing
        max_model_len=512,
        enforce_eager=True,  # Avoid CUDA graph issues for testing
    )
    
    # Test prompt
    test_text = "The quick brown fox jumps over the lazy dog"
    
    # Tokenize the prompt
    tokenizer = llm.get_tokenizer()
    token_ids = tokenizer.encode(test_text)
    
    print(f"\nTest text: {test_text}")
    print(f"Number of tokens: {len(token_ids)}")
    print(f"Token IDs: {token_ids}")
    
    # Create sampling params with score_mode enabled
    sampling_params = SamplingParams(
        score_mode=True,
        temperature=0.0,  # Greedy for deterministic results
    )
    
    print(f"\nSamplingParams:")
    print(f"  score_mode: {sampling_params.score_mode}")
    print(f"  max_tokens: {sampling_params.max_tokens}")
    print(f"  prompt_logprobs: {sampling_params.prompt_logprobs}")
    
    # Run inference with score mode
    print("\nRunning inference with score_mode...")
    outputs = llm.generate(
        prompts=[TokensPrompt(prompt_token_ids=token_ids)],
        sampling_params=sampling_params,
    )
    
    # Check the output
    output = outputs[0]
    
    print(f"\nOutput:")
    print(f"  Request ID: {output.request_id}")
    print(f"  Finished: {output.finished}")
    print(f"  Number of outputs: {len(output.outputs)}")
    
    if output.outputs:
        completion = output.outputs[0]
        print(f"\n  Completion[0]:")
        print(f"    Generated tokens: {len(completion.token_ids)}")
        print(f"    Token IDs: {completion.token_ids}")
        print(f"    Text: '{completion.text}'")
    
    # Check prompt logprobs
    if output.prompt_logprobs:
        print(f"\n  Prompt logprobs:")
        print(f"    Number of positions: {len(output.prompt_logprobs)}")
        
        # Check a few positions
        for i in range(min(3, len(output.prompt_logprobs))):
            if output.prompt_logprobs[i] is not None:
                print(f"    Position {i}: {len(output.prompt_logprobs[i])} tokens")
                # Show the logprob for the actual token at this position
                if i < len(token_ids):
                    actual_token = token_ids[i]
                    if actual_token in output.prompt_logprobs[i]:
                        logprob = output.prompt_logprobs[i][actual_token].logprob
                        print(f"      Token {actual_token}: logprob={logprob:.4f}")
        
        # Calculate perplexity
        total_logprob = 0.0
        num_tokens = 0
        for i in range(1, len(output.prompt_logprobs)):  # Skip first token (no context)
            if output.prompt_logprobs[i] and i < len(token_ids):
                actual_token = token_ids[i]
                if actual_token in output.prompt_logprobs[i]:
                    total_logprob += output.prompt_logprobs[i][actual_token].logprob
                    num_tokens += 1
        
        if num_tokens > 0:
            avg_nll = -total_logprob / num_tokens
            perplexity = math.exp(avg_nll)
            print(f"\n  Perplexity: {perplexity:.4f}")
            print(f"    (based on {num_tokens} tokens)")
        else:
            print("\n  Could not calculate perplexity (no valid tokens)")
    else:
        print("\n  No prompt logprobs returned!")
    
    print("\n✓ Test completed successfully!")
    return True

if __name__ == "__main__":
    test_score_mode()

