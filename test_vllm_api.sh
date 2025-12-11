#!/bin/bash

# vLLM API Diagnostic Script
# This script tests the vLLM API to diagnose why tokens are generated but not returned

API_KEY="26c2027f7cfe2c127b55ab02918ad3de454c50f9d21699806b34bf0621cdfa73"
BASE_URL="http://localhost:8000"

echo "=========================================="
echo "vLLM API Diagnostic Test"
echo "=========================================="
echo ""

# Step 1: Check if API is accessible
echo "=== Step 1: Checking API accessibility ==="
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X GET "${BASE_URL}/v1/models" \
  -H "Authorization: Bearer ${API_KEY}")
echo "HTTP Status Code: $HTTP_CODE"
if [ "$HTTP_CODE" != "200" ]; then
    echo "ERROR: API is not accessible. Check if vLLM is running."
    exit 1
fi
echo "✓ API is accessible"
echo ""

# Step 2: Get available models
echo "=== Step 2: Getting available models ==="
MODELS_RESPONSE=$(curl -s -X GET "${BASE_URL}/v1/models" \
  -H "Authorization: Bearer ${API_KEY}")

echo "Models endpoint response:"
echo "$MODELS_RESPONSE" | jq '.' 2>/dev/null || echo "$MODELS_RESPONSE"
echo ""

MODEL_NAME=$(echo "$MODELS_RESPONSE" | jq -r '.data[0].id' 2>/dev/null)
if [ -z "$MODEL_NAME" ] || [ "$MODEL_NAME" == "null" ]; then
    echo "ERROR: Could not determine model name"
    exit 1
fi
echo "Using model: $MODEL_NAME"
echo ""

# Step 3: Test non-streaming request (thorough analysis)
echo "=== Step 3: Testing NON-STREAMING request (Detailed Analysis) ==="
echo "Request: POST ${BASE_URL}/v1/chat/completions"
echo "Payload: {\"model\": \"$MODEL_NAME\", \"messages\": [{\"role\": \"user\", \"content\": \"Say hello.\"}], \"stream\": false, \"max_tokens\": 20}"
echo ""

NON_STREAM_RESPONSE=$(curl -s -w "\nHTTP_CODE:%{http_code}\nTIME_TOTAL:%{time_total}\n" \
  -X POST "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${API_KEY}" \
  -d "{
    \"model\": \"${MODEL_NAME}\",
    \"messages\": [
      {\"role\": \"user\", \"content\": \"Say hello.\"}
    ],
    \"stream\": false,
    \"max_tokens\": 20
  }")

RESPONSE_BODY=$(echo "$NON_STREAM_RESPONSE" | head -n -2)
HTTP_CODE=$(echo "$NON_STREAM_RESPONSE" | grep "HTTP_CODE" | cut -d: -f2)
TIME_TOTAL=$(echo "$NON_STREAM_RESPONSE" | grep "TIME_TOTAL" | cut -d: -f2)

echo "HTTP Status: $HTTP_CODE"
echo "Time taken: ${TIME_TOTAL}s"
echo ""

echo "=== Full Response Structure ==="
echo "$RESPONSE_BODY" | jq '.' 2>/dev/null || echo "$RESPONSE_BODY"
echo ""

# Detailed analysis
echo "=== Detailed Response Analysis ==="
CONTENT=$(echo "$RESPONSE_BODY" | jq -r '.choices[0].message.content' 2>/dev/null)
ROLE=$(echo "$RESPONSE_BODY" | jq -r '.choices[0].message.role' 2>/dev/null)
FINISH_REASON=$(echo "$RESPONSE_BODY" | jq -r '.choices[0].finish_reason' 2>/dev/null)
TOKEN_IDS=$(echo "$RESPONSE_BODY" | jq -r '.choices[0].message.token_ids // empty' 2>/dev/null)
USAGE_PROMPT=$(echo "$RESPONSE_BODY" | jq -r '.usage.prompt_tokens // empty' 2>/dev/null)
USAGE_COMPLETION=$(echo "$RESPONSE_BODY" | jq -r '.usage.completion_tokens // empty' 2>/dev/null)
USAGE_TOTAL=$(echo "$RESPONSE_BODY" | jq -r '.usage.total_tokens // empty' 2>/dev/null)

echo "Content: '$CONTENT'"
echo "Role: '$ROLE'"
echo "Finish Reason: '$FINISH_REASON'"
echo "Token IDs: ${TOKEN_IDS:-null}"
echo "Usage - Prompt tokens: ${USAGE_PROMPT:-null}"
echo "Usage - Completion tokens: ${USAGE_COMPLETION:-null}"
echo "Usage - Total tokens: ${USAGE_TOTAL:-null}"
echo ""

if [ -n "$CONTENT" ] && [ "$CONTENT" != "null" ] && [ "$CONTENT" != "" ]; then
    echo "✓ Response contains content: '$CONTENT'"
    CONTENT_LEN=${#CONTENT}
    echo "  Content length: $CONTENT_LEN characters"
else
    echo "✗ ERROR: Response does not contain content!"
    echo "  Content value: '$CONTENT'"
fi

if [ -n "$TOKEN_IDS" ] && [ "$TOKEN_IDS" != "null" ] && [ "$TOKEN_IDS" != "" ]; then
    echo "✓ Response contains token_ids"
else
    echo "✗ ERROR: Response does not contain token_ids!"
fi

if [ -n "$USAGE_COMPLETION" ] && [ "$USAGE_COMPLETION" != "null" ] && [ "$USAGE_COMPLETION" != "0" ]; then
    echo "✓ Usage shows completion tokens: $USAGE_COMPLETION"
else
    echo "⚠ WARNING: Usage shows no completion tokens (or zero)"
fi
echo ""

# Step 4: Test streaming request
echo "=== Step 4: Testing STREAMING request ==="
echo "Request: POST ${BASE_URL}/v1/chat/completions (stream=true)"
echo ""

STREAM_RESPONSE=$(timeout 10 curl -s -N \
  -X POST "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${API_KEY}" \
  -d "{
    \"model\": \"${MODEL_NAME}\",
    \"messages\": [
      {\"role\": \"user\", \"content\": \"Say hello.\"}
    ],
    \"stream\": true,
    \"max_tokens\": 20
  }" 2>&1)

echo "Streaming response (first 10 lines):"
echo "$STREAM_RESPONSE" | head -n 10
echo ""

# Count SSE events
EVENT_COUNT=$(echo "$STREAM_RESPONSE" | grep -c "^data: " || echo "0")
echo "Number of SSE events received: $EVENT_COUNT"

if [ "$EVENT_COUNT" -gt 0 ]; then
    echo "✓ Streaming is working"
    echo "First few events:"
    echo "$STREAM_RESPONSE" | grep "^data: " | head -n 3
else
    echo "✗ ERROR: No streaming events received!"
fi
echo ""

# Step 5: Check for errors in response
echo "=== Step 5: Checking for errors ==="
ERROR=$(echo "$NON_STREAM_RESPONSE" | head -n -2 | jq -r '.error' 2>/dev/null)
if [ -n "$ERROR" ] && [ "$ERROR" != "null" ]; then
    echo "✗ ERROR found in response:"
    echo "$NON_STREAM_RESPONSE" | head -n -2 | jq '.error' 2>/dev/null
else
    echo "✓ No errors in response"
fi
echo ""

echo "=========================================="
echo "Diagnostic complete"
echo "=========================================="

