#!/bin/bash

# Load .env if it exists
if [ -f .env ]; then
    # This sources the .env file, making variables available.
    # It assumes .env contains lines like VAR=value or export VAR=value
    set -a # Automatically export all subsequent variables
    . ./.env
    set +a # Turn off automatic exporting
fi

# Configuration
MODEL_PATH="${MODEL_PATH:-/path/to/your/model.gguf}"
LLAMA_SERVER="${LLAMA_SERVER:-/path/to/llama.cpp/llama-server}"
PORT=8080
HOST="0.0.0.0"
CONTEXT_SIZE=3072
BATCH_SIZE=1024

# Dynamic VRAM Layer Offloading Calculation
if command -v nvidia-smi &> /dev/null; then
    VRAM_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1)
    # Reserve 800MB for context, KV cache, and OS overhead
    RESERVED_MB=800
    AVAILABLE_FOR_LAYERS=$((VRAM_MB - RESERVED_MB))

    # Each layer ~140MB for 8B Q4_K_M
    LAYER_SIZE_MB=140
    MAX_LAYERS=32

    # Calculate optimal layers
    OPTIMAL_NGL=$((AVAILABLE_FOR_LAYERS / LAYER_SIZE_MB))
    FINAL_NGL=$((OPTIMAL_NGL < MAX_LAYERS ? OPTIMAL_NGL : MAX_LAYERS))
    if [ "$FINAL_NGL" -lt 0 ]; then FINAL_NGL=0; fi
else
    # Fallback if no NVIDIA gpu
    VRAM_MB="UNKNOWN"
    FINAL_NGL=32
fi

echo "Starting Llama 3 Server (Optimized for Stability & Caching)..."
echo "Model: $MODEL_PATH"
echo "URL: http://localhost:$PORT"
echo "VRAM Available: ${VRAM_MB}MB | Calculated Safe Layers: -ngl ${FINAL_NGL}"

# Run Server with Advanced V2.1 Context Optimization Flags
"$LLAMA_SERVER" \
    -m "$MODEL_PATH" \
    -c "$CONTEXT_SIZE" \
    -ngl "$FINAL_NGL" \
    -b "$BATCH_SIZE" \
    --parallel 1 \
    --cache-ram 512 \
    --mlock \
    --rope-scaling linear \
    --flash-attn on \
    --defrag-thold 0.1 \
    --metrics \
    --port "$PORT" \
    --host "$HOST" \
    --temp 0.6 \
    --top-k 40 \
    --top-p 0.9
