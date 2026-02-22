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
GPU_LAYERS=32
BATCH_SIZE=1024

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    exit 1
fi

# Check if server binary exists
if [ ! -f "$LLAMA_SERVER" ]; then
    echo "Error: llama-server not found at $LLAMA_SERVER"
    exit 1
fi

echo "Starting Llama 3 Server (Optimized for RTX 3050 6GB)..."
echo "Model: $MODEL_PATH"
echo "URL: http://localhost:$PORT"
echo "Config: -ngl $GPU_LAYERS | -b $BATCH_SIZE | --parallel 1 | --cache-ram 0"

# Run Server
"$LLAMA_SERVER" \
    -m "$MODEL_PATH" \
    -c "$CONTEXT_SIZE" \
    -ngl "$GPU_LAYERS" \
    -b "$BATCH_SIZE" \
    --parallel 1 \
    --cache-ram 0 \
    --port "$PORT" \
    --host "$HOST" \
    --temp 0.6 \
    --top-k 40 \
    --top-p 0.9
