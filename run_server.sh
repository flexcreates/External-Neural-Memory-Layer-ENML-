#!/bin/bash

# Configuration
MODEL_PATH="/home/flex/Ai-Models/models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
LLAMA_SERVER="/home/flex/Ai-Models/llama.cpp/build/bin/llama-server"
PORT=8080
HOST="0.0.0.0"
CONTEXT_SIZE=4096
GPU_LAYERS=38
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
