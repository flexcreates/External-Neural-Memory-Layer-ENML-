#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# ENML — Llama.cpp Inference Server Startup
# ═══════════════════════════════════════════════════════════════════════
# Starts llama-server with optimized settings for ENML.
# All configuration is read from .env
# ═══════════════════════════════════════════════════════════════════════

# Load .env
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Configuration from .env
MODEL_PATH="${MODEL_PATH:-/path/to/your/model.gguf}"
LLAMA_SERVER="${LLAMA_SERVER:-/path/to/llama.cpp/llama-server}"
LLAMA_URL="${LLAMA_SERVER_URL:-http://localhost:8080}"
PORT=$(echo "$LLAMA_URL" | grep -oP ':\K[0-9]+$' || echo "8080")
HOST="0.0.0.0"
CONTEXT_SIZE=3072
BATCH_SIZE=1024

# Validate paths
if [[ "$MODEL_PATH" == "/path/to/"* ]]; then
    echo "⚠ MODEL_PATH is not configured!"
    echo "  Edit .env and set MODEL_PATH to your GGUF model file."
    exit 1
fi

if [[ "$LLAMA_SERVER" == "/path/to/"* ]]; then
    echo "⚠ LLAMA_SERVER is not configured!"
    echo "  Edit .env and set LLAMA_SERVER to your llama-server binary."
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "✗ Model file not found: $MODEL_PATH"
    exit 1
fi

if [ ! -x "$LLAMA_SERVER" ]; then
    echo "✗ llama-server not found or not executable: $LLAMA_SERVER"
    exit 1
fi

# Dynamic VRAM Layer Offloading
if command -v nvidia-smi &>/dev/null; then
    VRAM_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1)
    RESERVED_MB=800
    AVAILABLE=$((VRAM_MB - RESERVED_MB))
    LAYER_SIZE_MB=140
    MAX_LAYERS=32
    OPTIMAL=$((AVAILABLE / LAYER_SIZE_MB))
    FINAL_NGL=$((OPTIMAL < MAX_LAYERS ? OPTIMAL : MAX_LAYERS))
    if [ "$FINAL_NGL" -lt 0 ]; then FINAL_NGL=0; fi
else
    VRAM_MB="N/A"
    FINAL_NGL=32
fi

echo "╔════════════════════════════════════════════════════════════╗"
echo "║              ENML — Llama.cpp Server                      ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo "  Model:   $(basename "$MODEL_PATH")"
echo "  URL:     http://localhost:$PORT"
echo "  VRAM:    ${VRAM_MB}MB free | GPU Layers: $FINAL_NGL"
echo ""

"$LLAMA_SERVER" \
    -m "$MODEL_PATH" \
    -c "$CONTEXT_SIZE" \
    -ngl "$FINAL_NGL" \
    -b "$BATCH_SIZE" \
    --parallel 1 \
    --mlock \
    --flash-attn on \
    --defrag-thold 0.1 \
    --metrics \
    --port "$PORT" \
    --host "$HOST" \
    --temp 0.6 \
    --top-k 40 \
    --top-p 0.9
