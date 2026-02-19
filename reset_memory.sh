#!/bin/bash

# Configuration
DATA_DIR="./data"
LOGS_DIR="./logs"

echo "⚠️  WARNING: This will delete ALL memory, sessions, and logs."
echo "     - $DATA_DIR/sessions/*"
echo "     - $DATA_DIR/vectors/*"
echo "     - $LOGS_DIR/*"
echo ""
read -p "Are you sure you want to proceed? (y/N): " confirm

if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "Aborted."
    exit 0
fi

echo "Cleaning up..."

# Remove Sessions
if [ -d "$DATA_DIR/sessions" ]; then
    rm -rf "$DATA_DIR/sessions"/*
    echo "✔ Cleared Sessions"
else
    echo "- No sessions directory found"
fi

# Remove Vectors
if [ -d "$DATA_DIR/vectors" ]; then
    rm -rf "$DATA_DIR/vectors"/*
    echo "✔ Cleared Vector Memory"
else
    echo "- No vectors directory found"
fi

# Remove Logs
if [ -d "$LOGS_DIR" ]; then
    rm -rf "$LOGS_DIR"/*
    echo "✔ Cleared Logs"
else
    echo "- No logs directory found"
fi

# Remove PyCache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
echo "✔ Cleared Python Cache"

echo ""
echo "✨ System Reset Complete. It is now fresh."
