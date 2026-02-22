#!/bin/bash

# run_qdrant.sh
# Script to manage Qdrant Vector DB container lifecycle for ENML

set -e

CONTAINER_NAME="enml-qdrant"
PORT=6333
STORAGE_DIR="$(pwd)/qdrant_storage"

echo "=== ENML Qdrant Startup Script ==="

# 1. Create storage directory if missing
if [ ! -d "$STORAGE_DIR" ]; then
    echo "[Info] Creating storage directory at $STORAGE_DIR"
    mkdir -p "$STORAGE_DIR"
fi

# 2. Check if a container with our name already exists
if sudo docker ps -a --format '{{.Names}}' | grep -Eq "^${CONTAINER_NAME}\$"; then
    echo "[Info] Container '$CONTAINER_NAME' already exists."

    # Check if it's currently running
    if sudo docker ps --format '{{.Names}}' | grep -Eq "^${CONTAINER_NAME}\$"; then
        echo "[Success] Qdrant is already running on port $PORT."
        exit 0
    else
        echo "[Info] Container exists but is stopped. Starting it now..."
        sudo docker start "$CONTAINER_NAME"
        echo "[Success] Qdrant started on port $PORT."
        exit 0
    fi
fi

# 3. Handle Port Conflicts (if something ELSE is using port 6333)
if sudo lsof -i :$PORT > /dev/null; then
    echo "[Warning] Port $PORT is already in use by another process."
    echo "[Action] Attempting to find and stop the conflicting Docker container..."
    
    CONFLICTING_CONTAINER=$(sudo docker ps -q --filter "publish=$PORT")
    
    if [ -n "$CONFLICTING_CONTAINER" ]; then
        echo "[Info] Stopping conflicting container: $CONFLICTING_CONTAINER"
        sudo docker stop "$CONFLICTING_CONTAINER"
        sudo docker rm "$CONFLICTING_CONTAINER"
    else
        echo "[Error] Port $PORT is in use by a NON-Docker process. Please close it manually."
        exit 1
    fi
fi

# 4. Start fresh container
echo "[Info] Starting fresh Qdrant container..."
sudo docker run -d \
  --name "$CONTAINER_NAME" \
  -p "$PORT:6333" \
  -v "$STORAGE_DIR:/qdrant/storage" \
  qdrant/qdrant

echo "[Success] Qdrant is now running on http://localhost:$PORT"
