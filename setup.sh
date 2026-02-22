#!/bin/bash
# setup.sh - ENML Automated Installation

set -e

echo "=== ENML System Setup Script ==="

echo "[1] Checking Requirements..."
command -v python3 >/dev/null 2>&1 || { echo >&2 "Python 3 is required but not installed. Aborting."; exit 1; }
command -v docker >/dev/null 2>&1 || { echo >&2 "Docker is required for Qdrant but not installed. Aborting."; exit 1; }

echo "[2] Creating Python Virtual Environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "  - Created .venv successfully."
else
    echo "  - .venv already exists."
fi

echo "[3] Installing Pip Requirements..."
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

echo "[4] Setting up Environment Variables..."
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "  - Created .env from .env.example."
        echo "  ! Please review '.env' and adjust MODEL_PATH and LLAMA_SERVER locations !"
    else
        echo "  - .env.example not found. Please create .env manually."
    fi
else
    echo "  - .env already exists."
fi

echo "[5] Booting Qdrant Vector Engine..."
if [ -f "run_qdrant.sh" ]; then
    chmod +x run_qdrant.sh
    ./run_qdrant.sh
else
    echo "  - run_qdrant.sh not found. Skipping vector DB startup."
fi

echo ""
echo "=== Setup Complete! ==="
echo "Next Steps:"
echo "1. Configure your MODEL_PATH and LLAMA_SERVER in '.env'"
echo "2. Run Llama Server: ./run_server.sh"
echo "3. Join ENML Chat: .venv/bin/python3 chat.py"
