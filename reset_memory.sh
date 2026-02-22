#!/bin/bash

# Configuration
MEMORY_DIR="memory"

echo "⚠️  WARNING: This will delete ALL memory, sessions, Authority JSON, and Qdrant Collections."
echo "     - $MEMORY_DIR/conversations/*"
echo "     - $MEMORY_DIR/authority/profile.json"
echo "     - Qdrant Vector Collections"
echo ""
read -p "Are you sure you want to proceed? (y/N): " confirm

if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "Aborted."
    exit 0
fi

echo "Cleaning up..."

if [ -d "$MEMORY_DIR/conversations" ]; then
    rm -rf "$MEMORY_DIR/conversations"/*
    echo "✔ Cleared ENML Conversations"
fi

if [ -d "$MEMORY_DIR/projects" ]; then
    rm -rf "$MEMORY_DIR/projects"/*
    echo "✔ Cleared Projects"
fi

if [ -d "$MEMORY_DIR/research" ]; then
    rm -rf "$MEMORY_DIR/research"/*
    echo "✔ Cleared Research Data"
fi

# Reset Authority JSON
AUTHORITY_FILE="$MEMORY_DIR/authority/profile.json"
if [ -f "$AUTHORITY_FILE" ]; then
    python3 -c "
import json
clean_profile = {
  'identity': { 'legal_name': None, 'preferred_name': None },
  'assistant': { 'name': 'Jarvis' },
  'system': { 'device': None, 'cpu': None, 'gpu': None, 'os': None, 'full_specs': None },
  'medical': [],
  'personal': { 'body_type': None },
  'family': { 'father': None, 'mother': None },
  'projects': [],
  'interests': [],
  'dislikes': []
}
with open('$AUTHORITY_FILE', 'w') as f:
    json.dump(clean_profile, f, indent=2)
"
    echo "✔ Resetted Authority Profile"
fi

# Reset Qdrant Collections
echo "Resetting Qdrant Collections..."
.venv/bin/python3 -c "
try:
    from qdrant_client import QdrantClient
    client = QdrantClient(url='http://localhost:6333')
    collections = ['research_collection', 'project_collection', 'conversation_collection', 'profile_collection', 'knowledge_collection']
    
    for c in collections:
        try:
            client.delete_collection(collection_name=c)
            print(f'  - Dropped {c}')
        except Exception as e:
            pass
except ImportError:
    print('  - Qdrant Client not installed. Skipping.')
except Exception as e:
    print(f'  - Qdrant connection omitted: {e}')
"
echo "✔ Cleared Vector Memory (Qdrant)"

# Remove PyCache and Test Artifacts
find . -not -path "*/\.venv/*" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -not -path "*/\.venv/*" -maxdepth 1 -type f -name "test_*.py" -delete 2>/dev/null
echo "✔ Cleared Caches & Test Scrips"

echo ""
echo "✨ System Reset Complete. Infinite Learning v2.0 is now fresh."
