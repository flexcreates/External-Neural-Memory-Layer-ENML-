# Universal Memory System

This directory contains the universal memory system for local AI models.

## Setup

1.  **Initialize Virtual Environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

2.  **Configuration:**
    Copy `.env.example` to `.env` and configure your settings.

## Usage

### 1. Initialize Memory Manager
```python
from src.memory_manager import MemoryManager

# Initialize (creates directories if needed)
memory = MemoryManager()
```

### 2. Save a Session
Log a full conversation for history.
```python
session_id = "session_123"
messages = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"}
]
memory.save_session(session_id, messages)
```

### 3. Add Long-Term Memory
Store specific facts or summaries for retrieval.
```python
memory.add_memory(
    "User's name is Flex.",
    metadata={"source": "session_123"}
)
```

### 4. Retrieve Context
Get relevant information based on a query.
```python
results = memory.retrieve_context("What is my name?")
print(results)
# Output: ["User's name is Flex."]
```

## Testing
Run the demo script:
```bash
python3 demo.py
```

Run unit tests:
```bash
pytest
```
