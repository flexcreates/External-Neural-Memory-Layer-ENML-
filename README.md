<p align="center">
  <h1 align="center">🧠 ENML — External Neural Memory Layer</h1>
  <p align="center">
    <em>Infinite memory for local AI systems. Your AI remembers everything, forever.</em>
  </p>
  <p align="center">
    <a href="#-quick-start">Quick Start</a> •
    <a href="#-features">Features</a> •
    <a href="#-architecture">Architecture</a> •
    <a href="#-configuration">Configuration</a> •
    <a href="docs/USER_GUIDE.md">User Guide</a> •
    <a href="docs/DEVELOPMENT.md">Dev Guide</a>
  </p>
</p>

---

## What is ENML?

**ENML** gives your local LLM (Llama 3, Mistral, etc.) persistent, long-term memory that survives across sessions. It builds a personal **Knowledge Graph** from your conversations — your name, hobbies, pets, hardware specs, projects — and recalls them instantly, forever.

No cloud. No API keys. **100% local.** Your data stays on your machine.

```
Session 1:  "My name is Flex, I have a pet lizard named Colu"
            → Extracted & stored as semantic triples

Session 2:  "What's my pet's name?"
            → "Your pet lizard's name is Colu."  ✅
```

---

## ⚡ Quick Start

### Prerequisites

| Requirement | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Runtime |
| Docker | 20+ | Qdrant vector database |
| llama.cpp | Latest | Local LLM inference server |
| GGUF Model | Any | Recommended: `Meta-Llama-3-8B-Instruct.Q4_K_M.gguf` |

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/flexcreates/ENML.git
cd ENML

# 2. Run the setup script
chmod +x setup.sh
./setup.sh

# 3. Configure your environment
nano .env
# ↳ Set MODEL_PATH, LLAMA_SERVER, and AI_NAME

# 4. Start the services
./run_server.sh        # Terminal 1: Start Llama server
./run_qdrant.sh        # Terminal 2: Start Qdrant (if not started by setup)

# 5. Start chatting!
source .venv/bin/activate
python3 chat.py
```

---

## ✨ Features

### 🧠 Real-Time Fact Extraction
Every message is analyzed by the LLM to extract semantic triples:
- `user has_name Flex` · `user has_pet lizard` · `user has_interest vibe-coding`
- Multi-value support: hobbies, pets, and projects stack without overwriting

### 🔍 Smart Query Routing
The context builder searches Qdrant for relevant memories and injects them into the system prompt — so the AI always has context about you.

### 🛡️ Identity Separation
AI identity and user identity are stored separately:
- `"My name is Flex"` → stored as `user has_name Flex` (Knowledge Graph)
- `"You are Jarvis"` → stored as `assistant.name = Jarvis` (Authority Memory)
- No collisions between user and AI identity

### 📊 Knowledge Graph
Facts are stored as semantic triples with contradiction detection:
- `user has_name Flex` (active) → `user has_name David` (supersedes old value)
- Multi-value predicates (hobbies, pets) allow multiple active values

### 🧹 Intelligent Filtering
- **Question pre-check**: Questions and commands skip extraction entirely
- **Noise filter**: Greetings and filler are rejected before storage
- **Name guard**: Device/brand names can't overwrite your identity
- **Predicate normalization**: `uses Ubuntu` → `uses_os Ubuntu` (prevents collisions)

### 🔒 100% Local & Private
- All processing happens on your machine
- No cloud APIs, no telemetry, no data leaves your system
- Qdrant runs in a local Docker container

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        chat.py (CLI)                        │
├─────────────────────────────────────────────────────────────┤
│                   core/orchestrator.py                      │
│        ┌──────────┬───────────────┬───────────────┐         │
│        │ Extractor│ MemoryManager │ ContextBuilder│         │
│        │(LLM-based│ (routes facts │ (builds prompt│         │
│        │ fact     │  to correct   │  with memory  │         │
│        │ mining)  │  storage)     │  context)     │         │
│        └────┬─────┴──────┬────────┴──────┬────────┘         │
│             │            │               │                  │
│   ┌─────────┴──┐  ┌──────┴─────┐  ┌──────┴──────┐           │
│   │ Knowledge  │  │ Authority  │  │   Qdrant    │           │
│   │   Graph    │  │  Memory    │  │  (Vector    │           │
│   │(Triples +  │  │ (JSON for  │  │   Search)   │           │
│   │Enrichment) │  │ AI name/   │  │             │           │
│   │            │  │ role)      │  │             │           │
│   └────────────┘  └────────────┘  └─────────────┘           │
├─────────────────────────────────────────────────────────────┤
│  Llama.cpp Server (localhost:8080) │ Qdrant (localhost:6333)│
└─────────────────────────────────────────────────────────────┘
```

### Core Modules

| Module | Purpose |
|---|---|
| `core/orchestrator.py` | Main pipeline: extract → store → build context → stream response |
| `core/memory/extractor.py` | LLM-based fact extraction with regex safety nets |
| `core/memory_manager.py` | Routes facts to Knowledge Graph, Authority Memory, or Qdrant |
| `core/knowledge_graph.py` | Semantic triple storage with contradiction detection |
| `core/memory/authority_memory.py` | Deterministic JSON profile for AI and user identity |
| `core/context_builder.py` | Builds LLM prompt with memory context injection |
| `core/vector/retriever.py` | Qdrant semantic search with re-ranking |
| `core/vector/embeddings.py` | Thread-safe singleton embedding service |
| `core/vector/qdrant_client.py` | Thread-safe singleton Qdrant connection manager |

---

## ⚙️ Configuration

All configuration is managed through `.env`. No hardcoded values.

### Key Settings

```bash
# Your AI's name
AI_NAME=Jarvis

# LLM model and server
MODEL_PATH=/path/to/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf
LLAMA_SERVER=/path/to/llama.cpp/build/bin/llama-server
LLAMA_SERVER_URL=http://localhost:8080

# Qdrant vector database
QDRANT_URL=http://localhost:6333

# Embedding model
EMBED_MODEL=all-MiniLM-L6-v2
EMBED_DIM=384
```

See [`.env.example`](.env.example) for the complete configuration reference.

---

## 📁 Project Structure

```
ENML/
├── chat.py                 # CLI chat interface
├── setup.sh                # One-command installer
├── run_server.sh           # Llama.cpp server launcher
├── run_qdrant.sh           # Qdrant Docker manager
├── reset_memory.sh         # Memory wipe utility
├── requirements.txt        # Python dependencies
├── .env.example            # Configuration template
│
├── core/                   # Core engine
│   ├── config.py           # .env loader
│   ├── orchestrator.py     # Main pipeline
│   ├── memory_manager.py   # Fact routing
│   ├── context_builder.py  # Prompt builder
│   ├── knowledge_graph.py  # Triple store
│   ├── logger.py           # Logging system
│   ├── memory/             # Memory subsystem
│   │   ├── extractor.py    # LLM fact extraction
│   │   ├── authority_memory.py  # JSON identity store
│   │   └── triple_memory.py     # Triple data class
│   ├── vector/             # Vector subsystem
│   │   ├── retriever.py    # Qdrant search
│   │   ├── embeddings.py   # Embedding service
│   │   └── qdrant_client.py # Connection manager
│   ├── router/             # Query routing
│   │   └── query_router.py
│   └── storage/            # Session storage
│       └── json_storage.py
│
├── research/               # Web research module
│   └── web_ingestor.py
├── tools/                  # File operations
│   └── file_tool.py
├── docs/                   # Documentation
│   ├── USER_GUIDE.md
│   ├── ARCHITECTURE.md
│   ├── DEVELOPMENT.md
│   └── WEB_CONNECTIVITY.md
│
├── ingest_conversation.py  # Batch conversation import
├── ingest_project.py       # Project codebase import
└── ingest_research.py      # Research data import
```

---

## 🧪 Chat Commands

| Command | Description |
|---|---|
| `/remember <text>` | Manually save a fact to memory |
| `exit` | End the session and save conversation |
| `Ctrl+C` | Force quit (session still saves) |

---

## 🔧 Shell Scripts

| Script | What it does |
|---|---|
| `setup.sh` | Complete installation: venv, deps, .env, dirs, Qdrant |
| `run_server.sh` | Starts llama-server with VRAM-optimized GPU layers |
| `run_qdrant.sh` | Manages Qdrant Docker container lifecycle |
| `reset_memory.sh` | Wipes all memory, sessions, and vector collections |

---

## 📖 Documentation

| Document | Description |
|---|---|
| [User Guide](docs/USER_GUIDE.md) | Step-by-step usage workflows |
| [Architecture](docs/ARCHITECTURE.md) | System design & data flow diagrams |
| [Development Guide](docs/DEVELOPMENT.md) | How to extend ENML |
| [Web Connectivity](docs/WEB_CONNECTIVITY.md) | Internet research integration |

---

## 🛣️ Roadmap

- [x] Real-time fact extraction from conversations
- [x] Persistent knowledge graph with contradiction detection
- [x] Multi-value predicate support (hobbies, pets, projects)
- [x] AI/user identity separation
- [x] Question/command pre-filtering
- [x] Predicate normalization
- [ ] Web research ingestion pipeline
- [ ] Multi-modal memory (images, documents)
- [ ] Conversation summarization
- [ ] Memory decay and confidence aging
- [ ] Plugin system for custom extractors

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

**Created by [Flex](https://github.com/flexcreates)** · 2024–2026

---

<p align="center">
  <em>ENML — Because your AI should remember who you are.</em>
</p>
