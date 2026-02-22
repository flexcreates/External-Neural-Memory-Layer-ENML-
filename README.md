# ENML - Infinite Learning v2.0 Cognitive Architecture

**ENML (Evolving Network Memory Layer)** is a locally hosted, LLM-agnostic framework that implements a purely deterministic, self-evolving **Knowledge Graph Memory** for Local AI models (like LLaMA 3). 

Instead of traditional Retrieval-Augmented Generation (RAG) which dumps raw chat histories into a vector database, ENML deploys an active "Cognitive Fact Extractor" that condenses your conversations into structured Semantic Triples (`Subject -> Predicate -> Object`) stored universally in a Qdrant Knowledge Collection, ensuring your AI learns about you infinitely with zero code schemas and zero hallucination.

## Features
- **Triple-Based Memory Engine**: Extracts user intent into strict Knowledge Graph nodes rather than massive unorganized texts.
- **Strict Grounding**: Context injection heavily limits the LLM to only respond based on the retrieved facts if asked personal or systemic questions, preventing hallucination.
- **Privacy First**: Fully self-hosted architecture using Llama.cpp and Local Qdrant Docker containers. Absolutely no data leaves your machine.
- **Multi-Layer Hierarchy**: Separates strict parameter storage (like CPU specs/Name constraints) from organic behavioral history.

## 🚀 Installation

Ensure you have **Python 3.12+** and **Docker** installed on your system.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/flexcreates/ENML.git
   cd ENML
   ```

2. **Automated Setup:**
   Run the included `setup.sh` script to configure the Python `.venv`, install requirements, generate the environment variables, and create the Qdrant container:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

3. **Configure Environment:**
   Edit the `.env` file created in your root directory and point it towards your LLaMA model and `llama-server` binaries:
   ```env
   MODEL_PATH=/path/to/models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf
   LLAMA_SERVER=/path/to/llama.cpp/build/bin/llama-server
   ```

## 🛠 Usage

1. **Start the Llama.cpp Inference Server:**
   ```bash
   ./run_server.sh
   ```

2. **Start the Infinite Learning Chat:**
   Open a new terminal window inside the `.venv` and execute:
   ```bash
   source .venv/bin/activate
   python3 chat.py
   ```
   *Try telling the bot about your family, hobbies, or PC specs, then reset your chat and ask it to recall!*

## 📖 Architecture Documentation
For deep-dive technical specifics regarding the Extractor, `MemoryTriple` framework, and Qdrant ingestion rules, refer to [core/README.md](core/README.md).

## Credits
**Author**: Flex  
**Socials**: You can find me on Discord, Instagram, and GitHub at **`flexcreates`**.

---

*This project is released under the [MIT License](LICENSE).*
