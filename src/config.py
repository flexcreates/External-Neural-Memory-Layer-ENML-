
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
MEMORY_DIR = Path(os.getenv("MEMORY_DIR", BASE_DIR / "data"))
SESSIONS_DIR = MEMORY_DIR / "sessions"
VECTORS_DIR = MEMORY_DIR / "vectors"

# Ensure directories exist
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
VECTORS_DIR.mkdir(parents=True, exist_ok=True)

# Model configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHROMA_DB_IMPL = os.getenv("CHROMA_DB_IMPL", "duckdb+parquet")
