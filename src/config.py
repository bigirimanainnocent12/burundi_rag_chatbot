
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# Chemins
ROOT_DIR   = Path(__file__).parent.parent
DATA_DIR   = ROOT_DIR / "data" / "raw"
CHROMA_DIR = ROOT_DIR / "data" / "chroma_db"

# Modèles
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CLAUDE_MODEL    = "claude-haiku-4-5-20251001"

# Paramètres RAG
CHUNK_SIZE    = 512
CHUNK_OVERLAP = 64
TOP_K_DOCS    = 4

# Clé API
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

if not ANTHROPIC_API_KEY:
    raise ValueError(
        "ANTHROPIC_API_KEY manquante. "
        "Vérifie ton fichier .env"
    )