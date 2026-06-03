# src/vectorstore.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from config import CHROMA_DIR, EMBEDDING_MODEL

_vectorstore = None


def get_vectorstore() -> Chroma:
    """Charge le vectorstore ChromaDB (singleton)"""
    global _vectorstore

    if _vectorstore is None:
        if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
            raise RuntimeError(
                "\n❌ Vectorstore introuvable !\n"
                "Lance d'abord : python src/ingest.py\n"
            )

        print("  Chargement du vectorstore ChromaDB...")
        embeddings = HuggingFaceEmbeddings(
            model_name    = EMBEDDING_MODEL,
            model_kwargs  = {"device": "cpu"},
            encode_kwargs = {"normalize_embeddings": True},
        )
        _vectorstore = Chroma(
            persist_directory  = str(CHROMA_DIR),
            embedding_function = embeddings,
        )
        count = _vectorstore._collection.count()
        print(f"  ✓ {count} chunks disponibles")

    return _vectorstore