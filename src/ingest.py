# src/ingest.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from config import DATA_DIR, CHROMA_DIR, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP


def load_pdfs() -> list:
    """Charge tous les PDFs du dossier data/raw/"""
    pdf_files = list(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"\nAucun PDF trouvé dans : {DATA_DIR}\n"
            "Télécharge des rapports sur le Burundi et place-les dans data/raw/\n"
            "Sources :\n"
            "  https://documents.worldbank.org (recherche Burundi)\n"
            "  https://www.undp.org/burundi/publications\n"
            "  https://www.imf.org (recherche Burundi Article IV)\n"
        )

    documents = []
    for pdf_path in pdf_files:
        print(f"  → Chargement : {pdf_path.name}")
        try:
            loader = PyPDFLoader(str(pdf_path))
            pages  = loader.load()
            for page in pages:
                texte = (
                    page.page_content
                    .replace("\x00", "")
                    .replace("\xa0", " ")
                    .strip()
                )
                if len(texte) > 20:
                    page.page_content       = texte
                    page.metadata["source"] = pdf_path.name
                    documents.append(page)
        except Exception as e:
            print(f"  ⚠ Erreur sur {pdf_path.name} : {e}")

    print(f"\n  ✓ {len(documents)} pages chargées depuis {len(pdf_files)} PDF(s)")
    return documents


def split_documents(documents: list) -> list:
    """Découpe les documents en chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size      = CHUNK_SIZE,
        chunk_overlap   = CHUNK_OVERLAP,
        separators      = ["\n\n", "\n", ". ", " ", ""],
        length_function = len,
    )
    chunks = splitter.split_documents(documents)
    chunks = [c for c in chunks if len(c.page_content.strip()) > 50]
    print(f"  ✓ {len(chunks)} chunks créés")
    return chunks


def create_vectorstore(chunks: list) -> Chroma:
    """Encode les chunks et stocke dans ChromaDB"""
    print(f"\n  Chargement du modèle d'embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name    = EMBEDDING_MODEL,
        model_kwargs  = {"device": "cpu"},
        encode_kwargs = {"normalize_embeddings": True},
    )
    print("  Création du vectorstore ChromaDB...")
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore = Chroma.from_documents(
        documents         = chunks,
        embedding         = embeddings,
        persist_directory = str(CHROMA_DIR),
    )
    print(f"  ✓ Vectorstore sauvegardé dans : {CHROMA_DIR}")
    return vectorstore


def run_ingestion():
    print("\n" + "=" * 55)
    print("  BURUNDI RAG CHATBOT — Pipeline d'ingestion")
    print("=" * 55)

    print("\n[1/3] Chargement des PDFs...")
    documents = load_pdfs()

    print("\n[2/3] Découpage en chunks...")
    chunks = split_documents(documents)

    print("\n[3/3] Encodage et stockage dans ChromaDB...")
    create_vectorstore(chunks)

    print("\n" + "=" * 55)
    print("  ✅ Ingestion terminée avec succès !")
    print(f"     {len(chunks)} chunks indexés dans ChromaDB")
    print("     Lance : uvicorn src.api:app --reload --port 8000")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    run_ingestion()