# src/rag_chain.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from langchain_anthropic import ChatAnthropic
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from vectorstore import get_vectorstore
from config import CLAUDE_MODEL, ANTHROPIC_API_KEY, TOP_K_DOCS

PROMPT_TEMPLATE = """Tu es un assistant expert sur le Burundi, \
spécialisé dans les données économiques, démographiques, sociales \
et du secteur des télécommunications.

Tu aides les utilisateurs à trouver des informations précises \
à partir de rapports officiels (Banque Mondiale, PNUD, OMS, INSBU, FMI).

Règles strictes :
1. Réponds UNIQUEMENT en te basant sur le contexte fourni ci-dessous.
2. Si la réponse n'est pas dans le contexte, réponds exactement :
   "Je ne trouve pas cette information dans les documents disponibles."
3. Réponds toujours en français, de façon claire et structurée.
4. Cite les chiffres et statistiques précis quand ils sont disponibles.
5. Si pertinent, mentionne l'année ou la source de la donnée.

Contexte :
{context}

Question : {question}

Réponse :"""

_chain = None


def get_chain() -> RetrievalQA:
    """Construit la chaîne RAG (singleton)"""
    global _chain

    if _chain is None:
        print("  Initialisation de la chaîne RAG...")

        # Retriever
        vectorstore = get_vectorstore()
        retriever   = vectorstore.as_retriever(
            search_type   = "similarity",
            search_kwargs = {"k": TOP_K_DOCS},
        )

        # LLM Claude
        llm = ChatAnthropic(
            model       = CLAUDE_MODEL,
            temperature = 0.1,
            max_tokens  = 1024,
            api_key     = ANTHROPIC_API_KEY,
        )

        # Prompt
        prompt = PromptTemplate(
            template        = PROMPT_TEMPLATE,
            input_variables = ["context", "question"],
        )

        # Chaîne complète
        _chain = RetrievalQA.from_chain_type(
            llm                     = llm,
            chain_type              = "stuff",
            retriever               = retriever,
            return_source_documents = True,
            chain_type_kwargs       = {"prompt": prompt},
        )
        print("  ✓ Chaîne RAG prête")

    return _chain


def answer(question: str) -> dict:
    """Répond à une question via le pipeline RAG"""
    chain  = get_chain()
    result = chain.invoke({"query": question})

    sources = list({
        doc.metadata.get("source", "source inconnue")
        for doc in result.get("source_documents", [])
    })

    return {
        "question"  : question,
        "answer"    : result["result"].strip(),
        "sources"   : sorted(sources),
        "nb_sources": len(sources),
    }