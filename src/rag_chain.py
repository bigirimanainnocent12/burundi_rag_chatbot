# src/rag_chain.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from vectorstore import get_vectorstore
from config import CLAUDE_MODEL, ANTHROPIC_API_KEY, TOP_K_DOCS

PROMPT_TEMPLATE = """Tu es un assistant expert sur le Burundi, \
spécialisé dans les données économiques, démographiques, sociales \
et du secteur des télécommunications.

Règles strictes :
1. Réponds UNIQUEMENT en te basant sur le contexte fourni.
2. Si la réponse n'est pas dans le contexte, réponds :
   "Je ne trouve pas cette information dans les documents disponibles."
3. Réponds toujours en français, de façon claire et en phrases complètes.
4. N'utilise JAMAIS de formatage Markdown (pas de **, pas de ##, pas de -, pas de *).
5. Écris uniquement des phrases fluides et naturelles.
6. Cite les chiffres précis quand ils sont disponibles en les intégrant dans les phrases.

Contexte :
{context}

Question : {question}

Réponse :"""

_chain = None


def format_docs(docs):
    """Formate les documents récupérés en texte."""
    return "\n\n".join(doc.page_content for doc in docs)


def get_chain():
    """Construit la chaîne RAG (singleton) avec LCEL."""
    global _chain

    if _chain is None:
        print("  Initialisation de la chaîne RAG...")

        vectorstore = get_vectorstore()
        retriever   = vectorstore.as_retriever(
            search_type   = "similarity",
            search_kwargs = {"k": TOP_K_DOCS},
        )

        llm = ChatAnthropic(
            model       = CLAUDE_MODEL,
            temperature = 0.1,
            max_tokens  = 1024,
            api_key     = ANTHROPIC_API_KEY,
        )

        prompt = PromptTemplate(
            template        = PROMPT_TEMPLATE,
            input_variables = ["context", "question"],
        )

        # LCEL — nouvelle syntaxe LangChain
        _chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        print("  ✓ Chaîne RAG prête")

    return _chain


def answer(question: str) -> dict:
    """Répond à une question via le pipeline RAG."""
    vectorstore = get_vectorstore()
    retriever   = vectorstore.as_retriever(
        search_type   = "similarity",
        search_kwargs = {"k": TOP_K_DOCS},
    )

    # Récupère les documents sources
    source_docs = retriever.invoke(question)

    # Génère la réponse
    chain    = get_chain()
    response = chain.invoke(question)

    sources = list({
        doc.metadata.get("source", "source inconnue")
        for doc in source_docs
    })

    return {
        "question"  : question,
        "answer"    : response.strip(),
        "sources"   : sorted(sources),
        "nb_sources": len(sources),
    }