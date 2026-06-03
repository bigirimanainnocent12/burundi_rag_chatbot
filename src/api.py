# src/api.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from rag_chain import answer, get_chain


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n🚀 Démarrage de l'API Burundi RAG Chatbot...")
    try:
        get_chain()
        print("✅ API prête !\n")
    except Exception as e:
        print(f"❌ Erreur au démarrage : {e}\n")
    yield
    print("API arrêtée.")


app = FastAPI(
    title       = "🇧🇮 Burundi RAG Chatbot API",
    description = "Questions-réponses intelligentes sur le Burundi — Claude (Anthropic) + ChromaDB",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length  = 5,
        max_length  = 500,
        description = "Question sur le Burundi (en français)",
        examples    = ["Quel est le PIB du Burundi ?"],
    )


class AnswerResponse(BaseModel):
    question   : str
    answer     : str
    sources    : list[str]
    nb_sources : int


@app.get("/", tags=["Général"])
def root():
    return {
        "service"      : "Burundi RAG Chatbot API",
        "version"      : "1.0.0",
        "documentation": "http://localhost:8000/docs",
        "status"       : "✅ opérationnel",
        "auteur"       : "Innocent BIGIRIMANA",
    }


@app.get("/health", tags=["Général"])
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AnswerResponse, tags=["RAG"])
def ask(request: QuestionRequest):
    """Pose une question sur le Burundi et obtiens une réponse sourcée."""
    try:
        result = answer(request.question)
        return AnswerResponse(**result)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))