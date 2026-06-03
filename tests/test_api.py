# tests/test_api.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from unittest.mock import patch
from fastapi.testclient import TestClient

MOCK = {
    "question"  : "Quel est le PIB du Burundi ?",
    "answer"    : "Le PIB du Burundi est d'environ 3,5 milliards USD.",
    "sources"   : ["world_bank_burundi.pdf"],
    "nb_sources": 1,
}


def get_client():
    with patch("rag_chain.get_chain"), patch("rag_chain.answer", return_value=MOCK):
        from api import app
        return TestClient(app)


def test_root():
    client = get_client()
    r = client.get("/")
    assert r.status_code == 200
    assert "service" in r.json()


def test_health():
    client = get_client()
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ask_valide():
    client = get_client()
    r = client.post("/ask", json={"question": "Quel est le PIB du Burundi ?"})
    assert r.status_code == 200
    data = r.json()
    assert "question"   in data
    assert "answer"     in data
    assert "sources"    in data
    assert "nb_sources" in data


def test_ask_question_courte():
    client = get_client()
    r = client.post("/ask", json={"question": "ok"})
    assert r.status_code == 422


def test_ask_question_vide():
    client = get_client()
    r = client.post("/ask", json={"question": ""})
    assert r.status_code == 422