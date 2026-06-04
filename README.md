# 🇧🇮 Burundi RAG Chatbot

> Système de questions-réponses intelligent sur le Burundi basé sur une architecture **RAG (Retrieval-Augmented Generation)**.

Les réponses sont générées par **Claude (Anthropic)** à partir de rapports officiels (Banque Mondiale, PNUD, OMS, INSBU, FMI). Le système retrouve automatiquement les passages les plus pertinents dans les documents et génère une réponse en français sous forme de phrases naturelles.

---

## 📋 Table des matières

- [À propos du projet](#-à-propos-du-projet)
- [Architecture](#-architecture)
- [Technologies utilisées](#-technologies-utilisées)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Ajouter de nouveaux documents](#-ajouter-de-nouveaux-documents)
- [Sources de données](#-sources-de-données)
- [Tests](#-tests)
- [Questions de test recommandées](#-questions-de-test-recommandées)
- [Paramètres de configuration](#-paramètres-de-configuration)
- [Auteur](#-auteur)

---

## 🎯 À propos du projet

Ce projet vise à construire un système de questions-réponses intelligent sur le Burundi en exploitant des rapports officiels (Banque Mondiale, PNUD, OMS, INSBU, FMI). L'objectif est de permettre à tout utilisateur de poser des questions sur le Burundi en français et d'obtenir des réponses précises, sourcées et formulées en phrases naturelles.

**Problématique**

Comment rendre accessibles et exploitables des informations complexes contenues dans des rapports officiels volumineux sur le Burundi, grâce à l'Intelligence Artificielle ?

**Démonstration**

```bash
POST /ask
{
  "question": "Quel est le PIB du Burundi ?"
}
```

```json
{
  "question": "Quel est le PIB du Burundi ?",
  "answer": "Le PIB par habitant du Burundi s'élève à 254 USD en 2023. Le taux de croissance du PIB réel était de 3,3% en 2023 et devrait atteindre 3,9% en 2024 selon les projections disponibles.",
  "sources": ["burundi_cfr_2025.pdf"],
  "nb_sources": 1
}
```

---

## 🏗️ Architecture

```
Rapports officiels PDF
(Banque Mondiale · PNUD · OMS · INSBU · FMI)
            ↓
    PyPDFLoader
    (extraction du texte)
            ↓
    RecursiveCharacterTextSplitter
    (chunks de 512 caractères, overlap 64)
            ↓
    sentence-transformers
    paraphrase-multilingual-MiniLM-L12-v2
    (embeddings vectoriels multilingues)
            ↓
    ChromaDB
    (base de données vectorielle persistée)
            ↓
    LangChain LCEL
    (récupère les 4 chunks les plus pertinents)
            ↓
    Claude Haiku — Anthropic
    (génère la réponse en français)
            ↓
    FastAPI /ask
    (API REST documentée Swagger)
```

**Structure du projet**

```
burundi-rag-chatbot/
├── data/
│   ├── raw/                  # PDFs sources (non versionné)
│   ├── chroma_db/            # Vectorstore persisté (non versionné)
│   └── indexed_files.txt     # Suivi des fichiers indexés
├── src/
│   ├── __init__.py
│   ├── config.py             # Configuration centralisée
│   ├── ingest.py             # Pipeline ingestion PDF → ChromaDB
│   ├── vectorstore.py        # Chargement ChromaDB (singleton)
│   ├── rag_chain.py          # Pipeline RAG LangChain + Claude
│   └── api.py                # API FastAPI
├── tests/
│   ├── __init__.py
│   └── test_api.py           # Tests des endpoints
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🛠️ Technologies utilisées

**Intelligence Artificielle & RAG**

1. LangChain LCEL — Orchestration du pipeline RAG
2. ChromaDB — Base de données vectorielle persistée
3. sentence-transformers — Modèle d'embeddings multilingue (français + anglais)
4. Claude Haiku (Anthropic) — Modèle de langage pour la génération des réponses

**Data Engineering**

1. PyPDFLoader — Extraction du texte depuis les PDFs
2. RecursiveCharacterTextSplitter — Découpage intelligent des documents

**Déploiement**

1. FastAPI — API REST documentée
2. Uvicorn — Serveur ASGI
3. Pydantic — Validation des données

**Tests & Outils**

1. Pytest — Tests des endpoints
2. uv — Gestionnaire de packages
3. Git — Versioning

**Installer les dépendances**

```bash
uv pip install -r requirements.txt
```

Ou manuellement :

```bash
uv pip install langchain langchain-community langchain-anthropic langchain-huggingface
uv pip install chromadb sentence-transformers pypdf
uv pip install fastapi uvicorn python-dotenv pydantic pytest httpx
```

---

## 🚀 Installation

**Prérequis**

- Python 3.10 ou 3.11
- uv (gestionnaire de packages) — https://github.com/astral-sh/uv
- Compte Anthropic (clé API) — https://console.anthropic.com
- Git

**Étapes**

```bash
# 1. Cloner le projet
git clone https://github.com/bigirimanainnocent12/burundi-rag-chatbot
cd burundi-rag-chatbot

# 2. Créer l'environnement virtuel
uv venv llmvenv
source llmvenv/bin/activate     # Mac/Linux
llmvenv\Scripts\activate        # Windows

# 3. Installer les dépendances
export UV_LINK_MODE=copy
uv pip install -r requirements.txt

# 4. Configurer la clé API
cp .env.example .env
# Edite .env et ajoute ta clé Anthropic :
# ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxxxxxx

# 5. Ajouter des PDFs dans data/raw/

# 6. Lancer l'ingestion
python src/ingest.py

# 7. Lancer l'API
uvicorn src.api:app --reload --port 8000 --host 0.0.0.0
```

Le script d'ingestion effectue :

✅ Chargement des PDFs depuis data/raw/

✅ Découpage en chunks (512 caractères, overlap 64)

✅ Encodage en embeddings vectoriels multilingues

✅ Stockage dans ChromaDB (persisté sur disque)

✅ Détection automatique des nouveaux fichiers (ingestion incrémentale)

---

## 📖 Utilisation

**1. Interface Swagger (recommandé)**

Ouvre http://localhost:8000/docs dans ton navigateur.

**2. Requête cURL**

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Quels sont les défis économiques du Burundi ?"}'
```

**3. Endpoints disponibles**

| Méthode | Endpoint | Description |
|---|---|---|
| GET | `/` | Informations sur l'API |
| GET | `/health` | Vérification de l'état de l'API |
| POST | `/ask` | Poser une question sur le Burundi |

**Documentation interactive**

- Swagger UI : http://localhost:8000/docs
- ReDoc : http://localhost:8000/redoc

---

## 📂 Ajouter de nouveaux documents

Il suffit de placer les nouveaux PDFs dans `data/raw/` et de relancer l'ingestion. Le système détecte automatiquement les nouveaux fichiers et n'indexe que ceux-ci — les documents déjà indexés ne sont pas retraités.

```bash
# Ajoute tes PDFs dans data/raw/
python src/ingest.py
```

Exemple de sortie :

```
1 nouveau(x) PDF(s) détecté(s) :
   - nouveau_rapport_burundi.pdf

✅ Ingestion terminée — 1 nouveau fichier indexé
   Total dans ChromaDB : 895 chunks
```

---

## 🌍 Sources de données

| Source | URL |
|---|---|
| Banque Mondiale — Burundi | https://documents.worldbank.org |
| PNUD Burundi | https://www.undp.org/burundi/publications |
| FMI — Article IV Burundi | https://www.imf.org |
| OMS Burundi | https://www.afro.who.int/countries/burundi |
| INSBU | https://www.isteebu.bi |

---

## ✅ Tests

```bash
pytest tests/ -v
```

---

## 💬 Questions de test recommandées

- *"Quel est le PIB du Burundi ?"*
- *"Quelle est la population du Burundi ?"*
- *"Quels sont les principaux défis économiques du Burundi ?"*
- *"Quel est le taux de mortalité infantile au Burundi ?"*
- *"Quels sont les indicateurs de santé au Burundi en 2024 ?"*
- *"Quels sont les objectifs de la vision Burundi 2024 ?"*
- *"Quel est le taux de pénétration mobile au Burundi ?"*

---

## ⚙️ Paramètres de configuration

Tous les paramètres sont centralisés dans `src/config.py` :

| Paramètre | Valeur par défaut | Description |
|---|---|---|
| `EMBEDDING_MODEL` | paraphrase-multilingual-MiniLM-L12-v2 | Modèle d'embeddings |
| `CLAUDE_MODEL` | claude-haiku-4-5-20251001 | Modèle Claude utilisé |
| `CHUNK_SIZE` | 512 | Taille des chunks en caractères |
| `CHUNK_OVERLAP` | 64 | Chevauchement entre chunks |
| `TOP_K_DOCS` | 4 | Nombre de chunks retournés |

Variables d'environnement :

| Variable | Description | Exemple |
|---|---|---|
| `ANTHROPIC_API_KEY` | Clé API Anthropic (obligatoire) | `sk-ant-xxxxx` |

---

## 👤 Auteur

**Innocent BIGIRIMANA** — Data Engineer / Data Analyst

- Portfolio : https://bigirimanainnocent12.github.io/Porfolio/
- LinkedIn : https://linkedin.com/in/bigirimana-innocent
- GitHub : https://github.com/bigirimanainnocent12

---

