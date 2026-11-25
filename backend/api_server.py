"""
api_server.py
-------------
API FastAPI exposant :
  - le mode conversationnel RAG (/ask)
  - les indicateurs d'analyse statistique du corpus et de l'index FAISS (/analytics/*)

LANCEMENT :
  uvicorn backend.api_server:app --reload --port 8000
"""

import os
import re
import sys
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BACKEND_DIR)
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from e_rag_pipeline import (
    NB_CHUNKS,
    NOM_MODELE_EMB,
    NOM_MODELE_OPENAI,
    PROMPT_TEMPLATE,
    appeler_openai,
    charger_ressources,
    construire_contexte,
    initialiser_openai,
    rechercher_chunks,
)
from g_corpus_analytics import analyser_corpus_et_index


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question posée au système RAG")
    top_k: int = Field(default=NB_CHUNKS, ge=1, le=15, description="Nombre de chunks à récupérer")


class SourceItem(BaseModel):
    rang: int
    chunk_id: str
    source_id: str
    type: str
    score: float
    extrait: str
    url: str | None = None
    source_ref: str | None = None


class AskResponse(BaseModel):
    question: str
    reponse: str
    top_k: int
    temps_retrieval_s: float
    temps_generation_s: float
    temps_total_s: float
    sources: list[SourceItem]


state: dict = {
    "client": None,
    "documents": None,
    "index": None,
    "modele": None,
    "analytics_cache": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge les ressources RAG et statistiques au démarrage de l'API."""
    client = initialiser_openai()
    documents, index, modele = charger_ressources()
    analytics = analyser_corpus_et_index(documents=documents)

    state["client"] = client
    state["documents"] = documents
    state["index"] = index
    state["modele"] = modele
    state["analytics_cache"] = analytics
    yield


app = FastAPI(
    title="Finances Publiques RAG & Analytics API",
    description="API FastAPI pour la recherche RAG conversationnelle et l'analyse statistique du corpus.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    """Vérifie l'état de l'API et des ressources chargées."""
    docs = state.get("documents") or []
    idx = state.get("index")
    return {
        "status": "ok",
        "modele_llm": NOM_MODELE_OPENAI,
        "modele_embedding": NOM_MODELE_EMB,
        "nb_chunks": len(docs),
        "nb_vecteurs_faiss": int(idx.ntotal) if idx is not None else 0,
        "default_top_k": NB_CHUNKS,
    }


def _extraire_lien_source(doc: dict) -> tuple[str | None, str]:
    """Retourne (url, source_ref) à partir des métadonnées disponibles du chunk."""
    url = None
    for champ_url in ("url", "source_url", "lien", "link", "href"):
        val = doc.get(champ_url)
        if isinstance(val, str) and val.strip().startswith(("http://", "https://")):
            url = val.strip()
            break

    source_ref = ""
    for champ_ref in ("source_file", "filename", "fichier", "document", "pdf", "docx", "source_name", "source_id"):
        val = doc.get(champ_ref)
        if isinstance(val, str) and val.strip():
            source_ref = val.strip()
            break

    if not source_ref:
        source_ref = str(doc.get("chunk_id", "Source"))
    return url, source_ref


def _formater_reponse_avec_sources(reponse: str, chunks: list[tuple[dict, float]]) -> str:
    """Remplace les citations [CHUNK-ID] par l'URL ou la référence fichier (PDF/DOCX/source) à côté de la réponse."""
    import re

    mapping: dict[str, str] = {}
    top_source_cible = ""
    for i, (doc, _) in enumerate(chunks):
        cid = str(doc.get("chunk_id", "")).strip()
        url, sref = _extraire_lien_source(doc)
        cible = url if url else sref
        if i == 0 and cible:
            top_source_cible = cible
        if cid and cible:
            mapping[cid] = cible

    replaced_any = False

    def _repl(match):
        nonlocal replaced_any
        raw_inner = match.group(1)
        tokens = [t.strip() for t in re.split(r"[,;]", raw_inner) if t.strip()]
        resolved = []
        for tok in tokens:
            if tok in mapping:
                resolved.append(mapping[tok])
                replaced_any = True
            elif re.fullmatch(r"DOC-\d+-C\d+", tok, flags=re.IGNORECASE):
                if top_source_cible:
                    resolved.append(top_source_cible)
                    replaced_any = True
            else:
                resolved.append(tok)
        if not resolved:
            return f" ({top_source_cible})" if top_source_cible else ""
        seen = []
        for item in resolved:
            if item not in seen:
                seen.append(item)
        return f" ({', '.join(seen)})"

    cleaned = re.sub(r"\[([^\]]+)\]", _repl, reponse)
    cleaned = re.sub(r"\s+([.,;:])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()

    # Pas de lien ajouté d'office : une réponse sans citation n'a pas de source.
    return cleaned


@app.post("/ask", response_model=AskResponse)
def ask_question(payload: AskRequest):
    """Exécute la recherche FAISS + génération LLM et retourne la réponse et les sources."""
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="La question ne peut pas être vide.")

    client: Optional[object] = state.get("client")
    documents = state.get("documents")
    index = state.get("index")
    modele = state.get("modele")

    if not all([client, documents is not None, index is not None, modele is not None]):
        raise HTTPException(status_code=503, detail="Ressources RAG non initialisées.")

    t0_ret = time.perf_counter()
    chunks = rechercher_chunks(question, documents, index, modele, k=payload.top_k)
    temps_ret = time.perf_counter() - t0_ret

    if not chunks:
        raise HTTPException(
            status_code=404,
            detail="Aucun chunk pertinent trouvé au-dessus du seuil de similarité.",
        )

    contexte = construire_contexte(chunks)
    prompt = PROMPT_TEMPLATE.format(contexte=contexte, question=question)

    t0_gen = time.perf_counter()
    reponse = appeler_openai(prompt, client)
    temps_gen = time.perf_counter() - t0_gen

    if not reponse:
        raise HTTPException(status_code=502, detail="Échec de génération auprès du modèle OpenAI.")

    reponse_formatee = _formater_reponse_avec_sources(reponse, chunks)

    # Chunks affichés seulement si la réponse cite au moins un chunk ([DOC-xxx-Cxxx]).
    # Si le contexte ne contient pas la réponse, aucune source n'est renvoyée.
    reponse_citee = bool(re.search(r"DOC-\d+-C\d+", reponse, flags=re.IGNORECASE))
    chunks_affiches = chunks[:3] if reponse_citee else []

    sources_formatees: list[SourceItem] = []
    for rang, (doc, score) in enumerate(chunks_affiches, start=1):
        contenu = doc.get("content", "")
        extrait = contenu[:320] + ("..." if len(contenu) > 320 else "")
        url, sref = _extraire_lien_source(doc)
        sources_formatees.append(
            SourceItem(
                rang=rang,
                chunk_id=str(doc.get("chunk_id", "N/A")),
                source_id=str(doc.get("source_id", "N/A")),
                type=str(doc.get("type", "text")),
                score=round(float(score), 4),
                extrait=extrait,
                url=url,
                source_ref=sref,
            )
        )

    return AskResponse(
        question=question,
        reponse=reponse_formatee,
        top_k=payload.top_k,
        temps_retrieval_s=round(temps_ret, 3),
        temps_generation_s=round(temps_gen, 3),
        temps_total_s=round(temps_ret + temps_gen, 3),
        sources=sources_formatees,
    )


@app.get("/analytics/overview")
def analytics_overview():
    """Retourne les KPIs, formats, types, distribution des tailles et index FAISS."""
    if state.get("analytics_cache") is None:
        state["analytics_cache"] = analyser_corpus_et_index(state.get("documents"))
    data = state["analytics_cache"]
    return {
        "kpis": data["kpis"],
        "repartition_formats": data["repartition_formats"],
        "repartition_types": data["repartition_types"],
        "histogramme_taille_chunks": data["histogramme_taille_chunks"],
        "faiss_index": data["faiss_index"],
    }


@app.get("/analytics/keywords")
def analytics_keywords():
    """Retourne la fréquence des mots-clés stratégiques dans le corpus."""
    if state.get("analytics_cache") is None:
        state["analytics_cache"] = analyser_corpus_et_index(state.get("documents"))
    return {"frequence_mots_cles": state["analytics_cache"]["frequence_mots_cles"]}


@app.get("/analytics/documents")
def analytics_documents():
    """Retourne les documents les plus représentés en nombre de chunks."""
    if state.get("analytics_cache") is None:
        state["analytics_cache"] = analyser_corpus_et_index(state.get("documents"))
    return {"top_documents": state["analytics_cache"]["top_documents"]}


@app.get("/analytics/ragas")
def analytics_ragas():
    """Retourne le dernier rapport d'évaluation RAGAS depuis artifacts/evaluation_report.json."""
    import json

    rapport_path = os.path.join(ROOT_DIR, "artifacts", "evaluation_report.json")
    if not os.path.exists(rapport_path):
        raise HTTPException(
            status_code=404,
            detail="Rapport RAGAS introuvable (artifacts/evaluation_report.json).",
        )
    with open(rapport_path, "r", encoding="utf-8") as f:
        return json.load(f)