"""
g_corpus_analytics.py
---------------------
Module d'analyse statistique sur le corpus et la base vectorielle FAISS.
Extrait les indicateurs liés aux finances publiques, formats/types de sources,
fréquence de mots-clés stratégiques et distribution de la taille des segments.

UTILISATION :
  python src/g_corpus_analytics.py
"""

import json
import os
import re
import sys
from collections import Counter

import faiss
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(ROOT, "artifacts")
DATA_DIR = os.path.join(ROOT, "data")

FICHIER_DOCUMENTS = os.path.join(ARTIFACTS_DIR, "documents.json")
FICHIER_CHUNKS_RAW = os.path.join(DATA_DIR, "chunks.json")
FICHIER_INDEX = os.path.join(ARTIFACTS_DIR, "faiss_index.bin")
FICHIER_STATS_OUT = os.path.join(ARTIFACTS_DIR, "corpus_stats.json")

MOTS_CLES_STRATEGIQUES = [
    "dette",
    "maastricht",
    "déficit",
    "budget",
    "loi de finances",
    "pib",
    "tva",
    "impôt",
    "fiscal",
    "sécurité sociale",
    "retraite",
    "fonction publique",
    "santé",
    "dcsi",
    "collectivités",
    "inflation",
    "prélèvements obligatoires",
]


def _charger_documents() -> list[dict]:
    """Charge les chunks depuis artifacts/documents.json en priorité, sinon data/chunks.json."""
    if os.path.exists(FICHIER_DOCUMENTS):
        with open(FICHIER_DOCUMENTS, "r", encoding="utf-8") as f:
            return json.load(f)
    if os.path.exists(FICHIER_CHUNKS_RAW):
        with open(FICHIER_CHUNKS_RAW, "r", encoding="utf-8") as f:
            return json.load(f)
    raise FileNotFoundError(
        f"Aucun fichier de chunks trouvé ({FICHIER_DOCUMENTS} ou {FICHIER_CHUNKS_RAW})."
    )


def _inferer_format_source(doc: dict) -> str:
    """Déduit le format/type de source à partir des champs disponibles du chunk."""
    for champ in ("format", "source_format", "extension", "file_type", "doc_type"):
        val = doc.get(champ)
        if isinstance(val, str) and val.strip():
            return val.strip().upper()

    source_ref = str(doc.get("source_file") or doc.get("filename") or doc.get("url") or "").lower()
    if source_ref.endswith(".pdf"):
        return "PDF"
    if source_ref.endswith((".html", ".htm")) or source_ref.startswith(("http://", "https://")):
        return "HTML / WEB"
    if source_ref.endswith((".csv", ".xlsx", ".xls")):
        return "TABLEUR"
    if source_ref.endswith(".json"):
        return "JSON"

    ctype = str(doc.get("type", "text")).strip().lower()
    if ctype == "table":
        return "TABLEAU (JSON/HTML)"
    return "TEXTE / WEB"


def analyser_corpus_et_index(
    documents: list[dict] | None = None,
    index_path: str = FICHIER_INDEX,
    mots_cles: list[str] | None = None,
) -> dict:
    """Calcule l'ensemble des indicateurs statistiques sur le corpus et l'index FAISS."""
    if documents is None:
        documents = _charger_documents()
    if mots_cles is None:
        mots_cles = MOTS_CLES_STRATEGIQUES

    nb_chunks = len(documents)
    sources_ids = [str(d.get("source_id", "INCONNU")) for d in documents]
    nb_documents_uniques = len(set(sources_ids))

    # 1. Répartition par format / type de chunk
    formats_counter = Counter(_inferer_format_source(d) for d in documents)
    types_counter = Counter(str(d.get("type", "text")).strip() or "text" for d in documents)
    chunks_par_doc_counter = Counter(sources_ids)

    # 2. Distribution de la taille des segments (en caractères et mots)
    longueurs_chars = [
        int(d.get("longueur") if d.get("longueur") is not None else len(d.get("content", "")))
        for d in documents
    ]
    longueurs_mots = [
        len(re.findall(r"\w+", d.get("content", "")))
        for d in documents
    ]

    arr_chars = np.array(longueurs_chars, dtype=np.float64) if longueurs_chars else np.array([0.0])
    arr_mots = np.array(longueurs_mots, dtype=np.float64) if longueurs_mots else np.array([0.0])

    bins = [0, 250, 500, 750, 1000, 1500, float("inf")]
    labels = ["0-250", "251-500", "501-750", "751-1000", "1001-1500", "1500+"]
    hist_counts, _ = np.histogram(arr_chars, bins=bins)
    histogramme_taille = [
        {"tranche": lbl, "nb_chunks": int(cnt)}
        for lbl, cnt in zip(labels, hist_counts)
    ]

    # 3. Fréquence des mots-clés stratégiques
    corpus_texte_lower = "\n".join(str(d.get("content", "")).lower() for d in documents)
    frequence_mots_cles = []
    for kw in mots_cles:
        pattern = re.escape(kw.lower())
        occurrences = len(re.findall(pattern, corpus_texte_lower))
        chunks_concernes = sum(
            1 for d in documents if re.search(pattern, str(d.get("content", "")).lower())
        )
        frequence_mots_cles.append(
            {
                "mot_cle": kw,
                "occurrences": int(occurrences),
                "nb_chunks": int(chunks_concernes),
                "part_chunks_pct": round((chunks_concernes / nb_chunks * 100.0) if nb_chunks else 0.0, 2),
            }
        )
    frequence_mots_cles.sort(key=lambda x: x["occurrences"], reverse=True)

    # 4. Top documents par nombre de chunks
    top_documents = [
        {
            "source_id": sid,
            "nb_chunks": int(cnt),
            "part_pct": round((cnt / nb_chunks * 100.0) if nb_chunks else 0.0, 2),
        }
        for sid, cnt in chunks_par_doc_counter.most_common(15)
    ]

    # 5. Informations index FAISS
    faiss_info = {
        "disponible": os.path.exists(index_path),
        "nb_vecteurs": None,
        "dimension": None,
        "type_index": None,
    }
    if os.path.exists(index_path):
        idx = faiss.read_index(index_path)
        faiss_info = {
            "disponible": True,
            "nb_vecteurs": int(idx.ntotal),
            "dimension": int(idx.d),
            "type_index": type(idx).__name__,
        }

    return {
        "kpis": {
            "total_chunks": int(nb_chunks),
            "total_documents": int(nb_documents_uniques),
            "chunks_moyens_par_doc": round(nb_chunks / nb_documents_uniques, 2) if nb_documents_uniques else 0.0,
            "taille_moyenne_chars": round(float(np.mean(arr_chars)), 1),
            "taille_mediane_chars": round(float(np.median(arr_chars)), 1),
            "taille_min_chars": int(np.min(arr_chars)),
            "taille_max_chars": int(np.max(arr_chars)),
            "mots_moyens_par_chunk": round(float(np.mean(arr_mots)), 1),
        },
        "repartition_formats": [
            {
                "format": fmt,
                "nb_chunks": int(cnt),
                "part_pct": round((cnt / nb_chunks * 100.0) if nb_chunks else 0.0, 2),
            }
            for fmt, cnt in formats_counter.most_common()
        ],
        "repartition_types": [
            {
                "type": t,
                "nb_chunks": int(cnt),
                "part_pct": round((cnt / nb_chunks * 100.0) if nb_chunks else 0.0, 2),
            }
            for t, cnt in types_counter.most_common()
        ],
        "histogramme_taille_chunks": histogramme_taille,
        "frequence_mots_cles": frequence_mots_cles,
        "top_documents": top_documents,
        "faiss_index": faiss_info,
    }


def sauvegarder_statistiques(output_path: str = FICHIER_STATS_OUT) -> dict:
    """Calcule et enregistre les statistiques dans artifacts/corpus_stats.json."""
    stats = analyser_corpus_et_index()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    return stats


if __name__ == "__main__":
    try:
        resultat = sauvegarder_statistiques()
        print("[OK] Statistiques du corpus générées avec succès.")
        print(f"  - Documents : {resultat['kpis']['total_documents']}")
        print(f"  - Chunks    : {resultat['kpis']['total_chunks']}")
        print(f"  - Taille moy: {resultat['kpis']['taille_moyenne_chars']} chars")
        print(f"  - Index FAISS: {resultat['faiss_index']}")
        print(f"  - Fichier   : {FICHIER_STATS_OUT}")
    except Exception as exc:
        print(f"[ERREUR] {exc}")
        sys.exit(1)
