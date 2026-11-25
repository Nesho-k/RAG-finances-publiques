"""
b_create_embeddings.py
----------------------
Étape 2 : Transformer les chunks en vecteurs numériques (embeddings).

UTILISATION :
  python src/b_create_embeddings.py
"""

import json
import os
import sys
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# CONFIGURATION
# ============================================================

FICHIER_CHUNKS     = os.path.join(ROOT, "data", "chunks.json")
ARTIFACTS_DIR    = os.path.join(ROOT, "artifacts")
FICHIER_EMBEDDINGS = os.path.join(ARTIFACTS_DIR, "embeddings.npy")
FICHIER_DOCUMENTS  = os.path.join(ARTIFACTS_DIR, "documents.json")

NOM_MODELE = "intfloat/multilingual-e5-base"
BATCH_SIZE = 32


# ============================================================
# ÉTAPE 1 : Charger les chunks
# ============================================================

def charger_corpus():
    """Charge le fichier data/chunks.json depuis la racine du projet."""

    if not os.path.exists(FICHIER_CHUNKS):
        print(f"[ERREUR] Fichier introuvable : '{FICHIER_CHUNKS}'")
        sys.exit(1)

    with open(FICHIER_CHUNKS, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not isinstance(chunks, list) or len(chunks) == 0:
        print(f"[ERREUR] Le fichier '{FICHIER_CHUNKS}' est vide ou invalide.")
        sys.exit(1)

    print(f"[OK] {len(chunks)} chunks chargés depuis {FICHIER_CHUNKS}")
    return chunks


# ============================================================
# ÉTAPE 2 : Charger le modèle d'embedding
# ============================================================

def charger_modele():
    """Charge le modèle sentence-transformers."""

    print(f"\n[...] Chargement du modèle '{NOM_MODELE}'...")
    modele = SentenceTransformer(NOM_MODELE)
    dimension = modele.get_sentence_embedding_dimension()
    print(f"[OK] Modèle chargé — dimension des vecteurs : {dimension}")
    return modele


# ============================================================
# ÉTAPE 3 : Calculer les embeddings
# ============================================================

def creer_embeddings(modele, documents):
    """
    Transforme chaque chunk en embedding normalisé (L2) pour la similarité cosinus.
    Utilise le préfixe 'passage:' recommandé pour multilingual-e5-base.
    """

    print("\n[...] Calcul des embeddings en cours...")

    textes = [
        f"passage: Document: {doc.get('source_id', 'N/A')} | "
        f"Chunk: {doc.get('chunk_id', f'C{i:04d}')} | Type: {doc.get('type', 'text')}\n"
        f"{doc.get('content', '')}"
        for i, doc in enumerate(documents)
    ]

    embeddings = modele.encode(
        textes,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=BATCH_SIZE,
    ).astype(np.float32)

    print(f"\n[OK] Embeddings calculés — forme : {embeddings.shape}")
    return embeddings


# ============================================================
# ÉTAPE 4 : Sauvegarder les résultats dans artifacts/
# ============================================================

def sauvegarder_resultats(embeddings, documents):
    """Sauvegarde embeddings.npy et documents.json dans artifacts/."""

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    np.save(FICHIER_EMBEDDINGS, embeddings)
    taille_ko = os.path.getsize(FICHIER_EMBEDDINGS) / 1024
    print(f"\n[OK] Embeddings sauvegardés → '{FICHIER_EMBEDDINGS}' ({taille_ko:.1f} Ko)")

    documents_indexes = []
    for i, doc in enumerate(documents):
        doc_enrichi = dict(doc)
        doc_enrichi["index"] = i
        documents_indexes.append(doc_enrichi)

    with open(FICHIER_DOCUMENTS, "w", encoding="utf-8") as f:
        json.dump(documents_indexes, f, ensure_ascii=False, indent=2)

    print(f"[OK] Documents sauvegardés  → '{FICHIER_DOCUMENTS}'")


# ============================================================
# ÉTAPE 5 : Contrôle qualité
# ============================================================

def afficher_verification(embeddings, documents):
    """Vérifie l'absence de NaN et la norme L2 des vecteurs."""

    has_nan = bool(np.isnan(embeddings).any())
    normes = np.linalg.norm(embeddings, axis=1)

    print(f"\n{'='*55}")
    print("  CONTRÔLE QUALITÉ DES EMBEDDINGS")
    print(f"{'='*55}")
    print(f"  Chunks vérifiés : {len(documents)}")
    print(f"  NaN détectés    : {'OUI [ALERTE]' if has_nan else 'Non [OK]'}")
    print(f"  Norme L2 min/moy/max : {normes.min():.4f} / {normes.mean():.4f} / {normes.max():.4f}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    print(f"\n{'='*55}")
    print("  CRÉATION DES EMBEDDINGS — FINANCES PUBLIQUES RAG")
    print(f"{'='*55}")

    corpus     = charger_corpus()
    modele     = charger_modele()
    embeddings = creer_embeddings(modele, corpus)
    sauvegarder_resultats(embeddings, corpus)
    afficher_verification(embeddings, corpus)
