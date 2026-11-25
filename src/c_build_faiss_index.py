"""
c_build_faiss_index.py
----------------------
Étape 3 : Construire l'index FAISS à partir des embeddings dans artifacts/.

UTILISATION :
  python src/c_build_faiss_index.py
"""

import os
import sys
import faiss
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# CONFIGURATION
# ============================================================

ARTIFACTS_DIR    = os.path.join(ROOT, "artifacts")
FICHIER_EMBEDDINGS = os.path.join(ARTIFACTS_DIR, "embeddings.npy")
FICHIER_INDEX      = os.path.join(ARTIFACTS_DIR, "faiss_index.bin")


# ============================================================
# ÉTAPE 1 : Charger les embeddings
# ============================================================

def charger_embeddings():
    """Charge artifacts/embeddings.npy et s'assure du type float32."""

    if not os.path.exists(FICHIER_EMBEDDINGS):
        print(f"[ERREUR] Fichier introuvable : '{FICHIER_EMBEDDINGS}'")
        print("  → Lancez d'abord : python src/b_create_embeddings.py")
        sys.exit(1)

    embeddings = np.load(FICHIER_EMBEDDINGS).astype(np.float32)
    print(f"[OK] Embeddings chargés — forme : {embeddings.shape}")
    return embeddings


# ============================================================
# ÉTAPE 2 : Normaliser les vecteurs (L2)
# ============================================================

def normaliser(embeddings):
    """Normalise chaque vecteur L2 à 1 pour que IndexFlatIP = similarité cosinus."""

    faiss.normalize_L2(embeddings)
    normes = np.linalg.norm(embeddings, axis=1)
    print(f"[OK] Vecteurs normalisés — norme min : {normes.min():.6f} | max : {normes.max():.6f}")
    return embeddings


# ============================================================
# ÉTAPE 3 : Construire l'index FAISS
# ============================================================

def construire_index(embeddings):
    """Construit un index exact IndexFlatIP."""

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    print(f"[OK] Index FAISS construit — {index.ntotal} vecteurs (dim={dimension})")
    return index


# ============================================================
# ÉTAPE 4 : Sauvegarder l'index sur disque
# ============================================================

def sauvegarder_index(index):
    """Enregistre l'index FAISS dans artifacts/faiss_index.bin."""

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    faiss.write_index(index, FICHIER_INDEX)
    taille_ko = os.path.getsize(FICHIER_INDEX) / 1024
    print(f"[OK] Index sauvegardé → '{FICHIER_INDEX}' ({taille_ko:.1f} Ko)")


# ============================================================
# ÉTAPE 5 : Auto-vérification
# ============================================================

def tester_index(index, embeddings):
    """Vérifie que le premier vecteur se retrouve lui-même au rang 1."""

    scores, indices = index.search(embeddings[0:1], k=3)
    print("\n--- TEST DE RECHERCHE (vecteur 0) ---")
    for rang, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
        print(f"  Rang {rang} : indice={idx}  score={score:.4f}")

    if indices[0][0] == 0 and abs(scores[0][0] - 1.0) < 0.01:
        print("[OK] Auto-vérification réussie.\n")
    else:
        print("[ALERTE] Vérifiez la normalisation des embeddings.\n")


if __name__ == "__main__":
    print(f"\n{'='*55}")
    print("  CONSTRUCTION DE L'INDEX FAISS — FINANCES PUBLIQUES RAG")
    print(f"{'='*55}\n")

    emb = charger_embeddings()
    emb = normaliser(emb)
    idx = construire_index(emb)
    sauvegarder_index(idx)
    tester_index(idx, emb)
