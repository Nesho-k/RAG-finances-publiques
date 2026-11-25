"""
e_rag_pipeline.py
-----------------
Étape 5 : Pipeline RAG complet — Recherche FAISS + Génération avec OpenAI.

UTILISATION :
  python src/e_rag_pipeline.py
"""

import json
import os
import re
import sys
import time
from urllib.parse import unquote

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.dirname(os.path.abspath(__file__))

load_dotenv(dotenv_path=os.path.join(ROOT, ".env"), override=False)

# ============================================================
# CONFIGURATION
# ============================================================

ARTIFACTS_DIR    = os.path.join(ROOT, "artifacts")
FICHIER_DOCUMENTS  = os.path.join(ARTIFACTS_DIR, "documents.json")
FICHIER_INDEX      = os.path.join(ARTIFACTS_DIR, "faiss_index.bin")
NOM_MODELE_EMB     = "intfloat/multilingual-e5-base"
NOM_MODELE_OPENAI  = "gpt-4o-mini"
NB_CHUNKS          = 5
NB_CHUNKS_AFFICHES = 5
SCORE_MIN          = 0.25
MAX_TENTATIVES     = 3
DELAI_RETRY        = 10

SYSTEM_INSTRUCTION = """Tu es un analyste rigoureux spécialisé dans les finances publiques, le budget de l'État et la dette publique en France.
Réponds uniquement à partir des extraits fournis dans le CONTEXTE.

Règles strictes :
1. Ne devine et n'invente aucun chiffre, pourcentage, date, périmètre comptable (comptabilité nationale, budgétaire, générale, dette brute vs nette, Maastricht) ou texte de loi.
2. Si l'information n'est pas présente ou est incomplète dans le contexte, indique-le clairement sans extrapoler.
3. Distingue toujours les différents périmètres et années mentionnés dans les sources.
4. Cite l'identifiant du chunk source entre crochets (ex. [DOC-009-C0001]) après chaque fait repris.
5. Réponds directement à la valeur ou information demandée (avec sa date/année ou périmètre en vigueur), sans ajouter d'anciens montants ou d'historique sauf si la question demande explicitement une évolution ou une comparaison.
6. Sois concis et synthétique : va droit au but sans répéter les mêmes informations ni développer inutilement."""

PROMPT_TEMPLATE = """CONTEXTE :
{contexte}

QUESTION : {question}

RÉPONSE COURTE ET FACTUELLE (en français) :"""


# ============================================================
# INITIALISATION OPENAI
# ============================================================

def initialiser_openai() -> OpenAI:
    """Initialise le client OpenAI avec OPENAI_API_KEY."""

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("[ERREUR] Variable d'environnement OPENAI_API_KEY non définie.")
        print("  Ajoutez OPENAI_API_KEY=sk-... dans votre fichier .env à la racine du projet.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    print(f"[OK] API OpenAI configurée (modèle : {NOM_MODELE_OPENAI})")
    return client


# ============================================================
# LIENS VERS LES SOURCES (URL, sinon nom du fichier PDF/DOCX)
# ============================================================

FICHIER_CORPUS    = os.path.join(ROOT, "data", "corpus_final.json")
DOSSIER_SOURCES   = os.path.join(ROOT, "data", "1_source")
FICHIER_URLS_HTML = os.path.join(DOSSIER_SOURCES, "html", "sources_html.json")
FICHIER_URLS_PDF  = os.path.join(DOSSIER_SOURCES, "pdf", "sources_pdf_url.json")


def _charger_json(chemin: str, defaut):
    """Lit un fichier JSON ; renvoie `defaut` si absent ou invalide."""
    try:
        with open(chemin, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return defaut


def _normaliser(texte: str) -> str:
    """Clé de comparaison : minuscules, sans ponctuation ni caractères mal encodés."""
    return re.sub(r"\W|_", "", (texte or "").lower())


def _lister_fichiers_source() -> list[str]:
    """Liste les noms de fichiers présents dans data/1_source (hors JSON)."""
    fichiers = []
    for _, _, noms in os.walk(DOSSIER_SOURCES):
        fichiers.extend(n for n in noms if not n.lower().endswith(".json"))
    return fichiers


def _nom_fichier_reel(nom_corpus: str, fichiers_reels: list[str]) -> str:
    """Retrouve le vrai nom sur disque : le corpus a remplacé les accents par '�'."""
    motif = "".join("." if c == "�" else re.escape(c) for c in nom_corpus)
    for nom in fichiers_reels:
        if re.fullmatch(motif, nom):
            return nom
    return nom_corpus


def _trouver_url(nom_fichier: str, sous_theme: str, urls_html: list, urls_pdf: list) -> str | None:
    """Renvoie l'URL d'un document : pour un PDF, correspondance sur le nom ; pour un HTML,
    1re ou 2e URL du sous-thème selon le suffixe '_html' / '_html_2'. DOCX : pas d'URL."""
    nom_min = nom_fichier.lower()
    cle_theme = _normaliser(sous_theme)

    if nom_min.endswith(".docx"):
        return None

    if nom_min.endswith(".pdf"):
        for entree in urls_pdf:
            url = entree.get("url", "")
            if unquote(url.rsplit("/", 1)[-1]).lower() == nom_min:
                return url
        candidats_pdf = [e["url"] for e in urls_pdf if _normaliser(e.get("sous_theme")) == cle_theme and e.get("url")]
        return candidats_pdf[0] if len(candidats_pdf) == 1 else None

    candidats = [e["url"] for e in urls_html if _normaliser(e.get("sous_theme")) == cle_theme and e.get("url")]
    if not candidats:
        return None
    index = 1 if "_html_2" in nom_min and len(candidats) > 1 else 0
    return candidats[index]


def enrichir_sources(documents: list[dict]) -> int:
    """Ajoute 'filename' (vrai nom du fichier source) et 'url' (si connue) à chaque chunk.

    Ces champs sont lus par l'API pour afficher le lien ou le nom du PDF/DOCX
    à la place de l'identifiant DOC-xxx. Renvoie le nombre de chunks enrichis.
    """
    corpus = _charger_json(FICHIER_CORPUS, [])
    urls_html = _charger_json(FICHIER_URLS_HTML, [])
    urls_pdf = _charger_json(FICHIER_URLS_PDF, [])
    fichiers_reels = _lister_fichiers_source()

    infos_par_doc = {}
    for entree in corpus:
        nom_reel = _nom_fichier_reel(entree.get("nom_document", ""), fichiers_reels)
        infos_par_doc[entree["id"]] = {
            "filename": nom_reel,
            "url": _trouver_url(nom_reel, entree.get("sous_theme", ""), urls_html, urls_pdf),
        }

    enrichis = 0
    for doc in documents:
        infos = infos_par_doc.get(doc.get("source_id"))
        if not infos:
            continue
        doc["filename"] = infos["filename"]
        if infos["url"]:
            doc["url"] = infos["url"]
        enrichis += 1
    return enrichis


# ============================================================
# CHARGEMENT DES RESSOURCES
# ============================================================

def charger_ressources():
    """Charge documents.json, faiss_index.bin et le modèle d'embedding."""

    for fichier in [FICHIER_DOCUMENTS, FICHIER_INDEX]:
        if not os.path.exists(fichier):
            print(f"[ERREUR] Fichier introuvable : '{fichier}'")
            print("  → Lancez d'abord b_create_embeddings.py puis c_build_faiss_index.py.")
            sys.exit(1)

    with open(FICHIER_DOCUMENTS, "r", encoding="utf-8") as f:
        documents = json.load(f)
    print(f"[OK] {len(documents)} chunks chargés")

    nb_lies = enrichir_sources(documents)
    print(f"[OK] Sources liées (fichier / URL) : {nb_lies} chunks")

    index = faiss.read_index(FICHIER_INDEX)
    print(f"[OK] Index FAISS chargé ({index.ntotal} vecteurs)")

    if index.ntotal != len(documents):
        print(f"[ERREUR] Désynchronisation index ({index.ntotal}) vs documents ({len(documents)}).")
        sys.exit(1)

    print("[...] Chargement du modèle d'embedding...")
    modele = SentenceTransformer(NOM_MODELE_EMB)
    print("[OK] Modèle d'embedding chargé\n")

    return documents, index, modele


# ============================================================
# RECHERCHE FAISS
# ============================================================

def rechercher_chunks(question: str, documents, index, modele, k: int = NB_CHUNKS):
    """Encode la question avec le préfixe 'query:' et retourne les k meilleurs chunks."""

    vecteur = modele.encode(
        [f"query: {question}"],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)

    faiss.normalize_L2(vecteur)
    scores, indices = index.search(vecteur, k=k)

    resultats = []
    for idx, score in zip(indices[0], scores[0]):
        if idx == -1 or float(score) < SCORE_MIN:
            continue
        resultats.append((documents[int(idx)], float(score)))

    return resultats


# ============================================================
# CONSTRUCTION DU CONTEXTE
# ============================================================

def construire_contexte(chunks) -> str:
    """Assemble les chunks récupérés avec leurs métadonnées."""

    parties = []
    for doc, score in chunks:
        cid = doc.get("chunk_id", f"idx_{doc.get('index', '?')}")
        sid = doc.get("source_id", "N/A")
        ctype = doc.get("type", "text")
        contenu = doc.get("content", "")
        parties.append(
            f"[Chunk: {cid} | Source: {sid} | Type: {ctype} | Score: {score:.4f}]\n{contenu}"
        )
    return "\n\n---\n\n".join(parties)


# ============================================================
# APPEL OPENAI
# ============================================================

def appeler_openai(prompt: str, client: OpenAI) -> str | None:
    """Envoie le prompt à OpenAI avec gestion des limites de débit (429)."""

    for tentative in range(1, MAX_TENTATIVES + 1):
        try:
            response = client.chat.completions.create(
                model=NOM_MODELE_OPENAI,
                messages=[
                    {"role": "system", "content": SYSTEM_INSTRUCTION},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            return response.choices[0].message.content
        except Exception as e:
            msg = str(e)
            if "429" in msg or "quota" in msg.lower() or "rate" in msg.lower():
                if tentative < MAX_TENTATIVES:
                    print(f"\n  [QUOTA] Limite API atteinte ({tentative}/{MAX_TENTATIVES}). Attente {DELAI_RETRY}s...")
                    time.sleep(DELAI_RETRY)
                else:
                    print("\n  [ERREUR] Quota API OpenAI dépassé.")
                    return None
            else:
                print(f"\n  [ERREUR] Appel OpenAI : {msg}")
                return None

    return None


# ============================================================
# AFFICHAGE ET BOUCLE INTERACTIVE
# ============================================================

def afficher_reponse(reponse: str, chunks, temps_retrieval: float, temps_generation: float):
    """Affiche la réponse générée, les temps mesurés et les chunks sources utilisés."""

    total = temps_retrieval + temps_generation
    print(f"\n{'='*55}")
    print("  RÉPONSE")
    print(f"{'='*55}\n")
    print(reponse)

    print(f"\n{'─'*55}")
    print(
        f"  CHRONOMÈTRE : retrieval={temps_retrieval:.2f}s | "
        f"génération={temps_generation:.2f}s | total={total:.2f}s"
    )
    print("  SOURCES UTILISÉES\n")
    for i, (doc, score) in enumerate(chunks[:NB_CHUNKS_AFFICHES], start=1):
        cid = doc.get("chunk_id", "N/A")
        sid = doc.get("source_id", "N/A")
        print(f"  [{i}] {cid} (Source: {sid}) — score: {score:.2%}")
    print(f"{'─'*55}\n")


def lancer_pipeline(documents, index, modele, client: OpenAI):
    """Boucle interactive en ligne de commande."""

    print("=" * 55)
    print("  ASSISTANT FINANCES PUBLIQUES — PIPELINE RAG (OPENAI)")
    print("=" * 55)
    print("  Posez vos questions en français ('quit' ou 'exit' pour quitter).\n")

    while True:
        try:
            question = input("  Votre question : ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n  Au revoir !")
            break

        if question.lower() in ("quit", "exit", "q"):
            print("\n  Au revoir !")
            break

        if not question:
            print("  (question vide, réessayez)\n")
            continue

        t0_ret = time.perf_counter()
        chunks = rechercher_chunks(question, documents, index, modele)
        temps_ret = time.perf_counter() - t0_ret

        if not chunks:
            print(f"\n  Aucun chunk pertinent trouvé (seuil : {SCORE_MIN:.0%}) en {temps_ret:.2f}s.\n")
            continue

        contexte = construire_contexte(chunks)
        prompt = PROMPT_TEMPLATE.format(contexte=contexte, question=question)

        t0_gen = time.perf_counter()
        reponse = appeler_openai(prompt, client)
        temps_gen = time.perf_counter() - t0_gen

        if reponse is None:
            print(f"  Impossible de générer une réponse (après {temps_gen:.2f}s).\n")
            continue

        afficher_reponse(reponse, chunks, temps_ret, temps_gen)


if __name__ == "__main__":
    print(f"\n{'='*55}")
    print("  DÉMARRAGE DU PIPELINE RAG")
    print(f"{'='*55}\n")

    oai_client = initialiser_openai()
    docs, idx, emb_model = charger_ressources()
    lancer_pipeline(docs, idx, emb_model, oai_client)