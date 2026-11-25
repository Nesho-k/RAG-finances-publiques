"""
f_evaluate_rag.py
-----------------
Étape 6 : Évaluation du pipeline RAG avec RAGAS (sans warnings dépréciés,
          timeouts stabilisés, top_k=5 et pertinence hors questions pièges).

UTILISATION :
  python src/f_evaluate_rag.py
"""

import json
import os
import sys
import time
from dotenv import load_dotenv
from openai import OpenAI

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

load_dotenv(dotenv_path=os.path.join(ROOT, ".env"), override=False)

from e_rag_pipeline import (
    NB_CHUNKS,
    NOM_MODELE_OPENAI,
    PROMPT_TEMPLATE,
    appeler_openai,
    charger_ressources,
    construire_contexte,
    initialiser_openai,
    rechercher_chunks,
)

# RAGAS moderne (sans LangchainLLMWrapper / LangchainEmbeddingsWrapper dépréciés)
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from langchain_openai import OpenAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import llm_factory
from ragas.metrics._answer_relevance import ResponseRelevancy
from ragas.metrics._context_precision import LLMContextPrecisionWithReference
from ragas.metrics._context_recall import LLMContextRecall
from ragas.metrics._faithfulness import Faithfulness
from ragas.run_config import RunConfig

FICHIER_DATASET = os.path.join(ROOT, "validation", "questions.json")
ARTIFACTS_DIR   = os.path.join(ROOT, "artifacts")
FICHIER_RAPPORT = os.path.join(ARTIFACTS_DIR, "evaluation_report.json")
DELAI_ENTRE_QUESTIONS = 1

NOMS_METRIQUES = {
    "faithfulness":                         "Fidélité                   (pas d'hallucination)",
    "answer_relevancy":                     "Pertinence réponse (toutes questions)",
    "answer_relevancy_hors_pieges":         "Pertinence réponse (hors pièges)",
    "llm_context_precision_with_reference": "Précision contexte         (qualité FAISS)",
    "context_recall":                       "Rappel contexte            (couverture FAISS)",
}


# ============================================================
# CONFIGURATION RAGAS AVEC OPENAI (API MODERNE)
# ============================================================

def configurer_ragas(client: OpenAI):
    """Configure les 4 métriques RAGAS avec OpenAI."""

    ragas_llm = llm_factory(NOM_MODELE_OPENAI, client=client)
    lc_emb = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=client.api_key,
    )
    ragas_emb = LangchainEmbeddingsWrapper(lc_emb)

    metriques = [
        Faithfulness(llm=ragas_llm),
        ResponseRelevancy(llm=ragas_llm, embeddings=ragas_emb),
        LLMContextPrecisionWithReference(llm=ragas_llm),
        LLMContextRecall(llm=ragas_llm),
    ]

    print(f"[OK] RAGAS configuré avec OpenAI ({NOM_MODELE_OPENAI}, top_k={NB_CHUNKS})\n")
    return metriques


# ============================================================
# EXÉCUTION DU RAG SUR LE DATASET
# ============================================================

def executer_rag_sur_dataset(dataset, documents, index, modele, client: OpenAI):
    """Exécute le retrieval FAISS (top_k=5) + génération OpenAI pour chaque question."""

    samples = []
    metadonnees_questions = []
    total = len(dataset)

    for i, item in enumerate(dataset, start=1):
        q_id = item.get("id", f"Q{i:02d}")
        q_type = item.get("type", "normale")
        question = item["question"]
        ground_truth = item.get("reponse_attendue") or item.get("ground_truth", "")

        print(f"  [{i}/{total}] ({q_id} | {q_type}) {question}")
        chunks = rechercher_chunks(question, documents, index, modele, k=NB_CHUNKS)

        textes_chunks = [doc.get("content", "") for doc, _ in chunks]
        if chunks:
            contexte = construire_contexte(chunks)
            prompt = PROMPT_TEMPLATE.format(contexte=contexte, question=question)
        else:
            prompt = PROMPT_TEMPLATE.format(
                contexte="Aucun extrait pertinent trouvé dans le corpus.",
                question=question,
            )

        reponse = appeler_openai(prompt, client)
        if not reponse:
            print("    → Pas de réponse, question ignorée\n")
            continue

        samples.append(
            SingleTurnSample(
                user_input=question,
                response=reponse,
                retrieved_contexts=textes_chunks if textes_chunks else [""],
                reference=ground_truth,
            )
        )
        metadonnees_questions.append({"id": q_id, "type": q_type})
        print(f"    → OK ({len(reponse)} caractères, {len(chunks)} chunk(s))\n")

        if i < total:
            time.sleep(DELAI_ENTRE_QUESTIONS)

    return EvaluationDataset(samples=samples), metadonnees_questions


# ============================================================
# CALCUL ET AFFICHAGE DU RAPPORT
# ============================================================

def calculer_scores_globaux(df) -> dict:
    """Calcule les moyennes globales et la pertinence hors questions pièges."""

    scores = {}
    for cle in [
        "faithfulness",
        "answer_relevancy",
        "llm_context_precision_with_reference",
        "context_recall",
    ]:
        if cle in df.columns:
            val = df[cle].mean()
            if val == val:  # non-NaN
                scores[cle] = round(float(val), 4)

    # Pertinence calculée uniquement sur les questions non-pièges
    if "answer_relevancy" in df.columns and "type" in df.columns:
        df_normales = df[df["type"] != "piege"]
        if len(df_normales) > 0:
            val_normales = df_normales["answer_relevancy"].mean()
            if val_normales == val_normales:
                scores["answer_relevancy_hors_pieges"] = round(float(val_normales), 4)

    return scores


def afficher_rapport(scores_globaux: dict):
    """Affiche les scores moyens RAGAS."""

    print(f"\n{'='*65}")
    print("  RAPPORT D'ÉVALUATION RAGAS (OPENAI)")
    print(f"{'='*65}\n")

    for cle, label in NOMS_METRIQUES.items():
        if cle not in scores_globaux:
            print(f"  {label}")
            print("  ░░░░░░░░░░░░░░░░░░░░  N/A\n")
            continue
        score = scores_globaux[cle]
        nb_blocs = int(score * 20)
        barre = "█" * nb_blocs + "░" * (20 - nb_blocs)
        print(f"  {label}")
        print(f"  {barre}  {score:.2%}\n")

    # Moyenne globale sur les 4 métriques principales (en utilisant la pertinence hors pièges si dispo)
    cles_moyenne = [
        "faithfulness",
        "answer_relevancy_hors_pieges" if "answer_relevancy_hors_pieges" in scores_globaux else "answer_relevancy",
        "llm_context_precision_with_reference",
        "context_recall",
    ]
    valeurs = [scores_globaux[c] for c in cles_moyenne if c in scores_globaux]
    if valeurs:
        moyenne = sum(valeurs) / len(valeurs)
        print(f"{'─'*65}")
        print(f"  Score global moyen (pertinence hors pièges) : {moyenne:.2%}")
        print(f"{'─'*65}\n")


# ============================================================
# POINT D'ENTRÉE
# ============================================================

def main():
    print(f"\n{'='*65}")
    print("  ÉVALUATION RAG — RAGAS + OPENAI")
    print(f"{'='*65}\n")

    client = initialiser_openai()

    if not os.path.exists(FICHIER_DATASET):
        print(f"[ERREUR] Dataset introuvable : '{FICHIER_DATASET}'")
        print("  → Vérifiez la présence de validation/questions.json à la racine.")
        sys.exit(1)

    with open(FICHIER_DATASET, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"[OK] {len(dataset)} questions chargées depuis {FICHIER_DATASET}\n")

    metriques = configurer_ragas(client)
    documents, index, modele = charger_ressources()

    print(f"[...] Exécution du pipeline RAG sur {len(dataset)} questions (top_k={NB_CHUNKS})...\n")
    eval_dataset, metadonnees = executer_rag_sur_dataset(dataset, documents, index, modele, client)

    if len(eval_dataset.samples) == 0:
        print("[ERREUR] Aucun échantillon valide produit pour RAGAS.")
        sys.exit(1)

    print("[...] Évaluation RAGAS en cours (max_workers=4, timeout=300s)...\n")
    run_config = RunConfig(timeout=300, max_retries=5, max_wait=60, max_workers=4)
    resultats = evaluate(dataset=eval_dataset, metrics=metriques, run_config=run_config)

    df = resultats.to_pandas()
    df["id"] = [m["id"] for m in metadonnees]
    df["type"] = [m["type"] for m in metadonnees]

    scores_globaux = calculer_scores_globaux(df)
    afficher_rapport(scores_globaux)

    rapport = {
        "config": {
            "modele_openai": NOM_MODELE_OPENAI,
            "top_k": NB_CHUNKS,
        },
        "scores_globaux": scores_globaux,
        "details_par_question": df.to_dict(orient="records"),
    }
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    with open(FICHIER_RAPPORT, "w", encoding="utf-8") as f:
        json.dump(rapport, f, ensure_ascii=False, indent=2)

    print(f"[OK] Rapport sauvegardé dans : {FICHIER_RAPPORT}")


if __name__ == "__main__":
    main()