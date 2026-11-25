"""
app_streamlit.py
----------------
Interface web Streamlit à double mode :
  1. Mode Conversationnel : recherche RAG précise avec liens sources (URL / PDF / DOCX) et top 3 chunks
  2. Mode Analytique : exploration statistique du corpus, de l'index FAISS et des scores RAGAS

LANCEMENT :
  streamlit run frontend/app_streamlit.py
"""

import os
import re
import sys

import pandas as pd
import plotly.express as px
import requests
import uuid

import streamlit as st
import streamlit.components.v1 as components

FRONTEND_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(FRONTEND_DIR)
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from e_rag_pipeline import NB_CHUNKS

st.set_page_config(
    page_title="Finances Publiques : RAG & Data Viz",
    layout="wide",
)

API_URL = os.environ.get("RAG_API_URL", "http://localhost:8000").rstrip("/")


def _source_principale(sources: list[dict]) -> str:
    """Retourne l'URL ou la référence fichier (PDF/DOCX/source) de la première source."""
    for s in sources or []:
        cible = (s.get("url") or s.get("source_ref") or s.get("source_id") or "").strip()
        if cible:
            return cible
    return ""


def _nettoyer_citations_texte(texte: str, sources: list[dict]) -> str:
    """Remplace [DOC-xxx-Cxxx] par l'URL ou le fichier (PDF/DOCX/source) directement à côté de la réponse."""
    mapping: dict[str, str] = {}
    for s in sources or []:
        cid = str(s.get("chunk_id", "")).strip()
        cible = (s.get("url") or s.get("source_ref") or s.get("source_id") or "").strip()
        if cid and cible:
            mapping[cid] = cible

    fallback_cible = _source_principale(sources)
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
                if fallback_cible:
                    resolved.append(fallback_cible)
                    replaced_any = True
            else:
                resolved.append(tok)
        if not resolved:
            return f" ({fallback_cible})" if fallback_cible else ""
        uniq = []
        for item in resolved:
            if item not in uniq:
                uniq.append(item)
        return f" ({', '.join(uniq)})"

    out = re.sub(r"\[([^\]]+)\]", _repl, texte)
    out = re.sub(r"\s+([.,;:])", r"\1", out)
    out = re.sub(r"[ \t]{2,}", " ", out).strip()
    return out


def _formater_etiquette_source(s: dict) -> str:
    """Formate l'affichage d'un chunk source : identifiant chunk + score."""
    cid = s.get("chunk_id", "N/A")
    score = float(s.get("score", 0.0))
    return f"**[{s['rang']}] `{cid}`** *(score: `{score:.2%}`)*"


def appeler_rag(question: str) -> dict:
    """Interroge l'API FastAPI avec le top_k configuré côté pipeline."""
    resp = requests.post(
        f"{API_URL}/ask",
        json={"question": question, "top_k": NB_CHUNKS},
        timeout=90,
    )
    resp.raise_for_status()
    data = resp.json()
    sources = (data.get("sources") or [])[:3]
    data["sources"] = sources
    data["reponse"] = _nettoyer_citations_texte(data.get("reponse", ""), sources)
    return data


def recuperer_analytics() -> dict:
    """Récupère les indicateurs analytiques via l'API FastAPI."""
    ov = requests.get(f"{API_URL}/analytics/overview", timeout=30)
    ov.raise_for_status()
    kw = requests.get(f"{API_URL}/analytics/keywords", timeout=30)
    kw.raise_for_status()
    docs = requests.get(f"{API_URL}/analytics/documents", timeout=30)
    docs.raise_for_status()
    merged = dict(ov.json())
    merged.update(kw.json())
    merged.update(docs.json())
    return merged


def charger_rapport_ragas() -> dict | None:
    """Charge le dernier rapport d'évaluation RAGAS via l'API FastAPI ou le fichier local."""
    import json

    try:
        resp = requests.get(f"{API_URL}/analytics/ragas", timeout=15)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()
    except Exception:
        rapport_local = os.path.join(ROOT_DIR, "artifacts", "evaluation_report.json")
        if os.path.exists(rapport_local):
            with open(rapport_local, "r", encoding="utf-8") as f:
                return json.load(f)
        return None


SCRIPT_DEFILEMENT_BAS = """
<script>
(function () {
  const doc = window.parent.document;
  const descendre = () => {
    // Tous les conteneurs réellement défilants de la page (la structure de Streamlit varie selon la version)
    doc.querySelectorAll("*").forEach(el => {
      const style = window.parent.getComputedStyle(el);
      if ((style.overflowY === "auto" || style.overflowY === "scroll") && el.scrollHeight > el.clientHeight) {
        el.scrollTop = el.scrollHeight;
      }
    });
    window.parent.scrollTo(0, doc.body.scrollHeight);
  };
  [0, 150, 400, 800, 1500, 3000].forEach(delai => setTimeout(descendre, delai));
})();
</script>
"""


def defiler_en_bas():
    """Fait défiler la page vers le bas après chaque question.

    Le commentaire unique force Streamlit à réexécuter le script à chaque question
    (un contenu identique à celui du rerun précédent n'est pas relancé).
    """
    components.html(f"<!-- {uuid.uuid4()} -->{SCRIPT_DEFILEMENT_BAS}", height=0)


def afficher_mode_conversationnel():
    """Mode 1 : Interface conversationnelle RAG avec liens sources et top 3 chunks."""
    st.subheader("💬 Mode Conversationnel")
    st.caption("Posez une question sur les finances publiques, le budget ou la dette. Les 3 premiers chunks sont affichés.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    exemples = [
        "Quel est le montant de la dette publique à la fin du deuxième trimestre 2026 ?",
        "Quelle hausse budgétaire la loi de finances 2026 accorde-t-elle à l'Éducation nationale ?",
        "Quel est le premier impôt collecté en France et quel montant rapporte-t-il en 2025 ?",
        "Combien de personnes comptait la fonction publique à la fin de 2024 ?",
    ]
    question_rapide = None
    for ligne in range(0, len(exemples), 2):
        cols_ex = st.columns(2)
        for colonne, idx in enumerate(range(ligne, ligne + 2)):
            if cols_ex[colonne].button(exemples[idx], key=f"ex_{idx}", use_container_width=True):
                question_rapide = exemples[idx]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("meta"):
                st.caption(msg["meta"])
            if msg.get("sources"):
                with st.expander("Sources récupérées (Top 3)"):
                    for s in msg["sources"][:3]:
                        st.markdown(_formater_etiquette_source(s))
                        st.write(s["extrait"])

    prompt_utilisateur = st.chat_input("Posez votre question sur les finances publiques...") or question_rapide
    if prompt_utilisateur:
        st.session_state.messages.append({"role": "user", "content": prompt_utilisateur})
        with st.chat_message("user"):
            st.markdown(prompt_utilisateur)

        with st.chat_message("assistant"):
            with st.spinner("Recherche et génération en cours..."):
                try:
                    res = appeler_rag(prompt_utilisateur)
                    reponse = res["reponse"]
                    sources = (res.get("sources") or [])[:3]
                    meta = (
                        f"Retrieval: {res['temps_retrieval_s']}s | "
                        f"Génération: {res['temps_generation_s']}s | "
                        f"Total: {res['temps_total_s']}s"
                    )
                    st.markdown(reponse)
                    st.caption(meta)
                    if sources:
                        with st.expander("Sources récupérées (Top 3)"):
                            for s in sources:
                                st.markdown(_formater_etiquette_source(s))
                                st.write(s["extrait"])

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": reponse,
                            "meta": meta,
                            "sources": sources,
                        }
                    )
                except Exception as exc:
                    st.error(f"Erreur lors de l'appel API : {exc}")

        defiler_en_bas()


def afficher_mode_analytique():
    """Mode 2 : Tableau de bord analytique du corpus, de la base vectorielle et de RAGAS."""
    st.subheader("📈 Mode Analytique")
    st.caption("Indicateurs statistiques extraits du corpus, de l'index FAISS et de l'évaluation RAGAS.")

    try:
        data = recuperer_analytics()
    except Exception as exc:
        st.error(f"Impossible de charger les statistiques depuis l'API : {exc}")
        return

    kpis = data.get("kpis", {})
    faiss_info = data.get("faiss_index", {})

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Documents sources", kpis.get("total_documents", 0))
    c2.metric("Segments (chunks)", kpis.get("total_chunks", 0))
    c3.metric("Taille moyenne d'un chunk (caractères)", f"{kpis.get('taille_moyenne_chars', 0)}")
    c4.metric("Vecteurs FAISS", faiss_info.get("nb_vecteurs") or kpis.get("total_chunks", 0))

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### Répartition par format / type de source")
        df_fmt = pd.DataFrame(data.get("repartition_formats", []))
        if not df_fmt.empty:
            fig_pie = px.pie(
                df_fmt,
                names="format",
                values="nb_chunks",
                hole=0.35,
                title="Part des segments par format détecté",
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Aucune donnée de format disponible.")

    with col_right:
        st.markdown("#### Volume de segments par document source")
        df_docs = pd.DataFrame(data.get("top_documents", [])).head(10)
        if not df_docs.empty:
            fig_docs = px.bar(
                df_docs,
                x="source_id",
                y="nb_chunks",
                title="Top 10 documents par nombre de chunks",
                labels={"source_id": "Document source", "nb_chunks": "Nombre de chunks"},
            )
            st.plotly_chart(fig_docs, use_container_width=True)
        else:
            st.info("Aucune donnée documentaire disponible.")

    col_kw, col_hist = st.columns(2)

    with col_kw:
        st.markdown("#### Fréquence des mots-clés stratégiques")
        df_kw = pd.DataFrame(data.get("frequence_mots_cles", [])).head(12)
        if not df_kw.empty:
            fig_kw = px.bar(
                df_kw.sort_values("occurrences", ascending=True),
                x="occurrences",
                y="mot_cle",
                orientation="h",
                title="Occurrences des mots-clés stratégiques dans le corpus",
                labels={"occurrences": "Occurrences", "mot_cle": "Mot-clé"},
            )
            st.plotly_chart(fig_kw, use_container_width=True)
        else:
            st.info("Aucun mot-clé calculé.")

    with col_hist:
        st.markdown("#### Distribution de la taille des segments (caractères)")
        df_hist = pd.DataFrame(data.get("histogramme_taille_chunks", []))
        if not df_hist.empty:
            fig_hist = px.bar(
                df_hist,
                x="tranche",
                y="nb_chunks",
                title="Répartition des segments par tranche de longueur",
                labels={"tranche": "Taille (caractères)", "nb_chunks": "Nombre de chunks"},
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("Aucun histogramme disponible.")

    st.divider()
    st.markdown("#### 🧪 Résultats d'évaluation RAGAS")
    rapport_ragas = charger_rapport_ragas()
    if rapport_ragas:
        scores = rapport_ragas.get("scores_globaux", {})
        cfg = rapport_ragas.get("config", {})
        st.caption(
            f"Modèle juge : `{cfg.get('modele_openai', 'N/A')}` | `top_k={cfg.get('top_k', 'N/A')}`"
        )
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Fidélité", f"{scores.get('faithfulness', 0):.2%}")
        r2.metric(
            "Pertinence (hors pièges)",
            f"{scores.get('answer_relevancy_hors_pieges', scores.get('answer_relevancy', 0)):.2%}",
        )
        r3.metric("Précision contexte", f"{scores.get('llm_context_precision_with_reference', 0):.2%}")
        r4.metric("Rappel contexte", f"{scores.get('context_recall', 0):.2%}")

        details = rapport_ragas.get("details_par_question", [])
        if details:
            df_ragas = pd.DataFrame(details)
            cols_dispo = [
                c
                for c in [
                    "id",
                    "type",
                    "user_input",
                    "faithfulness",
                    "answer_relevancy",
                    "llm_context_precision_with_reference",
                    "context_recall",
                ]
                if c in df_ragas.columns
            ]
            fig_q = px.bar(
                df_ragas,
                x="id" if "id" in df_ragas.columns else df_ragas.index,
                y=[
                    c
                    for c in [
                        "faithfulness",
                        "answer_relevancy",
                        "llm_context_precision_with_reference",
                        "context_recall",
                    ]
                    if c in df_ragas.columns
                ],
                barmode="group",
                title="Scores RAGAS par question",
                labels={"value": "Score", "id": "Question"},
            )
            st.plotly_chart(fig_q, use_container_width=True)
            st.dataframe(df_ragas[cols_dispo], use_container_width=True)
    else:
        st.info("Aucun fichier `artifacts/evaluation_report.json` trouvé. Lancez `python src/f_evaluate_rag.py` pour afficher les scores RAGAS.")


def main():
    with st.sidebar:
        st.title("⚙️ Navigation")
        if st.button("Vider l'historique de chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.title("Finances Publiques : RAG & Data Viz")
    mode_vue = st.radio(
        "Choix du mode",
        options=["Mode Conversationnel", "Mode Analytique"],
        index=0,
        horizontal=True,
        label_visibility="collapsed",
    )

    if mode_vue == "Mode Conversationnel":
        afficher_mode_conversationnel()
    else:
        afficher_mode_analytique()


if __name__ == "__main__":
    main()