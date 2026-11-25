# Finances publiques : RAG & Data Viz

Ce projet est un RAG (assistant questions-réponses) sur les finances publiques françaises (budget de l'État, dette, fiscalité, retraites, fonction publique…).
Le corpus est constitué de pages web, de PDF et de DOCX. AU total il y a 32 sources. Il est découpé en passages (810 chunks), indexé avec des embeddings (`intfloat/multilingual-e5-base`) dans un index FAISS. 
Pour chaque question, les passages les plus proches sont envoyés à OpenAI (`gpt-4o-mini`), qui répond en citant ses sources. 
**Important :** Si la question posée n'est pas dans les sources, le LLM ne répond pas.

L'interface Streamlit affiche la réponse, les liens vers les documents ainsi qu'un tableau de bord analytique du corpus et de l'évaluation RAGAS.

## Structure du projet

```
rag_m1/
├── requirements.txt          # dépendances Python (versions figées)
├── .env                      # clé OPENAI_API_KEY (non versionnée)
├── config/config.yaml        # paramètres : taille des passages, top_k, modèles, mots-clés
├── data/
│   ├── 1_source/             # documents d'origine (html, pdf, doc) + sources_*.json (URL)
│   ├── 2_raw/                # texte brut extrait (étape A)
│   ├── 3_clean/              # texte nettoyé (étape B)
│   ├── corpus_final.json     # métadonnées par document : DOC-xxx, sous-thème, nom de fichier
│   └── chunks.json           # passages produits par le découpage (étape E)
├── scripts/                  # préparation des données (téléchargement, extraction, nettoyage)
├── src/                      # étapes du pipeline RAG
│   ├── a_chunk.py            # découpage en passages
│   ├── b_create_embeddings.py# calcul des embeddings
│   ├── c_build_faiss_index.py# construction de l'index FAISS
│   ├── e_rag_pipeline.py     # recherche FAISS + génération OpenAI (+ liens vers les sources)
│   ├── f_evaluate_rag.py     # évaluation RAGAS
│   └── g_corpus_analytics.py # indicateurs statistiques du corpus et de l'index
├── artifacts/                # embeddings.npy, documents.json, faiss_index.bin, evaluation_report.json
├── backend/api_server.py     # API FastAPI : /ask et /analytics/*
└── frontend/app_streamlit.py # interface Streamlit (mode conversationnel et mode analytique)
```

## Installation

Prérequis : Python 3.12 (le projet a été développé avec cette version) et une clé API OpenAI.

```powershell
# 1. Créer et activer l'environnement virtuel (dans le dossier du projet)
python -m venv rag_m1
.\rag_m1\Scripts\Activate.ps1

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Créer le fichier .env à la racine du projet
#    contenu :  OPENAI_API_KEY=sk-...
```

Le premier lancement télécharge le modèle d'embedding `intfloat/multilingual-e5-base` (quelques centaines de Mo).

## Lancer le projet

Les fichiers `artifacts/` et `data/` sont déjà présents : pour utiliser l'application, il suffit de lancer l'API puis l'interface.

**1. Lancer l'API** (dans un premier terminal, environnement activé) :

```powershell
uvicorn backend.api_server:app --reload --port 8000
```

L'API charge l'index FAISS, le modèle d'embedding et calcule les statistiques au démarrage. Attendre le message de démarrage avant de passer à l'étape suivante.

**2. Lancer l'interface** (dans un second terminal, environnement activé) :

```powershell
streamlit run frontend/app_streamlit.py
```

L'interface s'ouvre sur `http://localhost:8501`. Elle appelle l'API sur `http://localhost:8000` (modifiable avec la variable d'environnement `RAG_API_URL`).

**Reconstruire les données (optionnel)** : à lancer seulement si les documents sources changent, dans cet ordre :

**Constitution du corpus** 

```powershell
python scripts/fetch_html.py          # télécharge les pages HTML (nécessite internet)
python scripts/extract_html.py        # étape A : extraction brute
python scripts/extract_pdf.py
python scripts/extract_docx.py
python scripts/clean.py               # étape B : nettoyage
python scripts/build_corpus.py        # étape C : corpus_final.json
python src/a_chunk.py                 # étape E : découpage en passages
python src/b_create_embeddings.py     # embeddings -> artifacts/
python src/c_build_faiss_index.py     # index FAISS -> artifacts/faiss_index.bin
```

**Pipeline RAG** 

```powershell
python src/a_chunk.py                 # étape E : découpage en passages
python src/b_create_embeddings.py     # embeddings -> artifacts/
python src/c_build_faiss_index.py     # index FAISS -> artifacts/faiss_index.bin
python src/e_rag_pipeline.py          # question en ligne de commande, sans interface
python src/f_evaluate_rag.py          # évaluation RAGAS -> artifacts/evaluation_report.json
python src/g_corpus_analytics.py      # indicateurs du corpus en console
```

## Difficultés rencontrées

- **Liens vers les sources absents.** `documents.json` ne contenait que l'identifiant `DOC-009`, sans URL ni nom de fichier. Le lien est maintenant reconstitué au chargement (`enrichir_sources()` dans `e_rag_pipeline.py`) à partir de `corpus_final.json` et des fichiers `sources_*.json`.
- **Lien ajouté à une réponse hors sujet.** Quand le contexte ne contenait pas la réponse, le backend ajoutait quand même un lien et les chunks. Maintenant, les sources ne sont renvoyées que si la réponse cite au moins un chunk.
- **Pages non téléchargeables.** Certaines URL renvoient une erreur 403 ou une erreur SSL (voir `data/2_raw/extraction.log`). Les documents correspondants ne sont pas dans le corpus.
- **Limites de débit OpenAI.** En cas d'erreur 429, `appeler_openai()` réessaie jusqu'à 3 fois avec 10 secondes d'attente.
- **Amélioration des scores RAGAS.** Au début, la fidélité était plombée par des ajouts hors contexte du modèle et par des passages peu pertinents récupérés dans l'index :
  - *Consignes plus strictes* : le prompt a été resserré pour coller uniquement aux extraits fournis et refuser clairement de répondre quand l'info n'y est pas (notamment sur Q08 pour éviter de ressortir d'anciens montants quand on ne demande pas l'historique).
  - *Filtrage de similarité* : ajout d'un seuil minimal de score (`SCORE_MIN`) dans `rechercher_chunks()` pour écarter les passages trop éloignés de la question.
  - *Cas des questions pièges* : calcul séparé de la pertinence hors questions pièges afin de ne pas fausser la moyenne quand le modèle refuse à juste titre de répondre.
