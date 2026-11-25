"""Construction de data/corpus_final.json : métadonnées des documents (étape C).

Une entrée par document, à partir des fichiers de data/3_clean/ (un fichier
= un document). Champs produits :
  - "id"              : DOC-001, DOC-002, ... (utilisé partout ensuite) ;
  - "type_document"   : pdf, html, docx ou txt ;
  - "id_sous_theme"   : ST-01, ST-02, ... (identifiant du sous-thème) ;
  - "sous_theme"      : libellé du sous-thème ;
  - "nom_document"    : nom du fichier d'origine (dans data/1_source/) ;
  - "date_extraction" : AAAA-MM-JJ (date du fichier de data/2_raw/).

Le sous-thème et le nom d'origine des PDF et DOCX viennent de scripts/noms.py.
Pour les HTML, ils viennent du nom du fichier.

Usage :
    python scripts/build_corpus.py
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

from noms import CORRESPONDANCE

RACINE = Path(__file__).resolve().parents[1]
MOTIF_NOM = re.compile(r"^(?P<theme>.+)_(?P<format>pdf|doc|html)(?:_\d+)?$")
TYPE_DOCUMENT = {"pdf": "pdf", "html": "html", "doc": "docx"}


def decomposer(stem: str) -> tuple[str, str]:
    """Renvoie (sous-thème, format) à partir du nom d'un fichier de 3_clean."""
    correspondance = MOTIF_NOM.match(stem)
    if correspondance is None:
        raise ValueError(f"Nom de fichier non reconnu : {stem}")
    return correspondance["theme"], correspondance["format"]


def nom_origine(theme: str, format_: str, stem: str) -> str:
    """Nom du fichier d'origine dans data/1_source/."""
    if format_ == "html":
        return f"{stem}.html"
    extension = {"pdf": ".pdf", "doc": ".docx"}[format_]
    for fichier, libelle in CORRESPONDANCE.items():
        if libelle == theme and fichier.endswith(extension):
            return fichier
    raise KeyError(f"Aucun fichier d'origine pour {theme} ({format_})")


def date_extraction(stem: str) -> str:
    """Date du fichier extrait dans data/2_raw/ (même nom, sous le dossier du format)."""
    _, format_ = decomposer(stem)
    dossier = {"pdf": "pdf", "html": "html", "doc": "doc"}[format_]
    fichier = RACINE / "data" / "2_raw" / dossier / f"{stem}.txt"
    return datetime.fromtimestamp(fichier.stat().st_mtime).strftime("%Y-%m-%d")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Construction de corpus_final.json.")
    parser.add_argument("--nettoye", default=str(RACINE / "data" / "3_clean"))
    parser.add_argument("--sortie", default=str(RACINE / "data" / "corpus_final.json"))
    args = parser.parse_args()

    fichiers = sorted(Path(args.nettoye).glob("*.txt"))
    sous_themes = sorted({decomposer(f.stem)[0] for f in fichiers})
    identifiants = {theme: f"ST-{rang:02d}" for rang, theme in enumerate(sous_themes, 1)}

    corpus = []
    for rang, fichier in enumerate(fichiers, 1):
        theme, format_ = decomposer(fichier.stem)
        corpus.append(
            {
                "id": f"DOC-{rang:03d}",
                "type_document": TYPE_DOCUMENT[format_],
                "id_sous_theme": identifiants[theme],
                "sous_theme": theme,
                "nom_document": nom_origine(theme, format_, fichier.stem),
                "date_extraction": date_extraction(fichier.stem),
            }
        )

    sortie = Path(args.sortie)
    sortie.parent.mkdir(parents=True, exist_ok=True)
    sortie.write_text(json.dumps(corpus, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{len(corpus)} document(s), {len(sous_themes)} sous-thème(s) -> {sortie}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
