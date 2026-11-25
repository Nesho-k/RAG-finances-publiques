"""Extraction brute des fichiers Word .docx (étape A).

Lit data/1_source/doc/*.docx avec python-docx :
  - les paragraphes, dans l'ordre ;
  - puis les tableaux, ligne par ligne, cellules séparées par " | ".
Écrit data/2_raw/doc/<nom>.txt. Aucun nettoyage n'est fait ici.

Usage :
    python scripts/extract_docx.py
    python scripts/extract_docx.py --source data/1_source/doc --sortie data/2_raw/doc
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from docx import Document
from noms import nom_sortie

RACINE = Path(__file__).resolve().parents[1]
LOG = RACINE / "data" / "2_raw" / "extraction.log"


def journaliser(message: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    horodatage = datetime.now().isoformat(timespec="seconds")
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"{horodatage} [docx] {message}\n")


def extraire_docx(fichier: Path) -> str:
    document = Document(fichier)
    lignes = [paragraphe.text for paragraphe in document.paragraphs]
    for tableau in document.tables:
        for ligne in tableau.rows:
            lignes.append(" | ".join(cellule.text for cellule in ligne.cells))
    return "\n".join(lignes)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Extraction brute des fichiers .docx.")
    parser.add_argument("--source", default=str(RACINE / "data" / "1_source" / "doc"))
    parser.add_argument("--sortie", default=str(RACINE / "data" / "2_raw" / "doc"))
    args = parser.parse_args()

    source = Path(args.source)
    sortie = Path(args.sortie)
    sortie.mkdir(parents=True, exist_ok=True)

    fichiers = sorted(source.glob("*.docx"))
    ok, erreurs = 0, 0
    for fichier in fichiers:
        try:
            texte = extraire_docx(fichier)
            (sortie / f"{nom_sortie(fichier.name, 'doc')}.txt").write_text(
                texte, encoding="utf-8", newline="\n"
            )
            ok += 1
            print(f"OK    {fichier.name} ({len(texte)} caractères)")
        except Exception as exc:  # un fichier en erreur ne bloque pas les autres
            erreurs += 1
            journaliser(f"ERREUR {fichier.name} : {exc}")
            print(f"ERREUR {fichier.name} : {exc}")

    print(f"DOCX : {ok} extrait(s), {erreurs} erreur(s) sur {len(fichiers)} fichier(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
