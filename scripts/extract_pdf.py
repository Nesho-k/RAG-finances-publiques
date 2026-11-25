"""Extraction brute des PDF (étape A).

Lit data/1_source/pdf/*.pdf avec pdfplumber, concatène le texte de chaque
page dans l'ordre, et écrit data/2_raw/pdf/<nom>.txt.
Aucun marqueur de position n'est écrit. Aucun nettoyage n'est fait ici.

Usage :
    python scripts/extract_pdf.py
    python scripts/extract_pdf.py --source data/1_source/pdf --sortie data/2_raw/pdf
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pdfplumber
from noms import nom_sortie

RACINE = Path(__file__).resolve().parents[1]
LOG = RACINE / "data" / "2_raw" / "extraction.log"


def journaliser(message: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    horodatage = datetime.now().isoformat(timespec="seconds")
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"{horodatage} [pdf] {message}\n")


def extraire_pdf(fichier: Path) -> str:
    morceaux = []
    with pdfplumber.open(fichier) as pdf:
        for feuille in pdf.pages:
            texte = feuille.extract_text() or ""
            morceaux.append(texte)
    return "\n".join(morceaux)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Extraction brute des PDF.")
    parser.add_argument("--source", default=str(RACINE / "data" / "1_source" / "pdf"))
    parser.add_argument("--sortie", default=str(RACINE / "data" / "2_raw" / "pdf"))
    args = parser.parse_args()

    source = Path(args.source)
    sortie = Path(args.sortie)
    sortie.mkdir(parents=True, exist_ok=True)

    fichiers = sorted(source.glob("*.pdf"))
    ok, erreurs = 0, 0
    for fichier in fichiers:
        try:
            texte = extraire_pdf(fichier)
            (sortie / f"{nom_sortie(fichier.name, 'pdf')}.txt").write_text(
                texte, encoding="utf-8", newline="\n"
            )
            ok += 1
            print(f"OK    {fichier.name} ({len(texte)} caractères)")
        except Exception as exc:  # un fichier en erreur ne bloque pas les autres
            erreurs += 1
            journaliser(f"ERREUR {fichier.name} : {exc}")
            print(f"ERREUR {fichier.name} : {exc}")

    print(f"PDF : {ok} extrait(s), {erreurs} erreur(s) sur {len(fichiers)} fichier(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
