"""Extraction brute des pages HTML (étape A).

Lit data/1_source/html/*.html avec BeautifulSoup (parseur lxml) :
  - supprime script, style, nav, header, footer, aside et les blocs de publicité ;
  - garde le contenu de <main> ou <article> s'il existe, sinon celui de <body> ;
  - écrit data/2_raw/html/<nom>.txt. Aucun nettoyage n'est fait ici.

Le fichier data/1_source/html/sources_html.json ne contient que des URL : il
n'est pas lu comme un document HTML.

Usage :
    python scripts/extract_html.py
    python scripts/extract_html.py --source data/1_source/html --sortie data/2_raw/html
"""

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

from bs4 import BeautifulSoup

RACINE = Path(__file__).resolve().parents[1]
LOG = RACINE / "data" / "2_raw" / "extraction.log"

SEUIL_CONTENU_MINI = 500  # caractères : en dessous, <main>/<article> est jugé insuffisant
BALISES_A_SUPPRIMER =["script", "style", "nav", "header", "footer", "aside"]
# Blocs publicitaires : classe ou identifiant contenant « pub » ou « ad-».
MOTIF_PUB = re.compile(r"(^|[-_ ])(pub|ads?|advert)([-_ ]|$)", re.IGNORECASE)


def journaliser(message: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    horodatage = datetime.now().isoformat(timespec="seconds")
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"{horodatage} [html] {message}\n")


def extraire_html(fichier: Path) -> str:
    soupe = BeautifulSoup(fichier.read_bytes(), "lxml")

    for balise in soupe(BALISES_A_SUPPRIMER):
        balise.decompose()
    for bloc in soupe.find_all(
        attrs={"class": MOTIF_PUB}
    ) + soupe.find_all(attrs={"id": MOTIF_PUB}):
        bloc.decompose()

    racine = soupe.find("main") or soupe.find("article")
    if racine is None or len(racine.get_text(strip=True)) < SEUIL_CONTENU_MINI:
        # <main>/<article> absent ou vide (contenu placé ailleurs) : on prend tout le body
        racine = soupe.body or soupe
    lignes = [ligne.strip() for ligne in racine.get_text("\n").splitlines()]
    return "\n".join(lignes)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Extraction brute des fichiers HTML.")
    parser.add_argument("--source", default=str(RACINE / "data" / "1_source" / "html"))
    parser.add_argument("--sortie", default=str(RACINE / "data" / "2_raw" / "html"))
    args = parser.parse_args()

    source = Path(args.source)
    sortie = Path(args.sortie)
    sortie.mkdir(parents=True, exist_ok=True)

    fichiers = sorted(source.glob("*.html"))
    if not fichiers:
        print(f"Aucun fichier .html trouvé dans {source}.")
    ok, erreurs = 0, 0
    for fichier in fichiers:
        try:
            texte = extraire_html(fichier)
            (sortie / f"{fichier.stem}.txt").write_text(texte, encoding="utf-8", newline="\n")
            ok += 1
            print(f"OK    {fichier.name} ({len(texte)} caractères)")
        except Exception as exc:  # un fichier en erreur ne bloque pas les autres
            erreurs += 1
            journaliser(f"ERREUR {fichier.name} : {exc}")
            print(f"ERREUR {fichier.name} : {exc}")

    print(f"HTML : {ok} extrait(s), {erreurs} erreur(s) sur {len(fichiers)} fichier(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
