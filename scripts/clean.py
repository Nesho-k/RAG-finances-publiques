"""Nettoyage des textes bruts (étape B).

Lit data/2_raw/**/*.txt et écrit data/3_clean/<nom>.txt (même nom, dossier plat).

Traitements :
  - caractères de contrôle et de format invisibles supprimés (ex. BOM, espaces de
    largeur nulle, tiret conditionnel) ; forme Unicode NFC (les accents restent) ;
  - espaces insécables, tabulations et espaces multiples réduits à un espace ;
  - lignes vides en excès réduites à une seule entre deux paragraphes.

Non modifié : les accents, la ponctuation, et les lignes contenant " | ".
Les en-têtes et pieds de page répétés ne sont PAS supprimés (choix validé).

Usage :
    python scripts/clean.py
"""

import argparse
import re
import sys
import unicodedata
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
SEPARATEUR_TABLEAU = " | "
ESPACES_MULTIPLES = re.compile(r" {2,}")


def supprimer_caracteres_invisibles(ligne: str) -> str:
    """Remplace tabulations et espaces spéciaux par un espace, retire contrôle et format invisible."""
    sortie = []
    for car in ligne:
        if car == "\t" or unicodedata.category(car) == "Zs":
            sortie.append(" ")
        elif unicodedata.category(car) in ("Cc", "Cf"):
            continue
        else:
            sortie.append(car)
    return "".join(sortie)


def normaliser_espaces(ligne: str) -> str:
    return ESPACES_MULTIPLES.sub(" ", ligne).strip()


def reduire_lignes_vides(lignes: list[str]) -> list[str]:
    sortie: list[str] = []
    for ligne in lignes:
        if ligne == "" and (not sortie or sortie[-1] == ""):
            continue
        sortie.append(ligne)
    while sortie and sortie[-1] == "":
        sortie.pop()
    while sortie and sortie[0] == "":
        sortie.pop(0)
    return sortie


def nettoyer_ligne(ligne: str) -> str:
    if SEPARATEUR_TABLEAU in ligne:
        return ligne  # lignes de tableau : aucune modification
    ligne = unicodedata.normalize("NFC", ligne)
    return normaliser_espaces(supprimer_caracteres_invisibles(ligne))


def nettoyer(texte: str) -> str:
    lignes = [nettoyer_ligne(ligne) for ligne in texte.replace("\r", "").split("\n")]
    lignes = reduire_lignes_vides(lignes)
    return "\n".join(lignes) + "\n"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Nettoyage des textes bruts.")
    parser.add_argument("--source", default=str(RACINE / "data" / "2_raw"))
    parser.add_argument("--sortie", default=str(RACINE / "data" / "3_clean"))
    args = parser.parse_args()

    source = Path(args.source)
    sortie = Path(args.sortie)
    sortie.mkdir(parents=True, exist_ok=True)

    fichiers = sorted(source.glob("**/*.txt"))
    total_avant = total_apres = 0
    for fichier in fichiers:
        brut = fichier.read_text(encoding="utf-8")
        propre = nettoyer(brut)
        (sortie / fichier.name).write_text(propre, encoding="utf-8", newline="\n")
        total_avant += len(brut)
        total_apres += len(propre)
        print(f"{len(brut):>7} -> {len(propre):>7}  {fichier.name}")

    print(f"Fichiers : {len(fichiers)}. Caractères : {total_avant} -> {total_apres}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
