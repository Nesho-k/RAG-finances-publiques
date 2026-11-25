"""Découpage des documents nettoyés en passages (étape E).

Lit data/3_clean/*.txt et data/corpus_final.json, écrit data/chunks.json.

Règles :
  - texte : passages d'environ chunk_cible caractères (400). Le passage finit à la
    fin de la phrase qui atteint ou dépasse 400 caractères : une phrase n'est jamais
    coupée en plein milieu. Pas de taille maximale. Les paragraphes ne sont pas une
    unité de découpage : le texte d'un document est traité en continu.
  - recouvrement (texte) : chaque passage commence par la dernière phrase du passage
    précédent. Si cette phrase dépasse recouvrement_phrase_max (100) caractères, elle
    est réduite à ses 100 derniers caractères (début de mot).
  - tableau (lignes contenant " | ") : passages d'environ chunk_cible caractères,
    coupés entre les lignes uniquement. Une ligne n'est jamais coupée ; la ligne qui
    fait dépasser 400 caractères est gardée entière. Recouvrement : la ligne précédente
    du tableau, sans limite de taille.
  - un passage ne mélange jamais texte et tableau.

Usage :
    python scripts/chunk.py
"""

import argparse
import bisect
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import yaml

from build_corpus import decomposer, nom_origine

RACINE = Path(__file__).resolve().parents[1]
SEPARATEUR_TABLEAU = " | "
MOTIF_FIN_PHRASE = re.compile(r"[.!?]\s+")


def charger_config(chemin: Path) -> dict:
    with chemin.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def journaliser(chemin: Path, message: str) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    horodatage = datetime.now().isoformat(timespec="seconds")
    with chemin.open("a", encoding="utf-8") as f:
        f.write(f"{horodatage} {message}\n")


def decouper_blocs(lignes: list[str]) -> list[tuple[str, list[tuple[int, str]]]]:
    """Groupe les lignes consécutives de même nature : ("table" ou "text", [(n° de ligne, texte)])."""
    blocs: list[tuple[str, list[tuple[int, str]]]] = []
    for numero, ligne in enumerate(lignes):
        nature = "table" if SEPARATEUR_TABLEAU in ligne else "text"
        if blocs and blocs[-1][0] == nature:
            blocs[-1][1].append((numero, ligne))
        else:
            blocs.append((nature, [(numero, ligne)]))
    return blocs


def recouvrement_phrase(phrase: str, maximum: int) -> str:
    """Phrase reprise en recouvrement : entière, ou ses derniers caractères si elle est trop longue."""
    if len(phrase) <= maximum:
        return phrase
    queue = phrase[-maximum:]
    espace = queue.find(" ")
    return queue[espace + 1:] if espace != -1 else queue


def trouver_fins_de_phrase(texte: str, abbreviations: set[str]) -> list[tuple[int, int]]:
    """Positions (début, fin) des phrases de texte. Un point après une abréviation ne termine pas la phrase."""
    phrases = []
    debut = 0
    for correspondance in MOTIF_FIN_PHRASE.finditer(texte):
        debut_mot = texte.rfind(" ", 0, correspondance.start()) + 1
        mot = texte[debut_mot:correspondance.start() + 1]
        if correspondance.group()[0] == "." and mot in abbreviations:
            continue
        phrases.append((debut, correspondance.start() + 1))
        debut = correspondance.end()
    if debut < len(texte.rstrip()):
        phrases.append((debut, len(texte.rstrip())))
    return phrases


def passages_texte(bloc: list[tuple[int, str]], cible: int, cap: int, abbreviations: set[str]):
    """Passages d'un bloc de texte : (n° de ligne de départ, contenu)."""
    numeros = [numero for numero, _ in bloc]
    debuts_lignes = []
    morceaux = []
    position = 0
    for _, ligne in bloc:
        debuts_lignes.append(position)
        morceaux.append(ligne)
        position += len(ligne) + 1
    texte = " ".join(morceaux)
    phrases = trouver_fins_de_phrase(texte, abbreviations)

    passages = []
    recouvrement = None
    i = 0
    while i < len(phrases):
        debut_passage = phrases[i][0]
        j = i
        while j < len(phrases):
            fin = phrases[j][1]
            j += 1
            if fin - debut_passage >= cible:
                break
        nouveau = re.sub(r"\s+", " ", texte[debut_passage:phrases[j - 1][1]]).strip()
        contenu = f"{recouvrement} {nouveau}" if recouvrement else nouveau
        ligne_debut = numeros[bisect.bisect_right(debuts_lignes, debut_passage) - 1]
        passages.append((ligne_debut, contenu))
        derniere = re.sub(r"\s+", " ", texte[phrases[j - 1][0]:phrases[j - 1][1]]).strip()
        recouvrement = recouvrement_phrase(derniere, cap)
        i = j
    return passages


def passages_tableau(bloc: list[tuple[int, str]], cible: int, journal: Path):
    """Passages d'un bloc de tableau : coupe entre les lignes, recouvrement = ligne précédente."""
    passages = []
    recouvrement = None
    i = 0
    while i < len(bloc):
        acc: list[str] = []
        j = i
        while j < len(bloc):
            acc.append(bloc[j][1])
            j += 1
            if len("\n".join(acc)) >= cible:
                break
        if len(acc) == 1 and len(acc[0]) > cible:
            journaliser(journal, f"AVERTISSEMENT ligne de tableau seule de {len(acc[0])} caractères (gardée entière)")
        contenu = "\n".join(acc)
        if recouvrement is not None:
            contenu = f"{recouvrement}\n{contenu}"
        passages.append((bloc[i][0], contenu))
        recouvrement = bloc[j - 1][1]
        i = j
    return passages


def correspondances_fichiers(dossier: Path, corpus: list[dict]) -> dict[str, Path]:
    """Associe l'id de chaque document (DOC-xxx) au fichier de data/3_clean/."""
    par_nom = {entree["nom_document"]: entree["id"] for entree in corpus}
    resultat = {}
    for fichier in sorted(dossier.glob("*.txt")):
        theme, format_ = decomposer(fichier.stem)
        nom = nom_origine(theme, format_, fichier.stem)
        resultat[par_nom[nom]] = fichier
    return resultat


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Découpage des documents nettoyés en passages.")
    parser.add_argument("--config", default=str(RACINE / "config" / "config.yaml"))
    parser.add_argument("--nettoye", default=str(RACINE / "data" / "3_clean"))
    parser.add_argument("--corpus", default=str(RACINE / "data" / "corpus_final.json"))
    parser.add_argument("--sortie", default=str(RACINE / "data" / "chunks.json"))
    parser.add_argument("--journal", default=str(RACINE / "logs" / "chunking.log"))
    args = parser.parse_args()

    config = charger_config(Path(args.config))
    cible = int(config["chunk_cible"])
    cap = int(config["recouvrement_phrase_max"])
    abbreviations = set(config["abbreviations"])
    journal = Path(args.journal)

    corpus = json.loads(Path(args.corpus).read_text(encoding="utf-8"))
    fichiers = correspondances_fichiers(Path(args.nettoye), corpus)

    passages_tous = []
    for entree in corpus:
        doc_id = entree["id"]
        lignes = fichiers[doc_id].read_text(encoding="utf-8").splitlines()
        total_lignes = len(lignes)
        resultats = []
        for nature, bloc in decouper_blocs(lignes):
            if nature == "table":
                resultats += [("table", ligne, contenu) for ligne, contenu in passages_tableau(bloc, cible, journal)]
            else:
                resultats += [("text", ligne, contenu) for ligne, contenu in passages_texte(bloc, cible, cap, abbreviations)]

        for rang, (type_, ligne, contenu) in enumerate(resultats, 1):
            if len(contenu) > 800:
                journaliser(journal, f"INFO {doc_id} passage {rang} de {len(contenu)} caractères (phrase longue)")
            passages_tous.append(
                {
                    "chunk_id": f"{doc_id}-C{rang:04d}",
                    "source_id": doc_id,
                    "type": type_,
                    "content": contenu,
                    "longueur": len(contenu),
                    "index_dans_document": rang,
                    "position_approx": round(100 * ligne / total_lignes, 1),
                }
            )
        journaliser(journal, f"{doc_id} : {len(resultats)} passage(s)")

    sortie = Path(args.sortie)
    sortie.parent.mkdir(parents=True, exist_ok=True)
    sortie.write_text(json.dumps(passages_tous, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{len(passages_tous)} passage(s) pour {len(corpus)} document(s) -> {sortie}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
