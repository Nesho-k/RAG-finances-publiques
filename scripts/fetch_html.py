"""Téléchargement des pages HTML listées dans sources_html.json (étape A).

Lit data/1_source/html/sources_html.json (clé "url"), télécharge chaque page
et l'enregistre dans data/1_source/html/ sous le nom « <sous_thème>_html.html ».
Si un même sous-thème a plusieurs pages, les suivantes reçoivent « _2 », « _3 »…
Les échecs sont journalisés dans data/2_raw/extraction.log.

Bibliothèque standard uniquement (urllib). Aucune donnée personnelle n'est envoyée.

Usage :
    python scripts/fetch_html.py
"""

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
LOG = RACINE / "data" / "2_raw" / "extraction.log"
ENTETE_UA = {"User-Agent": "Mozilla/5.0"}
DELAI_ENTRE_REQUETES = 1.0  # secondes, pour ne pas surcharger les sites
TENTATIVES = 3
DELAI_TIMEOUT = 30
TAILLE_MINI = 1000  # en dessous, on considère que ce n'est pas une vraie page
TAILLE_MARQUEURS = 3000  # les marqueurs de rejet ne sont cherchés que sous cette taille
MARQUEURS_REJET = [
    b"window.location.href",   # redirection JavaScript sans contenu
    b"_Incapsula_Resource",    # protection anti-bot
    b"Request Rejected",       # requête rejetée par le serveur
]
CARACTERES_INTERDITS = re.compile(r'[<>:"/\\|?*]')


def journaliser(message: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    horodatage = datetime.now().isoformat(timespec="seconds")
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"{horodatage} [fetch] {message}\n")


def nom_fichier(theme: str) -> str:
    """Nettoie les caractères interdits dans un nom de fichier Windows."""
    return CARACTERES_INTERDITS.sub("-", theme)


def telecharger_urllib(url: str) -> bytes:
    requete = urllib.request.Request(url, headers=ENTETE_UA)
    derniere_erreur: Exception | None = None
    for essai in range(1, TENTATIVES + 1):
        try:
            with urllib.request.urlopen(requete, timeout=DELAI_TIMEOUT) as reponse:
                return reponse.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            derniere_erreur = exc
            time.sleep(2 * essai)
    raise derniere_erreur  # type: ignore[misc]


def telecharger_curl(url: str) -> bytes:
    """Repli pour les erreurs TLS de Python : même requête, faite par curl."""
    resultat = subprocess.run(
        ["curl", "-sSL", "-A", ENTETE_UA["User-Agent"], "--max-time", str(DELAI_TIMEOUT), url],
        capture_output=True,
        check=False,
    )
    if resultat.returncode != 0:
        raise RuntimeError(resultat.stderr.decode("utf-8", "replace").strip())
    return resultat.stdout


def telecharger(url: str) -> bytes:
    try:
        return telecharger_urllib(url)
    except Exception as exc:
        if "SSL" in str(exc):
            return telecharger_curl(url)
        raise


def motif_rejet(contenu: bytes) -> str | None:
    """Retourne la raison du rejet si le contenu n'est pas une vraie page.

    Les marqueurs ne sont testés que sur les petits fichiers : une vraie page
    peut contenir du JavaScript qui utilise ces mêmes mots.
    """
    if len(contenu) < TAILLE_MINI:
        return f"contenu trop court ({len(contenu)} octets), probable redirection ou blocage"
    if len(contenu) < TAILLE_MARQUEURS:
        for marqueur in MARQUEURS_REJET:
            if marqueur in contenu:
                return f"page de redirection ou de blocage ({marqueur.decode()})"
    return None


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Téléchargement des pages HTML sources.")
    parser.add_argument(
        "--liste", default=str(RACINE / "data" / "1_source" / "html" / "sources_html.json")
    )
    parser.add_argument("--sortie", default=str(RACINE / "data" / "1_source" / "html"))
    args = parser.parse_args()

    entrees = json.loads(Path(args.liste).read_text(encoding="utf-8"))
    sortie = Path(args.sortie)
    sortie.mkdir(parents=True, exist_ok=True)

    rang = defaultdict(int)  # rang de l'URL dans son sous-thème (ordre du JSON)
    ok, erreurs = 0, 0
    for entree in entrees:
        theme = entree["sous_theme"]
        url = entree["url"]
        rang[theme] += 1
        suffixe = "" if rang[theme] == 1 else f"_{rang[theme]}"
        nom = f"{nom_fichier(theme)}_html{suffixe}.html"
        try:
            contenu = telecharger(url)
            raison = motif_rejet(contenu)
            if raison:
                raise ValueError(raison)
            (sortie / nom).write_bytes(contenu)
            ok += 1
            print(f"OK    {nom} ({len(contenu)} octets)")
        except Exception as exc:
            erreurs += 1
            journaliser(f"ERREUR {url} : {exc}")
            print(f"ERREUR {url} : {exc}")
        time.sleep(DELAI_ENTRE_REQUETES)

    print(f"HTML téléchargés : {ok} réussi(s), {erreurs} erreur(s) sur {len(entrees)} URL.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
