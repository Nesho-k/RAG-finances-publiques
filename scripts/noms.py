"""Nommage des fichiers de data/2_raw/ à partir du sous-thème (étape A).

Les fichiers sont nommés « <sous-thème>_<format>.txt », où <format> vaut
pdf, doc ou html. Les caractères interdits dans un nom de fichier Windows
sont remplacés par un tiret.

Correspondance des fichiers d'origine (pdf et docx) vers leur sous-thème,
d'après sources_pdf_url.json et le contenu des documents. Ce tableau est
aussi la référence pour retrouver le nom d'origine (nom_document) à l'étape C.
"""

import re

CORRESPONDANCE = {
    "9782729888930_extrait.pdf": "Histoire des finances publiques",
    "CELEX_52026DC0950_FR_TXT.pdf": "Règles budgétaires européennes",
    "DépensesEtat.pdf": "Dépenses de l'État par mission",
    "FinancementSecuNote.pdf": "Financement de la Sécurité sociale",
    "RegimesSpeciaux.pdf": "Régimes spéciaux de retraite",
    "emploispublics2024.pdf": "Effectifs de la fonction publique",
    "1_Generalites.docx": "Fraude fiscale",
    "LOLF.docx": "Cadre juridique (LOLF)",
}

CARACTERES_INTERDITS = re.compile(r'[<>:"/\\|?*]')


def nettoyer(theme: str) -> str:
    return CARACTERES_INTERDITS.sub("-", theme)


def nom_sortie(nom_origine: str, format_: str) -> str:
    """Nom du .txt de sortie, sans extension, pour un fichier source pdf ou docx."""
    if nom_origine not in CORRESPONDANCE:
        raise KeyError(f"Aucun sous-thème défini pour {nom_origine}")
    return f"{nettoyer(CORRESPONDANCE[nom_origine])}_{format_}"
