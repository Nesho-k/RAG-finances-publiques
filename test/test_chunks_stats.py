import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

CHUNKS_PATH = Path(__file__).resolve().parent.parent / "data" / "chunks.json"
REPORT_PATH = Path(__file__).resolve().parent / "chunks_stats_report.json"


def compute_stats(values: list[int | float]) -> dict:
    arr = np.array(values, dtype=float)
    q1, median, q3 = np.percentile(arr, [25, 50, 75])
    p5, p95 = np.percentile(arr, [5, 95])
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return {
        "count": int(len(arr)),
        "min": round(float(np.min(arr)), 2),
        "p5": round(float(p5), 2),
        "q1": round(float(q1), 2),
        "median": round(float(median), 2),
        "mean": round(mean, 2),
        "q3": round(float(q3), 2),
        "p95": round(float(p95), 2),
        "max": round(float(np.max(arr)), 2),
        "std": round(std, 2),
        "iqr": round(float(q3 - q1), 2),
        "cv_pct": round((std / mean * 100) if mean > 0 else 0.0, 2),
    }


def analyze_chunks(chunks: list[dict]) -> dict:
    total = len(chunks)
    if total == 0:
        raise ValueError("Le fichier chunks.json est vide.")

    char_lengths = [len(c.get("content", "")) for c in chunks]
    declared_lengths = [c.get("longueur", len(c.get("content", ""))) for c in chunks]
    word_counts = [len(c.get("content", "").split()) for c in chunks]
    approx_tokens = [round(len(c.get("content", "")) / 4) for c in chunks]

    by_source = defaultdict(list)
    by_type = Counter()
    anomalies = {
        "vide_ou_trop_court_lt_50_chars": [],
        "tres_long_gt_1500_chars": [],
        "commence_par_minuscule_ou_ponctuation": [],
        "finit_sans_ponctuation_forte": [],
        "ecart_longueur_declaree": [],
        "doublons_exacts": [],
    }

    end_puncts = set('.!?»:)]}"\'')

    seen_contents = {}
    for idx, c in enumerate(chunks):
        cid = c.get("chunk_id", f"idx_{idx}")
        sid = c.get("source_id", "UNKNOWN")
        ctype = c.get("type", "unknown")
        content = c.get("content", "")
        stripped = content.strip()
        real_len = len(content)

        by_source[sid].append(real_len)
        by_type[ctype] += 1

        if real_len < 50:
            anomalies["vide_ou_trop_court_lt_50_chars"].append((cid, real_len, stripped[:40]))
        if real_len > 1500:
            anomalies["tres_long_gt_1500_chars"].append((cid, real_len))
        if "longueur" in c and c["longueur"] != real_len:
            anomalies["ecart_longueur_declaree"].append((cid, c["longueur"], real_len))

        if stripped:
            first_char = stripped[0]
            if first_char.islower() or first_char in ",.;:)]}%":
                anomalies["commence_par_minuscule_ou_ponctuation"].append((cid, stripped[:45]))
            if stripped[-1] not in end_puncts:
                anomalies["finit_sans_ponctuation_forte"].append((cid, stripped[-45:]))

            if stripped in seen_contents:
                anomalies["doublons_exacts"].append((cid, seen_contents[stripped]))
            else:
                seen_contents[stripped] = cid

    source_summary = {}
    for sid, lengths in sorted(by_source.items()):
        source_summary[sid] = {
            "nb_chunks": len(lengths),
            "total_chars": int(sum(lengths)),
            "mean_chars": round(float(np.mean(lengths)), 2),
            "median_chars": round(float(np.median(lengths)), 2),
            "min_chars": int(min(lengths)),
            "max_chars": int(max(lengths)),
        }

    return {
        "global": {
            "total_chunks": total,
            "total_documents": len(by_source),
            "total_characters": int(sum(char_lengths)),
            "total_words": int(sum(word_counts)),
            "repartition_par_type": dict(by_type),
        },
        "distributions": {
            "caracteres_reels": compute_stats(char_lengths),
            "longueur_declaree": compute_stats(declared_lengths),
            "mots": compute_stats(word_counts),
            "tokens_estimes_div4": compute_stats(approx_tokens),
        },
        "qualite_et_anomalies": {
            k: {"count": len(v), "pct": round(len(v) / total * 100, 2), "exemples": v[:5]}
            for k, v in anomalies.items()
        },
        "par_document": source_summary,
    }


def print_report(report: dict) -> None:
    g = report["global"]
    d = report["distributions"]
    q = report["qualite_et_anomalies"]

    print("=" * 65)
    print("AUDIT STATISTIQUE DES CHUNKS (RAG)")
    print("=" * 65)
    print(f"Documents sources : {g['total_documents']} | Chunks totaux : {g['total_chunks']}")
    print(f"Volume total      : {g['total_characters']:,} caractères | {g['total_words']:,} mots")
    print(f"Types de chunks   : {g['repartition_par_type']}")

    print("\n--- DISTRIBUTION DES TAILLES ---")
    for metric, s in d.items():
        print(
            f"[{metric}] Moyenne={s['mean']} | Médiane={s['median']} | "
            f"Écart-type={s['std']} (CV={s['cv_pct']}%) | "
            f"Min={s['min']} | Q1={s['q1']} | Q3={s['q3']} | P95={s['p95']} | Max={s['max']}"
        )

    print("\n--- QUALITÉ DE DÉCOUPAGE & ANOMALIES ---")
    for name, info in q.items():
        print(f"- {name}: {info['count']} ({info['pct']}%)")
        for ex in info["exemples"][:2]:
            print(f"    ex: {ex}")

    print("\n--- RÉPARTITION PAR DOCUMENT (TOP 10 PAR NOMBRE DE CHUNKS) ---")
    sorted_docs = sorted(
        report["par_document"].items(), key=lambda x: x[1]["nb_chunks"], reverse=True
    )
    for sid, st in sorted_docs[:10]:
        print(
            f"{sid:<10} : {st['nb_chunks']:>3} chunks | "
            f"moy={st['mean_chars']:>6.1f} | med={st['median_chars']:>6.1f} | "
            f"min={st['min_chars']:>4} | max={st['max_chars']:>4}"
        )


def main() -> None:
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    report = analyze_chunks(chunks)
    print_report(report)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRapport JSON détaillé sauvegardé dans : {REPORT_PATH}")


if __name__ == "__main__":
    main()