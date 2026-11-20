"""
Prepara i dataset delle esercitazioni a partire da BCCC-CIC-IDS2017.

Legge i file per giorno (*_complete.csv) e produce due parquet sottocampionati:

  data/processed/static.parquet   Lezione 1 (NIDS statici)
  data/processed/stream.parquet   Lezione 2 (concept drift, apprendimento online)

Entrambi sono campioni stratificati per classe che conservano le proporzioni
originali (attacchi ~15%). Le classi con meno di --min-whole esempi in totale
vengono tenute intere, altrimenti sparirebbero. I record sono ordinati per
timestamp.

Lo script fa il minimo indispensabile e non pulisce i dati: identificativi
(flow_id, IP, porte, timestamp), duplicati, valori negativi e flussi da un solo
pacchetto restano dove sono, perché sono materia delle esercitazioni.

Uso (dalla radice del repository):

    python scripts/prepare_data.py
    python scripts/prepare_data.py --frac-static 0.10 --suffix _large
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

FAMILY = {
    "Benign": "Benign",
    "DoS_Hulk": "DoS",
    "DoS_GoldenEye": "DoS",
    "DoS_Slowhttptest": "DoS",
    "DoS_Slowloris": "DoS",
    "DDoS_LOIT": "DDoS",
    "Port_Scan": "PortScan",
    "FTP-Patator": "BruteForce",
    "SSH-Patator": "BruteForce",
    "Web_Brute_Force": "WebAttack",
    "Web_XSS": "WebAttack",
    "Web_SQL_Injection": "WebAttack",
    "Botnet_ARES": "Botnet",
    "Heartbleed": "Heartbleed",
}

STRING_COLS = ["flow_id", "timestamp", "src_ip", "dst_ip", "protocol", "label"]
CATEGORY_COLS = ["src_ip", "dst_ip", "protocol", "label"]

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data" / "raw",
                   help="cartella in cui cercare (anche nelle sottocartelle) i file *_complete.csv")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "processed")
    p.add_argument("--frac-static", type=float, default=0.07,
                   help="frazione campionata per static.parquet (default 0.07)")
    p.add_argument("--frac-stream", type=float, default=0.035,
                   help="frazione campionata per stream.parquet (default 0.035)")
    p.add_argument("--min-whole", type=int, default=100,
                   help="classi con meno esempi di così vengono tenute intere (default 100)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--suffix", default="",
                   help="suffisso per i nomi dei file, es. _large -> static_large.parquet")
    return p.parse_args()


def find_day_files(raw_dir: Path) -> dict[str, Path]:
    files = {}
    for day in DAYS:
        found = sorted(raw_dir.rglob(f"{day}_complete.csv"))
        if not found:
            sys.exit(f"ERRORE: non trovo {day}_complete.csv sotto {raw_dir}")
        if len(found) > 1:
            sys.exit(f"ERRORE: più file {day}_complete.csv sotto {raw_dir}: {found}")
        files[day] = found[0]
    return files


def read_csv(path: Path, **kw) -> pd.DataFrame:
    dtype = {c: "string" for c in STRING_COLS}
    if "usecols" in kw:
        dtype = {c: t for c, t in dtype.items() if c in kw["usecols"]}
    try:
        return pd.read_csv(path, engine="pyarrow", dtype=dtype, **kw)
    except (ImportError, ValueError):
        return pd.read_csv(path, dtype=dtype, low_memory=False, **kw)


def count_labels(files: dict[str, Path]) -> pd.Series:
    counts = [read_csv(f, usecols=["label"])["label"].value_counts() for f in files.values()]
    total = pd.concat(counts, axis=1).fillna(0).sum(axis=1).astype(int)
    unknown = sorted(set(total.index) - set(FAMILY))
    if unknown:
        sys.exit(f"ERRORE: etichette non previste, aggiorna FAMILY nello script: {unknown}")
    return total


def sample_day(df: pd.DataFrame, frac: float, whole: set[str], rng: np.random.Generator) -> pd.DataFrame:
    parts = []
    for label, g in df.groupby("label", observed=True, sort=True):
        if label in whole:
            parts.append(g)
        else:
            n = int(round(len(g) * frac))
            if n > 0:
                parts.append(g.sample(n=n, random_state=rng))
    return pd.concat(parts)


def finalize(df: pd.DataFrame, days: pd.Series) -> pd.DataFrame:
    # Tutte le conversioni di tipo in un'unica chiamata, e le colonne nuove
    # aggiunte in blocco: aggiungerle una alla volta frammenta il DataFrame.
    # int32 e non più piccoli: con int8/int16 un calcolo degli studenti
    # (es. packets_count * 1000) andrebbe in overflow senza avvisi.
    i32 = np.iinfo(np.int32)
    types = {c: "float32" for c in df.select_dtypes(include="float64").columns}
    for c in df.select_dtypes(include="integer").columns:
        if df[c].min() >= i32.min and df[c].max() <= i32.max:
            types[c] = "int32"
    types.update({c: "category" for c in CATEGORY_COLS})
    types["flow_id"] = "string"
    df = df.astype(types)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    extra = pd.DataFrame({
        "day": pd.Categorical(days, categories=DAYS, ordered=True),
        "label_binary": (df["label"] != "Benign").astype("int8").to_numpy(),
        "family": pd.Categorical(df["label"].map(FAMILY)),
    }, index=df.index)
    df = pd.concat([df, extra], axis=1)
    return df.sort_values("timestamp", kind="stable").reset_index(drop=True)


def main() -> None:
    args = parse_args()
    rng_static = np.random.default_rng(args.seed)
    rng_stream = np.random.default_rng(args.seed + 1)

    files = find_day_files(args.raw_dir)
    print("File trovati:")
    for day, f in files.items():
        print(f"  {day:<10} {f.relative_to(args.raw_dir)}  ({f.stat().st_size / 1e6:.0f} MB)")

    print("\nConteggio delle classi...")
    totals = count_labels(files)
    whole = set(totals[totals < args.min_whole].index)
    print(f"Classi tenute intere (< {args.min_whole} esempi): {sorted(whole) or 'nessuna'}")

    static_parts, stream_parts, stats = [], [], []
    for day, f in files.items():
        print(f"\nLeggo {day}...")
        df = read_csv(f)

        g = df.groupby("label", observed=True)
        stats.append(pd.DataFrame({
            "originale": g.size(),
            "flussi_1_pacchetto": g["packets_count"].apply(lambda s: int((s == 1).sum())),
            "durata_negativa": g["duration"].apply(lambda s: int((s < 0).sum())),
        }))

        static_parts.append((day, sample_day(df, args.frac_static, whole, rng_static)))
        stream_parts.append((day, sample_day(df, args.frac_stream, whole, rng_stream)))
        del df

    def combine(parts):
        df = pd.concat([p for _, p in parts], ignore_index=True)
        days = pd.Series(np.concatenate([[d] * len(p) for d, p in parts]), index=df.index)
        return finalize(df, days)

    static = combine(static_parts)
    stream = combine(stream_parts)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_static = args.out_dir / f"static{args.suffix}.parquet"
    out_stream = args.out_dir / f"stream{args.suffix}.parquet"
    static.to_parquet(out_static, compression="zstd", index=False)
    stream.to_parquet(out_stream, compression="zstd", index=False)

    summary = pd.concat(stats).groupby(level=0).sum()
    summary["static"] = static["label"].value_counts().reindex(summary.index).fillna(0).astype(int)
    summary["stream"] = stream["label"].value_counts().reindex(summary.index).fillna(0).astype(int)
    summary["pct_1_pacchetto"] = (100 * summary["flussi_1_pacchetto"] / summary["originale"]).round(1)
    summary["famiglia"] = summary.index.map(FAMILY)
    summary = summary.sort_values("originale", ascending=False)
    summary.index.name = "label"
    out_summary = args.out_dir / f"class_counts{args.suffix}.csv"
    summary.to_csv(out_summary, encoding="utf-8")

    print("\n" + summary.to_string())
    for name, d, path in [("static", static, out_static), ("stream", stream, out_stream)]:
        size = path.stat().st_size / 1e6
        warn = "  <-- ATTENZIONE: oltre i limiti di GitHub" if size > 95 else ""
        print(f"\n{name}: {len(d):,} righe, attacchi {100 * d['label_binary'].mean():.1f}%, "
              f"{size:.1f} MB su disco, {d.memory_usage(deep=True).sum() / 1e6:.0f} MB in memoria{warn}")
        print(f"  -> {path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path}")
    print(f"\nRiepilogo per classe: {out_summary}")


if __name__ == "__main__":
    main()
