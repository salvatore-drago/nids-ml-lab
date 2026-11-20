"""Percorsi del progetto, uguali su Colab e in locale, su qualunque sistema."""

from __future__ import annotations

from pathlib import Path

# nidslab sta in <radice>/src/nidslab/, quindi la radice è due livelli sopra.
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "processed"
FIGURES_DIR = REPO_ROOT / "figures"


def data_file(name: str) -> Path:
    """Percorso di un file in data/processed/, con un errore chiaro se manca."""
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"Non trovo {path}.\n"
            "Se lavori in locale: aggiorna il repository con 'git pull'.\n"
            "Se lavori su Colab: riesegui la cella di setup in cima al notebook."
        )
    return path
