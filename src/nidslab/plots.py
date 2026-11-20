"""Stile grafico comune e grafici ricorrenti.

Le stesse figure finiscono nei notebook, nelle slide e nella dispensa,
quindi hanno tutte lo stesso aspetto.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix

from .paths import FIGURES_DIR

# Colori di serie in ordine fisso: la serie i-esima ha sempre lo stesso colore.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#6250d6", "#e34948"]
BENIGN_COLOR, ATTACK_COLOR = CATEGORICAL[0], CATEGORICAL[1]

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

# Scala sequenziale a una tinta (blu), da chiaro a scuro.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "nidslab_blue",
    ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)


def set_style() -> None:
    """Applica lo stile comune a tutti i grafici successivi."""
    mpl.rcParams.update({
        "figure.dpi": 110,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.prop_cycle": mpl.cycler(color=CATEGORICAL),
        "axes.edgecolor": INK_SECONDARY,
        "axes.labelcolor": INK,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "text.color": INK,
        "legend.frameon": False,
        "lines.linewidth": 2,
        "font.size": 10,
    })


def plot_confusion_matrix(y_true, y_pred, labels=None, normalize: bool = True,
                          ax=None, title: str | None = None):
    """Matrice di confusione.

    Con normalize=True ogni riga somma a 1: la diagonale è la recall di ciascuna
    classe, ed è leggibile anche quando le classi hanno dimensioni molto diverse.
    Nelle celle compaiono la percentuale e, sotto, il numero di flussi.
    """
    if labels is None:
        labels = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
    counts = confusion_matrix(y_true, y_pred, labels=labels)
    with np.errstate(invalid="ignore", divide="ignore"):
        rates = counts / counts.sum(axis=1, keepdims=True)
    rates = np.nan_to_num(rates)
    values = rates if normalize else counts

    n = len(labels)
    if ax is None:
        size = max(4.0, 0.55 * n + 2)
        _, ax = plt.subplots(figsize=(size + 1, size))
    im = ax.imshow(values, cmap=SEQUENTIAL, vmin=0, vmax=1 if normalize else None)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fs = 10 if n <= 6 else 8 if n <= 10 else 7
    threshold = 0.55 * (1 if normalize else values.max())
    for i in range(n):
        for j in range(n):
            if counts[i, j] == 0:
                continue
            color = "white" if values[i, j] > threshold else INK
            text = f"{rates[i, j]:.0%}\n{counts[i, j]:,}" if normalize else f"{counts[i, j]:,}"
            ax.text(j, i, text, ha="center", va="center", fontsize=fs, color=color)

    ax.set_xticks(range(n), labels, rotation=45 if n > 2 else 0, ha="right" if n > 2 else "center")
    ax.set_yticks(range(n), labels)
    ax.set_xlabel("Classe predetta")
    ax.set_ylabel("Classe vera")
    if title:
        ax.set_title(title, loc="left")
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors=INK_SECONDARY)
    if normalize:
        cbar.set_label("Quota della classe vera")
    return ax


def save_figure(fig, name: str) -> Path:
    """Salva la figura in figures/ come PDF (per dispensa) e PNG (per slide)."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIGURES_DIR / f"{name}.pdf"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{name}.png", bbox_inches="tight", dpi=200)
    return pdf
