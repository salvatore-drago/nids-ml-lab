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


def fmt_int(n) -> str:
    """Numero intero con il separatore delle migliaia all'italiana: 169390 -> '169.390'."""
    return f"{int(n):,}".replace(",", ".")


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

    def pct(r):
        # Un decimale vicino agli estremi, così 345 falsi allarmi non diventano "0%"
        # e un 99,8% non diventa "100%".
        if r == 0 or r == 1:
            return f"{100 * r:.0f}%"
        if r < 0.001:
            return "<0,1%"
        if r > 0.999:
            return ">99,9%"
        if r < 0.10 or r > 0.90:
            return f"{100 * r:.1f}%".replace(".", ",")
        return f"{100 * r:.0f}%"

    fs = 10 if n <= 6 else 8 if n <= 10 else 7
    threshold = 0.55 * (1 if normalize else values.max())
    for i in range(n):
        for j in range(n):
            if counts[i, j] == 0:
                continue
            color = "white" if values[i, j] > threshold else INK
            text = f"{pct(rates[i, j])}\n{fmt_int(counts[i, j])}" if normalize else fmt_int(counts[i, j])
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


def plot_decision_regions(model, X, y, ax=None, feature_names=None, resolution: int = 300,
                          class_names=("Benign", "Attacco"), title: str | None = None,
                          alpha: float = 0.35):
    """Frontiera di decisione di un classificatore binario su due feature.

    X: array o DataFrame con esattamente due colonne; y: etichette 0/1.
    Il modello deve essere già addestrato su queste due colonne.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    pad_x = 0.05 * (X[:, 0].max() - X[:, 0].min() or 1)
    pad_y = 0.05 * (X[:, 1].max() - X[:, 1].min() or 1)
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - pad_x, X[:, 0].max() + pad_x, resolution),
        np.linspace(X[:, 1].min() - pad_y, X[:, 1].max() + pad_y, resolution),
    )
    zz = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    regions = mpl.colors.ListedColormap([BENIGN_COLOR, ATTACK_COLOR])
    ax.contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5], cmap=regions, alpha=0.15)
    ax.contour(xx, yy, zz, levels=[0.5], colors=INK_SECONDARY, linewidths=1)
    for value, color, name in [(0, BENIGN_COLOR, class_names[0]), (1, ATTACK_COLOR, class_names[1])]:
        m = y == value
        ax.scatter(X[m, 0], X[m, 1], s=12, color=color, alpha=alpha, linewidths=0, label=name)
    ax.grid(False)
    if feature_names is not None:
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
    if title:
        ax.set_title(title, loc="left")
    legend = ax.legend(loc="best", markerscale=2)
    for handle in legend.legend_handles:
        handle.set_alpha(1)
    return ax


def _compact(v: float) -> str:
    """1234 -> '1,2k', 5000000 -> '5M': etichette corte per assi con valori molto grandi."""
    for size, suffix in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
        if v >= size:
            return f"{v / size:.3g}".replace(".", ",") + suffix
    return f"{v:.0f}"


def set_log1p_ticks(ax, axis: str = "both", max_ticks: int = 6) -> None:
    """Per assi in scala log(1 + x): mette le tacche su valori "tondi" di x
    (0, 1, 2, 5, 10, 20, 50...) e scrive il valore originale, non il logaritmo."""
    nice = np.array([0, 1, 2, 3, 5] + [m * 10 ** e for e in range(1, 10) for m in (1, 2, 3, 5)], dtype=float)
    for which in (["x", "y"] if axis == "both" else [axis]):
        lo, hi = ax.get_xlim() if which == "x" else ax.get_ylim()
        vals = nice[(np.log1p(nice) >= lo) & (np.log1p(nice) <= hi)]
        if len(vals) < 3:  # intervallo stretto: valori intermedi
            vals = np.unique(np.round(np.linspace(np.expm1(max(lo, 0)), np.expm1(hi), 5)))
        # Tacche distanziate in modo uniforme sull'asse: ne teniamo una solo se è
        # abbastanza lontana dalla precedente.
        min_gap = (hi - lo) / max_ticks
        kept, last = [], -np.inf
        for v in vals:
            pos = np.log1p(v)
            if pos - last >= min_gap:
                kept.append(v)
                last = pos
        vals = np.array(kept)
        labels = [_compact(v) for v in vals]
        if which == "x":
            ax.set_xticks(np.log1p(vals), labels)
            ax.set_xlim(lo, hi)
        else:
            ax.set_yticks(np.log1p(vals), labels)
            ax.set_ylim(lo, hi)
