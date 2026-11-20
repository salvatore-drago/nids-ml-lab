"""Funzioni di supporto per le esercitazioni NIDS ML-based.

Il pacchetto si importa anche se mancano delle librerie: i controlli
dell'ambiente (env) non dipendono da nient'altro, e il resto viene caricato
solo al primo utilizzo. Così la cella di setup può installare ciò che manca.
"""

from importlib import import_module

from .env import check_environment, ensure_packages, in_colab
from .paths import DATA_DIR, FIGURES_DIR, REPO_ROOT

_LAZY = {
    "CATEGORICAL_COLS": "data", "DAYS": "data", "ID_COLS": "data", "LABEL_COLS": "data",
    "feature_columns": "data", "load_class_counts": "data",
    "load_static": "data", "load_stream": "data",
    "plot_confusion_matrix": "plots", "save_figure": "plots", "set_style": "plots",
    "CATEGORICAL": "plots", "BENIGN_COLOR": "plots", "ATTACK_COLOR": "plots",
    "INK": "plots", "INK_SECONDARY": "plots",
}

__all__ = ["check_environment", "ensure_packages", "in_colab",
           "DATA_DIR", "FIGURES_DIR", "REPO_ROOT", *_LAZY]


def __getattr__(name):
    if name in _LAZY:
        return getattr(import_module(f".{_LAZY[name]}", __name__), name)
    raise AttributeError(f"module 'nidslab' has no attribute {name!r}")
