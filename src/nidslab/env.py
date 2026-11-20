"""Controlli e installazione dell'ambiente, su Colab e in locale."""

from __future__ import annotations

import importlib
import importlib.metadata as md
import platform
import subprocess
import sys

from .paths import DATA_DIR

# (nome per pip, nome del modulo, versione minima)
PACKAGES = [
    ("numpy", "numpy", "1.26"),
    ("pandas", "pandas", "2.2"),
    ("pyarrow", "pyarrow", "15.0"),
    ("scikit-learn", "sklearn", "1.5"),
    ("imbalanced-learn", "imblearn", "0.12"),
    ("river", "river", "0.21"),
    ("shap", "shap", "0.45"),
    ("matplotlib", "matplotlib", "3.8"),
    ("ipywidgets", "ipywidgets", "8.0"),
    ("psutil", "psutil", "5.9"),
]

DATA_FILES = ["static.parquet", "stream.parquet", "class_counts.csv"]

OK, KO, WARN = "✓", "✗", "!"


def in_colab() -> bool:
    return "google.colab" in sys.modules


def _version_tuple(v: str) -> tuple[int, ...]:
    parts = []
    for p in v.split("."):
        digits = "".join(ch for ch in p if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _installed(pip_name: str) -> str | None:
    try:
        return md.version(pip_name)
    except md.PackageNotFoundError:
        return None


def ensure_packages(quiet: bool = True) -> None:
    """Su Colab installa le librerie mancanti o troppo vecchie. In locale non installa
    niente: indica solo il comando da lanciare."""
    missing = [f"{pip}>={vmin}" for pip, _, vmin in PACKAGES
               if (v := _installed(pip)) is None or _version_tuple(v) < _version_tuple(vmin)]
    if not missing:
        return
    if in_colab():
        print("Installo:", ", ".join(missing))
        cmd = [sys.executable, "-m", "pip", "install", *missing]
        if quiet:
            cmd.insert(4, "-q")
        subprocess.run(cmd, check=True)
        importlib.invalidate_caches()
    else:
        print(f"{WARN} Mancano o sono troppo vecchie: {', '.join(missing)}")
        print("  Dalla radice del repository, con il venv attivo: pip install -r requirements.txt")


def check_environment() -> bool:
    """Stampa lo stato dell'ambiente e restituisce True se è tutto a posto."""
    all_ok = True

    def row(ok, what, detail=""):
        nonlocal all_ok
        mark = OK if ok is True else WARN if ok is None else KO
        if ok is False:
            all_ok = False
        print(f"  {mark} {what:<22} {detail}")

    print("Ambiente")
    system = {"Darwin": "macOS"}.get(platform.system(), platform.system())
    row(True, "Dove", "Google Colab" if in_colab() else f"locale ({system})")
    py = sys.version.split()[0]
    row(sys.version_info >= (3, 10), "Python", py + ("" if sys.version_info >= (3, 10) else "  serve almeno 3.10"))

    print("\nLibrerie")
    for pip, _, vmin in PACKAGES:
        v = _installed(pip)
        if v is None:
            row(False, pip, "non installata")
        elif _version_tuple(v) < _version_tuple(vmin):
            row(False, pip, f"{v}  serve almeno {vmin}")
        else:
            row(True, pip, v)

    print("\nDati")
    for name in DATA_FILES:
        path = DATA_DIR / name
        if path.exists():
            size = path.stat().st_size
            row(True, name, f"{size / 1e6:.1f} MB" if size >= 1e6 else f"{size / 1e3:.0f} KB")
        else:
            row(False, name, "mancante")

    print("\nMemoria")
    try:
        import psutil
        avail = psutil.virtual_memory().available / 1e9
        total = psutil.virtual_memory().total / 1e9
        enough = True if avail >= 2 else None if avail >= 1 else False
        row(enough, "RAM disponibile", f"{avail:.1f} GB su {total:.1f} GB")
    except ImportError:
        row(None, "RAM disponibile", "non verificabile (manca psutil)")

    print("\n" + ("Tutto pronto." if all_ok else "Qualcosa non va: controlla le righe con ✗."))
    return all_ok
