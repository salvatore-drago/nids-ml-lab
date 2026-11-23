"""Punteggio della gara del notebook 01c.

Compito: classificare ogni flusso nella sua classe (Benign o uno dei 13 attacchi).
Punteggio: F1 macro sulle 14 classi, cioè la media dell'F1 di ciascuna classe,
dove ogni classe pesa uguale, rara o frequente che sia.
Vincolo: i falsi allarmi (flussi benigni classificati come un qualsiasi attacco)
non devono superare lo 0,5% dei flussi benigni.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score

MAX_FALSE_ALARMS = 0.005


def challenge_score(y_true, y_pred, verbose: bool = True) -> dict:
    """Calcola il punteggio della gara. y_true e y_pred sono le etichette di dettaglio
    (colonna `label`), come stringhe: 'Benign', 'DoS_Hulk', ..."""
    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred).astype(str)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Dimensioni diverse: {y_true.shape} etichette vere e {y_pred.shape} predizioni")

    classes = sorted(set(y_true))
    macro_f1 = f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)
    benign = y_true == "Benign"
    false_alarms = float((y_pred[benign] != "Benign").mean())
    valid = false_alarms <= MAX_FALSE_ALARMS

    if verbose:
        print(f"F1 macro ({len(classes)} classi): {macro_f1:.4f}")
        print(f"Falsi allarmi:            {false_alarms:.4%} (massimo consentito {MAX_FALSE_ALARMS:.1%})")
        print("Risultato valido." if valid else "Risultato NON valido: troppi falsi allarmi.")
    return {"F1 macro": macro_f1, "falsi allarmi": false_alarms, "valido": valid}
