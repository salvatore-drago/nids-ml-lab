# NIDS ML-based: esercitazioni

Materiale delle esercitazioni su Network Intrusion Detection flow-based
basati su machine learning.

## Come lavorare

**Consigliato: Google Colab.** Basta un account Google: apri il notebook
dal link e segui la prima cella. Su Colab lavori su una copia temporanea:
per tenere le tue modifiche usa *File → Salva una copia in Drive*.

### Lezione 1: NIDS statici

| Notebook | Contenuto | Colab |
|---|---|---|
| `00_setup` | Controllo dell'ambiente (da fare **prima** della lezione) | [![Apri in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salvatore-drago/nids-ml-lab/blob/main/notebooks/00_setup.ipynb) |
| `01a_primo_modello` | Dai flussi al primo classificatore | [![Apri in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salvatore-drago/nids-ml-lab/blob/main/notebooks/01a_primo_modello.ipynb) |
| `01b_confronto_modelli` | Modelli supervisionati e detector di anomalie a confronto | [![Apri in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salvatore-drago/nids-ml-lab/blob/main/notebooks/01b_confronto_modelli.ipynb) |
| `01c_multiclasse_sbilanciamento` | Metriche, multiclasse e sbilanciamento | [![Apri in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salvatore-drago/nids-ml-lab/blob/main/notebooks/01c_multiclasse_sbilanciamento.ipynb) |

### Lezione 2: concept drift e apprendimento online

In arrivo.

**In locale** (Python 3.x, a tuo rischio):

```bash
git clone https://github.com/salvatore-drago/nids-ml-lab.git
cd nids-ml-lab
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

## Struttura

- `notebooks/`: le esercitazioni
- `data/`: i dataset (vedi `data/README.md`)
- `src/nidslab/`: funzioni di supporto usate dai notebook
- `scripts/`: preparazione dei dati