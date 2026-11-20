# NIDS ML-based: esercitazioni

Materiale delle esercitazioni su Network Intrusion Detection flow-based
basati su machine learning.

## Come lavorare

**Consigliato: Google Colab.** Basta un account Google: apri il notebook
dal link e segui la prima cella. I link arriveranno a breve.

**In locale** (Python 3.x, a tuo rischio):

```bash
git clone https://github.com/<utente>/nids-ml-lab.git
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