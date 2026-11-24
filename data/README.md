# Dati
**BCCC-CIC-IDS2017** è una nuova estrazione dei flussi di rete
del dataset CIC-IDS2017 ottenuta con lo strumento NTLFlowLyzer. 

Le esercitazioni usano una versione sottocampionata e modificata di questo dataset pensata per scopi didattici.

## Riferimenti

- M. Shafi, A. H. Lashkari, A. H. Roudsari. *NTLFlowLyzer: Toward generating an
  intrusion detection dataset and intruders behavior profiling through network
  and transport layers traffic analysis and pattern extraction*. Computers &
  Security 148 (2025), 104160. https://doi.org/10.1016/j.cose.2024.104160
- I. Sharafaldin, A. H. Lashkari, A. A. Ghorbani. *Toward Generating a New
  Intrusion Detection Dataset and Intrusion Traffic Characterization*. ICISSP 2018.
  (Il dataset CIC-IDS2017 originale.)
- Pagina del dataset: https://www.yorku.ca/research/bccc/ucs-technical/cybersecurity-datasets-cds/intrusion-detection-dataset-bccc-cic-ids2017

<!-- TODO: licenza e condizioni di ridistribuzione da verificare prima di rendere pubblico il repository -->

## Cosa c'è in `processed/`

| File | Uso | Contenuto |
|---|---|---|
| `static.parquet` | Lezione 1 | Campione stratificato al 7% |
| `stream.parquet` | Lezione 2 | Campione stratificato al 3,5% |
| `class_counts.csv` | Riferimento | Conteggi per classe: originali, nei due campioni, flussi da 1 pacchetto, durate negative |

Entrambi i parquet:

- **conservano le proporzioni originali delle classi** (attacchi circa 15%). Le
  classi con meno di 100 esempi in totale (Heartbleed, Web_SQL_Injection) sono
  tenute intere;
- **sono ordinati per timestamp**, dal lunedì al venerdì;
- **non sono puliti**: identificativi (`flow_id`, IP, porte, `timestamp`),
  duplicati, valori negativi e flussi da un solo pacchetto restano, perché
  vanno trattati nelle esercitazioni.

Rispetto ai CSV originali ci sono tre colonne in più:

| Colonna | Significato |
|---|---|
| `day` | Giorno di cattura (Monday ... Friday), categoria ordinata |
| `label_binary` | 0 = Benign, 1 = attacco |
| `family` | Famiglia dell'attacco, vedi tabella sotto |

I numeri decimali sono in `float32` e gli interi in `int32`, per ridurre la memoria.

## Classi

| Giorno | `label` | `family` |
|---|---|---|
| tutti | Benign | Benign |
| Tuesday | FTP-Patator, SSH-Patator | BruteForce |
| Wednesday | DoS_Hulk, DoS_GoldenEye, DoS_Slowhttptest, DoS_Slowloris | DoS |
| Wednesday | Heartbleed | Heartbleed |
| Thursday | Web_Brute_Force, Web_XSS, Web_SQL_Injection | WebAttack |
| Friday | Port_Scan | PortScan |
| Friday | DDoS_LOIT | DDoS |
| Friday | Botnet_ARES | Botnet |

Il lunedì contiene solo traffico benigno. 

I conteggi esatti sono in `class_counts.csv`.

## Rigenerare i file

1. Scarica i CSV per giorno (`Monday_complete.csv` ... `Friday_complete.csv`) e
   mettili in `data/raw/`, anche in una sottocartella.
2. Dalla radice del repository:

   ```bash
   python scripts/prepare_data.py
   ```

Opzioni utili:

```bash
python scripts/prepare_data.py --help
python scripts/prepare_data.py --frac-static 0.10 --suffix _large   # versione più grande
```

Il campionamento è riproducibile: con lo stesso `--seed` (default 42) si
ottengono gli stessi file.
