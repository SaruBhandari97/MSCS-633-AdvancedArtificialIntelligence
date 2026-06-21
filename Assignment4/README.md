# Fraud Detection with a Deep AutoEncoder (PyOD)

**MSCS-633 — Advanced Artificial Intelligence**
Author: Saru Bhandari ([SaruBhandari97](https://github.com/SaruBhandari97))

An unsupervised credit-card fraud detector built with the **AutoEncoder**
anomaly detector from [PyOD](https://pyod.readthedocs.io). The model learns to
reconstruct ordinary transactions and flags the ones it reconstructs poorly —
high reconstruction error is used as the anomaly (fraud) score.

## Dataset

Anonymized credit-card transactions from Kaggle:
<https://www.kaggle.com/datasets/whenamancodes/fraud-detection>

Download `creditcard.csv` and place it in the project root. The file holds
284,807 transactions with 28 PCA-anonymized features (`V1`–`V28`) plus `Time`,
`Amount`, and a `Class` label (0 = genuine, 1 = fraud). Fraud is ~0.17% of rows.

> The CSV is not committed to the repository because of its size and Kaggle's
> licensing — download it directly from the link above.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python fraud_detection_autoencoder.py --data creditcard.csv
```

The script prints the evaluation metrics to the console and saves a summary
figure to `outputs/fraud_detection_results.png`.

## How it works

1. **Load & inspect** — read the CSV and report the class balance.
2. **Preprocess** — stratified train/test split, then standardize every feature
   so `Time`/`Amount` do not dominate the PCA components.
3. **Train** — fit a PyOD `AutoEncoder` (30 → 32 → 16 → 8 bottleneck → mirror)
   for 30 epochs. Training is unsupervised; labels are never shown to the model.
4. **Score & evaluate** — convert reconstruction error into anomaly scores and
   report ROC-AUC, Average Precision, a confusion matrix, and a classification
   report. ROC-AUC and Average Precision are emphasized because accuracy is
   meaningless under 0.17% fraud.
5. **Visualize** — reconstruction-error distribution, ROC curve, precision-recall
   curve, and confusion matrix.

## Project layout

```
.
├── fraud_detection_autoencoder.py   # main script
├── requirements.txt                 # dependency manifest
├── README.md
├── .gitignore
└── outputs/                         # figures saved at runtime
```

