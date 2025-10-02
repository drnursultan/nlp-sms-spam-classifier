# SMS Spam Classifier (NLP – Easy Project)

A clean, reproducible **spam vs ham** SMS classifier that showcases end‑to‑end NLP skills recruiters care about: EDA, modeling, evaluation, interpretability, and reproducibility.

## 🔎 Overview
**Task:** Binary text classification (spam vs ham) on the classic _SMS Spam Collection_ dataset (5,574 messages).  
**Approach:** `TF‑IDF` features over unigrams + bigrams → `Logistic Regression` baseline.  
**Why this matters:** Strong, interpretable baseline; great canvas to discuss trade‑offs (precision vs recall), thresholds, and deployment.

<p align="center">
<img alt="ROC Curve" src="reports/figures/roc_test.png" width="55%">
<img alt="Confusion Matrix" src="reports/figures/confusion_matrix_test.png" width="43%">
</p>

## Project Structure
```
sms-spam-classifier/
├── data/
│   ├── raw/SMSSpamCollection            # raw dataset (TSV: label \t text)
│   └── processed/                       # (optional) train/val/test splits
├── models/
│   └── latest/model.joblib              # trained pipeline (TF-IDF + LR)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_eval_plots.ipynb
│   └── 03_threshold_tuning.ipynb
├── reports/
│   ├── figures/                         # exported plots
│   ├── top_features_ham.txt             # most negative weights (ham)
│   └── top_features_spam.txt            # most positive weights (spam)
├── src/
│   ├── train.py                         # train + evaluate + save model
│   ├── infer.py                         # CLI scoring, prints label & p_spam
│   └── top_ngrams.py                    # prints top weighted n-grams
├── environment.yml                      # conda environment (recommended)
├── requirements.txt                     # pip alternative
└── README.md
```

## Quickstart
```bash
# 0) Create & activate isolated environment
conda env create -f environment.yml
conda activate nlp-easy
# or: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

# 1) Put the data in place (TSV, no header)
#    Path should be: data/raw/SMSSpamCollection

# 2) Train & evaluate (saves models/latest/model.joblib)
python src/train.py --data data/raw/SMSSpamCollection

# 3) Inspect top features
python src/top_ngrams.py --model_dir models/latest --k 25

# 4) Score new messages
python src/infer.py --model_dir models/latest --text "Free entry to win £1000! Reply STOP to opt-out"
printf '%s\n%s\n' 'win cash now!!!' 'Hey, are we meeting at 5?' | python src/infer.py --model_dir models/latest
```

## Results (reproducible baseline)
On a held‑out split (15% val, 15% test, stratified; random_state=42):

| Split | Accuracy | Spam Precision | Spam Recall | F1 (spam) | ROC‑AUC |
|---|---:|---:|---:|---:|---:|
| **Validation** | 0.968 | 1.000 | 0.759 | 0.863 | **0.996** |
| **Test**       | 0.962 | 1.000 | 0.714 | 0.833 | **0.986** |

**Interpretation:** The baseline is highly precise for spam (few false positives) but misses some spam (recall ~0.71). Use **threshold tuning** (Notebook `03_threshold_tuning.ipynb`) to trade precision for recall as needed.

## Plots & Analysis
- **ROC curve** and **Confusion matrix** → run `02_eval_plots.ipynb`.
- **Threshold sweep (val)** → run `03_threshold_tuning.ipynb` for precision/recall/F1 curves and a suggested threshold.
- **Top weighted n‑grams** → `python src/top_ngrams.py` (writes into `reports/`).

## Interactive Notebooks (HTML Exports)

- [01_eda.html](notebooks/exports/01_eda.html)
- [02_eval_plots.html](notebooks/exports/02_eval_plots.html)
- [03_threshold_tuning.html](notebooks/exports/03_threshold_tuning.html)

## Model Card (Short)
**Intended use.** Educational demo for SMS spam detection; not production‑hardened.  
**Data.** SMS Spam Collection v1 (English; UK/Singapore origin). One message per line: `label \t text` (labels: `ham`, `spam`).  
**Preprocessing.** Lowercasing, accent stripping; tokenization via `TfidfVectorizer` with `ngram_range=(1,2)`, `min_df=2`, `max_df=0.95`. No stemming/lemmatization.  
**Model.** Logistic Regression (scikit‑learn) on TF‑IDF features; outputs `p(spam)`; default threshold `0.5`.  
**Metrics.** Accuracy, precision, recall, F1 (class‑wise), ROC‑AUC; confusion matrix on held‑out test set.  
**Limitations.** Domain/temporal bias in dataset; limited to English SMS; might miss obfuscated/modern spam patterns.  
**Responsible use.** Calibrate threshold to your tolerance for false positives/negatives; monitor drift; consider human‑in‑the‑loop review in production.

## Reproducibility
- Deterministic split with `random_state=42`.  
- Frozen environment via `environment.yml` / `requirements.txt`.  
- All figures and artifacts saved in `reports/` and `models/`.

## Roadmap / Extensions
- DistilBERT baseline comparison (few‑epoch fine‑tune).  
- Add **--threshold** flag to `infer.py`.  
- Simple **FastAPI** endpoint for scoring.  
- Adversarial spam patterns / robustness checks.

---

**Author:** Nursultan Azhimuratov ([@drnursultan](https://github.com/drnursultan))  
If you use this repo, a star ⭐️ or a note is appreciated!
