import argparse
import os
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    df = None
    # Try UCI TSV format first
    try:
        tmp = pd.read_csv(path, sep='\t', header=None, names=['label','text'], encoding='utf-8')
        if {'label','text'}.issubset(tmp.columns) and tmp['text'].notna().any():
            df = tmp
    except Exception:
        pass

    # Fallback to CSV with header
    if df is None:
        tmp = pd.read_csv(path, encoding='utf-8', engine='python')
        cols = {c.lower(): c for c in tmp.columns}
        label_col = next((cols[c] for c in ['label','class','target','v1'] if c in cols), None)
        text_col  = next((cols[c] for c in ['text','message','sms','v2'] if c in cols), None)
        if label_col is None or text_col is None:
            raise ValueError(f"Could not infer text/label columns from CSV. Found columns: {list(tmp.columns)}")
        df = tmp[[label_col, text_col]].rename(columns={label_col:'label', text_col:'text'})

    df = df.dropna(subset=['label','text']).copy()
    df['label'] = (
        df['label'].astype(str).str.strip().str.lower()
        .map({'ham':0,'spam':1,'0':0,'1':1})
        .fillna(df['label'])
    )
    if df['label'].dtype == 'O':
        codes, uniques = pd.factorize(df['label'])
        if len(uniques) != 2:
            raise ValueError(f"Label column appears to have {len(uniques)} classes, expected 2.")
        df['label'] = codes
    df['text'] = df['text'].astype(str)
    return df[['text','label']]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--max_iter", type=int, default=500)
    args = parser.parse_args()

    df = load_dataset(args.data)
    X, y = df['text'].values, df['label'].values

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    val_rel = args.val_size / (1.0 - args.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_rel, random_state=42, stratify=y_trainval
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(strip_accents="unicode", lowercase=True,
                                  ngram_range=(1,2), min_df=2, max_df=0.95)),
        ("clf", LogisticRegression(max_iter=args.max_iter, n_jobs=None))
    ])

    pipe.fit(X_train, y_train)

    def evaluate(name, y_true, y_pred, y_prob):
        try:
            auc = roc_auc_score(y_true, y_prob)
        except Exception:
            auc = np.nan
        print(f"\n=== {name} ===")
        print(classification_report(y_true, y_pred, digits=4))
        print("ROC-AUC:", round(float(auc), 4) if auc==auc else "nan")
        print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

    val_pred = pipe.predict(X_val)
    val_prob = pipe.predict_proba(X_val)[:,1]
    evaluate("Validation", y_val, val_pred, val_prob)

    test_pred = pipe.predict(X_test)
    test_prob = pipe.predict_proba(X_test)[:,1]
    evaluate("Test", y_test, test_pred, test_prob)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    latest_dir = os.path.join("models", "latest")
    run_dir = os.path.join("models", f"run-{timestamp}")
    os.makedirs(latest_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)
    joblib.dump(pipe, os.path.join(latest_dir, "model.joblib"))
    joblib.dump(pipe, os.path.join(run_dir, "model.joblib"))
    print(f"\nSaved model to: {latest_dir} and {run_dir}")

if __name__ == "__main__":
    main()
