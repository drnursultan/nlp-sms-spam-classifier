import argparse
import sys, os
import joblib

def load_model(model_dir: str):
    path = os.path.join(model_dir, "model.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}. Train first.")
    return joblib.load(path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="models/latest")
    parser.add_argument("--text", type=str, default=None, help="Single text to score; else read from STDIN (one message per line)")
    args = parser.parse_args()

    pipe = load_model(args.model_dir)

    if args.text is not None:
        texts = [args.text]
    else:
        raw = sys.stdin.read().strip()
        if not raw:
            print("No input text provided. Use --text or pipe text via STDIN.")
            return
        texts = [line for line in raw.splitlines() if line.strip()]

    preds = pipe.predict(texts)
    probas = pipe.predict_proba(texts)[:,1]
    for t, p, pr in zip(texts, preds, probas):
        label = "spam" if p == 1 else "ham"
        print(f"[{label}] (p_spam={pr:.3f}) :: {t}")

if __name__ == "__main__":
    main()
