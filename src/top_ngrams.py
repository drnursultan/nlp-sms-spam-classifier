import argparse, os, joblib, numpy as np

def top_weights(model_dir: str, k: int = 25):
    pipe = joblib.load(os.path.join(model_dir, 'model.joblib'))
    vec = pipe.named_steps['tfidf']
    clf = pipe.named_steps['clf']
    feature_names = np.array(vec.get_feature_names_out())
    coefs = clf.coef_[0]  # positive -> spam, negative -> ham

    idx_spam = np.argsort(coefs)[::-1][:k]
    idx_ham  = np.argsort(coefs)[:k]

    top_spam = list(zip(feature_names[idx_spam], coefs[idx_spam]))
    top_ham  = list(zip(feature_names[idx_ham],  coefs[idx_ham]))

    os.makedirs('reports', exist_ok=True)
    with open('reports/top_features_spam.txt', 'w', encoding='utf-8') as f:
        for tok, w in top_spam:
            f.write(f"{tok}\t{w:.4f}\n")
    with open('reports/top_features_ham.txt', 'w', encoding='utf-8') as f:
        for tok, w in top_ham:
            f.write(f"{tok}\t{w:.4f}\n")

    print('Top spam-weighted n-grams:')
    for tok, w in top_spam:
        print(f"{tok:30s} {w: .4f}")
    print('\nTop ham-weighted n-grams:')
    for tok, w in top_ham:
        print(f"{tok:30s} {w: .4f}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir', type=str, default='models/latest')
    ap.add_argument('--k', type=int, default=25)
    args = ap.parse_args()
    top_weights(args.model_dir, args.k)
