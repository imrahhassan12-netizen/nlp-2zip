import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def run_count_reps(csv_path="data/processed/cleaned_reviews.csv"):
    df = pd.read_csv(csv_path).dropna(subset=["cleaned_review"])
    texts = df["cleaned_review"].astype(str).tolist()
    tokenized = [t.split() for t in texts]

    print("=== Cleaned Corpus Recap ===")
    print(f"Documents      : {len(texts):,}")
    print(f"Total tokens   : {sum(len(t) for t in tokenized):,}")
    print(f"Unique tokens  : {len(set(w for t in tokenized for w in t)):,}")
    print(f"Avg doc length : {sum(len(t) for t in tokenized)/len(texts):.2f}\n")

    # A1: Bag of Words
    bow = CountVectorizer()
    X_bow = bow.fit_transform(texts)
    r, c = X_bow.shape
    sparsity = 100.0 * (1.0 - X_bow.nnz / (r * c))
    print("=== A1: Bag of Words ===")
    print(f"Shape           : {X_bow.shape}")
    print(f"Non-zero entries: {X_bow.nnz:,}")
    print(f"Sparsity        : {sparsity:.4f}%\n")

    # A2: TF-IDF High vs Near-Zero Weights (Document 0)
    tfidf = TfidfVectorizer()
    X_tfidf = tfidf.fit_transform(texts)
    names = np.array(tfidf.get_feature_names_out())
    df_counts = np.asarray((X_bow > 0).sum(axis=0)).ravel()
    vocab = bow.vocabulary_

    doc0 = X_tfidf[0].toarray().ravel()
    nz = np.where(doc0 > 0)[0]
    sorted_nz = nz[np.argsort(doc0[nz])]

    print("=== A2: TF-IDF Example (Document 0) ===")
    print("High-weight words:")
    for i in sorted_nz[-3:][::-1]:
        w = names[i]
        print(f"  {w:<15} | Weight: {doc0[i]:.4f} | DF: {df_counts[vocab[w]]:,}")

    print("Near-zero (common) words:")
    for i in sorted_nz[:2]:
        w = names[i]
        print(f"  {w:<15} | Weight: {doc0[i]:.4f} | DF: {df_counts[vocab[w]]:,}")
    print()

    # A3: N-grams
    print("=== A3: N-gram Growth & Sparsity (min_df=5) ===")
    for ng in [(1, 1), (1, 2), (1, 3)]:
        vec = TfidfVectorizer(ngram_range=ng, min_df=5)
        X_ng = vec.fit_transform(texts)
        nr, nc = X_ng.shape
        nsp = 100.0 * (1.0 - X_ng.nnz / (nr * nc))
        print(f"Range {ng} -> Features: {nc:>7,} | Shape: {X_ng.shape} | Sparsity: {nsp:.4f}%")

if __name__ == "__main__":
    run_count_reps()