from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec, FastText

def run_probing_and_math():
    # --- Part A4: Hand-Worked Toy Verification ---
    toy_docs = [
        "great movie great acting",
        "great movie terrible plot",
        "terrible movie boring plot",
        "masterpiece movie great story"
    ]
    vec_raw = TfidfVectorizer(norm=None).fit(toy_docs)
    vec_l2 = TfidfVectorizer(norm="l2").fit(toy_docs)
    idx = vec_raw.vocabulary_["great"]

    print("=== Part A4: Toy Verification ===")
    print(f"Raw TF-IDF ('great', D1): {vec_raw.transform(toy_docs)[0, idx]:.6f} (Hand Math: 2.446287)")
    print(f"L2 TF-IDF  ('great', D1): {vec_l2.transform(toy_docs)[0, idx]:.6f} (Hand Math: 0.749377)")

    bi = CountVectorizer(ngram_range=(2, 2)).fit(toy_docs)
    gm = bi.vocabulary_["great movie"]
    print(f"Bigram 'great movie'    : {bi.transform(toy_docs)[:, gm].toarray().ravel().tolist()} (Hand Math: [1, 1, 0, 0])\n")

    # Load saved models for Part C
    cbow = Word2Vec.load("models/cbow.model")
    skipgram = Word2Vec.load("models/skipgram.model")
    fasttext = FastText.load("models/fasttext.model")

    # --- Part C1: Most Similar Words ---
    print("=== Part C1: Top-5 Most Similar Words ===")
    for q in ["act", "horror", "masterpiec", "bore"]:
        if q in cbow.wv:
            print(f"Query: '{q}'")
            for name, m in [("CBOW", cbow), ("Skip-gram", skipgram), ("fastText", fasttext)]:
                nbrs = [f"{w} ({s:.2f})" for w, s in m.wv.most_similar(q, topn=5)]
                print(f"  {name:<10}: {', '.join(nbrs)}")
    print()

    # --- Part C2: Cosine Similarity ---
    print("=== Part C2: Cosine Similarity ===")
    pairs = [
        ("film", "movi", "Similar"),
        ("excel", "superb", "Similar"),
        ("horror", "popcorn", "Unrelated"),
        ("script", "spaceship", "Unrelated")
    ]
    for w1, w2, tag in pairs:
        if w1 in cbow.wv and w2 in cbow.wv:
            print(f"{w1:<8} / {w2:<10} ({tag:<9}) -> "
                  f"CBOW: {cbow.wv.similarity(w1, w2):.4f} | "
                  f"SG: {skipgram.wv.similarity(w1, w2):.4f} | "
                  f"FT: {fasttext.wv.similarity(w1, w2):.4f}")
    print()

    # --- Part C3: Word Analogy ---
    print("=== Part C3: Word Analogy (actor - man + woman) ===")
    for name, m in [("CBOW", cbow), ("Skip-gram", skipgram), ("fastText", fasttext)]:
        ans = [(w, round(s, 3)) for w, s in m.wv.most_similar(positive=["actor", "woman"], negative=["man"], topn=3)]
        print(f"  {name:<10}: {ans}")

if __name__ == "__main__":
    run_probing_and_math()