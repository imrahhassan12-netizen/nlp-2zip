import time
import pandas as pd
from gensim.models import Word2Vec, FastText

PARAMS = dict(vector_size=100, window=5, min_count=5, epochs=5, seed=42, workers=4)

def train_embeddings(csv_path="data/processed/cleaned_reviews.csv"):
    df = pd.read_csv(csv_path).dropna(subset=["cleaned_review"])
    sentences = [str(t).split() for t in df["cleaned_review"]]

    # B1: Word2Vec CBOW
    t0 = time.time()
    cbow = Word2Vec(sentences, sg=0, **PARAMS)
    t_cbow = time.time() - t0

    # B2: Word2Vec Skip-gram
    t0 = time.time()
    skipgram = Word2Vec(sentences, sg=1, **PARAMS)
    t_sg = time.time() - t0

    # B4: fastText
    t0 = time.time()
    fasttext = FastText(sentences, sg=0, **PARAMS)
    t_ft = time.time() - t0

    print("=== Part B: Learned Embeddings Summary ===")
    print(f"Hyperparameters: {PARAMS}")
    print(f"CBOW (sg=0)      | Time: {t_cbow:.2f}s | Vocab: {len(cbow.wv):,}")
    print(f"Skip-gram (sg=1) | Time: {t_sg:.2f}s | Vocab: {len(skipgram.wv):,}")
    print(f"fastText (sg=0)  | Time: {t_ft:.2f}s | Vocab: {len(fasttext.wv):,}\n")

    # B4: OOV / Misspelled Word Demonstration
    print("=== Part B4: OOV / Misspelled Word Demo ===")
    for oov in ["horrrorrr", "cinematographiii"]:
        print(f"Query: '{oov}'")
        for name, m in [("CBOW", cbow), ("Skip-gram", skipgram)]:
            try:
                _ = m.wv[oov]
            except KeyError:
                print(f"  {name:<10}: KeyError (not in vocabulary)")
        sims = [(w, round(s, 4)) for w, s in fasttext.wv.most_similar(oov, topn=5)]
        print(f"  fastText  : {sims}\n")

    return cbow, skipgram, fasttext

if __name__ == "__main__":
    train_embeddings()