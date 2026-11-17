import joblib
import numpy as np
from gensim.models import Word2Vec, FastText
from src.clean_text import clean_text  # Assignment 1 ka function

_CACHE = {}

def vectorize(raw_doc: str, method: str = "tfidf") -> np.ndarray:
    cleaned = clean_text(raw_doc)

    if method == "tfidf":
        if "tfidf" not in _CACHE:
            _CACHE["tfidf"] = joblib.load("models/tfidf.pkl")
        return _CACHE["tfidf"].transform([cleaned]).toarray()[0]

    loaders = {
        "cbow": ("models/cbow.model", Word2Vec),
        "skipgram": ("models/skipgram.model", Word2Vec),
        "fasttext": ("models/fasttext.model", FastText),
    }
    if method not in loaders:
        raise ValueError("Choose from 'tfidf', 'cbow', 'skipgram', or 'fasttext'")

    if method not in _CACHE:
        path, cls = loaders[method]
        _CACHE[method] = cls.load(path)

    model = _CACHE[method]
    tokens = cleaned.split()
    vecs = [model.wv[t] for t in tokens if t in model.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(model.vector_size, dtype=np.float32)

if __name__ == "__main__":
    sample = "An amazing movie! The acting was superb and the story was gripping."
    for m in ["tfidf", "cbow", "skipgram", "fasttext"]:
        v = vectorize(sample, method=m)
        print(f"Method: {m:<10} | Shape: {v.shape}")