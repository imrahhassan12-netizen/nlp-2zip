import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from src.embeddings import train_embeddings

def save_all(csv_path="data/processed/cleaned_reviews.csv"):
    os.makedirs("models", exist_ok=True)
    df = pd.read_csv(csv_path).dropna(subset=["cleaned_review"])
    texts = df["cleaned_review"].astype(str).tolist()

    tfidf = TfidfVectorizer(max_features=50000)
    tfidf.fit(texts)
    joblib.dump(tfidf, "models/tfidf.pkl")

    cbow, skipgram, fasttext = train_embeddings(csv_path)
    cbow.save("models/cbow.model")
    skipgram.save("models/skipgram.model")
    fasttext.save("models/fasttext.model")
    print("All models saved in models/ folder.")

if __name__ == "__main__":
    save_all()