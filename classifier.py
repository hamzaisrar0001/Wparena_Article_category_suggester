"""Article category classifier: TF-IDF (word + char n-grams) + Logistic Regression."""
import pandas as pd
from sklearn.pipeline import make_pipeline, make_union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def build_model():
    features = make_union(
        TfidfVectorizer(lowercase=True, ngram_range=(1, 2), stop_words="english"),
        TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(3, 5)),
    )
    return make_pipeline(features, LogisticRegression(C=10, max_iter=1000))


def train(path="data/train.csv"):
    df = pd.read_csv(path)
    model = build_model()
    model.fit(df["title"], df["category"])
    return model


def suggest(model, title, top_k=3):
    """Return the top_k (category, probability) pairs for a title."""
    probs = model.predict_proba([title])[0]
    ranked = sorted(zip(model.classes_, probs), key=lambda x: -x[1])
    return ranked[:top_k]


if __name__ == "__main__":
    import sys
    m = train()
    title = " ".join(sys.argv[1:]) or "How to secure your WordPress login page"
    for cat, p in suggest(m, title):
        print(f"{cat:22s} {p:.1%}")
