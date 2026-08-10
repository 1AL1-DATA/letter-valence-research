"""Train the word-level cheap tier (TF-IDF + VADER + keyword -> logistic regression).

Run from the repo root:
    python -m scripts.train_cheap_tier
or
    /home/a/.venv/bin/python -m scripts.train_cheap_tier
    (add PYTHONPATH=/tmp/.../vader if vaderSentiment is not installed)

Saves the fitted vectorizer + model to models/cheap_tier.pkl, which the
Vercel API loads for on-demand classification.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scipy import sparse  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from api.cheap_tier import keyword_valence, vader_valence  # noqa: E402

SENTENCES_PATH = REPO / "data" / "Sentences_50Agree.txt"
MODEL_PATH = REPO / "models" / "cheap_tier.pkl"

LABEL_CODE = {"negative": 0, "neutral": 1, "positive": 2}


def load_fpb() -> pd.DataFrame:
    rows = []
    with open(SENTENCES_PATH, encoding="latin-1") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if "@" not in line:
                continue
            text, label = line.rsplit("@", 1)
            label = label.strip().lower()
            if label not in LABEL_CODE:
                continue
            rows.append({"text": text.strip().strip('"'), "label": label})
    df = pd.DataFrame(rows)
    df["y"] = df["label"].map(LABEL_CODE)
    return df


def main() -> None:
    df = load_fpb()
    print(f"Loaded {len(df)} FinancialPhraseBank sentences")
    print(df["label"].value_counts().to_dict())

    texts = df["text"].tolist()
    y3 = df["y"].to_numpy()

    print("Scoring VADER + keyword extras...")
    extra = np.asarray([[vader_valence(t), keyword_valence(t)] for t in texts])

    vec = TfidfVectorizer(
        lowercase=True, token_pattern=r"[a-z]+",
        ngram_range=(1, 2), min_df=2, max_features=20_000,
    )
    x = vec.fit_transform(texts)
    x = sparse.hstack([x, sparse.csr_matrix(extra)]).tocsr()
    print(f"TF-IDF matrix: {x.shape[0]} x {x.shape[1]}")

    lr = LogisticRegression(
        C=1.0, max_iter=2000, random_state=42, class_weight="balanced"
    )
    lr.fit(x, y3)
    print(f"Train accuracy: {lr.score(x, y3):.4f}")

    bundle = {"vectorizer": vec, "model": lr, "classes": list(lr.classes_)}
    with open(MODEL_PATH, "wb") as fh:
        pickle.dump(bundle, fh)
    print(f"Saved {MODEL_PATH} ({MODEL_PATH.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
