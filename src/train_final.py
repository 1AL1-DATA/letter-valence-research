"""Train a final letter-feature sentiment classifier and save it to disk.

This trains a Random Forest on ALL of the FinancialPhraseBank binary data
(1,967 sentences), then persists the fitted model + the feature pipeline
as a single pickle. The saved model can be loaded by `src.classify` and
used to score new, unseen paragraphs.

We do NOT do CV here — that was done in `src.analyze` and reported as
0.7377 ± 0.0058. This script just builds a single production model.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.data import load_articles_binary
from src.features import get_feature_names
from src.train import train, save

OUTPUT_DIR = REPO / "models"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "letter_sentiment_rf.pkl"


def main() -> None:
    print("=" * 78)
    print("TRAINING FINAL CLASSIFIER (letter-only, random forest, 68 features)")
    print("=" * 78)
    print()

    # Load all the data
    print("Loading FinancialPhraseBank binary (1,967 sentences)...")
    articles_df = load_articles_binary()
    articles = articles_df["words"].tolist()
    y = articles_df["label"].values
    print(f"  n_articles: {len(articles)}")
    print(f"  positive: {(y == 1).sum()}, negative: {(y == 0).sum()}")
    print()

    # Train on the full dataset
    print("Training Random Forest on full dataset (n_estimators=100, random_state=42)...")
    feature_names = get_feature_names()
    model, pipeline = train(articles, y, feature_names=feature_names, strategy="all",
                            model_name="rf", model_kwargs={"n_estimators": 100,
                                                            "random_state": 42,
                                                            "n_jobs": -1})
    print(f"  feature_names: {len(feature_names)} features")
    print(f"  strategy: all (mean + max + min + std = 272-dim vector per article)")
    print()

    # Save
    print(f"Saving to {OUTPUT_PATH}...")
    save(model, pipeline, str(OUTPUT_PATH))
    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"  Saved. Size: {size_kb:.1f} KB")
    print()

    # Sanity: in-sample accuracy (this is a memorization check, not a generalization claim)
    print("Sanity check: in-sample accuracy (should be ~1.0 if the model fits perfectly)...")
    X_in = pipeline.transform(articles)
    y_in_pred = model.predict(X_in)
    in_acc = (y_in_pred == y).mean()
    print(f"  in-sample accuracy: {in_acc:.4f}")
    print()

    # The generalisation claim comes from the 5-fold CV done in src.analyze.
    # That said, let me also do a proper held-out 20% test to make the
    # number concrete for this script's output.
    print("Held-out 20% test (reproducing the 0.7377 claim)...")
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    X_all = pipeline.transform(articles)
    X_tr, X_te, y_tr, y_te = train_test_split(X_all, y, test_size=0.2, random_state=42, stratify=y)
    from sklearn.ensemble import RandomForestClassifier
    rf_test = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_test.fit(X_tr, y_tr)
    y_pred = rf_test.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    f1 = f1_score(y_te, y_pred)
    print(f"  Held-out test accuracy: {acc:.4f}")
    print(f"  Held-out test F1:      {f1:.4f}")
    print()
    print("Classification report on held-out 20% test:")
    print(classification_report(y_te, y_pred, target_names=["negative", "positive"], digits=3))
    print()
    print("DONE. Model is at:", OUTPUT_PATH)
    print("Use it with: python -m src.classify --text '...' ")


if __name__ == "__main__":
    main()
