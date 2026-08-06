"""Model training for the letter-valence analysis.

Provides:
- compute_article_features: extract per-word features and aggregate to one vector
- train_random_forest: train a Random Forest classifier on aggregated features
- save_model / load_model: serialise the fitted pipeline
- predict_article: predict sentiment for a single article
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.preprocessing import StandardScaler

from .features import (
    aggregate_article,
    compute_features_for_words,
    get_feature_names,
)


# ---- feature extraction pipeline ----
@dataclass
class FeaturePipeline:
    """Bundles the scaler + the list of feature names used at training time.

    Use `fit` to compute features on a training corpus, then `transform`
    to apply the same feature set (and scaler) to new data.
    """
    feature_names: list[str] = field(default_factory=list)
    strategy: str = "all"
    scaler: StandardScaler = field(default_factory=StandardScaler)
    _is_fitted: bool = False

    def fit(self, articles: list[list[str]]) -> np.ndarray:
        """Compute features for the given training articles.

        Returns the (n_articles, n_features) feature matrix and fits the scaler.
        """
        # Pre-compute features for all words
        all_words = set()
        for ws in articles:
            all_words.update(ws)
        word_feats = compute_features_for_words(list(all_words))

        # Aggregate per article
        X = np.array([
            aggregate_article(ws, word_feats, feats=self.feature_names, strategy=self.strategy)
            for ws in articles
        ])
        self.scaler.fit(X)
        self._is_fitted = True
        return X

    def transform(self, articles: list[list[str]]) -> np.ndarray:
        """Apply the fitted pipeline to new articles."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() before transform()")
        all_words = set()
        for ws in articles:
            all_words.update(ws)
        word_feats = compute_features_for_words(list(all_words))
        X = np.array([
            aggregate_article(ws, word_feats, feats=self.feature_names, strategy=self.strategy)
            for ws in articles
        ])
        return self.scaler.transform(X)

    def fit_transform(self, articles: list[list[str]]) -> np.ndarray:
        X = self.fit(articles)
        return self.scaler.transform(X)


# ---- model factory ----
def make_model(name: str = "rf", **kwargs):
    """Factory for the supported classifiers.

    Supported:
        "rf"        - RandomForestClassifier (default)
        "logreg"    - LogisticRegression
        "ridge"     - RidgeClassifier (linear baseline)
    """
    if name == "rf":
        defaults = dict(n_estimators=100, random_state=42, n_jobs=-1)
        defaults.update(kwargs)
        return RandomForestClassifier(**defaults)
    if name == "logreg":
        defaults = dict(C=1.0, max_iter=2000, random_state=42)
        defaults.update(kwargs)
        return LogisticRegression(**defaults)
    if name == "ridge":
        defaults = dict(alpha=1.0, random_state=42)
        defaults.update(kwargs)
        return RidgeClassifier(**defaults)
    raise ValueError(f"Unknown model: {name}")


# ---- training ----
def train(
    articles_train: list[list[str]],
    y_train: np.ndarray,
    model_name: str = "rf",
    feature_names: Optional[list[str]] = None,
    strategy: str = "all",
    model_kwargs: Optional[dict] = None,
) -> tuple[object, FeaturePipeline]:
    """Train a classifier on a list of tokenized articles.

    Returns the fitted model and the feature pipeline.
    """
    if feature_names is None:
        feature_names = get_feature_names()
    if model_kwargs is None:
        model_kwargs = {}

    pipeline = FeaturePipeline(
        feature_names=feature_names,
        strategy=strategy,
    )
    X_train = pipeline.fit_transform(articles_train)
    model = make_model(model_name, **model_kwargs)
    model.fit(X_train, y_train)
    return model, pipeline


def predict(
    model,
    pipeline: FeaturePipeline,
    articles: list[list[str]],
) -> np.ndarray:
    """Predict labels for new articles."""
    X = pipeline.transform(articles)
    return model.predict(X)


def predict_proba(model, pipeline: FeaturePipeline, articles: list[list[str]]) -> np.ndarray:
    """Predict class probabilities. Raises AttributeError if model doesn't support predict_proba."""
    if not hasattr(model, "predict_proba"):
        raise AttributeError(f"{type(model).__name__} doesn't support predict_proba")
    X = pipeline.transform(articles)
    return model.predict_proba(X)


# ---- model persistence ----
def save(model, pipeline: FeaturePipeline, path: str | Path) -> None:
    """Save the (model, pipeline) pair to a single pickle file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {"model": model, "pipeline": pipeline}
    with open(path, "wb") as f:
        pickle.dump(bundle, f)


def load(path: str | Path) -> tuple[object, FeaturePipeline]:
    """Load a (model, pipeline) pair saved by `save`."""
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["pipeline"]
