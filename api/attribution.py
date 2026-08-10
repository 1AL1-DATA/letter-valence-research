"""Dependency-free per-instance feature attribution for the letter model.

For each of the 68 base letter features we measure the change in the
model's P(positive) when that feature is reset to its training mean
(the standardized vector's zero), then sum the four aggregation blocks
(mean, max, min, std) that share a base feature. The sign tells the
direction the feature pushes the prediction, the magnitude how much.

This is a SHAP-style signed attribution computed directly from
model.predict_proba, so it needs no shap/numba/llvmlite on the server.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.classify import _aggregate, _features_for_words, _load_model, _tokenize  # noqa: E402


def top_features(text: str, n: int = 8) -> list[dict]:
    """Return the top-n base letter features by |attribution|.

    Each item is {"name": str, "value": float}; a positive value means the
    feature pushed the prediction toward positive, negative toward negative.
    """
    model, feature_names, strategy, scaler = _load_model()

    words = _tokenize(text)
    word_feats = _features_for_words(words)
    vec = _aggregate(word_feats, feature_names, strategy)
    x = scaler.transform(vec.reshape(1, -1))[0]

    base = float(model.predict_proba(x.reshape(1, -1))[0, 1])

    k = len(feature_names)
    contrib = np.zeros(k)
    for j in range(x.shape[0]):
        if x[j] == 0.0:
            continue
        x_alt = x.copy()
        x_alt[j] = 0.0  # training mean in standardized space
        p_alt = float(model.predict_proba(x_alt.reshape(1, -1))[0, 1])
        contrib[j % k] += base - p_alt

    out = []
    for i in np.argsort(-np.abs(contrib)):
        if len(out) >= n:
            break
        out.append({"name": feature_names[i], "value": float(contrib[i])})
    return out
