"""Per-instance SHAP attribution for the letter-feature RandomForest.

SHAP values are computed on the scaled 272-dim aggregated vector, then
collapsed back onto the 68 base letter features by summing the four
aggregation blocks (mean, max, min, std) that share a base feature.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.classify import _aggregate, _features_for_words, _load_model, _tokenize  # noqa: E402

_explainer = None


def top_features(text: str, n: int = 8) -> list[dict]:
    """Return the top-n base letter features by |SHAP| contribution.

    Each item is {"name": str, "value": float} where a positive value means
    the feature pushed the prediction toward positive, negative toward negative.
    """
    global _explainer

    model, feature_names, strategy, scaler = _load_model()
    if _explainer is None:
        import shap  # noqa: PLC0415 (heavy import, keep off the hot path)

        _explainer = shap.TreeExplainer(model)

    words = _tokenize(text)
    word_feats = _features_for_words(words)
    vec = _aggregate(word_feats, feature_names, strategy)
    x = vec.reshape(1, -1)
    x_scaled = scaler.transform(x)

    sv = _explainer.shap_values(x_scaled)
    arr = sv[1] if isinstance(sv, list) else sv
    if arr.ndim == 3:
        arr = arr[:, :, 1]

    k = len(feature_names)
    contrib = np.zeros(k)
    for j in range(x.shape[1]):
        contrib[j % k] += float(arr[0, j])

    out = []
    for i in np.argsort(-np.abs(contrib)):
        if len(out) >= n:
            break
        out.append({"name": feature_names[i], "value": float(contrib[i])})
    return out
