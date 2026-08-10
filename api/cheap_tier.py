"""Self-contained word-level cheap tier: TF-IDF(1-2 grams) + VADER + keyword.

Replicates the cascade benchmark's cheap tier (src/benchmark_cascade.py):
  3-class logistic regression on TF-IDF 1-2 grams stacked with
  [VADER compound, keyword valence]; valence v = P(positive) - P(negative).
"""
from __future__ import annotations

import math
import pickle
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scipy import sparse  # noqa: E402

MODEL_PATH = REPO / "models" / "cheap_tier.pkl"

LABEL_BAND = 0.1
CHEAP_THRESHOLD = 0.6

POS_WORDS = {
    "beat", "surge", "gain", "gains", "record", "growth", "rally",
    "upgrade", "upgrades", "profit", "strong", "raise", "raises",
    "outperform", "boost", "boosted", "jump", "jumps", "climb",
    "wins", "positive", "buy", "soar", "tops", "milestone",
}
NEG_WORDS = {
    "miss", "misses", "drop", "drops", "fall", "falls", "decline",
    "declines", "cut", "cuts", "downgrade", "downgrades", "loss",
    "losses", "lawsuit", "weak", "risk", "warning", "below", "sell",
    "plunge", "slump", "probe", "investigation", "layoffs", "recall",
}

_BUNDLE = None
_VADER = None


def keyword_valence(text: str) -> float:
    """Keyword baseline valence in [-1, 1] (equal-weight pos/neg lexicon)."""
    words = set(re.findall(r"[a-z']+", (text or "").lower()))
    pos = len(words & POS_WORDS)
    neg = len(words & NEG_WORDS)
    if pos + neg == 0:
        return 0.0
    return (pos - neg) / (pos + neg)


def vader_valence(text: str) -> float:
    """VADER compound score mapped to [-1, 1]."""
    global _VADER
    if _VADER is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        _VADER = SentimentIntensityAnalyzer()
    return float(_VADER.polarity_scores(text or "")["compound"])


def quadratic_score(v: float) -> float:
    """Map a valence ``v`` in [-1, 1] to ``sign(v) * v**2``."""
    return math.copysign(v * v, v)


def label_for_score(score: float, band: float = LABEL_BAND) -> str:
    if score >= band:
        return "positive"
    if score <= -band:
        return "negative"
    return "neutral"


def _load() -> dict:
    global _BUNDLE
    if _BUNDLE is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Cheap tier model not found at {MODEL_PATH}. "
                "Run `python -m scripts.train_cheap_tier` first."
            )
        with open(MODEL_PATH, "rb") as fh:
            _BUNDLE = pickle.load(fh)
    return _BUNDLE


def predict(text: str) -> dict:
    """Classify with the word-level cheap tier.

    Returns: label, valence, score, proba {pos, neu, neg}, decided (bool).
    """
    bundle = _load()
    vec = bundle["vectorizer"]
    lr = bundle["model"]
    classes = bundle["classes"]

    x = vec.transform([text or ""])
    extra = np.asarray([[vader_valence(text), keyword_valence(text)]])
    x = sparse.hstack([x, sparse.csr_matrix(extra)]).tocsr()

    proba = lr.predict_proba(x)[0]
    pos_col = classes.index(2)
    neg_col = classes.index(0)
    neu_col = classes.index(1)
    valence = float(proba[pos_col] - proba[neg_col])
    score = quadratic_score(valence)

    return {
        "label": label_for_score(score),
        "valence": round(valence, 4),
        "score": round(score, 4),
        "proba": {
            "positive": round(float(proba[pos_col]), 4),
            "neutral": round(float(proba[neu_col]), 4),
            "negative": round(float(proba[neg_col]), 4),
        },
        "decided": abs(valence) >= CHEAP_THRESHOLD,
    }
