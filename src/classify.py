"""Use a trained letter-feature model to classify new paragraphs.

Two interfaces:
  1. CLI:  python -m src.classify --text "Some financial news here."
          python -m src.classify --file report.txt
          python -m src.classify --text "..." --compare  (also run VADER)
  2. Library:  from src.classify import classify, classify_batch
                result = classify("Some text", return_features=True)

The model expects 68 features per word, aggregated across the words in
the input (mean, max, min, std = 272-dim vector per paragraph).
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.features import (
    features as feature_one_word,
    get_feature_names,
)

DEFAULT_MODEL = REPO / "models" / "letter_sentiment_rf.pkl"


def _load_model(path: Path = DEFAULT_MODEL) -> tuple:
    """Load the saved model + feature pipeline.

    Returns: (model, feature_names, strategy, scaler)
    The pickle bundle is {"model": ..., "pipeline": FeaturePipeline}
    where the pipeline has .feature_names, .strategy, .scaler as attributes.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. "
            "Run `python -m src.train_final` to train and save the model first."
        )
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    pipe = bundle["pipeline"]
    return (
        bundle["model"],
        pipe.feature_names,
        pipe.strategy,
        pipe.scaler,
    )


def _tokenize(text: str) -> list[str]:
    """Lowercase, extract alphabetic words, drop single-character noise."""
    return [w for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 1]


def _features_for_words(words: list[str]) -> np.ndarray:
    """Compute 68 features for each word; return shape (n_words, 68)."""
    if not words:
        return np.zeros((1, len(get_feature_names())), dtype=float)
    out = np.array([list(feature_one_word(w).values()) for w in words], dtype=float)
    return out


def _aggregate(word_feats: np.ndarray, feature_names: list[str], strategy: str) -> np.ndarray:
    """Aggregate per-word features into a single 1D vector.

    Strategies (per src/features.py::aggregate_article):
      - 'mean': (68,)
      - 'max': (68,)
      - 'min': (68,)
      - 'std': (68,)
      - 'sign_count_pos': (68,)
      - 'all': mean + max + min + std = (272,)
    """
    if strategy == "mean":
        return word_feats.mean(axis=0)
    if strategy == "max":
        return word_feats.max(axis=0)
    if strategy == "min":
        return word_feats.min(axis=0)
    if strategy == "std":
        return word_feats.std(axis=0)
    if strategy == "sign_count_pos":
        return (word_feats > 0).mean(axis=0)
    if strategy == "all":
        return np.concatenate([
            word_feats.mean(axis=0),
            word_feats.max(axis=0),
            word_feats.min(axis=0),
            word_feats.std(axis=0),
        ])
    raise ValueError(f"Unknown strategy: {strategy}")


def classify(
    text: str,
    return_proba: bool = True,
    return_features: bool = False,
    model_path: Path = DEFAULT_MODEL,
) -> dict:
    """Classify a single paragraph of text.

    Returns a dict with:
        - label: str ("positive" or "negative")
        - label_code: int (1 or 0)
        - proba: dict with "negative" and "positive" probabilities (if return_proba)
        - confidence: float (max prob, if return_proba)
        - top_features: list of (name, value) (if return_features)
        - n_words: int

    The function loads the model from disk on first call and caches it.
    """
    if not hasattr(classify, "_model_cache"):
        classify._model_cache = {}

    if model_path not in classify._model_cache:
        classify._model_cache[model_path] = _load_model(model_path)

    model, feature_names, strategy, scaler = classify._model_cache[model_path]

    words = _tokenize(text)
    word_feats = _features_for_words(words)
    vec = _aggregate(word_feats, feature_names, strategy)
    X = vec.reshape(1, -1)
    X_scaled = scaler.transform(X)

    label_code = int(model.predict(X_scaled)[0])
    label = "positive" if label_code == 1 else "negative"

    out = {
        "text": text,
        "n_words": len(words),
        "label": label,
        "label_code": label_code,
    }

    if return_proba and hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_scaled)[0]
        out["proba"] = {
            "negative": float(proba[0]),
            "positive": float(proba[1]),
        }
        out["confidence"] = float(proba.max())

    if return_features:
        out["top_features"] = sorted(
            zip(feature_names, vec.tolist()),
            key=lambda kv: abs(kv[1]), reverse=True,
        )[:10]

    return out


def classify_batch(texts: list[str], model_path: Path = DEFAULT_MODEL) -> list[dict]:
    """Classify a list of paragraphs."""
    return [classify(t, model_path=model_path) for t in texts]


# ---- VADER baseline (for comparison) ----
def vader_score(text: str) -> dict:
    """Score with VADER; returns a dict with the compound score and the label.

    Threshold 0: compound >= 0 -> positive, else negative.
    Threshold -0.05 (tuned for FPB): compound >= -0.05 -> positive, else negative.
    """
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError:
        return {"error": "vaderSentiment not installed"}
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    return {
        "compound": scores["compound"],
        "neg": scores["neg"],
        "neu": scores["neu"],
        "pos": scores["pos"],
        "label_default": "positive" if scores["compound"] >= 0 else "negative",
        "label_tuned": "positive" if scores["compound"] >= -0.05 else "negative",
    }


# ---- CLI ----
def _print_result(result: dict, compare: bool = False, fp=sys.stdout) -> None:
    """Print a single classification result."""
    print("=" * 78, file=fp)
    print(f"TEXT: {result['text'][:200]}{'...' if len(result['text']) > 200 else ''}", file=fp)
    print(f"  n_words:  {result['n_words']}", file=fp)
    print(f"  LABEL:    {result['label'].upper()}", file=fp)
    if "proba" in result:
        print(f"  P(pos):   {result['proba']['positive']:.3f}", file=fp)
        print(f"  P(neg):   {result['proba']['negative']:.3f}", file=fp)
        print(f"  confidence: {result['confidence']:.3f}", file=fp)
    if "top_features" in result:
        print("  top features (by |value|):", file=fp)
        for name, val in result["top_features"]:
            print(f"    {name:30s} = {val:+.3f}", file=fp)
    if compare:
        v = vader_score(result["text"])
        if "error" not in v:
            print(f"  VADER compound: {v['compound']:+.3f}", file=fp)
            print(f"  VADER default:  {v['label_default']}", file=fp)
            print(f"  VADER tuned:    {v['label_tuned']}", file=fp)
    print(file=fp)


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify a financial paragraph using letter-derived features.")
    parser.add_argument("--text", help="The paragraph to classify")
    parser.add_argument("--file", help="Path to a file containing the paragraph(s), one per line")
    parser.add_argument("--compare", action="store_true", help="Also show VADER scores")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--features", action="store_true", help="Show top features")
    args = parser.parse_args()

    if not args.text and not args.file:
        parser.print_help()
        print()
        print("Error: provide --text or --file")
        sys.exit(1)

    texts = []
    if args.text:
        texts.append(args.text)
    if args.file:
        path = Path(args.file)
        if not path.exists():
            print(f"File not found: {path}")
            sys.exit(1)
        with open(path) as f:
            texts.extend(line.strip() for line in f if line.strip())

    results = classify_batch(texts)
    if args.json:
        out = [{k: v for k, v in r.items() if k != "top_features"} for r in results]
        if args.features:
            for r, o in zip(results, out):
                o["top_features"] = [{"name": n, "value": v} for n, v in r.get("top_features", [])]
        print(json.dumps(out, indent=2))
    else:
        for r in results:
            _print_result(r, compare=args.compare)


if __name__ == "__main__":
    main()
