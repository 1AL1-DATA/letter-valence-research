"""DEPRECATED three-tier letter cascade — kept for side-by-side evaluation only.

Tiers, cheapest-and-most-decisive first (this is the design ``engine.py``
replaced):

  1. ``dft``  - logistic probe on the 8 DFT spectral features; fires when ``|v| >= DFT_THRESHOLD``
  2. ``rf``   - letter-feature Random Forest (272-dim); fires when ``|v| >= RF_THRESHOLD``
  3. ``heavy`` - FinancialBERT 3-class (VADER fallback when the model is unavailable)

Valence ``v`` lives in ``[-1, 1]``:
  - binary tiers: ``v = 2 * p_positive - 1``
  - heavy tier:   ``v = p_positive - p_negative``
  - VADER:        ``v = compound``

``score = sign(v) * v**2`` (quadratic). Labels come from ``score`` thresholds
around the neutral band ``LABEL_BAND``.

Deprecation provenance (port from esg-dashboard, 2026-09-25):
the letter tiers were strictly dominated (README cascade follow-up): DFT
probe never fires on FPB (max |v| 0.922 < 0.95), letter RF fires 3.6%; on
240 live news headlines (LLM-judge gold, 92 neu / 89 pos / 59 neg) measured
precision-when-firing was flat 33-40% at EVERY band (chance ~37%, keyword
51.2%) -- confidence does not correlate with correctness on headline text.
Hand-set bands 0.30/0.10 (2026-09-24 experiment) reached 80.8% cheap
coverage at the same chance-level precision; superseded by the word-level
cheap tier in ``engine.py``. Original FPB-calibrated bands restored here.
"""

from __future__ import annotations

import logging
import math
import os
import pickle
import re
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

ASSET_DIR = Path(__file__).resolve().parent / "assets"

# Cascade thresholds (validated on 1,967 clear + 2,879 borderline FPB sentences):
# the DFT probe is unreliable below |v| >= 0.95 and the RF only pays off at
# |v| >= 0.8 (100% accuracy when it fires); the heavy tier carries the load.
DFT_THRESHOLD = 0.95
RF_THRESHOLD = 0.8
LABEL_BAND = 0.1

MODEL_ID = "ahmedrachid/FinancialBERT-Sentiment-Analysis"

# Heavy-tier inference memory budget (GB). Activation memory per chunk is capped
# so a full corpus can be scored without OOM-ing typical machines.
MAX_HEAVY_MEMORY_GB = float(os.environ.get("ESG_SENTIMENT_MEMORY_GB", "4"))
_HEAVY_FLOAT_BYTES = 4  # float32
_HEAVY_ACTIVATION_SAFETY = 2.5  # residual activations held per layer during fwd
_HEAVY_RUNTIME_OVERHEAD_BYTES = int(1.5e9)  # torch + transformers + runtime base

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

_BUNDLE_CACHE: dict[Path, dict] = {}
_VADER = None
_HEAVY_SINGLETON: _HeavyModel | None = None


def _tokenize(text: str) -> list[str]:
    """Lowercase alphabetic words (same as the RF training tokenizer)."""
    return re.findall(r"[a-z]+", (text or "").lower())


def keyword_valence(text: str) -> float:
    """Baseline keyword valence in [-1, 1] (equal-weight pos/neg lexicon)."""
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


def _norm_label(name: str) -> str:
    s = str(name).lower()
    if "pos" in s:
        return "positive"
    if "neg" in s:
        return "negative"
    return "neutral"


def _binary_proba(p: float) -> dict[str, float]:
    return {"negative": 1.0 - p, "positive": p}


class _HeavyModel:
    """Lazily-loaded FinancialBERT 3-class scorer with full softmax logits.

    Inference is batched dynamically: texts are grouped into chunks whose total
    token count keeps estimated activation memory under ``MAX_HEAVY_MEMORY_GB``,
    so arbitrarily large corpora score without exhausting RAM.
    """

    def __init__(self) -> None:
        self._tokenizer = None
        self._model = None
        self._label_names: dict[int, str] = {}
        self._max_tokens_per_chunk = 16_384

    def _ensure(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self._model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
        self._model.eval()
        id2label = getattr(self._model.config, "id2label", None) or {}
        for idx, name in id2label.items():
            self._label_names[int(idx)] = _norm_label(name)
        if not self._label_names:
            for idx, name in enumerate(("negative", "neutral", "positive")):
                self._label_names[idx] = name

        cfg = self._model.config
        layers = int(getattr(cfg, "num_hidden_layers", 12))
        hidden = int(getattr(cfg, "hidden_size", 768))
        n_params = sum(p.numel() for p in self._model.parameters())
        model_bytes = n_params * _HEAVY_FLOAT_BYTES
        budget = max(0.0, MAX_HEAVY_MEMORY_GB * 1e9 - model_bytes - _HEAVY_RUNTIME_OVERHEAD_BYTES)
        bytes_per_token = layers * hidden * _HEAVY_FLOAT_BYTES * _HEAVY_ACTIVATION_SAFETY
        self._max_tokens_per_chunk = max(64, int(budget // bytes_per_token))
        logger.info(
            "Heavy tier loaded: %s (CPU, %d chunks max %d tokens)",
            MODEL_ID, layers, self._max_tokens_per_chunk,
        )

    def _score_chunk(self, idxs: list[int], input_ids: list[list[int]], lengths: list[int],
                     results: list[dict | None], torch) -> None:
        max_len = max(lengths[i] for i in idxs)
        batch_ids = torch.full((len(idxs), max_len), self._tokenizer.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(idxs), max_len), dtype=torch.long)
        for r, i in enumerate(idxs):
            seq = torch.tensor(input_ids[i], dtype=torch.long)
            batch_ids[r, : seq.shape[0]] = seq
            mask[r, : seq.shape[0]] = 1
        with torch.inference_mode():
            logits = self._model(input_ids=batch_ids, attention_mask=mask).logits
        probs = torch.softmax(logits, dim=-1).numpy()
        for j, row in zip(idxs, probs):
            proba = {"negative": 0.0, "neutral": 0.0, "positive": 0.0}
            for idx, p in enumerate(row):
                proba[self._label_names.get(idx, "neutral")] = float(p)
            results[j] = proba

    def score(self, texts: list[str]) -> list[dict[str, float]]:
        import torch

        self._ensure()
        if not texts:
            return []
        enc = self._tokenizer(texts, padding=False, truncation=True, max_length=512)
        input_ids = enc["input_ids"]
        lengths = [len(ids) for ids in input_ids]
        order = sorted(range(len(texts)), key=lambda i: lengths[i])
        results: list[dict[str, float] | None] = [None] * len(texts)
        chunk: list[int] = []
        chunk_tokens = 0
        for i in order:
            n = lengths[i]
            if chunk and chunk_tokens + n > self._max_tokens_per_chunk:
                self._score_chunk(chunk, input_ids, lengths, results, torch)
                chunk, chunk_tokens = [], 0
            chunk.append(i)
            chunk_tokens += n
        if chunk:
            self._score_chunk(chunk, input_ids, lengths, results, torch)
        return results  # type: ignore[return-value]


def heavy_score_batch(texts: list[str]) -> list[dict[str, float]]:
    """Score texts with the 3-class FinancialBERT model (lazy, shared instance)."""
    global _HEAVY_SINGLETON
    if _HEAVY_SINGLETON is None:
        _HEAVY_SINGLETON = _HeavyModel()
    return _HEAVY_SINGLETON.score(texts)


def route_components(
    dft_p: np.ndarray,
    rf_p: np.ndarray,
    heavy_proba: list[dict[str, float] | None],
    vader_v: np.ndarray,
    n_words: list[int],
    *,
    dft_threshold: float = DFT_THRESHOLD,
    rf_threshold: float = RF_THRESHOLD,
    label_band: float = LABEL_BAND,
) -> list[dict]:
    """Route per-instance component scores through the cascade.

    ``dft_p`` / ``rf_p`` are P(positive) from the two trained tiers (empty to skip),
    ``heavy_proba`` a per-instance 3-class dict (``None`` when the heavy tier is
    unavailable for that instance), ``vader_v`` the VADER fallback valence.
    """
    dft_v = 2 * np.asarray(dft_p, dtype=float) - 1
    rf_v = 2 * np.asarray(rf_p, dtype=float) - 1
    out: list[dict] = []
    for i in range(len(dft_v)):
        if abs(dft_v[i]) >= dft_threshold:
            v, tier, proba = dft_v[i], "dft", _binary_proba(float(dft_p[i]))
        elif abs(rf_v[i]) >= rf_threshold:
            v, tier, proba = rf_v[i], "rf", _binary_proba(float(rf_p[i]))
        elif heavy_proba[i] is not None:
            p = heavy_proba[i]
            v = p["positive"] - p["negative"]
            tier, proba = "heavy", p
        else:
            v, tier, proba = float(vader_v[i]), "vader", None
        score = quadratic_score(v)
        out.append({
            "score": round(score, 6),
            "label": label_for_score(score, label_band),
            "v": round(float(v), 6),
            "tier": tier,
            "proba": proba,
            "confidence": round(max(proba.values()), 6) if proba else 1.0,
            "n_words": int(n_words[i]),
        })
    return out


class SentimentEngine:
    """The 3-tier cascade. Models are loaded lazily on first use."""

    def __init__(
        self,
        *,
        rf_bundle: dict | None = None,
        dft_bundle: dict | None = None,
        rf_path: Path | str | None = None,
        dft_path: Path | str | None = None,
        use_heavy: bool = True,
        dft_threshold: float = DFT_THRESHOLD,
        rf_threshold: float = RF_THRESHOLD,
        label_band: float = LABEL_BAND,
    ) -> None:
        self._rf_bundle = rf_bundle
        self._dft_bundle = dft_bundle
        self.rf_path = Path(rf_path) if rf_path else ASSET_DIR / "letter_sentiment_rf.pkl"
        self.dft_path = Path(dft_path) if dft_path else ASSET_DIR / "dft_probe.pkl"
        self.use_heavy = use_heavy
        self.dft_threshold = dft_threshold
        self.rf_threshold = rf_threshold
        self.label_band = label_band
        self._heavy: _HeavyModel | None = None

    def _load(self) -> tuple[dict, dict]:
        if self._rf_bundle is None:
            if self.rf_path not in _BUNDLE_CACHE:
                with open(self.rf_path, "rb") as fh:
                    _BUNDLE_CACHE[self.rf_path] = pickle.load(fh)
            self._rf_bundle = _BUNDLE_CACHE[self.rf_path]
        if self._dft_bundle is None:
            if self.dft_path not in _BUNDLE_CACHE:
                with open(self.dft_path, "rb") as fh:
                    _BUNDLE_CACHE[self.dft_path] = pickle.load(fh)
            self._dft_bundle = _BUNDLE_CACHE[self.dft_path]
        return self._rf_bundle, self._dft_bundle

    def component_scores(
        self, texts: list[str], *, use_heavy: bool | None = None
    ) -> dict:
        """Per-instance component valences for threshold sweeps / benchmarking."""
        from . import features as feat

        n = len(texts)
        tokens = [_tokenize(t) for t in texts]
        n_words = [len(ws) for ws in tokens]

        all_words: set[str] = set()
        for ws in tokens:
            all_words.update(ws)
        word_feats = feat.compute_features_for_words(list(all_words))

        rf_b, dft_b = self._load()
        x_rf = np.array([
            feat.aggregate_article(
                ws, word_feats, feats=rf_b["feature_names"], strategy=rf_b["strategy"]
            )
            for ws in tokens
        ])
        rf_p = rf_b["rf"].predict_proba(rf_b["scaler"].transform(x_rf))[:, 1]

        x_dft = np.array([
            feat.aggregate_article(
                ws, word_feats, feats=dft_b["feature_names"], strategy=dft_b["strategy"]
            )
            for ws in tokens
        ])
        dft_p = dft_b["model"].predict_proba(dft_b["scaler"].transform(x_dft))[:, 1]

        use_heavy = self.use_heavy if use_heavy is None else use_heavy
        heavy_proba: list[dict[str, float] | None] = [None] * n
        if use_heavy and self._heavy is None:
            self._heavy = _HeavyModel()
        if self._heavy is not None:
            routed = [i for i in range(n) if abs(2 * dft_p[i] - 1) < self.dft_threshold
                      and abs(2 * rf_p[i] - 1) < self.rf_threshold]
            if routed:
                try:
                    batch = [texts[i] for i in routed]
                    scores = self._heavy.score(batch)
                    for idx, proba in zip(routed, scores):
                        heavy_proba[idx] = proba
                except Exception as e:  # pragma: no cover - env dependent
                    logger.warning("Heavy tier failed (%s); using VADER fallback", e)

        vader_v = np.array([vader_valence(t) for t in texts])
        return {
            "dft_p": dft_p,
            "rf_p": rf_p,
            "heavy_proba": heavy_proba,
            "vader_v": vader_v,
            "n_words": n_words,
        }

    def score_batch(self, texts: list[str], *, use_heavy: bool | None = None) -> list[dict]:
        comps = self.component_scores(texts, use_heavy=use_heavy)
        return route_components(
            comps["dft_p"],
            comps["rf_p"],
            comps["heavy_proba"],
            comps["vader_v"],
            comps["n_words"],
            dft_threshold=self.dft_threshold,
            rf_threshold=self.rf_threshold,
            label_band=self.label_band,
        )

    def score_text(self, text: str, *, use_heavy: bool | None = None) -> dict:
        return self.score_batch([text], use_heavy=use_heavy)[0]


default_engine = SentimentEngine()
