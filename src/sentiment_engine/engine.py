"""Two-tier sentiment cascade with quadratic scoring.

Tiers, cheapest-and-most-decisive first:

  1. ``cheap``  - word-level tier: TF-IDF (1-2 grams) + [VADER, keyword] extras
                  through a 3-class logistic regression; fires when
                  ``|v| >= CHEAP_THRESHOLD``.
  2. ``heavy``  - FinancialBERT 3-class (VADER fallback when the model is
                  unavailable), then the keyword lexicon as last resort.

Valence ``v`` lives in ``[-1, 1]``:
  - cheap tier: ``v = p_positive - p_negative``
  - heavy tier: ``v = p_positive - p_negative``
  - VADER:      ``v = compound``

``score = sign(v) * v**2`` (quadratic). Labels come from ``score`` thresholds
around the neutral band ``LABEL_BAND``.

Provenance (2026-09-25 port from letter-valence-research benchmarks):
the previous 3-tier letter cascade (dft/rf) was strictly dominated --
the DFT probe never fired on FPB (max |v| 0.922 < 0.95) and the letter RF
fired on 3.6%; on 240 live headlines measured precision was flat 33-40%
at every band (chance ~37%, keyword fallback 51%). See
results/cascade_benchmark.json for the 2-tier evaluation (cheap tier decides
36.6% of clear-polarity calls at 97.2% accuracy; cascade 0.9512 vs 0.9558
heavy-only, McNemar p = 0.15).
"""

from __future__ import annotations

import logging
import math
import os
import pickle
import re
from pathlib import Path

import numpy as np
from scipy import sparse

logger = logging.getLogger(__name__)

ASSET_DIR = Path(__file__).resolve().parent / "assets"

# Cheap-tier operating point (chosen in results/cascade_benchmark.json sweep):
# the cheap word tier decides only when |v| >= 0.6; everything else falls to
# the heavy tier.
CHEAP_THRESHOLD = 0.6
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

_CHEAP_CACHE: dict[Path, dict] = {}
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


def _load_cheap_bundle() -> dict:
    path = ASSET_DIR / "cheap_tier.pkl"
    if path not in _CHEAP_CACHE:
        with open(path, "rb") as fh:
            _CHEAP_CACHE[path] = pickle.load(fh)
    return _CHEAP_CACHE[path]


def cheap_valence(texts: list[str], vader_v: np.ndarray, kw_v: np.ndarray) -> np.ndarray:
    """Word-level cheap tier: TF-IDF(1-2) + [VADER, keyword] extras -> v in [-1, 1].

    The bundle (models/cheap_tier.pkl, trained on FinancialPhraseBank) is
    ``{"vectorizer", "model", "classes"}``; ``v = p_pos - p_neg``.
    """
    b = _load_cheap_bundle()
    x = b["vectorizer"].transform(texts)
    extra = sparse.csr_matrix(np.column_stack([vader_v, kw_v]))
    x = sparse.hstack([x, extra]).tocsr()
    proba = b["model"].predict_proba(x)
    classes = [int(c) for c in b["classes"]]
    return proba[:, classes.index(2)] - proba[:, classes.index(0)]


def route_components(
    cheap_v: np.ndarray,
    heavy_proba: list[dict[str, float] | None],
    vader_v: np.ndarray,
    n_words: list[int],
    *,
    cheap_threshold: float = CHEAP_THRESHOLD,
    label_band: float = LABEL_BAND,
) -> list[dict]:
    """Route per-instance component valences through the 2-tier cascade.

    ``cheap_v`` are the word-tier valences, ``heavy_proba`` a per-instance
    3-class dict (``None`` when the heavy tier is unavailable for that
    instance), ``vader_v`` the VADER fallback valence.
    """
    cheap_v = np.asarray(cheap_v, dtype=float)
    out: list[dict] = []
    for i in range(len(cheap_v)):
        if abs(cheap_v[i]) >= cheap_threshold:
            v, tier = cheap_v[i], "cheap"
        elif heavy_proba[i] is not None:
            p = heavy_proba[i]
            v, tier = p["positive"] - p["negative"], "heavy"
        else:
            v, tier = float(vader_v[i]), "vader"
        score = quadratic_score(v)
        out.append({
            "score": round(score, 6),
            "label": label_for_score(score, label_band),
            "v": round(float(v), 6),
            "tier": tier,
            "proba": heavy_proba[i] if tier == "heavy" else None,
            "confidence": round(max(heavy_proba[i].values()), 6)
            if tier == "heavy" and heavy_proba[i] else 1.0,
            "n_words": int(n_words[i]),
        })
    return out


class SentimentEngine:
    """The 2-tier cascade. Models are loaded lazily on first use."""

    def __init__(
        self,
        *,
        cheap_path: Path | str | None = None,
        use_heavy: bool = True,
        cheap_threshold: float = CHEAP_THRESHOLD,
        label_band: float = LABEL_BAND,
    ) -> None:
        self.cheap_path = Path(cheap_path) if cheap_path else ASSET_DIR / "cheap_tier.pkl"
        self.use_heavy = use_heavy
        self.cheap_threshold = cheap_threshold
        self.label_band = label_band
        self._heavy: _HeavyModel | None = None

    def component_scores(
        self, texts: list[str], *, use_heavy: bool | None = None
    ) -> dict:
        """Per-instance component valences for threshold sweeps / benchmarking."""
        n = len(texts)
        n_words = [len(_tokenize(t)) for t in texts]
        vader_v = np.array([vader_valence(t) for t in texts])
        kw_v = np.array([keyword_valence(t) for t in texts])
        cheap_v = cheap_valence(texts, vader_v, kw_v)

        use_heavy = self.use_heavy if use_heavy is None else use_heavy
        heavy_proba: list[dict[str, float] | None] = [None] * n
        if use_heavy and self._heavy is None:
            self._heavy = _HeavyModel()
        if self._heavy is not None:
            routed = [i for i in range(n) if abs(cheap_v[i]) < self.cheap_threshold]
            if routed:
                try:
                    batch = [texts[i] for i in routed]
                    scores = self._heavy.score(batch)
                    for idx, proba in zip(routed, scores):
                        heavy_proba[idx] = proba
                except Exception as e:  # pragma: no cover - env dependent
                    logger.warning("Heavy tier failed (%s); using VADER fallback", e)

        return {
            "cheap_v": cheap_v,
            "heavy_proba": heavy_proba,
            "vader_v": vader_v,
            "keyword_v": kw_v,
            "n_words": n_words,
        }

    def score_batch(self, texts: list[str], *, use_heavy: bool | None = None) -> list[dict]:
        comps = self.component_scores(texts, use_heavy=use_heavy)
        return route_components(
            comps["cheap_v"],
            comps["heavy_proba"],
            comps["vader_v"],
            comps["n_words"],
            cheap_threshold=self.cheap_threshold,
            label_band=self.label_band,
        )

    def score_text(self, text: str, *, use_heavy: bool | None = None) -> dict:
        return self.score_batch([text], use_heavy=use_heavy)[0]


default_engine = SentimentEngine()
