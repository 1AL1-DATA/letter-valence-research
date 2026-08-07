"""Statistically-significant evaluation of the 2-tier sentiment cascade.

Run from the research repo root with the dashboard workspace venv:
    cd /home/a/letter-valence-research
    /home/a/esg-dashboard/.venv/bin/python -m src.benchmark_cascade

Design
------
* Clear-polarity set (n = 1,967 pos/neg FinancialPhraseBank sentences):
  identical 5-fold stratified test predictions for every method
  {cheap, cascade, heavy, vader, keyword}. The cheap tier (word-level
  TF-IDF + VADER + keyword features, 3-class logistic regression) is
  retrained per fold on the train split; the transformer / VADER /
  keyword baselines are fixed.
* Borderline set (n = 2,879 held-out neutral sentences): false-polarity rate
  and score distributions for every method (deployed, full-data models).
* Metrics: accuracy + Wilson 95% CI, macro-F1, per-class precision/recall/F1,
  exact McNemar significance (Bonferroni-corrected), cascade tier routing.
* Threshold sweep over (cheap_threshold, label_band) re-routed cheaply
  on stored per-instance component valences.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

DASHBOARD_SRC = Path(os.environ.get("ESG_DASHBOARD_SRC", "/home/a/esg-dashboard/src"))
sys.path.insert(0, str(DASHBOARD_SRC))

from esg_dashboard.data.sentiment_engine import (  # noqa: E402
    LABEL_BAND,
    heavy_score_batch,
    keyword_valence,
    label_for_score,
    quadratic_score,
    vader_valence,
)

RESULTS_DIR = REPO / "results"
FIGURES_DIR = REPO / "figures"

# Chosen operating point: the cheap tier decides only when |v| >= 0.6;
# everything else falls through to the heavy (FinancialBERT) tier.
CHEAP_THRESHOLD = 0.6

METHODS = ("cheap", "cascade", "heavy", "vader", "keyword")

SENTENCES_PATH = REPO / "data" / "Sentences_50Agree.txt"


# ---- data loading ----
def load_fpb() -> pd.DataFrame:
    rows = []
    with open(SENTENCES_PATH, encoding="latin-1") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if "@" not in line:
                continue
            text, label = line.rsplit("@", 1)
            label = label.strip().lower()
            if label not in ("negative", "neutral", "positive"):
                continue
            rows.append({"id": i, "text": text.strip().strip('"'), "label": label})
    df = pd.DataFrame(rows)
    df["y"] = df["label"].map({"negative": 0, "neutral": 1, "positive": 2})
    df["words"] = df["text"].apply(lambda t: re.findall(r"[a-z]+", t.lower()))
    return df


# ---- metrics ----
def wilson_ci(n: int, k: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value (binomial), via scipy (numerically stable)."""
    from scipy.stats import binomtest

    m = b + c
    if m == 0:
        return 1.0
    return float(binomtest(min(b, c), m, 0.5, alternative="two-sided").pvalue)


def binary_metrics(y_true: np.ndarray, pred: np.ndarray, pos_label: int = 2) -> dict:
    n = len(y_true)
    acc = float((pred == y_true).mean())
    lo, hi = wilson_ci(n, int((pred == y_true).sum()))
    out = {"n": n, "accuracy": round(acc, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4)}
    for label, code in (("negative", 0), ("positive", 2)):
        tp = int(((y_true == code) & (pred == code)).sum())
        fp = int(((y_true != code) & (pred == code)).sum())
        fn = int(((y_true == code) & (pred != code)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[f"{label}_precision"] = round(prec, 4)
        out[f"{label}_recall"] = round(rec, 4)
        out[f"{label}_f1"] = round(f1, 4)
    out["macro_f1"] = round((out["negative_f1"] + out["positive_f1"]) / 2, 4)
    out["neutral_predicted"] = float((pred == 1).mean())
    return out


def predict_from_v(v: np.ndarray, band: float = LABEL_BAND) -> np.ndarray:
    codes = {"positive": 2, "negative": 0, "neutral": 1}
    return np.array([codes[label_for_score(quadratic_score(x), band)] for x in v])


def cascade_predict(cheap_v, heavy_v, vader_v, n_words, *, threshold, band):
    """Route per-instance cheap-tier valences through the 2-tier cascade."""
    cheap_v = np.asarray(cheap_v, dtype=float)
    heavy_v = np.asarray(heavy_v, dtype=float)
    labels = []
    tiers = []
    for i in range(len(cheap_v)):
        if abs(cheap_v[i]) >= threshold:
            v, tier = cheap_v[i], "cheap"
        elif not np.isnan(heavy_v[i]):
            v, tier = heavy_v[i], "heavy"
        else:
            v, tier = float(vader_v[i]), "vader"
        labels.append(label_for_score(quadratic_score(v), band))
        tiers.append(tier)
    label_code = np.array([
        {"positive": 2, "negative": 0, "neutral": 1}[label] for label in labels
    ])
    return label_code, tiers


def cheap_tier_cv(texts, y3, extra, *, n_splits=5, seed=42):
    """5-fold stratified CV valences for the word-level cheap tier.

    The cheap tier is a 3-class logistic regression on TF-IDF(1-2 grams) +
    [VADER compound, keyword valence]; v = p_positive - p_negative.
    """
    from scipy import sparse
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    n = len(texts)
    v = np.full(n, np.nan)
    fold_of = np.zeros(n, dtype=int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tr, te) in enumerate(skf.split(texts, y3)):
        fold_of[te] = fold
        vec = TfidfVectorizer(
            lowercase=True, token_pattern=r"[a-z]+",
            ngram_range=(1, 2), min_df=2, max_features=20_000,
        )
        x_tr = vec.fit_transform([texts[i] for i in tr])
        x_te = vec.transform([texts[i] for i in te])
        extra_tr = np.asarray([extra[i] for i in tr])
        extra_te = np.asarray([extra[i] for i in te])
        x_tr = sparse.hstack([x_tr, sparse.csr_matrix(extra_tr)]).tocsr()
        x_te = sparse.hstack([x_te, sparse.csr_matrix(extra_te)]).tocsr()
        lr = LogisticRegression(
            C=1.0, max_iter=2000, random_state=42, class_weight="balanced"
        )
        lr.fit(x_tr, y3[tr])
        proba = lr.predict_proba(x_te)
        classes = list(lr.classes_)
        pos_col = classes.index(2)
        neg_col = classes.index(0)
        v[te] = proba[:, pos_col] - proba[:, neg_col]
    return v, fold_of


# ---- main ----
def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    print("Loading FinancialPhraseBank 3-class...")
    df = load_fpb()
    clear = df[df["label"] != "neutral"].reset_index(drop=True)
    neut = df[df["label"] == "neutral"].reset_index(drop=True)
    y_true = clear["y"].to_numpy()
    print(f"  clear n={len(clear)} (pos={(y_true == 2).sum()}, neg={(y_true == 0).sum()})")
    print(f"  neutral n={len(neut)}")

    all_texts = df["text"].tolist()
    y3 = df["y"].to_numpy()

    print("Scoring VADER + keyword baselines...")
    vader_v_all = np.array([vader_valence(t) for t in all_texts])
    kw_v_all = np.array([keyword_valence(t) for t in all_texts])

    print("Scoring heavy tier on ALL 4,846 sentences (once)...")
    heavy_proba_all = heavy_score_batch(all_texts)
    heavy_v_all = np.array([p["positive"] - p["negative"] for p in heavy_proba_all])
    print("  done.")

    print("5-fold CV for the word-level cheap tier...")
    extra = list(zip(vader_v_all, kw_v_all))
    cheap_v_all, fold_of_all = cheap_tier_cv(all_texts, y3, extra)

    # ---- slice to the two evaluation sets ----
    clear_idx = (df["label"] != "neutral").to_numpy()
    neut_idx = ~clear_idx
    cheap_v_clear = cheap_v_all[clear_idx]
    heavy_v_clear = heavy_v_all[clear_idx]
    vader_v_clear = vader_v_all[clear_idx]
    kw_v_clear = kw_v_all[clear_idx]
    n_words_clear = clear["words"].str.len().to_numpy()

    print("Evaluating methods on the clear set...")
    y_code = y_true  # 0 / 2
    method_labels: dict[str, np.ndarray] = {
        "cheap": predict_from_v(cheap_v_clear),
        "heavy": predict_from_v(heavy_v_clear),
        "vader": predict_from_v(vader_v_clear),
        "keyword": predict_from_v(kw_v_clear),
    }
    cascade_code, cascade_tiers = cascade_predict(
        cheap_v_clear, heavy_v_clear, vader_v_clear, n_words_clear,
        threshold=CHEAP_THRESHOLD, band=LABEL_BAND,
    )
    method_labels["cascade"] = cascade_code

    metric_sets = {}
    for m in METHODS:
        metric_sets[m] = binary_metrics(y_code, method_labels[m])

    print("  accuracy / neg-recall / macro-F1:")
    for m in METHODS:
        s = metric_sets[m]
        msg = (
            f"    {m:>8} acc={s['accuracy']:.4f} "
            f"neg_rec={s['negative_recall']:.4f} macroF1={s['macro_f1']:.4f}"
        )
        print(msg)

    # ---- McNemar: cascade vs each method ----
    mcnemar = {}
    for m in METHODS:
        if m == "cascade":
            continue
        a = cascade_code != y_code
        b = method_labels[m] != y_code
        both_wrong = (a & b).sum()
        casc_only = (a & ~b).sum()
        other_only = (~a & b).sum()
        p = mcnemar_exact(int(casc_only), int(other_only))
        mcnemar[m] = {
            "both_wrong": int(both_wrong),
            "cascade_wrong_only": int(casc_only),
            "other_wrong_only": int(other_only),
            "p_exact": round(p, 6),
            "significant_0.01": p < 0.01,
        }

    # ---- cascade tier routing on the clear set ----
    cascade_tiers_arr = np.asarray(cascade_tiers)
    tiers, tier_counts = np.unique(cascade_tiers_arr, return_counts=True)
    tier_routing = {}
    for t, c in zip(tiers, tier_counts):
        idx = cascade_tiers_arr == t
        tier_routing[str(t)] = {
            "n": int(c),
            "share": round(c / len(clear), 4),
            "accuracy": round(float((cascade_code[idx] == y_code[idx]).mean()), 4),
        }

    # ---- threshold sweep (cheap re-route on stored valences) ----
    print("Threshold sweep...")
    sweep = []
    best = None
    for ct in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        for band in (0.05, 0.1, 0.15):
            code, tiers_s = cascade_predict(
                cheap_v_clear, heavy_v_clear, vader_v_clear, n_words_clear,
                threshold=ct, band=band,
            )
            acc = float((code == y_code).mean())
            heavy_share = float(np.mean([t == "heavy" for t in tiers_s]))
            entry = {
                "cheap_threshold": ct, "band": band, "accuracy": round(acc, 4),
                "heavy_share": round(heavy_share, 4),
            }
            sweep.append(entry)
            if best is None or acc > best["accuracy"]:
                best = entry
    sweep.sort(key=lambda e: e["accuracy"], reverse=True)

    # ---- borderline (neutral) set, deployed components ----
    print("Scoring neutral set...")
    nw = neut["words"].str.len().to_numpy()
    neutral_cascade_code, neutral_tiers = cascade_predict(
        cheap_v_all[neut_idx], heavy_v_all[neut_idx], vader_v_all[neut_idx], nw,
        threshold=CHEAP_THRESHOLD, band=LABEL_BAND,
    )
    neutral_methods = {
        "cheap": predict_from_v(cheap_v_all[neut_idx]),
        "heavy": predict_from_v(heavy_v_all[neut_idx]),
        "vader": predict_from_v(vader_v_all[neut_idx]),
        "keyword": predict_from_v(kw_v_all[neut_idx]),
        "cascade": neutral_cascade_code,
    }

    borderline = {}
    for m in METHODS:
        pred = neutral_methods[m]
        n = len(pred)
        false_pol = float(((pred == 0) | (pred == 2)).mean())
        borderline[m] = {
            "n": int(n),
            "pred_negative": float((pred == 0).mean()),
            "pred_neutral": float((pred == 1).mean()),
            "pred_positive": float((pred == 2).mean()),
            "false_polarity_rate": round(false_pol, 4),
        }
    nt, nt_counts = np.unique(np.asarray(neutral_tiers), return_counts=True)
    borderline_tier_routing = {
        str(t): {"n": int(c), "share": round(c / len(neutral_tiers), 4)}
        for t, c in zip(nt, nt_counts)
    }

    # ---- save ----
    report = {
        "config": {
            "cheap_threshold": CHEAP_THRESHOLD,
            "label_band": LABEL_BAND,
            "model": "ahmedrachid/FinancialBERT-Sentiment-Analysis",
            "clear_n": len(clear),
            "neutral_n": len(neut),
            "cv": "5-fold stratified, retrain cheap tier per fold",
        },
        "clear_set": metric_sets,
        "mcnemar_vs_cascade": mcnemar,
        "tier_routing_clear": tier_routing,
        "threshold_sweep_top": sweep[:8],
        "threshold_sweep_best": best,
        "borderline_set": borderline,
        "borderline_tier_routing": borderline_tier_routing,
    }
    with open(RESULTS_DIR / "cascade_benchmark.json", "w") as fh:
        json.dump(report, fh, indent=2)

    pred_rows = []
    for i in range(len(clear)):
        pred_rows.append({
            "set": "clear", "id": int(clear["id"].iloc[i]),
            "fold": int(fold_of_all[np.nonzero(clear_idx)[0][i]]),
            "true": int(y_code[i]),
            "cheap_v": round(float(cheap_v_clear[i]), 6),
            "heavy_v": round(float(heavy_v_clear[i]), 6),
            "vader_v": round(float(vader_v_clear[i]), 6),
            "kw_v": round(float(kw_v_clear[i]), 6),
            "cascade_label": int(cascade_code[i]), "cascade_tier": cascade_tiers[i],
        })
    for i in range(len(neut)):
        pred_rows.append({
            "set": "neutral", "id": int(neut["id"].iloc[i]), "fold": -1,
            "true": 1,
            "cheap_v": round(float(cheap_v_all[neut_idx][i]), 6),
            "heavy_v": round(float(heavy_v_all[neut_idx][i]), 6),
            "vader_v": round(float(vader_v_all[neut_idx][i]), 6),
            "kw_v": round(float(kw_v_all[neut_idx][i]), 6),
            "cascade_label": int(neutral_cascade_code[i]),
            "cascade_tier": neutral_tiers[i],
        })
    pd.DataFrame(pred_rows).to_csv(RESULTS_DIR / "cascade_predictions.csv", index=False)

    print()
    print("Saved results/cascade_benchmark.json and results/cascade_predictions.csv")
    print("Best sweep config:", best)
    print("Chosen operating point: cheap_threshold=0.6, label_band=0.1")


if __name__ == "__main__":
    main()
