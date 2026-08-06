"""Model evaluation utilities.

Provides:
- cross_validate: stratified k-fold CV with multiple metrics
- permutation_test: shuffle-label null distribution for significance testing
- learning_curve: train on subsets of increasing size, plot-style output
- family_ablation: leave-one-family-out
- baseline_metrics: dummy/majority-class baseline
- format_report: human-readable results
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold

from .train import FeaturePipeline, make_model, predict, train


# ---- baselines ----
def baseline_majority_class(y: np.ndarray) -> dict:
    """Score the always-predict-majority classifier."""
    majority = int(np.bincount(y).argmax())
    preds = np.full_like(y, majority)
    return _score(y, preds, name="majority_class")


def baseline_stratified_random(y: np.ndarray, n_repeats: int = 100, seed: int = 42) -> dict:
    """Score a stratified-random classifier (mean over n_repeats)."""
    rng = np.random.RandomState(seed)
    accs, f1s = [], []
    for _ in range(n_repeats):
        preds = rng.permutation(y)
        accs.append(accuracy_score(y, preds))
        f1s.append(f1_score(y, preds, average="binary", zero_division=0))
    return {
        "name": "stratified_random",
        "accuracy": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "f1": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
    }


# ---- single-fold scoring ----
def _score(y_true: np.ndarray, y_pred: np.ndarray, name: str = "model") -> dict:
    """Compute accuracy, F1, and a per-class breakdown."""
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    p, r, f, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    return {
        "name": name,
        "accuracy": float(acc),
        "f1": float(f1),
        "f1_macro": float(f1_macro),
        "precision_neg": float(p[0]),
        "recall_neg": float(r[0]),
        "f1_neg": float(f[0]),
        "precision_pos": float(p[1]),
        "recall_pos": float(r[1]),
        "f1_pos": float(f[1]),
        "support_neg": int(sup[0]),
        "support_pos": int(sup[1]),
    }


# ---- k-fold cross-validation ----
def cross_validate(
    articles: list[list[str]],
    y: np.ndarray,
    model_name: str = "rf",
    n_splits: int = 5,
    seed: int = 42,
    feature_names: Optional[list[str]] = None,
    strategy: str = "all",
    model_kwargs: Optional[dict] = None,
    n_repeats: int = 1,
) -> dict:
    """Stratified k-fold cross-validation, optionally repeated.

    Returns a dict with mean ± std for each metric across folds/repeats.
    """
    if feature_names is None:
        from .features import get_feature_names
        feature_names = get_feature_names()
    if model_kwargs is None:
        model_kwargs = {}

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    all_results: list[dict] = []
    for rep in range(n_repeats):
        for train_idx, test_idx in skf.split(articles, y):
            train_articles = [articles[i] for i in train_idx]
            test_articles = [articles[i] for i in test_idx]
            y_tr = y[train_idx]
            y_te = y[test_idx]
            model, pipeline = train(
                train_articles, y_tr, model_name=model_name,
                feature_names=feature_names, strategy=strategy,
                model_kwargs=model_kwargs,
            )
            preds = predict(model, pipeline, test_articles)
            res = _score(y_te, preds, name=model_name)
            res["rep"] = rep
            res["fold"] = len([r for r in all_results if r.get("rep") == rep])
            all_results.append(res)

    return _summarise(all_results, model_name=model_name)


def _summarise(results: list[dict], model_name: str) -> dict:
    """Aggregate per-fold results into mean ± std."""
    if not results:
        return {"name": model_name}
    out = {"name": model_name, "n_folds": len(results)}
    for k in ["accuracy", "f1", "f1_macro", "precision_neg", "recall_neg", "f1_neg",
              "precision_pos", "recall_pos", "f1_pos"]:
        vals = [r[k] for r in results]
        out[f"{k}_mean"] = float(np.mean(vals))
        out[f"{k}_std"] = float(np.std(vals))
    return out


# ---- permutation test ----
def permutation_test(
    articles: list[list[str]],
    y: np.ndarray,
    observed_acc: float,
    model_name: str = "rf",
    n_perm: int = 50,
    n_splits: int = 5,
    seed: int = 42,
    feature_names: Optional[list[str]] = None,
    strategy: str = "all",
    model_kwargs: Optional[dict] = None,
) -> dict:
    """Shuffle-label permutation test for the observed accuracy.

    Returns the null distribution and a p-value for P[null >= observed].
    """
    if feature_names is None:
        from .features import get_feature_names
        feature_names = get_feature_names()
    if model_kwargs is None:
        model_kwargs = {}

    rng = np.random.RandomState(seed)
    null_accs: list[float] = []
    for i in range(n_perm):
        yp = rng.permutation(y)
        res = cross_validate(
            articles, yp, model_name=model_name, n_splits=n_splits, seed=seed,
            feature_names=feature_names, strategy=strategy, model_kwargs=model_kwargs,
            n_repeats=1,
        )
        null_accs.append(res["accuracy_mean"])
    null = np.array(null_accs)
    p_value = float((null >= observed_acc).mean())
    return {
        "observed_accuracy": float(observed_acc),
        "n_permutations": n_perm,
        "null_mean": float(null.mean()),
        "null_std": float(null.std()),
        "null_min": float(null.min()),
        "null_max": float(null.max()),
        "null_95pct": float(np.percentile(null, 95)),
        "null_99pct": float(np.percentile(null, 99)),
        "p_value": p_value,
        "null_distribution": null.tolist(),
    }


# ---- learning curve ----
def learning_curve(
    articles: list[list[str]],
    y: np.ndarray,
    train_sizes: Optional[list[float]] = None,
    n_splits: int = 5,
    seed: int = 42,
    model_name: str = "rf",
    feature_names: Optional[list[str]] = None,
    strategy: str = "all",
    model_kwargs: Optional[dict] = None,
) -> pd.DataFrame:
    """Train on increasingly large subsets; record CV accuracy at each size.

    Returns a DataFrame with columns [train_size, n_train, accuracy_mean, accuracy_std, f1_mean, f1_std].
    """
    if feature_names is None:
        from .features import get_feature_names
        feature_names = get_feature_names()
    if model_kwargs is None:
        model_kwargs = {}
    if train_sizes is None:
        train_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    n = len(articles)
    rows = []
    for frac in train_sizes:
        n_train = int(n * frac)
        if n_train < 50:
            continue
        rng = np.random.RandomState(seed)
        idx = rng.permutation(n)[:n_train]
        sub_articles = [articles[i] for i in idx]
        sub_y = y[idx]
        res = cross_validate(
            sub_articles, sub_y, model_name=model_name, n_splits=n_splits,
            seed=seed, feature_names=feature_names, strategy=strategy,
            model_kwargs=model_kwargs, n_repeats=1,
        )
        rows.append({
            "train_size": frac,
            "n_train": n_train,
            "accuracy_mean": res["accuracy_mean"],
            "accuracy_std": res["accuracy_std"],
            "f1_mean": res["f1_mean"],
            "f1_std": res["f1_std"],
        })
    return pd.DataFrame(rows)


# ---- family ablation ----
# Feature families (matches the families defined in src/features.py)
FEATURE_FAMILIES: dict[str, list[str]] = {
    "f1_alphabet":       ["alpha_sum", "alpha_mean", "alpha_max", "alpha_min",
                          "alpha_range", "alpha_sum_mod9", "alpha_sum_mod26",
                          "alpha_sum_parity"],
    "f2_modular":        ["sum_mod3", "sum_mod7", "sum_mod11", "is_prime_sum",
                          "mod26_cyclic_cos", "mod26_cyclic_sin", "digital_root",
                          "sum_mod5", "sum_mod4"],
    "f3_letter_freq":    ["letter_freq_mean", "letter_freq_sum", "rare_letter_count"],
    "f4_bigram":         ["bigram_unique_ratio", "trigram_count"],
    "f5_phonetic":       ["phon_n", "phon_vowel_ratio", "phon_plosive_ratio",
                          "phon_fricative_ratio", "phon_voiceless_ratio", "phon_voice_ratio",
                          "phon_front_ratio", "phon_back_ratio", "phon_high_ratio",
                          "phon_low_ratio"],
    "f6_shape":          ["vowel_ratio", "consonant_ratio", "distinct_letter_ratio",
                          "plosive_count", "plosive_ratio", "fricative_count",
                          "nasal_count", "liquid_count"],
    "f7_length":         ["word_length", "log_word_length"],
    "f8_attractor":      ["alphabet_centeredness", "letter_position_skew"],
    "f9_spectral":       ["dft_power_k1", "dft_power_k2", "dft_power_k3",
                          "dft_high_freq_ratio", "dft_total_power",
                          "dft_spectral_entropy", "autocorr_lag1", "autocorr_lag2"],
    "f10_compression":   ["gzip_size", "gzip_size_per_char", "gzip_ratio_vs_random"],
    "f11_numeric":       ["letter_product_mod26", "word_value_mod_9",
                          "word_value_mod_26", "mispar_hechrechi_sum"],
    "f12_symmetry":      ["is_palindrome", "prefix_eq_suffix", "symmetry_density",
                          "max_run_length", "n_runs", "mean_run_length",
                          "run_length_entropy", "first_letter_lp", "last_letter_lp"],
}


def family_ablation(
    articles: list[list[str]],
    y: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
    model_name: str = "rf",
    feature_names: Optional[list[str]] = None,
    strategy: str = "all",
    model_kwargs: Optional[dict] = None,
) -> pd.DataFrame:
    """Leave-one-family-out: drop each feature family, measure CV accuracy.

    Returns a DataFrame with columns [family, n_features, accuracy_mean, accuracy_std, delta].
    """
    if feature_names is None:
        from .features import get_feature_names
        feature_names = get_feature_names()
    if model_kwargs is None:
        model_kwargs = {}

    # Baseline: all features
    baseline = cross_validate(
        articles, y, model_name=model_name, n_splits=n_splits, seed=seed,
        feature_names=feature_names, strategy=strategy, model_kwargs=model_kwargs,
    )
    base_acc = baseline["accuracy_mean"]

    rows = [{"family": "BASELINE_ALL", "n_features": len(feature_names),
             "accuracy_mean": base_acc, "accuracy_std": baseline["accuracy_std"],
             "delta": 0.0}]

    for family_name, family_features in FEATURE_FAMILIES.items():
        kept = [f for f in feature_names if f not in family_features]
        if not kept:
            continue
        res = cross_validate(
            articles, y, model_name=model_name, n_splits=n_splits, seed=seed,
            feature_names=kept, strategy=strategy, model_kwargs=model_kwargs,
        )
        rows.append({
            "family": f"DROP_{family_name}",
            "n_features": len(kept),
            "accuracy_mean": res["accuracy_mean"],
            "accuracy_std": res["accuracy_std"],
            "delta": res["accuracy_mean"] - base_acc,
        })

    return pd.DataFrame(rows)


# ---- single-family-only ----
def single_family_evaluation(
    articles: list[list[str]],
    y: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
    model_name: str = "rf",
    all_feature_names: Optional[list[str]] = None,
    strategy: str = "all",
    model_kwargs: Optional[dict] = None,
) -> pd.DataFrame:
    """Use only one family at a time; report CV accuracy for each."""
    if all_feature_names is None:
        from .features import get_feature_names
        all_feature_names = get_feature_names()
    if model_kwargs is None:
        model_kwargs = {}

    rows = []
    for family_name, family_features in FEATURE_FAMILIES.items():
        kept = [f for f in family_features if f in all_feature_names]
        if not kept:
            continue
        res = cross_validate(
            articles, y, model_name=model_name, n_splits=n_splits, seed=seed,
            feature_names=kept, strategy=strategy, model_kwargs=model_kwargs,
        )
        rows.append({
            "family": family_name,
            "n_features": len(kept),
            "accuracy_mean": res["accuracy_mean"],
            "accuracy_std": res["accuracy_std"],
            "f1_mean": res["f1_mean"],
        })
    return pd.DataFrame(rows).sort_values("accuracy_mean", ascending=False)


# ---- reporting ----
def format_report(cv_result: dict, baseline: Optional[dict] = None,
                  permutation: Optional[dict] = None) -> str:
    """Format a CV result as a human-readable multi-line string."""
    lines = [
        f"=== {cv_result['name']} ===",
        f"  Accuracy:       {cv_result['accuracy_mean']:.4f} ± {cv_result['accuracy_std']:.4f}",
        f"  F1 (binary):    {cv_result['f1_mean']:.4f} ± {cv_result['f1_std']:.4f}",
        f"  F1 (macro):     {cv_result['f1_macro_mean']:.4f} ± {cv_result['f1_macro_std']:.4f}",
        f"  Negative class: P={cv_result['precision_neg_mean']:.3f}, R={cv_result['recall_neg_mean']:.3f}, F1={cv_result['f1_neg_mean']:.3f}",
        f"  Positive class: P={cv_result['precision_pos_mean']:.3f}, R={cv_result['recall_pos_mean']:.3f}, F1={cv_result['f1_pos_mean']:.3f}",
        f"  N folds:        {cv_result['n_folds']}",
    ]
    if baseline is not None:
        delta = cv_result["accuracy_mean"] - baseline["accuracy"]
        lines.append(f"  vs {baseline['name']}: {delta:+.4f}")
    if permutation is not None:
        lines.append(f"  Permutation p-value: {permutation['p_value']:.4f}")
        lines.append(f"  Null distribution: mean={permutation['null_mean']:.4f}, "
                     f"95%={permutation['null_95pct']:.4f}, max={permutation['null_max']:.4f}")
    return "\n".join(lines)
