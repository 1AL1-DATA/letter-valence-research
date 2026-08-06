"""Main analysis entry point for the letter-valence research project.

Run with:
    python -m src.analyze

This script:
1. Loads the data
2. Trains a baseline model (hand-crafted ridge formula on aggregated features)
3. Trains a random forest on the full 68 letter-derived features
4. Runs 5-fold cross-validation with permutation test
5. Computes the bias-variance learning curve
6. Runs the family ablation and single-family evaluation
7. Writes all results to /results/

Designed to be the single command that reproduces the headline finding.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data import load_articles_binary, load_warriner, tokenize
from src.evaluate import (
    baseline_majority_class,
    baseline_stratified_random,
    cross_validate,
    family_ablation,
    format_report,
    learning_curve,
    permutation_test,
    single_family_evaluation,
    _score,
)
from src.features import compute_features_for_words, get_feature_names
from src.train import FeaturePipeline, make_model, predict, train

RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def banner(msg: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {msg}")
    print("=" * 70)


def main(
    n_perm: int = 50,
    learning_curve_sizes: list[float] | None = None,
    skip_permutation: bool = False,
    skip_learning_curve: bool = False,
) -> None:
    t_start = time.time()

    banner("STEP 1: Load data")
    articles_df = load_articles_binary()
    warriner_df = load_warriner()
    print(f"  Loaded {len(articles_df)} FPB articles, {len(warriner_df)} Warriner lemmas")
    print(f"  Class distribution: {articles_df['label'].value_counts().to_dict()}")

    articles = articles_df["words"].tolist()
    y = articles_df["label"].values
    fns = get_feature_names()
    print(f"  Feature set: {len(fns)} features")

    banner("STEP 2: Baselines")
    majority = baseline_majority_class(y)
    random_b = baseline_stratified_random(y)
    print(f"  Majority class baseline: acc={majority['accuracy']:.4f}")
    print(f"  Stratified random baseline: acc={random_b['accuracy']:.4f} ± {random_b['accuracy_std']:.4f}")

    banner("STEP 3: Random Forest (5-fold CV)")
    cv_rf = cross_validate(articles, y, model_name="rf", n_splits=5, seed=42)
    print(format_report(cv_rf, baseline=majority))
    pd.DataFrame([cv_rf]).to_csv(RESULTS_DIR / "cv_random_forest.csv", index=False)

    banner("STEP 4: Logistic Regression (5-fold CV, baseline ML)")
    cv_lr = cross_validate(articles, y, model_name="logreg", n_splits=5, seed=42)
    print(format_report(cv_lr, baseline=majority))
    pd.DataFrame([cv_lr]).to_csv(RESULTS_DIR / "cv_logistic_regression.csv", index=False)

    banner("STEP 5: Ridge Classifier (5-fold CV, linear baseline)")
    cv_ridge = cross_validate(articles, y, model_name="ridge", n_splits=5, seed=42)
    print(format_report(cv_ridge, baseline=majority))
    pd.DataFrame([cv_ridge]).to_csv(RESULTS_DIR / "cv_ridge.csv", index=False)

    # Permutation test on the strongest model
    if not skip_permutation:
        banner(f"STEP 6: Permutation test (n={n_perm})")
        perm = permutation_test(
            articles, y, observed_acc=cv_rf["accuracy_mean"],
            model_name="rf", n_perm=n_perm, n_splits=5, seed=42,
        )
        print(f"  Observed accuracy: {perm['observed_accuracy']:.4f}")
        print(f"  Null mean ± std:   {perm['null_mean']:.4f} ± {perm['null_std']:.4f}")
        print(f"  Null 95th pct:     {perm['null_95pct']:.4f}")
        print(f"  Null max:          {perm['null_max']:.4f}")
        print(f"  p-value:           {perm['p_value']:.4f}")
        # Save without the distribution
        perm_save = {k: v for k, v in perm.items() if k != "null_distribution"}
        with open(RESULTS_DIR / "permutation_test.json", "w") as f:
            json.dump(perm_save, f, indent=2)
    else:
        perm = None

    # Learning curve (bias-variance decomposition)
    if not skip_learning_curve:
        banner("STEP 7: Learning curve (bias-variance decomposition)")
        if learning_curve_sizes is None:
            learning_curve_sizes = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
        lc = learning_curve(
            articles, y, train_sizes=learning_curve_sizes, n_splits=5, seed=42,
            model_name="rf",
        )
        print(lc.to_string(index=False))
        lc.to_csv(RESULTS_DIR / "learning_curve.csv", index=False)
        # Interpretation
        slope = (lc["accuracy_mean"].iloc[-1] - lc["accuracy_mean"].iloc[len(lc)//2]) / (lc["train_size"].iloc[-1] - lc["train_size"].iloc[len(lc)//2])
        print(f"\n  Approx accuracy gain from 50% to 100% training data: {slope:.4f} per unit")
        if slope < 0.02:
            print("  → FLAT curve: variance dominates, more data won't help. Need different features.")
        else:
            print("  → RISING curve: bias likely dominates, more data should help.")

    # Family ablation
    banner("STEP 8: Family ablation (leave-one-family-out)")
    abl = family_ablation(articles, y, n_splits=5, seed=42, model_name="rf")
    print(abl.to_string(index=False))
    abl.to_csv(RESULTS_DIR / "family_ablation.csv", index=False)

    # Single-family evaluation
    banner("STEP 9: Single-family evaluation")
    single = single_family_evaluation(articles, y, n_splits=5, seed=42, model_name="rf")
    print(single.to_string(index=False))
    single.to_csv(RESULTS_DIR / "single_family.csv", index=False)

    # Save timing
    banner("STEP 10: Timing")
    t_end = time.time()
    elapsed = t_end - t_start
    print(f"  Total wall-clock: {elapsed:.1f}s")
    timing = {"total_seconds": elapsed, "n_permutations": n_perm}
    with open(RESULTS_DIR / "timing.json", "w") as f:
        json.dump(timing, f, indent=2)

    # Save the final summary as JSON
    summary = {
        "n_articles": int(len(articles)),
        "n_features": len(fns),
        "class_distribution": {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        "baselines": {
            "majority_class": {k: float(v) for k, v in majority.items() if isinstance(v, (int, float))},
            "stratified_random": {k: float(v) for k, v in random_b.items() if isinstance(v, (int, float))},
        },
        "random_forest_5fold": {k: float(v) for k, v in cv_rf.items() if isinstance(v, (int, float))},
        "logistic_regression_5fold": {k: float(v) for k, v in cv_lr.items() if isinstance(v, (int, float))},
        "ridge_5fold": {k: float(v) for k, v in cv_ridge.items() if isinstance(v, (int, float))},
    }
    if perm is not None:
        summary["permutation_test"] = {k: float(v) for k, v in perm.items()
                                        if k != "null_distribution" and isinstance(v, (int, float))}
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    banner("DONE")
    print(f"  All outputs in {RESULTS_DIR}/")
    print(f"  Headline: RF 5-fold CV accuracy = {cv_rf['accuracy_mean']:.4f} ± {cv_rf['accuracy_std']:.4f}")
    if perm is not None:
        print(f"  Permutation p-value: {perm['p_value']:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the full letter-valence analysis")
    parser.add_argument("--n-perm", type=int, default=50, help="Number of permutations")
    parser.add_argument("--skip-permutation", action="store_true")
    parser.add_argument("--skip-learning-curve", action="store_true")
    args = parser.parse_args()
    main(
        n_perm=args.n_perm,
        skip_permutation=args.skip_permutation,
        skip_learning_curve=args.skip_learning_curve,
    )
