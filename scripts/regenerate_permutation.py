"""Regenerate results/permutation_test.json with 999 label shuffles.

History: the committed artifact was first generated with n_perm=5, later
regenerated with the documented default of 50. Both used the raw
(count/n_perm) estimator, which printed p_value = 0.0 — a claim a finite
Monte-Carlo test cannot support (resolution floor is 1/(n_perm + 1)).
This script reruns the permutation test with n_perm=999 and the add-one
estimator (Phipson & Smyth 2010) and rewrites permutation_test.json and the
permutation_test block of summary.json.

Usage:
    python -m scripts.regenerate_permutation
"""

from __future__ import annotations

import json

from src.analyze import RESULTS_DIR
from src.data import load_articles_binary
from src.evaluate import baseline_stratified_random, permutation_test
from src.features import get_feature_names


def main() -> None:
    articles_df = load_articles_binary()
    articles = articles_df["words"].tolist()
    y = articles_df["label"].values

    print(f"Loaded {len(articles)} articles, {len(get_feature_names())} features")

    random_b = baseline_stratified_random(y)
    print(
        f"Stratified random baseline: "
        f"acc={random_b['accuracy']:.4f} +/- {random_b['accuracy_std']:.4f}, "
        f"F1={random_b['f1']:.4f}"
    )

    print("Running permutation test with n_perm=999 ...")
    perm = permutation_test(
        articles, y, observed_acc=0.737673241110293,
        model_name="rf", n_perm=999, n_splits=5, seed=42,
    )
    print(f"  Observed accuracy: {perm['observed_accuracy']:.4f}")
    print(f"  Null mean +/- std: {perm['null_mean']:.4f} +/- {perm['null_std']:.4f}")
    print(f"  Null min/max:      {perm['null_min']:.4f} / {perm['null_max']:.4f}")
    print(f"  Null 95th/99th:    {perm['null_95pct']:.4f} / {perm['null_99pct']:.4f}")
    print(f"  p-value:           {perm['p_value']:.4f}")
    print(f"  n >= observed:     {sum(1 for v in perm['null_distribution'] if v >= perm['observed_accuracy'])} / {perm['n_permutations']}")

    perm_save = {k: v for k, v in perm.items() if k != "null_distribution"}
    perm_save["observed_accuracy"] = float(perm_save["observed_accuracy"])
    for key in ("null_mean", "null_std", "null_min", "null_max", "null_95pct", "null_99pct"):
        perm_save[key] = float(perm_save[key])

    with open(RESULTS_DIR / "permutation_test.json", "w") as f:
        json.dump(perm_save, f, indent=2)
    print("Wrote results/permutation_test.json")

    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path) as f:
        summary = json.load(f)
    summary["permutation_test"] = {k: float(v) for k, v in perm_save.items()}
    summary["baselines"]["stratified_random"] = {
        "accuracy_mean": float(random_b["accuracy"]),
        "accuracy_std": float(random_b["accuracy_std"]),
        "f1_mean": float(random_b["f1"]),
        "f1_std": float(random_b["f1_std"]),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=False)
    print("Updated results/summary.json (permutation_test + stratified_random F1)")


if __name__ == "__main__":
    main()
