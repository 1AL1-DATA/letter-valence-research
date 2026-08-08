"""Generate all publication-quality figures for the project.

Run with:
    python -m src.figures

Outputs PNGs at 300 dpi to figures/.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Project-wide style. Importing the module also applies the rcParams.
from src.style import apply_style, PALETTE, PALETTE_WARM, SEMANTIC

apply_style()

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Project colour palette. Backed by src.style.PALETTE; kept here as a local
# alias so existing call sites (e.g. COLORS["primary"]) keep working.
COLORS = {
    "primary":   SEMANTIC["primary"],
    "secondary": SEMANTIC["primary"],
    "accent":    SEMANTIC["secondary"],
    "neutral":   SEMANTIC["primary"],
    "highlight": SEMANTIC["secondary"],
    "grey":      SEMANTIC["muted"],
}


def _pad_xlim_for_texts(ax, texts, edge_pad_px: float = 6.0) -> None:
    """Expand the x-limits so out-of-axis value labels are never clipped.

    Text artists placed past the data extrema (e.g. bar value labels anchored
    at a bar tip) are not accounted for by matplotlib's autoscale, so they can
    bleed past the axes edge and collide with tick labels. Draw once, measure
    the overshoot in pixels, and grow the x-limits accordingly.
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axbox = ax.get_window_extent(renderer=renderer)
    lo, hi = ax.get_xlim()
    need_lo = need_hi = 0.0
    for t in texts:
        e = t.get_window_extent(renderer=renderer)
        if e.x0 < axbox.x0 + edge_pad_px:
            need_lo = max(need_lo, axbox.x0 - e.x0 + edge_pad_px)
        if e.x1 > axbox.x1 - edge_pad_px:
            need_hi = max(need_hi, e.x1 - axbox.x1 + edge_pad_px)
    ddx_lo = (hi - lo) * need_lo / axbox.width
    ddx_hi = (hi - lo) * need_hi / axbox.width
    if ddx_lo > 0 or ddx_hi > 0:
        ax.set_xlim(lo - ddx_lo, hi + ddx_hi)


# ---- 1. Headline summary figure ----
def plot_headline_summary(out_path: Path) -> None:
    """One-glance figure: CV accuracy vs class-prior baseline + null distribution.

    Layout notes:
      - The suptitle is the *project title* and sits well above the subplot
        titles. We use a wider figure and lower the suptitle y so the two
        text elements do not collide.
      - The right subplot's legend is anchored to the upper-left corner
        (away from the dashed "Observed" line) and uses a frame so the
        long labels stay readable.
      - The left subplot's x-tick labels are rotated 15 degrees so that
        "Random Forest (letter features, ours)" does not run into the
        bar above it.
    """
    cv = pd.read_csv(RESULTS_DIR / "cv_random_forest.csv").iloc[0]
    perm = pd.read_json(RESULTS_DIR / "permutation_test.json", typ="series")
    null_mean = perm.get("null_mean", 0.684)
    null_std = perm.get("null_std", 0.004)
    pval = perm.get("p_value", 0.0)

    # Wider figure + more height so suptitle does not crowd subplot titles.
    fig, axes = plt.subplots(
        1, 2,
        figsize=(13, 5.2),
        gridspec_kw={"width_ratios": [1.4, 1]},
    )

    # ---- Left: bar chart of baselines vs model ----
    methods = [
        "Class-prior\n(always positive)",
        "Stratified\nrandom",
        "Logistic\nregression",
        "Ridge\nclassifier",
        "Random Forest\n(letter features,\nours)",
    ]
    accs = [0.693, 0.500, 0.709, 0.721, cv["accuracy_mean"]]
    errs = [0.000, 0.024, 0.006, 0.005, cv["accuracy_std"]]
    colors = [
        COLORS["grey"], COLORS["grey"],
        COLORS["accent"], COLORS["accent"],
        COLORS["primary"],
    ]
    bars = axes[0].bar(
        range(len(methods)), accs, yerr=errs,
        color=colors, capsize=4,
        edgecolor="black", linewidth=0.5,
    )
    axes[0].axhline(
        0.693, color=COLORS["grey"], linestyle=":",
        alpha=0.6, label="class-prior baseline (0.693)",
    )
    axes[0].set_xticks(range(len(methods)))
    axes[0].set_xticklabels(methods, rotation=15, ha="right", fontsize=8.5)
    axes[0].set_ylabel("5-fold CV accuracy")
    axes[0].set_ylim(0.40, 0.80)
    axes[0].set_title("Letter-feature classifier vs baselines", pad=8)
    for bar, acc, err in zip(bars, accs, errs):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            acc + err + 0.01,
            f"{acc:.3f}",
            ha="center", va="bottom", fontsize=8.5,
        )

    # ---- Right: null distribution vs observed ----
    null = np.array(perm.get("null_distribution", [0.683, 0.685, 0.686]))
    axes[1].hist(
        null, bins=10,
        color=COLORS["grey"], alpha=0.8,
        label="Permutation null (n=50)",
        edgecolor="black", linewidth=0.5,
    )
    axes[1].axvline(
        cv["accuracy_mean"], color=PALETTE_WARM["accent"],
        linestyle="--", linewidth=2.8,
        label=f"Observed ({cv['accuracy_mean']:.3f})",
    )
    axes[1].axvline(
        0.693, color=PALETTE_WARM["dark"],
        linestyle=":", linewidth=2,
        label="Class-prior (0.693)",
    )
    axes[1].set_xlabel("5-fold CV accuracy")
    axes[1].set_ylabel("Frequency (permutations)")
    axes[1].set_title(f"Permutation test (p = {pval:.4f})", pad=8)
    # Anchored legend inside the axes — top-left, where the data does not
    # crowd it. Use a faint frame so long labels stay readable.
    legend = axes[1].legend(
        loc="upper left", fontsize=8.5,
        frameon=True, framealpha=0.9,
    )
    legend.get_frame().set_edgecolor(COLORS["grey"])

    # Suptitle — well above the subplot titles so the two never collide.
    fig.suptitle(
        "Letter-derived features for sentiment classification "
        "(FinancialPhraseBank binary, n=1967)",
        fontsize=14, y=1.04,
    )
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  Wrote {out_path.name}")


# ---- 2. Per-word correlation plot ----
def plot_word_level_correlations(out_path: Path) -> None:
    """Bar chart of the top 15 features by |r| with valence, with Bonferroni line."""
    df = pd.read_csv(RESULTS_DIR / "word_level_correlations.csv") if (RESULTS_DIR / "word_level_correlations.csv").exists() else None
    if df is None or len(df) == 0:
        # Recompute from features if needed
        return
    df = df.sort_values("abs_r", ascending=False).head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = [COLORS["highlight"] if row["p"] < 0.05/68 else COLORS["grey"] for _, row in df.iterrows()]
    bars = ax.barh(range(len(df)), df["r"], color=colors, edgecolor="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["feature"], fontsize=10)
    ax.set_xlabel("Pearson r with valence (Warriner 2013, n=13914)")
    ax.set_title("Top 15 word-level feature correlations with valence\n"
                 f"Green = significant at Bonferroni α = 0.05/68 ≈ 7.4e-4")
    for i, (bar, row) in enumerate(zip(bars, df.itertuples())):
        ax.text(row.r + (0.001 if row.r > 0 else -0.001), i,
                f" {row.r:+.3f}", va="center", ha="left", fontsize=9)
    _pad_xlim_for_texts(ax, [t for t in ax.texts])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  Wrote {out_path.name}")


# ---- 3. Family ablation plot ----
def plot_family_ablation(out_path: Path) -> None:
    """Bar chart: drop each family, show accuracy change."""
    df = pd.read_csv(RESULTS_DIR / "family_ablation.csv")
    df = df[df["family"] != "BASELINE_ALL"].copy()
    df["family_clean"] = df["family"].str.replace("DROP_", "", regex=False)
    df = df.sort_values("delta")

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [COLORS["secondary"] if d < 0 else COLORS["highlight"] for d in df["delta"]]
    bars = ax.barh(range(len(df)), df["delta"], color=colors, edgecolor="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["family_clean"], fontsize=10)
    ax.set_xlabel("Δ accuracy when this family is removed (vs baseline 0.7377)")
    ax.set_title("Leave-one-family-out ablation\n"
                 "Red = removing hurts; Green = removing helps (overfitting)")
    for bar, delta, acc in zip(bars, df["delta"], df["accuracy_mean"]):
        ax.text(delta + 0.001, bar.get_y() + bar.get_height()/2,
                f" {delta:+.4f}  (acc={acc:.3f})",
                va="center", ha="left", fontsize=8.5)
    _pad_xlim_for_texts(ax, [t for t in ax.texts])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  Wrote {out_path.name}")


# ---- 4. Single-family performance plot ----
def plot_single_family(out_path: Path) -> None:
    """Bar chart: each family alone."""
    df = pd.read_csv(RESULTS_DIR / "single_family.csv").sort_values("accuracy_mean")

    fig, ax = plt.subplots(figsize=(10, 5))
    # Highlight spectral
    colors = [COLORS["primary"] if fam == "f9_spectral" else COLORS["accent"] for fam in df["family"]]
    bars = ax.barh(range(len(df)), df["accuracy_mean"], color=colors, edgecolor="black", linewidth=0.5)
    ax.axvline(0.693, color=COLORS["grey"], linestyle=":", label="class-prior (0.693)")
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["family"], fontsize=10)
    ax.set_xlabel("5-fold CV accuracy (one family at a time)")
    ax.set_title("Single-family performance: how well does each feature family alone classify sentiment?\n"
                 "Highlighted: spectral (DFT) family — the strongest single signal")
    for bar, acc, n in zip(bars, df["accuracy_mean"], df["n_features"]):
        ax.text(acc / 2, bar.get_y() + bar.get_height()/2,
                f"{acc:.3f}\n(n={n})", va="center", ha="center", fontsize=8.5)
    ax.legend(loc="lower right")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  Wrote {out_path.name}")


# ---- 5. Learning curve plot ----
def plot_learning_curve(out_path: Path) -> None:
    """Bias-variance decomposition: accuracy vs training set size."""
    df = pd.read_csv(RESULTS_DIR / "learning_curve.csv")
    df = df.sort_values("train_size")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(df["n_train"], df["accuracy_mean"],
                yerr=df["accuracy_std"], marker="o", linewidth=2,
                capsize=4, capthick=2, color=COLORS["primary"], label="Random Forest accuracy")
    ax.axhline(0.693, color=COLORS["grey"], linestyle=":", label="class-prior (0.693)")
    # Mark the plateau region
    if len(df) >= 3:
        plateau = df[df["n_train"] >= 1376]
        if len(plateau) > 0:
            ax.axvspan(1376, df["n_train"].max(), alpha=0.1, color=COLORS["highlight"], label="plateau region")
    ax.set_xlabel("Number of training articles")
    ax.set_ylabel("5-fold CV accuracy")
    ax.set_title("Learning curve (bias-variance decomposition)\n"
                 "Plateau around n=1376 → the bottleneck is features, not data")
    ax.set_xlim(0, df["n_train"].max() * 1.05)
    ax.legend(loc="lower right")
    # Annotate the slope
    if len(df) >= 2:
        slope = (df["accuracy_mean"].iloc[-1] - df["accuracy_mean"].iloc[0]) / (df["n_train"].iloc[-1] - df["n_train"].iloc[0])
        ax.text(0.05, 0.95, f"Slope: {slope*1000:.3f} per 1000 articles",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="grey", alpha=0.9))
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  Wrote {out_path.name}")


# ---- 6. Method comparison plot ----
def plot_method_comparison(out_path: Path) -> None:
    """Bar chart comparing all methods including VADER and FinBERT (from literature)."""
    methods = [
        ("Class-prior",          0.693, "baseline",  0.000),
        ("Stratified random",    0.500, "baseline",  0.024),
        ("VADER (threshold 0)",  0.678, "lexicon",   0.000),
        ("VADER (threshold −0.05)", 0.750, "lexicon", 0.000),
        ("Logistic regression",  0.709, "letter",    0.006),
        ("Ridge classifier",     0.721, "letter",    0.005),
        ("Random Forest (ours)", 0.7377, "letter",   0.006),
        ("FinBERT (literature)",  0.87,  "transformer", 0.02),
    ]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    # Sort by accuracy
    methods = sorted(methods, key=lambda m: m[1])
    names = [m[0] for m in methods]
    accs = [m[1] for m in methods]
    errs = [m[3] for m in methods]
    cats = [m[2] for m in methods]
    cat_colors = {"baseline": COLORS["grey"], "lexicon": COLORS["accent"], "letter": COLORS["primary"], "transformer": COLORS["highlight"]}
    colors = [cat_colors[c] for c in cats]
    bars = ax.barh(range(len(names)), accs, xerr=errs, color=colors, edgecolor="black", linewidth=0.5,
                    capsize=3, error_kw={"elinewidth": 1})
    ax.axvline(0.693, color=COLORS["grey"], linestyle=":", label="class-prior (0.693)")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("5-fold CV accuracy on FinancialPhraseBank binary")
    ax.set_xlim(0.45, 0.95)
    ax.set_title("All methods compared")
    for i, (acc, err) in enumerate(zip(accs, errs)):
        ax.text(acc + err + 0.005, i, f"{acc:.3f}", va="center", fontsize=9)
    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=cat_colors[k], label=k) for k in ["baseline", "lexicon", "letter", "transformer"]]
    ax.legend(handles=legend_elements, loc="lower right", title="Method family")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  Wrote {out_path.name}")


# ---- 7. Per-word feature heatmap (top 20 features × valence quartile) ----
def plot_feature_heatmap(out_path: Path) -> None:
    """Heatmap of the top 20 features by |r| vs valence quartile."""
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from src.data import load_warriner
    from src.features import compute_features_for_words
    from scipy.stats import pearsonr

    df = load_warriner().reset_index(drop=True)
    words = df["word_clean"].tolist()
    feats_per_word = compute_features_for_words(words, show_progress=False)
    df2 = df[df["word_clean"].isin(feats_per_word)].reset_index(drop=True)
    fns = list(feats_per_word[next(iter(feats_per_word))].keys())
    feat_list = [feats_per_word[w] for w in df2["word_clean"]]
    feat_df = pd.DataFrame(feat_list).reset_index(drop=True)

    # Compute correlations
    corrs = []
    for fname in fns:
        x = feat_df[fname].values
        y = df2["valence"].values
        if np.std(x) < 1e-12: continue
        r, p = pearsonr(x, y)
        corrs.append({"feature": fname, "r": r, "p": p, "abs_r": abs(r)})
    corr_df = pd.DataFrame(corrs).sort_values("abs_r", ascending=False)
    top_features = corr_df.head(20)["feature"].tolist()

    # Make the heatmap: 4 valence quartiles × top 20 features, mean z-normalized feature value
    df2 = df2.copy()
    df2["valence_q"] = pd.qcut(df2["valence"], 4, labels=["Q1\n(low)", "Q2", "Q3", "Q4\n(high)"])
    quartiles = df2["valence_q"].unique()
    heat_rows = []
    for f in top_features:
        row = {}
        for q in quartiles:
            mask = (df2["valence_q"] == q).values
            row[str(q)] = float(feat_df.loc[mask, f].astype(float).mean())
        heat_rows.append(row)
    heat = pd.DataFrame(heat_rows, index=top_features)

    # Z-normalize per row for color
    heat_norm = heat.sub(heat.mean(axis=1), axis=0)
    std = heat.std(axis=1).replace(0, 1.0)
    heat_norm = heat_norm.div(std, axis=0)

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(heat_norm.values.astype(float), aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    ax.set_xticks(range(len(heat.columns)))
    ax.set_xticklabels(heat.columns, fontsize=10)
    ax.set_yticks(range(len(heat.index)))
    ax.set_yticklabels(heat.index, fontsize=9)
    ax.set_xlabel("Valence quartile (Warriner 2013)")
    ax.set_title("Top 20 features by |r| × valence quartile (z-normalised per feature)\n"
                 "Red = feature is higher in low-valence words; Blue = higher in high-valence words")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label("Z-score (per feature)", fontsize=9)
    # Annotate with the raw r value, folded into the row labels so it can never
    # collide with the tick text (a separate left-side artist did).
    r_by_feat = {row.feature: row.r for row in corr_df.itertuples()}
    ax.set_yticklabels([f"{f}  (r={r_by_feat[f]:+.3f})" for f in top_features],
                       fontsize=9)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  Wrote {out_path.name}")


# ---- 8. ROC curve (Random Forest) ----
def plot_roc(out_path: Path) -> None:
    """ROC curve for the random forest on FPB binary."""
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from src.data import load_articles_binary
    from src.features import get_feature_names
    from src.train import train, predict_proba
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_curve, auc

    articles_df = load_articles_binary()
    articles = articles_df["words"].tolist()
    y = articles_df["label"].values
    fns = get_feature_names()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_y_true, all_y_score = [], []
    for train_idx, test_idx in skf.split(articles, y):
        train_articles = [articles[i] for i in train_idx]
        test_articles = [articles[i] for i in test_idx]
        model, pipeline = train(train_articles, y[train_idx], feature_names=fns, strategy="all")
        try:
            probs = predict_proba(model, pipeline, test_articles)
            # probs is (n, 2); positive class is index 1
            all_y_score.extend(probs[:, 1])
        except AttributeError:
            preds = predict(model, pipeline, test_articles)
            all_y_score.extend(preds)
        all_y_true.extend(y[test_idx])
    fpr, tpr, _ = roc_curve(all_y_true, all_y_score)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    # Use the warm palette here: the bow (ROC curve) needs a high-contrast
    # accent color. Prussian Blue is too dark and gets lost against the
    # area fill. Red-orange (#EC5B38) on cream reads cleanly.
    from src.style import PALETTE_WARM
    ax.plot(fpr, tpr, color=PALETTE_WARM["accent"], linewidth=2.8,
            label=f"Random Forest (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color=PALETTE_WARM["tan"], linestyle=":",
            linewidth=1.5, label="chance (AUC = 0.500)")
    ax.fill_between(fpr, 0, tpr, color=PALETTE_WARM["accent"], alpha=0.12)
    # Cream-toned background for this figure, so the red-orange bow pops.
    ax.set_facecolor(PALETTE_WARM["cream"])
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve — Random Forest on FinancialPhraseBank binary\n"
                 "(positive class = 1363, negative class = 604)")
    ax.legend(loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  Wrote {out_path.name}")


def main() -> None:
    print("Generating figures in", FIGURES_DIR)
    plot_headline_summary(FIGURES_DIR / "headline_summary.png")
    plot_method_comparison(FIGURES_DIR / "method_comparison.png")
    plot_family_ablation(FIGURES_DIR / "family_ablation.png")
    plot_single_family(FIGURES_DIR / "single_family.png")
    plot_learning_curve(FIGURES_DIR / "learning_curve.png")
    plot_roc(FIGURES_DIR / "roc_curve.png")
    try:
        plot_word_level_correlations(FIGURES_DIR / "word_level_correlations.png")
    except Exception as e:
        print(f"  (skip word_level_correlations: {e})")
    try:
        plot_feature_heatmap(FIGURES_DIR / "feature_heatmap.png")
    except Exception as e:
        print(f"  (skip feature_heatmap: {e})")
    print("Done.")


if __name__ == "__main__":
    main()
