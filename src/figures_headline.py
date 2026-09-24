"""4-panel infographic: live-headline evaluation against LLM-judge gold.

Reads ``data/cascade_test/sentiment_cascade_results.csv`` (written by
``src.cascade_eval``) and writes ``figures/headline_gold_findings.png``.

Panels:
  A. Single-signal accuracy vs gold (n = 240 headlines)
  B. Cheap-tier band sweep: precision vs coverage
  C. Ensemble ladder: best single -> valence average -> 5-signal CV stacker -> oracle
  D. Fires at shipped bands: legacy letter tiers (0) vs word tier (16 @ 81.2%)

Run from the research repo root:
    python -m src.figures_headline
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.style import PALETTE, apply_style

REPO = Path(__file__).resolve().parent.parent
CSV = REPO / "data" / "cascade_test" / "sentiment_cascade_results.csv"
OUT = REPO / "figures" / "headline_gold_findings.png"

BAND = 0.1          # neutral band on the quadratic score (engine default)
CHEAP_THRESHOLD = 0.6
SIGNALS = ["cheap_v", "dft_v", "rf_v", "vader_v", "keyword_v"]
SIGNAL_LABELS = {
    "cheap_v": "word tier", "dft_v": "legacy DFT", "rf_v": "legacy RF",
    "vader_v": "VADER", "keyword_v": "keyword",
}


def label_from_v(v: np.ndarray) -> np.ndarray:
    score = np.copysign(v * v, v)
    return np.where(score >= BAND, "positive",
                    np.where(score <= -BAND, "negative", "neutral"))


def stacker_cv_acc(df: pd.DataFrame, gold: np.ndarray) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    X = df[SIGNALS].to_numpy()
    cv = StratifiedKFold(5, shuffle=True, random_state=7)
    clf = LogisticRegression(max_iter=1000)
    return float(cross_val_score(clf, X, gold, cv=cv).mean())


def main() -> None:
    apply_style()
    df = pd.read_csv(CSV)
    gold = df["gold"].to_numpy()
    n = len(df)
    majority = float(pd.Series(gold).value_counts(normalize=True).max())

    # A — single-signal accuracy
    singles = {SIGNAL_LABELS[c]: float((label_from_v(df[c].to_numpy()) == gold).mean())
               for c in SIGNALS}

    # B — cheap band sweep (coverage, precision)
    cv = df["cheap_v"].to_numpy()
    cheap_labels = label_from_v(cv)
    sweep = []
    for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        f = np.abs(cv) >= t
        if f.sum() < 5:
            continue
        sweep.append((float(f.mean()), float((cheap_labels[f] == gold[f]).mean()), t))

    # C — ensemble ladder
    pred_stack = stacker_cv_acc(df, gold)
    w = np.array([0.5, 0.0, 0.0, 0.25, 0.25])
    avg_labels = label_from_v((w * df[SIGNALS].to_numpy()).sum(axis=1))
    ladder = [
        ("best single\n(keyword)", max(singles.values())),
        ("valence avg\ncheap+vader+kw", float((avg_labels == gold).mean())),
        ("5-signal stacker\nCV-5", pred_stack),
        ("oracle\n(any-of-5)", float(np.logical_or.reduce(
            [label_from_v(df[c].to_numpy()) == gold for c in SIGNALS]).mean())),
    ]

    # D — fires at shipped bands
    legacy_dft = int((df["dft_v"].abs() >= 0.95).sum())
    legacy_rf = int((df["rf_v"].abs() >= 0.80).sum())
    word = int((np.abs(cv) >= CHEAP_THRESHOLD).sum())
    word_prec = float((cheap_labels[np.abs(cv) >= CHEAP_THRESHOLD] ==
                       gold[np.abs(cv) >= CHEAP_THRESHOLD]).mean())

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    fig.suptitle("Live-headline evaluation — 240 headlines vs LLM-judge gold",
                 fontsize=15, fontweight="bold", color=PALETTE["black"], y=0.985)
    fig.text(0.5, 0.945,
             "Google News RSS (12 tickers) · synthetic single-judge labels · "
             "word tier = TF-IDF + [VADER, keyword] → logistic",
             ha="center", fontsize=9.5, color="#555555", style="italic")

    # A — singles
    ax = axes[0, 0]
    names = list(singles)[::-1]
    vals = [singles[k] for k in names]
    cols = [PALETTE["orange"] if k == "word tier" else PALETTE["alabaster"]
            for k in names]
    ax.barh(names, vals, color=cols, edgecolor=PALETTE["prussian"], linewidth=0.6)
    ax.axvline(majority, color=PALETTE["prussian"], ls="--", lw=1.2)
    ax.text(0.02, 1.04, f"dashed = always-neutral baseline {majority:.0%}",
            transform=ax.transAxes, fontsize=8.5, color=PALETTE["prussian"])
    for i, v in enumerate(vals):
        ax.text(v + 0.012, i, f"{v:.1%}", va="center", fontsize=9.5,
                fontweight="bold")
    ax.set_xlim(0, max(max(vals), majority) + 0.16)
    ax.set_title("A. Single-signal accuracy", loc="left", fontsize=11,
                 fontweight="bold")
    ax.set_xlabel("accuracy vs gold")

    # B — band sweep
    ax = axes[0, 1]
    cov = [s[0] for s in sweep]
    pre = [s[1] for s in sweep]
    ax.plot(cov, pre, "-o", color=PALETTE["prussian"], lw=2, ms=7)
    for c, p, t in sweep:
        ax.annotate(f"{t:.1f}", (c, p), textcoords="offset points",
                    xytext=(6, 6), fontsize=8.5, color="#555555")
    ship = next(s for s in sweep if abs(s[2] - CHEAP_THRESHOLD) < 1e-9)
    ax.plot([ship[0]], [ship[1]], "o", ms=13, mfc="none",
            mec=PALETTE["orange"], mew=2.5)
    ax.annotate(f"shipped {CHEAP_THRESHOLD}", (ship[0], ship[1]),
                textcoords="offset points", xytext=(-72, -18), fontsize=9,
                color=PALETTE["orange"], fontweight="bold")
    ax.set_ylim(min(pre) - 0.05, max(pre) + 0.06)
    ax.set_title("B. Cheap-tier band sweep (precision vs coverage)",
                 loc="left", fontsize=11, fontweight="bold")
    ax.set_xlabel("coverage (share of headlines fired)")
    ax.set_ylabel("precision when firing")
    ax.grid(alpha=0.3)

    # C — ensemble ladder
    ax = axes[1, 0]
    lnames = [l[0] for l in ladder]
    lvals = [l[1] for l in ladder]
    lcols = [PALETTE["alabaster"], PALETTE["alabaster"],
             PALETTE["orange"], PALETTE["prussian"]]
    ax.bar(range(len(ladder)), lvals, color=lcols,
           edgecolor=PALETTE["prussian"], linewidth=0.6)
    ax.set_xticks(range(len(ladder)))
    ax.set_xticklabels(lnames, fontsize=8.5)
    for i, v in enumerate(lvals):
        ax.text(i, v + 0.015, f"{v:.1%}", ha="center", fontsize=10,
                fontweight="bold")
    ax.set_ylim(0, max(lvals) + 0.12)
    ax.set_title("C. Combinations of the 5 signals", loc="left", fontsize=11,
                 fontweight="bold")
    ax.set_ylabel("accuracy vs gold (CV-5 for stacker)")

    # D — fires at shipped bands
    ax = axes[1, 1]
    labels = ["legacy DFT\nband 0.95", "legacy RF\nband 0.80",
              f"word tier\nband {CHEAP_THRESHOLD}"]
    fires = [legacy_dft, legacy_rf, word]
    dcols = [PALETTE["alabaster"], PALETTE["alabaster"], PALETTE["orange"]]
    ax.bar(labels, fires, color=dcols, edgecolor=PALETTE["prussian"], linewidth=0.6)
    ax.text(2, word + 4, f"{word} fires · {word_prec:.0%} precision",
            ha="center", fontsize=9.5, fontweight="bold",
            color=PALETTE["prussian"])
    ax.text(0.5, 6, "0 fires — deprecated,\nnever reaches its band",
            ha="center", fontsize=9, color="#555555")
    ax.set_ylim(0, max(fires) * 1.35)
    ax.set_title("D. Who fires at shipped bands (of 240)", loc="left",
                 fontsize=11, fontweight="bold")
    ax.set_ylabel("headlines fired")

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT.name}")


if __name__ == "__main__":
    main()
