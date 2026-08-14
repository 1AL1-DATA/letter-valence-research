"""4-panel visual for the 2-tier sentiment cascade on general news.

Reads ``results/general_news_benchmark.json`` + ``results/general_news_predictions.csv``
and writes ``figures/general_news_eval.png``.

Panels:
  A. Accuracy +/- Wilson 95% CI per method (clear-polarity set, n=651):
     cheap tiers, cascade x 2 heavy tiers, both transformers, VADER, keyword
  B. Misclassification breakdown per method on the clear set
  C. Cascade tier routing + per-tier accuracy (cheap_news x general heavy)
  D. Threshold sweep: accuracy vs heavy-tier share across cheap tiers

Run from the research repo root:
    python -m src.figures_general
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.style import PALETTE, PALETTE_WARM, apply_style

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
FIGURES = REPO / "figures"
OUT = FIGURES / "general_news_eval.png"

METHODS = (
    "keyword",
    "vader",
    "cheap_fpb",
    "cheap_news",
    "cascade_fpb_gen",
    "cascade_news_gen",
    "heavy_fin",
    "heavy_gen",
)
METHOD_LABELS = {
    "keyword": "Keyword lexicon",
    "vader": "VADER",
    "cheap_fpb": "Cheap tier (trained on FPB)",
    "cheap_news": "Cheap tier (trained on news)",
    "cascade_fpb_gen": "Cascade FPB-cheap + gen-heavy",
    "cascade_news_gen": "Cascade news-cheap + gen-heavy",
    "heavy_fin": "FinancialBERT (finance-tuned)",
    "heavy_gen": "General BERT (twitter-roberta)",
}
METHOD_COLORS = {
    "keyword": "#A8A492",
    "vader": "#8FA6B0",
    "cheap_fpb": "#C9B8A8",
    "cheap_news": "#B89A76",
    "cascade_fpb_gen": "#E8A13C",
    "cascade_news_gen": PALETTE["orange"],
    "heavy_fin": PALETTE["alabaster"],
    "heavy_gen": PALETTE["prussian"],
}
BAND = 0.1  # neutral band on the quadratic score (matches engine default)
CODE_LABEL = {0: "negative", 1: "neutral", 2: "positive"}


def predict_from_v(v: np.ndarray) -> np.ndarray:
    score = np.copysign(v * v, v)
    return np.where(score >= BAND, 2, np.where(score <= -BAND, 0, 1))


def main() -> None:
    apply_style()
    report = json.load(open(RESULTS / "general_news_benchmark.json"))
    df = pd.read_csv(RESULTS / "general_news_predictions.csv")
    clear = df[df["set"] == "clear"].reset_index(drop=True)
    y_true = clear["true"].to_numpy()

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5))
    fig.suptitle(
        "2-tier sentiment cascade on general news — NewsMTSC (non-financial)\n"
        "clear-polarity n=651 + held-out neutral n=416; heavy tiers fixed",
        fontweight="bold",
    )

    # ---- Panel A: accuracy +/- Wilson CI ----
    ax = axes[0, 0]
    ms = report["clear_set"]
    names = [METHOD_LABELS[m] for m in METHODS]
    accs = [ms[m]["accuracy"] for m in METHODS]
    lows = [ms[m]["accuracy"] - ms[m]["ci_low"] for m in METHODS]
    highs = [ms[m]["ci_high"] - ms[m]["accuracy"] for m in METHODS]
    colors = [METHOD_COLORS[m] for m in METHODS]
    bars = ax.barh(names, accs, xerr=[lows, highs], color=colors, capsize=3,
                   error_kw={"elinewidth": 1.0, "ecolor": PALETTE["black"]})
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Accuracy (clear polarity, n=651)")
    ax.set_title("A. Accuracy ± 95% Wilson CI — general news", fontweight="bold")
    for bar, acc in zip(bars, accs):
        ax.text(acc + 0.012, bar.get_y() + bar.get_height() / 2, f"{acc:.3f}",
                va="center", ha="left", fontsize=8)
    ax.axvline(0.5, color=PALETTE["black"], ls=":", lw=0.8, alpha=0.5)
    ax.text(0.505, len(METHODS) - 0.4, "chance", fontsize=7, color=PALETTE["black"], alpha=0.6)
    ax.axvline(0.616, color=PALETTE["black"], ls="--", lw=0.9, alpha=0.7)
    ax.text(0.622, len(METHODS) - 0.4, "majority class\n(401 neg / 250 pos)", fontsize=6.5,
            color=PALETTE["black"], alpha=0.7, va="top", ha="left")

    # ---- Panel B: misclassification breakdown (clear set) ----
    ax = axes[0, 1]
    short = {
        "keyword": "keyword",
        "vader": "VADER",
        "cheap_fpb": "cheap-FPB",
        "cheap_news": "cheap-news",
        "cascade_fpb_gen": "cas-FPB",
        "cascade_news_gen": "cas-news",
        "heavy_fin": "FinBERT",
        "heavy_gen": "gen-BERT",
    }
    x = np.arange(len(METHODS))
    correct, wrong_pol, neut_miss = [], [], []
    for m in METHODS:
        if m.startswith("cascade_"):
            pred = clear[f"{m}_label"].to_numpy()
        else:
            col = {"cheap_fpb": "cheap_fpb_v", "cheap_news": "cheap_news_v",
                   "heavy_fin": "heavy_fin_v", "heavy_gen": "heavy_gen_v",
                   "vader": "vader_v", "keyword": "kw_v"}[m]
            pred = predict_from_v(clear[col].to_numpy())
        right = pred == y_true
        wrong = pred != y_true
        pred_neutral = pred == 1
        correct.append(right.mean())
        wrong_pol.append((wrong & ~pred_neutral).mean())
        neut_miss.append((wrong & pred_neutral).mean())
    ax.bar(x, correct, color=PALETTE["prussian"], label="correct")
    ax.bar(x, wrong_pol, bottom=correct, color=PALETTE_WARM["accent"], label="wrong polarity")
    ax.bar(x, neut_miss, bottom=np.array(correct) + np.array(wrong_pol),
           color=PALETTE["alabaster"], label="predicted neutral")
    ax.set_xticks(x)
    ax.set_xticklabels([short[m] for m in METHODS], rotation=32, ha="right", fontsize=7.5)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Fraction of clear set")
    ax.set_title("B. Misclassification (clear polarity)", fontweight="bold")
    ax.legend(loc="upper center", ncol=3, frameon=True, fontsize=8,
              bbox_to_anchor=(0.5, 1.02))

    # ---- Panel C: cascade tier routing (news-cheap x general heavy) ----
    ax = axes[1, 0]
    tr = report["tier_routing_clear"]["cascade_news_gen"]
    tiers = ["cheap", "heavy"]
    tier_colors = {"cheap": "#B89A76", "heavy": PALETTE["prussian"]}
    shares = [tr.get(t, {}).get("share", 0.0) for t in tiers]
    accs_t = [tr.get(t, {}).get("accuracy") for t in tiers]
    left = 0.0
    for share, t, acc in zip(shares, tiers, accs_t):
        ax.barh(0, share, left=left, color=tier_colors[t], height=0.45,
                label=f"{t} tier" if share else None)
        if share and acc is not None:
            ax.text(left + share / 2, 0, f"{share:.0%} · acc {acc:.3f}",
                    ha="center", va="center", fontsize=8,
                    color=PALETTE["white"], fontweight="bold")
        left += share
    ax.set_yticks([0])
    ax.set_yticklabels(["clear polarity\n(n=651)"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of decisions routed per tier")
    ax.set_title("C. Cascade routing (news-cheap + general heavy)", fontweight="bold")
    ax.legend(loc="lower right", frameon=True)

    # ---- Panel D: threshold sweep ----
    ax = axes[1, 1]
    sweep = report["threshold_sweep_top"] + [
        report["threshold_sweep_best"],
    ]
    for key, color in (("cascade_fpb_gen", "#E8A13C"), ("cascade_news_gen", PALETTE["orange"])):
        pts = [(e["heavy_share"], e["accuracy"]) for e in sweep if e["cascade"] == key]
        if pts:
            pts = sorted(pts)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=4,
                    color=color, label=key.replace("cascade_", "").replace("_", " "))
    ax.set_xlabel("Heavy-tier share")
    ax.set_ylabel("Cascade accuracy")
    ax.set_title("D. Threshold sweep (routing-only, band fixed; in-sample)", fontweight="bold")
    ax.legend(loc="lower left", frameon=True)
    ax.axhline(0.5, color=PALETTE["black"], ls=":", lw=0.8, alpha=0.5)

    fig.savefig(OUT)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
