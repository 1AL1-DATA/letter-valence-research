"""4-panel visual for the 2-tier sentiment cascade evaluation.

Reads ``results/cascade_benchmark.json`` + ``results/cascade_predictions.csv``
and writes ``figures/cascade_sentiment_eval.png``.

Panels:
  A. Accuracy +/- Wilson 95% CI per method (clear-polarity set, n=1,967)
  B. Misclassification breakdown per method on the clear set
  C. Borderline uncertainty: |score| distributions on the neutral set (n=2,879)
     with the neutral-band cutoff; false-polarity rates annotated
  D. Cascade tier routing + per-tier accuracy

Run from the research repo root:
    /home/a/esg-dashboard/.venv/bin/python -m src.figures_cascade
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
OUT = FIGURES / "cascade_sentiment_eval.png"

METHODS = ("keyword", "vader", "cheap", "cascade", "heavy")
METHOD_LABELS = {
    "keyword": "Keyword lexicon",
    "vader": "VADER",
    "cheap": "Cheap word tier",
    "cascade": "Cascade (2-tier)",
    "heavy": "FinancialBERT",
}
METHOD_COLORS = {
    "keyword": "#A8A492",
    "vader": "#8FA6B0",
    "cheap": "#C9B8A8",
    "cascade": PALETTE["orange"],
    "heavy": PALETTE["prussian"],
}
BAND = 0.1  # neutral band on the quadratic score (matches engine default)
CHEAP_THRESHOLD = 0.6  # cascade routing threshold (matches engine default)
CODE_LABEL = {0: "negative", 1: "neutral", 2: "positive"}


def predict_from_v(v: np.ndarray) -> np.ndarray:
    score = np.copysign(v * v, v)
    return np.where(score >= BAND, 2, np.where(score <= -BAND, 0, 1))


def method_v(rows: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "cheap": rows["cheap_v"].to_numpy(),
        "heavy": rows["heavy_v"].to_numpy(),
        "vader": rows["vader_v"].to_numpy(),
        "keyword": rows["kw_v"].to_numpy(),
    }


def cascade_v(rows: pd.DataFrame) -> np.ndarray:
    """Per-instance cascade |score| (cheap tier where it fires, else heavy)."""
    out = []
    for r in rows.itertuples(index=False):
        if abs(r.cheap_v) >= CHEAP_THRESHOLD:
            v = r.cheap_v
        elif not np.isnan(r.heavy_v):
            v = r.heavy_v
        else:
            v = r.vader_v
        out.append(v * v)
    return np.array(out)


def main() -> None:
    apply_style()
    report = json.load(open(RESULTS / "cascade_benchmark.json"))
    df = pd.read_csv(RESULTS / "cascade_predictions.csv")
    clear = df[df["set"] == "clear"].reset_index(drop=True)
    neut = df[df["set"] == "neutral"].reset_index(drop=True)
    y_true = clear["true"].to_numpy()

    clear_v = method_v(clear)
    clear_v["cascade"] = clear["cascade_label"].to_numpy().astype(float)  # used as code
    neut_v = method_v(neut)
    neut_v["cascade"] = cascade_v(neut)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5))
    fig.suptitle(
        "2-tier sentiment cascade vs baselines — FinancialPhraseBank\n"
        "clear-polarity n=1,967 (5-fold CV) + held-out neutral n=2,879",
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
    ax.set_xlabel("Accuracy (clear polarity)")
    ax.set_title("A. Accuracy ± 95% Wilson CI", fontweight="bold")
    for bar, acc in zip(bars, accs):
        ax.text(acc + 0.012, bar.get_y() + bar.get_height() / 2, f"{acc:.3f}",
                va="center", ha="left", fontsize=8)
    ax.axvline(0.5, color=PALETTE["black"], ls=":", lw=0.8, alpha=0.5)

    # ---- Panel B: misclassification breakdown (clear set) ----
    ax = axes[0, 1]
    x = np.arange(len(METHODS))
    correct, wrong_pol, neut_miss = [], [], []
    for m in METHODS:
        if m == "cascade":
            pred = clear["cascade_label"].to_numpy()
        else:
            pred = predict_from_v(clear_v[m])
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
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS], rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Fraction of clear set")
    ax.set_title("B. Misclassification (clear polarity)", fontweight="bold")
    ax.legend(loc="upper center", ncol=3, frameon=True)
    neg_rec = [ms[m]["negative_recall"] for m in METHODS]
    for i, nr in enumerate(neg_rec):
        ax.annotate(f"neg rec {nr:.2f}", xy=(i, 0.03), ha="center", fontsize=7,
                    color=PALETTE["white"], fontweight="bold")

    # ---- Panel C: borderline uncertainty (neutral set) ----
    ax = axes[1, 0]
    positions = np.arange(len(METHODS))
    vals = [np.abs(neut_v[m]) ** 2 for m in METHODS]
    vp = ax.violinplot(vals, positions=positions, showextrema=False, widths=0.8)
    for part, m in zip(vp["bodies"], METHODS):
        part.set_facecolor(METHOD_COLORS[m])
        part.set_alpha(0.65)
    ax.axhline(BAND, color=PALETTE_WARM["accent"], ls="--", lw=1.2)
    band_label = f"neutral band (|score| = {BAND:.2f})"
    ax.text(4.45, BAND + 0.02, band_label, fontsize=8, color=PALETTE_WARM["accent"])
    ax.set_xticks(positions)
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS], rotation=20, ha="right")
    ax.set_ylabel("|score| on neutral news (quadratic)")
    ax.set_ylim(0, 1.05)
    ax.set_title("C. Borderline uncertainty (neutral set)", fontweight="bold")
    for i, m in enumerate(METHODS):
        fp = report["borderline_set"][m]["false_polarity_rate"]
        ax.annotate(f"false pol {fp:.0%}", xy=(i, 1.02), ha="center", fontsize=8)

    # ---- Panel D: tier routing ----
    ax = axes[1, 1]
    tr = report["tier_routing_clear"]
    br = report["borderline_tier_routing"]
    tiers = ["cheap", "heavy"]
    tier_colors = {"cheap": "#C9B8A8", "heavy": PALETTE["prussian"]}
    clear_shares = [tr.get(t, {}).get("share", 0.0) for t in tiers]
    neut_shares = [br.get(t, {}).get("share", 0.0) for t in tiers]
    left_c = np.zeros(2)
    for share, t in zip(clear_shares, tiers):
        ax.barh(0, share, left=left_c[0], color=tier_colors[t], height=0.5,
                label=f"{t} tier" if share else None)
        acc = tr.get(t, {}).get("accuracy")
        if share and acc is not None:
            ax.text(left_c[0] + share / 2, 0, f"{share:.0%} · acc {acc:.3f}",
                    ha="center", va="center", fontsize=8, color=PALETTE["white"], fontweight="bold")
        left_c[0] += share
    left_n = np.zeros(2)
    for share, t in zip(neut_shares, tiers):
        ax.barh(1, share, left=left_n[1], color=tier_colors[t], height=0.5)
        if share:
            ax.text(left_n[1] + share / 2, 1, f"{share:.0%}",
                    ha="center", va="center", fontsize=8, color=PALETTE["white"], fontweight="bold")
        left_n[1] += share
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["clear polarity\n(n=1,967)", "neutral\n(n=2,879)"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of decisions routed per tier")
    ax.set_title("D. Cascade tier routing", fontweight="bold")
    ax.legend(loc="lower right", frameon=True)

    fig.savefig(OUT)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
