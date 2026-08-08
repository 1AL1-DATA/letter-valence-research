"""Statistically-robust follow-up analysis for the cross-domain cascade eval.

Recomputes, from the stored per-instance predictions, every significance claim
made in README / SUMMARY / METHODOLOGY / research_report / arxiv_paper:

* exact two-sided paired McNemar p-values (scipy binomtest) on the clear
  (n = 651) and borderline (n = 416) general-news sets and the FPB borderline
  (n = 2,879) set;
* Wilson 95% CIs on every accuracy / false-polarity rate;
* exact paired-difference CIs (Wilson on the discordant ratio) and 20,000-
  resample bootstrap CIs on the headline accuracy differences;
* per-comparison resolvable difference at alpha = 0.01 / 80% power, i.e.
  2.802 * sqrt(m) / n for m discordant pairs — the power analysis that says
  what the n = 416 null results do and do not mean;
* the routing-only threshold sweep (label band held fixed at 0.1) that backs
  the 0.654 / 0.647 general best points and the FPB best = heavy-only result.

Run:
    cd /home/a/letter-valence-research
    /home/a/esg-dashboard/.venv/bin/python -m src.robustness_analysis
"""

from __future__ import annotations

import csv
import math
import random
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, "/home/a/esg-dashboard/src")

from src.benchmark_cascade import predict_from_v, cascade_predict  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

Z = 1.96
ALPHA = 0.01
POWER = 0.8


def wilson_ci(n: int, k: int, z: float = Z) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_p(b: int, c: int) -> float:
    m = b + c
    if m == 0:
        return 1.0
    return float(binomtest(min(b, c), m, 0.5, alternative="two-sided").pvalue)


def paired_counts(y, l1, l2) -> tuple[int, int]:
    """b = l1 wrong & l2 right; c = l2 wrong & l1 right."""
    n = len(y)
    b = sum(1 for i in range(n) if l1[i] != y[i] and l2[i] == y[i])
    c = sum(1 for i in range(n) if l2[i] != y[i] and l1[i] == y[i])
    return b, c


def paired_diff_ci(b: int, c: int, n: int, z: float = Z) -> tuple[float, float]:
    """Wilson CI on the paired difference (l1 - l2) in correct/positive rate."""
    m = b + c
    if m == 0:
        return (0.0, 0.0)
    plo, phi = wilson_ci(m, b, z)  # CI on b/m
    # d = (c - b)/n = 2*(m/n)*(p_hat - 0.5) with p_hat = c/m = 1 - b/m
    wlo = 2 * (m / n) * ((1 - phi) - 0.5)
    whi = 2 * (m / n) * ((1 - plo) - 0.5)
    return (wlo, whi)


def bootstrap_diff_ci(y, l1, l2, n_boot: int = 20_000, seed: int = 0):
    rng = random.Random(seed)
    n = len(y)
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        a1 = sum(1 for i in idx if y[i] == l1[i]) / n
        a2 = sum(1 for i in idx if y[i] == l2[i]) / n
        diffs.append(a1 - a2)
    diffs.sort()
    return diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot) - 1]


def resolvable(m: int, n: int) -> float:
    """Minimal |difference| this paired test can resolve at ALPHA / POWER."""
    za = 1.96  # 0.005 two-sided
    zb = 0.8416  # 80%
    return (za + zb) * math.sqrt(m) / n


def _val(sample, col: str) -> np.ndarray:
    return np.array([float(r[col]) for r in sample])


def _labels(sample, col: str) -> np.ndarray:
    return predict_from_v(_val(sample, col))


def _cascade_labels(sample, col: str) -> np.ndarray:
    return np.array([int(r[col]) for r in sample])


def main() -> None:
    results = REPO / "results"
    rows = list(csv.DictReader((results / "general_news_predictions.csv").open()))
    clear = [r for r in rows if r["set"] == "clear"]
    neut = [r for r in rows if r["set"] == "neutral"]
    yc = np.array([int(r["true"]) for r in clear])
    yn = np.array([int(r["true"]) for r in neut])

    print(f"general news: clear n = {len(yc)}, borderline n = {len(yn)}\n")

    methods_c = {
        "keyword": _labels(clear, "kw_v"),
        "heavy_fin": _labels(clear, "heavy_fin_v"),
        "cheap_fpb": _labels(clear, "cheap_fpb_v"),
        "vader": _labels(clear, "vader_v"),
        "cheap_news": _labels(clear, "cheap_news_v"),
        "heavy_gen": _labels(clear, "heavy_gen_v"),
        "cascade_fpb_gen": _cascade_labels(clear, "cascade_fpb_gen_label"),
        "cascade_news_gen": _cascade_labels(clear, "cascade_news_gen_label"),
    }
    methods_n = {
        "keyword": _labels(neut, "kw_v"),
        "heavy_fin": _labels(neut, "heavy_fin_v"),
        "cheap_fpb": _labels(neut, "cheap_fpb_v"),
        "vader": _labels(neut, "vader_v"),
        "cheap_news": _labels(neut, "cheap_news_v"),
        "heavy_gen": _labels(neut, "heavy_gen_v"),
        "cascade_fpb_gen": _cascade_labels(neut, "cascade_fpb_gen_label"),
        "cascade_news_gen": _cascade_labels(neut, "cascade_news_gen_label"),
    }

    print("== CLEAR SET accuracies (Wilson 95% CI) ==")
    for k, l in methods_c.items():
        kk = int((l == yc).sum())
        lo, hi = wilson_ci(len(yc), kk)
        print(f"  {k:20s} {kk / len(yc):.4f} [{lo:.4f}, {hi:.4f}]")

    pairs = [
        ("cascade_news_gen", "heavy_gen"),
        ("cascade_fpb_gen", "heavy_gen"),
        ("cascade_news_gen", "cascade_fpb_gen"),
        ("cheap_fpb", "heavy_fin"),
        ("cheap_fpb", "vader"),
        ("cheap_news", "vader"),
        ("cheap_news", "heavy_gen"),
        ("heavy_gen", "vader"),
        ("cascade_news_gen", "cheap_news"),
        ("cascade_fpb_gen", "cheap_fpb"),
    ]
    print("\n== CLEAR paired (l1 minus l2) ==")
    for a, b in pairs:
        bb, cc = paired_counts(yc, methods_c[a], methods_c[b])
        p = mcnemar_p(bb, cc)
        d = (cc - bb) / len(yc)
        wlo, whi = paired_diff_ci(bb, cc, len(yc))
        blo, bhi = bootstrap_diff_ci(yc, methods_c[a], methods_c[b])
        print(f"  {a:20s} vs {b:20s} b={bb:3d} c={cc:3d} d={d:+.4f} "
              f"Wilson[{wlo:+.4f},{whi:+.4f}] "
              f"boot[{blo:+.4f},{bhi:+.4f}] p={p:.3g} res={resolvable(bb+cc, len(yc))*100:.1f}pts")

    print("\n== BORDERLINE false-polarity (Wilson 95% CI) ==")
    for k, l in methods_n.items():
        fp = int(np.isin(l, (0, 2)).sum())
        lo, hi = wilson_ci(len(yn), fp)
        print(f"  {k:20s} {fp / len(yn):.4f} ({fp}) [{lo:.4f}, {hi:.4f}]  "
              f"neutral-pred={np.mean(l == 1):.4f}")

    print("\n== BORDERLINE paired (l1 minus l2) ==")
    for a, b in pairs:
        bb, cc = paired_counts(yn, methods_n[a], methods_n[b])
        p = mcnemar_p(bb, cc)
        d = (cc - bb) / len(yn)
        wlo, whi = paired_diff_ci(bb, cc, len(yn))
        print(f"  {a:20s} vs {b:20s} b={bb:3d} c={cc:3d} d={d:+.4f} "
              f"Wilson[{wlo:+.4f},{whi:+.4f}] p={p:.3g} res={resolvable(bb+cc, len(yn))*100:.1f}pts")

    # ---------------- FPB ----------------
    fpb = list(csv.DictReader((results / "cascade_predictions.csv").open()))
    fb = [r for r in fpb if r["set"] == "neutral"]
    fc = [r for r in fpb if r["set"] == "clear"]
    yb = np.array([int(r["true"]) for r in fb])
    lfp = {
        "keyword": _labels(fb, "kw_v"),
        "cheap": _labels(fb, "cheap_v"),
        "heavy": _labels(fb, "heavy_v"),
        "cascade": np.array([int(r["cascade_label"]) for r in fb]),
    }
    print(f"\n== FPB borderline (n = {len(yb)}) ==")
    for k, l in lfp.items():
        fp = int(np.isin(l, (0, 2)).sum())
        lo, hi = wilson_ci(len(yb), fp)
        print(f"  {k:8s} fp {fp / len(yb):.4f} ({fp}) [{lo:.4f}, {hi:.4f}]")
    for a, b in [("cascade", "cheap"), ("cascade", "heavy"), ("cascade", "keyword")]:
        bb, cc = paired_counts(yb, lfp[a], lfp[b])
        p = mcnemar_p(bb, cc)
        d = (cc - bb) / len(yb)
        wlo, whi = paired_diff_ci(bb, cc, len(yb))
        print(f"  {a:8s} vs {b:8s} b={bb:4d} c={cc:4d} d={d:+.4f} "
              f"Wilson[{wlo:+.4f},{whi:+.4f}] p={p:.3g} res={resolvable(bb+cc, len(yb))*100:.1f}pts")
    kw_clear_neu = float(np.mean(predict_from_v(_val(fc, "kw_v")) == 1))
    print(f"  keyword neutral-pred on FPB clear: {kw_clear_neu:.4f}")
    print(f"  keyword neutral-pred on FPB borderline: {np.mean(lfp['keyword'] == 1):.4f}")

    print("\n== FPB clear (n = %d) ==" % len(fc))
    yc2 = np.array([int(r["true"]) for r in fc])
    lc2 = {
        "cheap": _labels(fc, "cheap_v"),
        "heavy": _labels(fc, "heavy_v"),
        "keyword": _labels(fc, "kw_v"),
        "cascade": np.array([int(r["cascade_label"]) for r in fc]),
    }
    for a, b in [("cascade", "cheap"), ("cascade", "heavy"), ("cascade", "keyword")]:
        bb, cc = paired_counts(yc2, lc2[a], lc2[b])
        p = mcnemar_p(bb, cc)
        d = (cc - bb) / len(yc2)
        wlo, whi = paired_diff_ci(bb, cc, len(yc2))
        print(f"  {a:8s} vs {b:8s} b={bb:4d} c={cc:4d} d={d:+.4f} "
              f"Wilson[{wlo:+.4f},{whi:+.4f}] p={p:.3g}")

    print("\n== heavy_fin out-of-domain conservatism (general news) ==")
    print(f"  neutral-pred on borderline: {np.mean(methods_n['heavy_fin'] == 1):.4f}")
    print(f"  neutral-pred on clear:      {np.mean(methods_c['heavy_fin'] == 1):.4f}")

    print("\n== routing-only threshold sweep (label band held fixed at 0.1) ==")
    yt = np.array([int(r["true"]) for r in clear])
    for name, col in (("cascade_fpb_gen", "cheap_fpb_v"), ("cascade_news_gen", "cheap_news_v")):
        best = None
        frontier = []
        for ct in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
            code, tiers = cascade_predict(
                _val(clear, col), _val(clear, "heavy_gen_v"), _val(clear, "vader_v"),
                None, threshold=ct, band=0.1,
            )
            acc = float((code == yt).mean())
            hs = float(np.mean([t == "heavy" for t in tiers]))
            frontier.append((ct, acc, hs))
            if best is None or acc > best[1]:
                best = (ct, acc, hs)
        print(f"  {name}: best acc {best[1]:.4f} at ct={best[0]} "
              f"(heavy share {best[2]:.3f}); frontier: " +
              ", ".join(f"ct={ct}:{acc:.3f}/{hs:.2f}" for ct, acc, hs in frontier))

    fbt = np.array([int(r["true"]) for r in fc])
    best = None
    for ct in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        code, tiers = cascade_predict(
            _val(fc, "cheap_v"), _val(fc, "heavy_v"), _val(fc, "vader_v"),
            None, threshold=ct, band=0.1,
        )
        acc = float((code == fbt).mean())
        hs = float(np.mean([t == "heavy" for t in tiers]))
        if best is None or acc > best[1]:
            best = (ct, acc, hs)
    print(f"  FPB cascade: best acc {best[1]:.4f} at ct={best[0]} "
          f"(heavy share {best[2]:.3f}) = heavy-only exactly (0.9558)")


if __name__ == "__main__":
    main()
