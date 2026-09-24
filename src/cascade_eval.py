"""Cascade evaluation on live test news — new 2-tier engine vs legacy letter tiers.

Data: free Google News RSS headlines (cached NDJSON in data/cascade_test/);
gold = LLM-judge synthetic labels (see data/cascade_test/gold_labels.json meta).
Heavy tier (FinancialBERT) runs offline-disabled — deferred rows are answered
by VADER, which is the engine's designed fallback.

Usage (from the repo root):
    python -m src.cascade_eval
Outputs:
    data/cascade_test/sentiment_cascade_results.csv
    figures/sentiment_cascade_{funnel,valence_bands,examples}.png
"""

from __future__ import annotations

import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from src.sentiment_engine import engine
from src.sentiment_engine import legacy_engine as legacy

REPO = Path(__file__).resolve().parent.parent
TEST_DIR = REPO / "data" / "cascade_test"
NEWS_CACHE = TEST_DIR / "news_test.ndjson"
GOLD_PATH = TEST_DIR / "gold_labels.json"
RESULTS_CSV = TEST_DIR / "sentiment_cascade_results.csv"
VISUALS_DIR = REPO / "figures"

GREEN, AMBER, RED, GRAY, BLUE, ORANGE = (
    "#2e7d32", "#f9a825", "#c62828", "#9e9e9e", "#1565c0", "#e65100")
TIER_COLORS = {"cheap": GREEN, "heavy": ORANGE, "vader": ORANGE,
               "dft": BLUE, "rf": BLUE, "keyword": GRAY}

# Test queries: recognizable single names across sectors.
QUERIES = [
    "Apple stock", "Microsoft stock", "Nvidia stock", "Tesla stock",
    "Amazon stock", "JPMorgan", "Exxon Mobil", "Johnson & Johnson",
    "Procter & Gamble", "Caterpillar", "Boeing", "Netflix stock",
]
PER_QUERY = 20
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"}


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def fetch_news() -> pd.DataFrame:
    """Headlines from Google News RSS; cached as NDJSON."""
    import requests

    rows: dict[str, dict] = {}
    if NEWS_CACHE.exists():
        for line in NEWS_CACHE.read_text().splitlines():
            if line.strip():
                rows[json.loads(line)["headline"]] = json.loads(line)
        if rows:
            print(f"  cached news: {len(rows)} articles ({NEWS_CACHE.name})")
    if not rows:
        import html

        for q in QUERIES:
            url = ("https://news.google.com/rss/search?q="
                   + requests.utils.quote(q + " when:90d")
                   + "&hl=en-US&gl=US&ceid=US:en")
            try:
                resp = requests.get(url, headers=UA, timeout=15)
                resp.raise_for_status()
                root = ET.fromstring(resp.content)
                items = root.findall(".//item")[:PER_QUERY]
            except Exception as e:
                print(f"  [warn] {q}: {e}")
                continue
            for it in items:
                title = html.unescape((it.findtext("title") or "").strip())
                if not title:
                    continue
                rows[title] = {
                    "ticker": q.split()[0],
                    "query": q,
                    "headline": title,
                    "source": (it.findtext("source") or "").strip(),
                    "published": (it.findtext("pubDate") or "").strip(),
                    "link": (it.findtext("link") or "").strip(),
                }
            time.sleep(1.2)
        TEST_DIR.mkdir(parents=True, exist_ok=True)
        with NEWS_CACHE.open("w") as fh:
            for r in rows.values():
                fh.write(json.dumps(r) + "\n")
        print(f"  fetched: {len(rows)} unique articles -> {NEWS_CACHE.name}")
    return pd.DataFrame(list(rows.values()))


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
def evaluate() -> pd.DataFrame:
    df = fetch_news()
    texts = df["headline"].tolist()

    comps = engine.default_engine.component_scores(texts, use_heavy=False)
    routed = engine.route_components(
        comps["cheap_v"], comps["heavy_proba"], comps["vader_v"], comps["n_words"],
    )
    # legacy letter tiers, fire-all for side-by-side
    lcomps = legacy.default_engine.component_scores(texts, use_heavy=False)
    dft_v = 2 * np.asarray(lcomps["dft_p"]) - 1
    rf_v = 2 * np.asarray(lcomps["rf_p"]) - 1

    out = df.copy()
    out["cheap_v"] = comps["cheap_v"]
    out["dft_v"] = dft_v
    out["rf_v"] = rf_v
    out["vader_v"] = comps["vader_v"]
    out["keyword_v"] = comps["keyword_v"]
    out["tier"] = [r["tier"] for r in routed]
    out["v"] = [r["v"] for r in routed]
    out["score"] = [r["score"] for r in routed]
    out["label"] = [r["label"] for r in routed]
    out["n_words"] = comps["n_words"]
    if GOLD_PATH.exists():
        gold = json.loads(GOLD_PATH.read_text())["labels"]
        out["gold"] = [gold.get(str(i), "neutral") for i in out.index]
    return out


def _lab(v: float) -> str:
    """Pipeline-faithful label: quadratic score through the +-0.1 band."""
    return engine.label_for_score(engine.quadratic_score(float(v)))


def report(df: pd.DataFrame) -> None:
    n = len(df)
    counts = df["tier"].value_counts()
    cheap_n = int(counts.get("cheap", 0))
    defer_n = n - cheap_n
    print("\n=== Cascade funnel on live test news (new 2-tier engine) ===")
    print(f"  articles: {n}")
    print(f"  cheap    fired {cheap_n:4d}  ({cheap_n / n:6.1%} of corpus)  "
          f"band |v| >= {engine.CHEAP_THRESHOLD}")
    print(f"  heavy    {defer_n:4d}  ({defer_n / n:6.1%} of corpus)  "
          "(offline: VADER answers)")
    if "gold" in df.columns:
        m = df["tier"] == "cheap"
        if m.any():
            prec = float((df.loc[m, "label"] == df.loc[m, "gold"]).mean())
            print(f"  cheap precision vs gold: {prec:.1%}  (n={int(m.sum())})")
    print(f"\n  max |cheap v| = {df['cheap_v'].abs().max():.3f}  "
          f"(legacy dft {df['dft_v'].abs().max():.3f}, "
          f"rf {df['rf_v'].abs().max():.3f})")
    print(f"  label counts (offline): {df['label'].value_counts().to_dict()}")
    print("\n=== Per-example firings (every article) ===")
    print(f"  {'tier':7s} {'cheap v':>8s} {'dft v':>7s} {'vdr v':>7s} "
          f"{'score':>7s} {'label':8s} headline")
    for _, r in df.sort_values(["tier", "cheap_v"], ascending=[True, False]).iterrows():
        h = r["headline"][:58]
        print(f"  {r['tier']:7s} {r['cheap_v']:+8.3f} {r['dft_v']:+7.3f} "
              f"{r['vader_v']:+7.3f} {r['score']:+7.3f} {r['label']:8s} {h}")
    print(f"\n  results -> {RESULTS_CSV}")


def fire_all(df: pd.DataFrame) -> None:
    """Fire every signal on every row (no cascade gating) and score vs gold."""
    if "gold" not in df.columns:
        print("\n=== Fire-all comparison: skipped (no gold_labels.json) ===")
        return
    gold = df["gold"]
    n = len(df)
    cheap_l = np.where(df["cheap_v"].abs() >= engine.CHEAP_THRESHOLD,
                       [ _lab(v) for v in df["cheap_v"] ], "defer")
    dft_l = np.where(df["dft_v"].abs() >= legacy.DFT_THRESHOLD,
                     [ _lab(v) for v in df["dft_v"] ], "defer")
    rf_l = np.where(df["rf_v"].abs() >= legacy.RF_THRESHOLD,
                    [ _lab(v) for v in df["rf_v"] ], "defer")
    vdr_l = np.array([_lab(v) for v in df["vader_v"]])
    kw_l = np.array([_lab(v) for v in df["keyword_v"]])

    print("\n=== Fire-all comparison vs LLM-judge gold ===")
    print(f"  n = {n}  gold = {gold.value_counts().to_dict()}")
    for name, pred in [("cheap", cheap_l), ("legacy_dft", dft_l), ("legacy_rf", rf_l)]:
        m = np.asarray(pred) != "defer"
        if m.any():
            ok = int((np.asarray(pred)[m] == gold[m]).sum())
            neu = int((gold[m] == "neutral").sum())
            print(f"  {name:11s} fires {int(m.sum()):4d} ({m.mean():5.1%})  "
                  f"precision {ok / int(m.sum()):6.1%}  "
                  f"fires-on-gold-neutral {neu:3d}  "
                  f"wrong-on-directional-gold {ok_wrong(m, pred, gold, neu)}")
        else:
            print(f"  {name:11s} fires 0  (legacy band never reached — as documented)")
    print(f"  vader      answers all rows (fallback)     accuracy {float((vdr_l == gold).mean()):6.1%}")
    print(f"  keyword    answers all rows (reference)    accuracy {float((kw_l == gold).mean()):6.1%}")
    m = np.asarray(cheap_l) != "defer"
    if m.any():
        print(f"  cheap-vs-vader label agreement where cheap fires: "
              f"{(np.asarray(cheap_l)[m] == vdr_l[m]).mean():.1%} (n={int(m.sum())})")


def ok_wrong(m, pred, gold, neu) -> int:
    pred = np.asarray(pred)
    return int((pred[m] != gold[m]).sum()) - int(neu)


def vader_profile(df: pd.DataFrame) -> None:
    """VADER behavior: distribution, accuracy vs gold, band sweep, decorrelation."""
    if "gold" not in df.columns:
        return
    gold = df["gold"].values
    vv = df["vader_v"].values
    kv = df["keyword_v"].values
    vl = np.array([_lab(v) for v in vv])
    kl = np.array([_lab(v) for v in kv])
    m_dir = gold != "neutral"
    print("\n=== VADER behavior vs LLM-judge gold ===")
    print(f"  compound: mean {vv.mean():+.2f}  p10 {np.quantile(vv, .1):+.2f}  "
          f"p50 {np.quantile(vv, .5):+.2f}  p90 {np.quantile(vv, .9):+.2f}")
    print(f"  vader    accuracy {float((vl == gold).mean()):.1%}  "
          f"neutral-abstain {float((vl == 'neutral').mean()):.1%}  "
          f"directional {float((vl[m_dir] == gold[m_dir]).mean()):.1%}")
    print(f"  keyword  accuracy {float((kl == gold).mean()):.1%}  "
          f"neutral-abstain {float((kl == 'neutral').mean()):.1%}  "
          f"directional {float((kl[m_dir] == gold[m_dir]).mean()):.1%}")
    dis = vl != kl
    if dis.any():
        print(f"  disagree on {int(dis.sum())} rows: "
              f"vader right {float((vl[dis] == gold[dis]).mean()):.1%} vs "
              f"keyword right {float((kl[dis] == gold[dis]).mean()):.1%}")
    print("  band sweep (hypothetical vader tier):")
    for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.8]:
        f = np.abs(vv) >= t
        if f.sum() < 5:
            continue
        print(f"    |compound|>={t:.1f}: fires {int(f.sum()):3d} ({f.mean():5.1%})  "
              f"precision {float((vl[f] == gold[f]).mean()):5.1%}")


def compare_ensemble(df: pd.DataFrame) -> None:
    """Fixed gates + valence averages + CV stacker over all signals, vs gold."""
    if "gold" not in df.columns:
        return
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    gold = df["gold"].values
    sig = {
        "cheap": np.array([_lab(v) for v in df["cheap_v"]]),
        "dft": np.array([_lab(v) for v in df["dft_v"]]),
        "rf": np.array([_lab(v) for v in df["rf_v"]]),
        "vader": np.array([_lab(v) for v in df["vader_v"]]),
        "keyword": np.array([_lab(v) for v in df["keyword_v"]]),
    }
    acc = lambda p: float(np.mean(np.asarray(p) == gold))
    print("\n=== Ensemble combinations vs gold (combinatronics) ===")
    for name, p in sig.items():
        print(f"  single: {name:8s} {acc(p):.1%}")

    d, r, k, vd = sig["dft"], sig["rf"], sig["keyword"], sig["vader"]
    ch = sig["cheap"]

    def majority3(a, b, c):
        out = []
        for x, y, z in zip(a, b, c):
            if x == y or x == z:
                out.append(x)
            elif y == z:
                out.append(y)
            else:
                out.append("neutral")
        return np.array(out)

    print("  gates:")
    print(f"    majority(cheap, vader, keyword)        {acc(majority3(ch, vd, k)):.1%}")
    print(f"    majority(dft, rf, keyword) [legacy]    {acc(majority3(d, r, k)):.1%}")
    print(f"    consensus all-5                        "
          f"{acc(np.array([t[0] if len(set(t)) == 1 else 'neutral' for t in zip(*sig.values())])):.1%}")
    print("  valence averages (quadratic+band on mean):")
    for wname, w in [("equal(5)", (0.2, 0.2, 0.2, 0.2, 0.2)),
                     ("cheap+vader+kw", (0.5, 0.0, 0.0, 0.25, 0.25))]:
        vw = sum(wi * df[c].values for wi, c in
                 zip(w, ["cheap_v", "dft_v", "rf_v", "vader_v", "keyword_v"]))
        print(f"    {wname:36s} {acc([_lab(v) for v in vw]):.1%}")

    X = df[["cheap_v", "dft_v", "rf_v", "vader_v", "keyword_v"]].values
    cv = StratifiedKFold(5, shuffle=True, random_state=7)
    clf = LogisticRegression(max_iter=1000)
    cv_acc = cross_val_score(clf, X, gold, cv=cv).mean()
    clf.fit(X, gold)
    print(f"  stacker (5-signal logit): CV-5 {cv_acc:.1%}  "
          f"resub {float((clf.predict(X) == gold).mean()):.1%}")
    oracle = np.mean(np.logical_or.reduce([p == gold for p in sig.values()]))
    print(f"  oracle ceiling (any-of-5 correct): {oracle:.1%}")


def frontier(df: pd.DataFrame) -> None:
    """Cheap-tier threshold sweep: coverage vs precision against gold."""
    if "gold" not in df.columns:
        return
    gold = df["gold"]
    cv = df["cheap_v"]
    print("\n=== Cheap-tier band sweep (precision vs LLM-judge gold) ===")
    print(f"  {'band':>5} {'fires':>6} {'coverage':>9} {'precision':>9} {'deferred':>9}")
    for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        f = cv.abs() >= t
        if int(f.sum()) < 5:
            continue
        pred = np.where(f, [_lab(v) for v in cv], "defer")
        prec = float((np.asarray(pred)[f] == gold[f]).mean())
        mark = "  <- shipped" if t == engine.CHEAP_THRESHOLD else ""
        print(f"  {t:5.1f} {int(f.sum()):6d} {float(f.mean()):9.1%} "
              f"{prec:9.1%} {1 - float(f.mean()):9.1%}{mark}")


# ---------------------------------------------------------------------------
# figures (quality_visuals style)
# ---------------------------------------------------------------------------
def _save(fig, name: str) -> None:
    fig.savefig(VISUALS_DIR / name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}")


def _box(ax, x, y, w, h, fc, ec, alpha=1.0, lw=1.2, round_=0.06):
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad={round_}",
        facecolor=fc, edgecolor=ec, alpha=alpha, linewidth=lw))


def _pct(x: float) -> str:
    return f"{x:.0%}" if x >= 0.005 else f"{x:.1%}"


def v1_funnel(df: pd.DataFrame) -> None:
    n = len(df)
    cheap_n = int((df["tier"] == "cheap").sum())
    defer_n = n - cheap_n
    rows = [
        ("INPUT", n, GRAY, "free Google News RSS headlines, deduped", None),
        ("T1 · cheap", cheap_n, GREEN,
         f"fires iff |v| ≥ {engine.CHEAP_THRESHOLD}  ·  {_pct(cheap_n / n)} of corpus",
         "word tier: TF-IDF(1-2) + [VADER, keyword] → logistic"),
        ("T2 · heavy", defer_n, ORANGE,
         f"everything the cheap tier defers  ·  {_pct(defer_n / n)} of corpus",
         "FinancialBERT 3-class (offline: VADER answers)"),
    ]
    fig, ax = plt.subplots(figsize=(11.5, 6.6))
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 7.0)
    ax.axis("off")
    ax.text(5.75, 6.6, "Sentiment Cascade — Real Coverage on Live News",
            ha="center", fontsize=15, fontweight="bold", color="#1a1a1a")
    sub = f"n = {n} headlines · word-tier operating point |v| ≥ {engine.CHEAP_THRESHOLD} (research benchmark)"
    if "gold" in df.columns:
        m = df["tier"] == "cheap"
        if m.any():
            prec = float((df.loc[m, "label"] == df.loc[m, "gold"]).mean())
            sub += f" · cheap precision vs gold {prec:.0%}"
    ax.text(5.75, 6.1, sub, ha="center", fontsize=10, color="#666666",
            style="italic")

    y_top = 5.0
    for name, c, col, band, cost in rows:
        w = 0.5 + 7.6 * (c / n)
        _box(ax, 0.8, y_top - 0.5, 8.6, 1.0, "#f0f0f0", "none", round_=0.05)
        _box(ax, 0.8, y_top - 0.5, w, 1.0, col, "none", alpha=0.30, round_=0.05)
        _box(ax, 0.8, y_top - 0.5, 1.9, 1.0, col, "none", alpha=0.92, round_=0.05)
        ax.text(1.75, y_top, name, ha="center", va="center", fontsize=10.5,
                fontweight="bold", color="white")
        ax.text(2.95, y_top, f"{c} article" + ("s" if c != 1 else ""), ha="left",
                va="center", fontsize=10.5, fontweight="bold", color="#1a1a1a")
        ax.text(4.9, y_top, band, ha="left", va="center", fontsize=9,
                color="#444444")
        if cost:
            ax.text(9.4, y_top - 0.88, cost, ha="right", va="center", fontsize=8,
                    color="#888888", style="italic")
        if name != "INPUT":
            ax.annotate("", xy=(2.7, y_top - 1.05), xytext=(2.7, y_top - 0.62),
                        arrowprops=dict(arrowstyle="-|>", color="#999999", lw=1.8))
        y_top -= 1.7

    ax.text(5.75, 0.4, f"Word tier answers {_pct(cheap_n / n)} on the cheap; "
                       f"transformer only the {_pct(defer_n / n)} residual",
            ha="center", fontsize=9.5, style="italic", color="#555555",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#f5f5f5",
                      edgecolor="#cccccc"))
    _save(fig, "sentiment_cascade_funnel.png")


def v2_valence_bands(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.0), sharex=True)
    fig.suptitle("New word tier vs legacy DFT probe — valence vs fire bands",
                 fontsize=15, fontweight="bold", color="#1a1a1a", y=0.98)
    specs = [
        ("cheap_v", engine.CHEAP_THRESHOLD, "Cheap word tier  v = p⁺ − p⁻",
         f"fires: {int((df['cheap_v'].abs() >= engine.CHEAP_THRESHOLD).sum())}/{len(df)} "
         "in green zones"),
        ("dft_v", legacy.DFT_THRESHOLD, "LEGACY DFT probe  v = 2p − 1",
         f"fires: {int((df['dft_v'].abs() >= legacy.DFT_THRESHOLD).sum())}/{len(df)} "
         "— never fires (deprecated)"),
    ]
    for ax, (col, thr, title, fire_txt) in zip(axes, specs):
        v = df[col].values
        order = np.argsort(np.abs(v))
        v = v[order]
        colors = [TIER_COLORS.get(t, GRAY) for t in df["tier"].values[order]]
        x = np.arange(len(v))
        ax.axhspan(-1.02, -thr, color="#e8f5e9", alpha=0.55)
        ax.axhspan(thr, 1.02, color="#e8f5e9", alpha=0.55)
        ax.axhline(thr, color=GREEN, ls="--", lw=1.2)
        ax.axhline(-thr, color=GREEN, ls="--", lw=1.2)
        ax.axhspan(-thr, thr, color="#fafafa")
        ax.scatter(x, v, s=22, c=colors, alpha=0.75, lw=0.2,
                   edgecolors="white", zorder=3)
        # keep the annotation clear of the band line (high bands need the bottom)
        if thr < 0.7:
            ax.text(0.012, 0.965, fire_txt, transform=ax.transAxes,
                    fontsize=9.5, color="#555555", va="top")
        else:
            ax.text(0.988, 0.05, fire_txt, transform=ax.transAxes,
                    fontsize=9.5, color="#555555", va="bottom", ha="right")
        ax.set_title(f"{title}   ·   band |v| ≥ {thr}", fontsize=10.5,
                     loc="left", color="#333333")
        ax.set_ylim(-1.02, 1.02)
        ax.set_ylabel("v")
        ax.grid(axis="x", color="#eeeeee", lw=0.5)
    axes[1].set_xlabel("articles, sorted by |v| (left = least confident)")
    handles = [mpatches.Patch(color=c, label=t)
               for t, c in TIER_COLORS.items() if t in ("cheap", "vader")]
    axes[1].legend(handles=handles, loc="lower left", fontsize=9, frameon=True,
                   framealpha=0.9, borderaxespad=0.4)
    _save(fig, "sentiment_cascade_valence_bands.png")


def v3_examples(df: pd.DataFrame) -> None:
    """Per-example table: cheap near-misses, legacy DFT, VADER disagreements."""
    picks = pd.concat([
        df.reindex(df["cheap_v"].abs().sort_values(ascending=False).index).head(7),
        df.reindex(df["dft_v"].abs().sort_values(ascending=False).index).head(5),
        df.reindex(df["vader_v"].abs().sort_values(ascending=False).index).head(4),
    ])
    show = picks[~picks.index.duplicated()].head(16)
    rows = len(show)
    rows_h = rows * 0.5
    fig, ax = plt.subplots(figsize=(13.6, 0.34 * (rows_h + 2.6)))
    ax.set_xlim(0, 13.6)
    ax.set_ylim(0, rows_h + 2.6)
    ax.axis("off")
    ax.text(6.8, rows_h + 2.1, "Per-Example Firings — new word tier vs legacy",
            ha="center", fontsize=14, fontweight="bold", color="#1a1a1a")
    ax.text(6.8, rows_h + 1.55,
            f"cheap fires at |v| ≥ {engine.CHEAP_THRESHOLD} (legacy dft band "
            f"{legacy.DFT_THRESHOLD} never reached); gold = LLM-judge (synthetic)",
            ha="center", fontsize=9.5, color="#666666", style="italic")
    cols = [("headline", 0.25, 6.6), ("cheap v", 6.95, 1.05),
            ("dft v", 8.05, 1.05), ("vdr v", 9.15, 1.05),
            ("tier", 10.3, 1.05), ("label", 11.4, 1.0), ("gold", 12.45, 1.0)]
    for name, x, w in cols:
        ax.text(x + w / 2, rows_h + 0.95, name, ha="center", fontsize=9.5,
                fontweight="bold", color="#1a1a1a")
    y = rows_h + 0.42
    for i, (_, r) in enumerate(show.iterrows()):
        if i % 2 == 0:
            ax.add_patch(mpatches.Rectangle((0.1, y - 0.19), 13.4, 0.5,
                                            facecolor="#f7f7f7", lw=0))
        h = r["headline"]
        if len(h) > 72:
            h = h[:69] + "…"
        ax.text(0.25, y + 0.05, h, ha="left", va="center", fontsize=8.3,
                color="#222222")
        ax.text(7.47, y + 0.05, f"{r['cheap_v']:+.3f}", ha="center", va="center",
                fontsize=8.3, color="#222222")
        ax.text(8.57, y + 0.05, f"{r['dft_v']:+.3f}", ha="center", va="center",
                fontsize=8.3, color="#888888")
        ax.text(9.67, y + 0.05, f"{r['vader_v']:+.3f}", ha="center", va="center",
                fontsize=8.3, color="#222222")
        _box(ax, 10.32, y - 0.09, 1.0, 0.37, TIER_COLORS.get(r["tier"], GRAY),
             "none", alpha=0.9, round_=0.05)
        ax.text(10.82, y + 0.09, r["tier"], ha="center", va="center",
                fontsize=7.5, fontweight="bold", color="white")
        lab_c = GREEN if r["label"] == "positive" else (
            RED if r["label"] == "negative" else GRAY)
        ax.text(11.9, y + 0.05, r["label"], ha="center", va="center",
                fontsize=8.3, fontweight="bold", color=lab_c)
        if "gold" in df.columns:
            gold_c = GREEN if r["gold"] == "positive" else (
                RED if r["gold"] == "negative" else GRAY)
            ax.text(12.95, y + 0.05, r["gold"], ha="center", va="center",
                    fontsize=8.3, fontweight="bold", color=gold_c)
        y -= 0.5
    _save(fig, "sentiment_cascade_examples.png")


def main() -> None:
    VISUALS_DIR.mkdir(exist_ok=True)
    TEST_DIR.mkdir(exist_ok=True)
    print("Sentiment cascade — new 2-tier engine vs legacy letter tiers")
    df = evaluate()
    fire_all(df)
    vader_profile(df)
    compare_ensemble(df)
    df.to_csv(RESULTS_CSV, index=False)
    report(df)
    frontier(df)
    print("\nVisuals:")
    v1_funnel(df)
    v2_valence_bands(df)
    v3_examples(df)


if __name__ == "__main__":
    main()
