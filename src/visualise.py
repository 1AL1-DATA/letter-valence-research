"""
Dedicated visualisations for the letter-valence research:
  1. DFT spectral heatmap  (positive vs negative words)
  2. DFT word fingerprints (individual word spectra)
  3. SHAP beeswarm plot    (feature attribution landscape)
  4. SHAP bar plot        (mean |SHAP| per feature)
  5. SHAP waterfall       (one positive and one negative prediction)

All figures are publication-quality PNG at 300 dpi, referenced in the
paper / blog post / LinkedIn post.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.data import load_articles_binary
from src.features import features as feature_one_word, get_feature_names
from src.style import apply_style, PALETTE, SEMANTIC

apply_style()

# Local palette aliases for use in plots.
PALETTE_PRIMARY   = SEMANTIC['primary']
PALETTE_HIGHLIGHT = SEMANTIC['highlight']
PALETTE_MUTED     = SEMANTIC['muted']
PALETTE_TEXT      = SEMANTIC['text']

# visualise.py has colorbars in some figures, which do not work with
# constrained_layout (matplotlib raises a RuntimeError on save). Switch
# to the classic layout engine so fig.tight_layout() calls below work.
plt.rcParams["figure.constrained_layout.use"] = False

OUT_DIR = REPO / "figures"
OUT_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _pad_to(vec: np.ndarray, n: int) -> np.ndarray:
    """Right-pad or truncate a vector to length n."""
    out = np.zeros(n)
    L = min(len(vec), n)
    out[:L] = vec[:L]
    return out


def dft_spectrum(word: str, n_bins: int = 32):
    """
    Compute the log-magnitude DFT spectrum of a word's alphabet-position signal.
    Returns (log_magnitude, phase), both length n_bins.
    """
    positions = [ord(c) - ord('a') + 1 for c in word.lower() if c.isalpha()]
    if not positions:
        positions = [0]
    sig = np.array(positions, dtype=float)
    sig = sig - sig.mean()
    f = np.fft.fft(sig)
    mag = np.abs(f)[:n_bins]
    phase = np.angle(f)[:n_bins]
    return np.log1p(_pad_to(mag, n_bins)), _pad_to(phase, n_bins)


def _all_log_magnitudes(words: list[str], n_bins: int = 32) -> np.ndarray:
    """Compute log-magnitude DFT for each word."""
    specs = []
    for w in words:
        mag, _ = dft_spectrum(w, n_bins)
        specs.append(mag)
    return np.array(specs)


def _load_words_by_label() -> tuple[list[str], list[str]]:
    """Return (positive_words, negative_words) — flat word lists."""
    df = load_articles_binary()
    pos_words, neg_words = [], []
    for _, row in df.iterrows():
        words = row["words"]
        (pos_words if row["label"] == 1 else neg_words).extend(words)
    return pos_words, neg_words


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — DFT spectral heatmap
# ══════════════════════════════════════════════════════════════════════════════

def make_spectral_heatmap(pos_words, neg_words, n_bins=32):
    pos_specs = _all_log_magnitudes(pos_words, n_bins)
    neg_specs = _all_log_magnitudes(neg_words, n_bins)

    pos_mean = pos_specs.mean(axis=0)
    neg_mean = neg_specs.mean(axis=0)

    matrix = np.stack([pos_mean, neg_mean])   # shape (2, n_bins)

    fig, ax = plt.subplots(figsize=(12, 2.8))
    cmap = plt.cm.RdYlBu_r
    vmax = max(abs(matrix).max(), 0.01)
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, interpolation="nearest",
                   vmin=-vmax, vmax=vmax)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["positive", "negative"], fontsize=11)
    ax.set_xlabel("Frequency bin  (0 = DC, 1 = fundamental, 2– = harmonics)", fontsize=10)
    ax.set_title(
        "Average log-magnitude DFT spectrum of word letter-position sequences\n"
        "(positive vs negative, FinancialPhraseBank binary)",
        fontsize=11, pad=8,
    )
    cbar = plt.colorbar(im, ax=ax, orientation="vertical", pad=0.02, aspect=30)
    cbar.set_label("Log(1 + |FFT|)", fontsize=9)
    ax.axhline(0.5, color="white", linewidth=0.8, linestyle="--", alpha=0.7)
    for x in [0, 1, 2, 3]:
        ax.axvline(x - 0.5, color="white", linewidth=0.3, linestyle=":", alpha=0.5)
    ax.set_xticks([0, 4, 8, 12, 16, 20, 24, 28, 31])
    ax.set_xticklabels(["0\n(DC)", "4", "8", "12", "16", "20", "24", "28", "31"])
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — DFT word fingerprints
# ══════════════════════════════════════════════════════════════════════════════

def make_word_fingerprints(pos_words, neg_words, n_bins=32, n_examples=8):
    def pick(words, seen=None):
        if seen is None:
            seen = set()
        out = []
        for w in words:
            if 5 <= len(w) <= 8 and w not in seen:
                seen.add(w); out.append(w)
                if len(out) == n_examples:
                    break
        return out, seen

    seen = None
    pos_sample, seen = pick(pos_words, seen)
    neg_sample, _   = pick(neg_words, seen)

    n = len(pos_sample)
    fig, axes = plt.subplots(2, n, figsize=(3.5 * n, 5.5), sharey=False)
    if n == 1:
        axes = axes.reshape(2, 1)

    x = np.arange(n_bins)
    pos_color = PALETTE_PRIMARY
    neg_color = PALETTE_HIGHLIGHT

    for ax, word in zip(axes[0], pos_sample):
        mag, _ = dft_spectrum(word, n_bins)
        ax.bar(x, mag, color=pos_color, width=0.8, alpha=0.85, linewidth=0)
        ax.set_title(f"+  {word}", fontsize=10, color=pos_color, fontweight="bold")
        ax.set_xticks([]); ax.spines[["top","right"]].set_visible(False)
        ax.set_xlim(-0.5, n_bins - 0.5)
        ax.set_ylabel("Log magnitude", fontsize=8)
        ax.tick_params(labelsize=7)

    for ax, word in zip(axes[1], neg_sample):
        mag, _ = dft_spectrum(word, n_bins)
        ax.bar(x, mag, color=neg_color, width=0.8, alpha=0.85, linewidth=0)
        ax.set_title(f"−  {word}", fontsize=10, color=neg_color, fontweight="bold")
        ax.set_xticks([]); ax.spines[["top","right"]].set_visible(False)
        ax.set_xlim(-0.5, n_bins - 0.5)
        ax.set_ylabel("Log magnitude", fontsize=8)
        ax.tick_params(labelsize=7)

    fig.suptitle(
        "DFT spectral fingerprints: individual word letter sequences\n"
        "(blue = positive sentiment, red = negative sentiment)",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SHAP helpers
# ══════════════════════════════════════════════════════════════════════════════

def _train_model_and_get_X(n_samples=400, random_state=42):
    """Train RF on FPB binary, return (model, X_test, feature_names)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    df = load_articles_binary()
    articles = df["words"].tolist()
    y = df["label"].values
    feature_names = get_feature_names()

    # Build the aggregation pipeline inline (no FeaturePipeline import needed)
    def _aggregate(words, fnames, strategy="all"):
        wfs = np.array([list(feature_one_word(w).values()) for w in words], dtype=float)
        vecs = []
        if strategy in ("mean", "all"):
            vecs.append(wfs.mean(axis=0))
        if strategy in ("max", "all"):
            vecs.append(wfs.max(axis=0))
        if strategy in ("min", "all"):
            vecs.append(wfs.min(axis=0))
        if strategy in ("std", "all"):
            vecs.append(wfs.std(axis=0))
        return np.concatenate(vecs)

    X_all = np.array([_aggregate(a, feature_names) for a in articles])
    _, X_test, _, _ = train_test_split(
        X_all, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # Use a small subsample for SHAP (faster)
    idx = np.random.RandomState(random_state).choice(len(X_test), min(n_samples, len(X_test)), replace=False)
    X_sample = X_test[idx]

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_all, y)
    return model, X_sample, feature_names


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — SHAP beeswarm (Feature importance landscape)
# ══════════════════════════════════════════════════════════════════════════════

def make_shap_beeswarm(shap_values, X_sample, feature_names, n_top=40):
    """Beeswarm plot: distribution of SHAP values per feature."""
    import shap

    # Shorten long feature names for readability
    short = {fn: fn[:28] for fn in feature_names}

    plt.figure()
    shap.plots.beeswarm(
        shap.Explanation(
            values=shap_values[:, :len(feature_names)],
            base_values=np.zeros(len(shap_values)),
            data=X_sample[:, :len(feature_names)],
            feature_names=[short[f] for f in feature_names],
        ),
        max_display=n_top,
        show=False,
        plot_size=(12, 10),
    )
    plt.title("SHAP beeswarm: feature attribution landscape\n"
              "(top 40 features by mean |SHAP|, random forest)",
              fontsize=11, pad=8)
    plt.tight_layout()
    fig = plt.gcf()
    plt.close()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — SHAP bar (mean |SHAP|)
# ══════════════════════════════════════════════════════════════════════════════

def make_shap_bar(shap_values, feature_names, n_top=30):
    """Bar plot: mean absolute SHAP value per feature."""
    import shap

    short = {fn: fn[:32] for fn in feature_names}
    mean_abs = np.abs(shap_values[:, :len(feature_names)]).mean(axis=0)

    fig, ax = plt.subplots(figsize=(10, max(6, n_top * 0.35)))
    idx = np.argsort(mean_abs)[::-1][:n_top]
    names = [short[feature_names[i]] for i in idx]
    vals = mean_abs[idx]

    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(vals)))
    ax.barh(range(len(vals)), vals, color=colors, height=0.7, linewidth=0)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value|  (higher = more important)", fontsize=9)
    ax.set_title("SHAP feature importance (mean |SHAP|, top 30)\nRandom Forest on 68 letter-derived features",
                  fontsize=11, pad=8)
    ax.spines[["top","right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — SHAP waterfall (one positive, one negative example)
# ══════════════════════════════════════════════════════════════════════════════

def make_shap_waterfall(shap_vals, X_sample, feature_names, explainer,
                        pos_idx=0, neg_idx=None, max_display=12):
    """
    Waterfall plots for one positive and one negative prediction.
    Shows how each feature pushes the prediction toward or away from positive.
    Pass the same TreeExplainer used in main() so we get the correct expected_value.

    Drawn manually with matplotlib instead of ``shap.plots.waterfall``: SHAP's
    internal layout places "value = feature" text over its own y-tick labels on
    narrow axes, which cannot be repositioned from outside.
    """
    short = {fn: fn[:28] for fn in feature_names}
    n_feats = len(feature_names)

    if neg_idx is None:
        neg_idx = pos_idx + 10

    # Extract expected_value (may be a 2-element array for binary; pick class-1)
    ev = explainer.expected_value
    if isinstance(ev, (list, np.ndarray)):
        ev = float(ev[1]) if len(ev) > 1 else float(ev[0])
    else:
        ev = float(ev)

    fig, axes = plt.subplots(1, 2, figsize=(22, 6))

    def _plot(idx, ax, label):
        sv = shap_vals[idx, :n_feats].astype(float)
        dr = X_sample[idx, :n_feats]
        order = np.argsort(-np.abs(sv))
        display = list(order[:max_display])
        other_sv = 0.0
        if len(order) > max_display:
            other_sv = float(sv[order[max_display:]].sum())
            display.append(-1)
        rows = np.arange(len(display))
        pos = float(ev)
        y = len(display) - 1 - rows  # top-down order
        for i, fi in enumerate(display):
            if fi == -1:
                val = other_sv
                name = f"{len(order) - max_display} other features"
            else:
                val = float(sv[fi])
                fn = short[feature_names[fi]]
                dv = float(dr[fi]) if not np.isnan(dr[fi]) else np.nan
                name = f"{fn} = {dv:g}" if not np.isnan(dv) else fn
            color = "#B04040" if val >= 0 else "#4A6FA5"
            ypos = y[i]
            ax.annotate(
                "", xy=(pos + val, ypos), xytext=(pos, ypos),
                arrowprops=dict(arrowstyle="-|>", lw=1.8, color=color),
                annotation_clip=False,
            )
            ax.text(pos + val / 2, ypos + 0.28, name, ha="center", va="bottom",
                    fontsize=8, color="#222222")
            pos += val
        ax.axvline(ev, color="#666666", ls="--", lw=0.9)
        xmin = min(ev, pos) - 0.5
        xmax = max(ev, pos) + 0.5
        ax.text(xmin + 0.02, -0.55, f"base {ev:.2f}", ha="left", va="top",
                fontsize=7.5, color="#666666")
        ax.text(xmax - 0.02, -0.55, f"f(x) = {pos:.2f}", ha="right", va="top",
                fontsize=9, fontweight="bold")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(-0.8, len(display) + 0.4)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_title(f"Predicted: {label.upper()}", fontsize=11, fontweight="bold")

    _plot(pos_idx, axes[0], "positive")
    _plot(neg_idx, axes[1], "negative")

    fig.suptitle("SHAP waterfall: per-prediction feature breakdown\n"
                 "(positive vs negative FinancialPhraseBank sentence)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 70)
    print("VISUALISATIONS: DFT spectral + SHAP")
    print("=" * 70)

    # ── Load words ──────────────────────────────────────────────────────────
    print("\n[1/5] Loading word data...")
    pos_words, neg_words = _load_words_by_label()
    print(f"  positive words: {len(pos_words):,}")
    print(f"  negative words: {len(neg_words):,}")

    # ── Figure 1: Spectral heatmap ──────────────────────────────────────────
    print("\n[2/5] Building spectral heatmap...")
    fig1 = make_spectral_heatmap(pos_words, neg_words)
    out1 = OUT_DIR / "dft_spectral_heatmap.png"
    fig1.savefig(out1, dpi=300, bbox_inches="tight")
    print(f"  → {out1.name}  ({out1.stat().st_size / 1024:.0f} KB)")

    # ── Figure 2: Word fingerprints ─────────────────────────────────────────
    print("\n[3/5] Building word fingerprints...")
    fig2 = make_word_fingerprints(pos_words, neg_words)
    out2 = OUT_DIR / "dft_word_fingerprints.png"
    fig2.savefig(out2, dpi=300, bbox_inches="tight")
    print(f"  → {out2.name}  ({out2.stat().st_size / 1024:.0f} KB)")

    # ── Train model for SHAP ────────────────────────────────────────────────
    print("\n[4/5] Training model for SHAP (n_estimators=100, 20% held-out)...")
    model, X_sample, feature_names = _train_model_and_get_X(n_samples=400)
    print(f"  X_sample shape: {X_sample.shape}")

    print("  Computing SHAP values (TreeExplainer, exact, 400 samples)...")
    import shap
    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(X_sample)

    # shap returns different shapes depending on version:
    #   list of 2 arrays  -> [shap_class0, shap_class1], each (n_samples, n_features)
    #   2D array          -> (n_samples, n_features)      [single class]
    #   3D array         -> (n_samples, n_features, 2)     [sklearn >= 1.4 with dense]
    # We always want class-1 (positive) values as a 2D array (n_samples, n_feats).
    if isinstance(raw, list):
        shap_vals = np.array(raw[1])          # list[array] -> pick positive class
    elif raw.ndim == 3:
        shap_vals = raw[:, :, 1]               # (N, F, 2) -> (N, F)
    else:
        shap_vals = np.array(raw)              # already 2D

    print(f"  SHAP values: raw shape={getattr(raw, 'shape', len(raw))}, "
          f"using class-1 shape={shap_vals.shape}")

    # ── Figure 3: Beeswarm ─────────────────────────────────────────────────
    print("\n[5/5] Building SHAP plots...")
    print("  - beeswarm (feature attribution landscape)...")
    fig3 = make_shap_beeswarm(shap_vals, X_sample, feature_names, n_top=40)
    out3 = OUT_DIR / "shap_beeswarm.png"
    fig3.savefig(out3, dpi=300, bbox_inches="tight")
    print(f"  → {out3.name}  ({out3.stat().st_size / 1024:.0f} KB)")

    # ── Figure 4: Bar ─────────────────────────────────────────────────────
    print("  - bar (mean |SHAP|)...")
    fig4 = make_shap_bar(shap_vals, feature_names, n_top=30)
    out4 = OUT_DIR / "shap_bar.png"
    fig4.savefig(out4, dpi=300, bbox_inches="tight")
    print(f"  → {out4.name}  ({out4.stat().st_size / 1024:.0f} KB)")

    # ── Figure 5: Waterfall ────────────────────────────────────────────────
    print("  - waterfall (one positive, one negative example)...")
    fig5 = make_shap_waterfall(shap_vals, X_sample, feature_names,
                                explainer=explainer,
                                pos_idx=0, neg_idx=None)
    out5 = OUT_DIR / "shap_waterfall.png"
    fig5.savefig(out5, dpi=300, bbox_inches="tight")
    print(f"  → {out5.name}  ({out5.stat().st_size / 1024:.0f} KB)")

    print("\n" + "=" * 70)
    print("DONE. All 5 figures saved to figures/")
    print("=" * 70)


if __name__ == "__main__":
    main()
