"""Cross-domain generalisation benchmark for the 2-tier sentiment cascade.

Run from the research repo root with the dashboard workspace venv:
    cd /home/a/letter-valence-research
    /home/a/esg-dashboard/.venv/bin/python -m src.benchmark_general

Dataset
-------
NewsMTSC (Hamborg et al., EACL 2021): 5-coder-labelled 3-class sentence
sentiment from real-world general news (AllSides: politics, tech, world, ...).
The recommended real-world held-out split ``devtest_rw.jsonl`` (n = 1,067) is
used for evaluation; ``train.jsonl`` (n = 7,758) is used only for the in-domain
cheap-tier reference model. The data are NOT financial.

Design
------
* Clear-polarity set (n = 651 pos/neg devtest_rw sentences): every method
  {cheap_fpb, cheap_news, cascade_fpb, cascade_news, heavy, vader, keyword}
  evaluated on identical held-out sentences.
    - ``cheap_fpb``  — word-level cheap tier (TF-IDF 1-2 grams + VADER + keyword
      features, 3-class logistic regression) trained ONLY on the 4,846
      FinancialPhraseBank sentences (cross-domain transfer, zero general-news
      supervision).
    - ``cheap_news`` — identical architecture retrained on the 7,758 NewsMTSC
      train sentences (in-domain upper reference).
    - ``cascade_fpb`` / ``cascade_news`` — the 2-tier cascade routing each cheap
      tier to the fixed heavy (FinancialBERT) / VADER fallback.
    - ``heavy`` = ahmedrachid/FinancialBERT-Sentiment-Analysis (fixed,
      finance-tuned — a conservative domain-mismatch test on general news).
* Borderline set (n = 416 held-out neutral devtest_rw sentences): false-polarity
  rate and predicted-class distribution for every method.
* Metrics: accuracy + Wilson 95% CI, macro-F1, per-class precision/recall/F1,
  exact McNemar significance (Bonferroni-corrected), cascade tier routing,
  threshold sweep re-routed cheaply on stored per-instance valences.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

DASHBOARD_SRC = Path(os.environ.get("ESG_DASHBOARD_SRC", "/home/a/esg-dashboard/src"))
sys.path.insert(0, str(DASHBOARD_SRC))

from esg_dashboard.data.sentiment_engine import (  # noqa: E402
    LABEL_BAND,
    heavy_score_batch,
    keyword_valence,
    vader_valence,
)

from src.benchmark_cascade import (  # noqa: E402
    binary_metrics,
    cascade_predict,
    load_fpb,
    mcnemar_exact,
    predict_from_v,
    wilson_ci,
)

RESULTS_DIR = REPO / "results"
FIGURES_DIR = REPO / "figures"

CHEAP_THRESHOLD = 0.6  # cascade routing threshold (matches FPB benchmark / engine)
METHODS = ("cheap_fpb", "cheap_news", "cascade_fpb", "cascade_news", "heavy", "vader", "keyword")

NEWSMTS = REPO / "data" / "newsmtsc"
NEWSMTS_TRAIN = NEWSMTS / "train.jsonl"
NEWSMTS_DEV = NEWSMTS / "devtest_rw.jsonl"


def load_newsmtsc(path: Path) -> pd.DataFrame:
    rows = []
    with open(path) as fh:
        for i, line in enumerate(fh):
            row = json.loads(line)
            text = row.get("sentence_normalized", "").strip()
            if not text:
                continue
            pol = None
            for t in row.get("targets", []):
                if t.get("Input.gid") == row.get("primary_gid"):
                    pol = t.get("polarity")
                    break
            if pol is None:
                continue
            rows.append({"id": i, "text": text, "polarity": float(pol)})
    df = pd.DataFrame(rows)
    df["label"] = df["polarity"].map({2.0: "negative", 4.0: "neutral", 6.0: "positive"})
    df["y"] = df["label"].map({"negative": 0, "neutral": 1, "positive": 2})
    return df


def train_cheap_tier(texts, y3, extra):
    """Fit the word-level cheap tier on an in-domain training set.

    Returns a callable ``(texts, extra) -> valence array`` mirroring the FPB
    benchmark's per-fold estimator (same vectorizer / LR config).
    """
    from scipy import sparse
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    vec = TfidfVectorizer(
        lowercase=True, token_pattern=r"[a-z]+",
        ngram_range=(1, 2), min_df=2, max_features=20_000,
    )
    x = vec.fit_transform([t for t in texts])
    extra = np.asarray(extra)
    x = sparse.hstack([x, sparse.csr_matrix(extra)]).tocsr()
    lr = LogisticRegression(C=1.0, max_iter=2000, random_state=42, class_weight="balanced")
    lr.fit(x, y3)
    classes = list(lr.classes_)
    pos_col = classes.index(2)
    neg_col = classes.index(0)

    def predict(new_texts, new_extra):
        x_new = vec.transform([t for t in new_texts])
        new_extra = np.asarray(new_extra)
        x_new = sparse.hstack([x_new, sparse.csr_matrix(new_extra)]).tocsr()
        proba = lr.predict_proba(x_new)
        return proba[:, pos_col] - proba[:, neg_col]

    return predict


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    print("Loading NewsMTSC (general news, real-world held-out split)...")
    news = load_newsmtsc(NEWSMTS_DEV)
    clear = news[news["label"] != "neutral"].reset_index(drop=True)
    neut = news[news["label"] == "neutral"].reset_index(drop=True)
    y_true = clear["y"].to_numpy()
    print(f"  devtest_rw n={len(news)} (pos={(news['y'] == 2).sum()}, "
          f"neg={(news['y'] == 0).sum()}, neutral={(news['y'] == 1).sum()})")
    print(f"  clear n={len(clear)} (pos={(y_true == 2).sum()}, neg={(y_true == 0).sum()})")
    print(f"  neutral n={len(neut)}")

    print("Loading FinancialPhraseBank for cross-domain cheap tier...")
    fpb = load_fpb()
    fpb_texts = fpb["text"].tolist()
    fpb_y = fpb["y"].to_numpy()
    fpb_extra = list(zip(
        [vader_valence(t) for t in fpb_texts],
        [keyword_valence(t) for t in fpb_texts],
    ))

    print("Loading NewsMTSC train for in-domain cheap tier...")
    news_train = load_newsmtsc(NEWSMTS_TRAIN)
    print(f"  train n={len(news_train)} (pos={(news_train['y'] == 2).sum()}, "
          f"neg={(news_train['y'] == 0).sum()}, neutral={(news_train['y'] == 1).sum()})")
    train_texts = news_train["text"].tolist()
    train_y = news_train["y"].to_numpy()
    train_extra = list(zip(
        [vader_valence(t) for t in train_texts],
        [keyword_valence(t) for t in train_texts],
    ))

    print("Training cheap tier variants...")
    cheap_fpb = train_cheap_tier(fpb_texts, fpb_y, fpb_extra)
    cheap_news = train_cheap_tier(train_texts, train_y, train_extra)

    print("Scoring devtest_rw with VADER + keyword baselines...")
    dev_texts = news["text"].tolist()
    vader_v_dev = np.array([vader_valence(t) for t in dev_texts])
    kw_v_dev = np.array([keyword_valence(t) for t in dev_texts])
    dev_extra = list(zip(vader_v_dev, kw_v_dev))

    print("Scoring heavy tier (FinancialBERT) on all 1,067 dev sentences...")
    heavy_proba_dev = heavy_score_batch(dev_texts)
    heavy_v_dev = np.array([p["positive"] - p["negative"] for p in heavy_proba_dev])
    print("  done.")

    cheap_fpb_v = cheap_fpb(dev_texts, dev_extra)
    cheap_news_v = cheap_news(dev_texts, dev_extra)

    clear_mask = (news["label"] != "neutral").to_numpy()
    y_code = y_true
    method_v = {
        "cheap_fpb": cheap_fpb_v[clear_mask],
        "cheap_news": cheap_news_v[clear_mask],
        "heavy": heavy_v_dev[clear_mask],
        "vader": vader_v_dev[clear_mask],
        "keyword": kw_v_dev[clear_mask],
    }

    print("Evaluating methods on the clear set (n=%d)..." % len(clear))
    method_labels = {m: predict_from_v(method_v[m]) for m in ("cheap_fpb", "cheap_news", "heavy", "vader", "keyword")}
    cascade_fpb_code, cascade_fpb_tiers = cascade_predict(
        cheap_fpb_v[clear_mask], heavy_v_dev[clear_mask], vader_v_dev[clear_mask],
        np.asarray([len(t.split()) for t in dev_texts])[clear_mask],
        threshold=CHEAP_THRESHOLD, band=LABEL_BAND,
    )
    cascade_news_code, cascade_news_tiers = cascade_predict(
        cheap_news_v[clear_mask], heavy_v_dev[clear_mask], vader_v_dev[clear_mask],
        np.asarray([len(t.split()) for t in dev_texts])[clear_mask],
        threshold=CHEAP_THRESHOLD, band=LABEL_BAND,
    )
    method_labels["cascade_fpb"] = cascade_fpb_code
    method_labels["cascade_news"] = cascade_news_code

    metric_sets = {}
    for m in METHODS:
        metric_sets[m] = binary_metrics(y_code, method_labels[m])

    print("  accuracy / neg-recall / macro-F1:")
    for m in METHODS:
        s = metric_sets[m]
        print(f"    {m:>13} acc={s['accuracy']:.4f} "
              f"neg_rec={s['negative_recall']:.4f} macroF1={s['macro_f1']:.4f}")

    mcnemar = {}
    for cascade_key in ("cascade_fpb", "cascade_news"):
        cc = method_labels[cascade_key]
        mcnemar[cascade_key] = {}
        for m in METHODS:
            if m == cascade_key:
                continue
            a = cc != y_code
            b = method_labels[m] != y_code
            both_wrong = (a & b).sum()
            casc_only = (a & ~b).sum()
            other_only = (~a & b).sum()
            p = mcnemar_exact(int(casc_only), int(other_only))
            mcnemar[cascade_key][m] = {
                "both_wrong": int(both_wrong),
                "cascade_wrong_only": int(casc_only),
                "other_wrong_only": int(other_only),
                "p_exact": round(p, 6),
                "significant_0.01": p < 0.01,
            }

    tier_routing = {}
    for cascade_key, tiers in (("cascade_fpb", cascade_fpb_tiers), ("cascade_news", cascade_news_tiers)):
        cc = method_labels[cascade_key]
        tiers_arr = np.asarray(tiers)
        tr = {}
        for t, c in zip(*np.unique(tiers_arr, return_counts=True)):
            idx = tiers_arr == t
            tr[str(t)] = {
                "n": int(c),
                "share": round(c / len(clear), 4),
                "accuracy": round(float((cc[idx] == y_code[idx]).mean()), 4),
            }
        tier_routing[cascade_key] = tr

    print("Threshold sweep (re-routed on stored valences)...")
    sweep = []
    for cascade_key, cheap_v in (("cascade_fpb", cheap_fpb_v), ("cascade_news", cheap_news_v)):
        for ct in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
            for band in (0.05, 0.1, 0.15):
                code, tiers_s = cascade_predict(
                    cheap_v[clear_mask], heavy_v_dev[clear_mask], vader_v_dev[clear_mask],
                    np.asarray([len(t.split()) for t in dev_texts])[clear_mask],
                    threshold=ct, band=band,
                )
                acc = float((code == y_code).mean())
                heavy_share = float(np.mean([t == "heavy" for t in tiers_s]))
                sweep.append({
                    "cascade": cascade_key, "cheap_threshold": ct, "band": band,
                    "accuracy": round(acc, 4), "heavy_share": round(heavy_share, 4),
                })
    sweep.sort(key=lambda e: e["accuracy"], reverse=True)

    print("Scoring borderline (neutral) set with deployed components...")
    neut_texts = neut["text"].tolist()
    neut_vader = np.array([vader_valence(t) for t in neut_texts])
    neut_kw = np.array([keyword_valence(t) for t in neut_texts])
    neut_extra = list(zip(neut_vader, neut_kw))
    neut_cheap_fpb = cheap_fpb(neut_texts, neut_extra)
    neut_cheap_news = cheap_news(neut_texts, neut_extra)
    neut_mask = (news["label"] == "neutral").to_numpy()
    neut_heavy = heavy_v_dev[neut_mask]

    neut_method_labels = {
        "cheap_fpb": predict_from_v(neut_cheap_fpb),
        "cheap_news": predict_from_v(neut_cheap_news),
        "heavy": predict_from_v(neut_heavy),
        "vader": predict_from_v(neut_vader),
        "keyword": predict_from_v(neut_kw),
    }
    neut_fpb_code, neut_fpb_tiers = cascade_predict(
        neut_cheap_fpb, neut_heavy, neut_vader, [len(t.split()) for t in neut_texts],
        threshold=CHEAP_THRESHOLD, band=LABEL_BAND,
    )
    neut_news_code, neut_news_tiers = cascade_predict(
        neut_cheap_news, neut_heavy, neut_vader, [len(t.split()) for t in neut_texts],
        threshold=CHEAP_THRESHOLD, band=LABEL_BAND,
    )
    neut_method_labels["cascade_fpb"] = neut_fpb_code
    neut_method_labels["cascade_news"] = neut_news_code

    borderline = {}
    for m in METHODS:
        pred = neut_method_labels[m]
        n = len(pred)
        false_pol = float(((pred == 0) | (pred == 2)).mean())
        borderline[m] = {
            "n": int(n),
            "pred_negative": float((pred == 0).mean()),
            "pred_neutral": float((pred == 1).mean()),
            "pred_positive": float((pred == 2).mean()),
            "false_polarity_rate": round(false_pol, 4),
        }

    # ---- save ----
    report = {
        "config": {
            "dataset": "NewsMTSC (Hamborg et al., EACL 2021), devtest_rw real-world split",
            "cheap_threshold": CHEAP_THRESHOLD,
            "label_band": LABEL_BAND,
            "model": "ahmedrachid/FinancialBERT-Sentiment-Analysis",
            "clear_n": len(clear),
            "neutral_n": len(neut),
            "cheap_fpb_train": "FinancialPhraseBank (n=4846, cross-domain transfer)",
            "cheap_news_train": "NewsMTSC train (n=%d, in-domain reference)" % len(news_train),
        },
        "clear_set": metric_sets,
        "mcnemar_vs_cascade": mcnemar,
        "tier_routing_clear": tier_routing,
        "threshold_sweep_top": sweep[:8],
        "threshold_sweep_best": max(sweep, key=lambda e: e["accuracy"]),
        "borderline_set": borderline,
    }
    with open(RESULTS_DIR / "general_news_benchmark.json", "w") as fh:
        json.dump(report, fh, indent=2)

    pred_rows = []
    for i in range(len(clear)):
        pred_rows.append({
            "set": "clear", "id": int(clear["id"].iloc[i]),
            "true": int(y_code[i]),
            "cheap_fpb_v": round(float(cheap_fpb_v[i]), 6),
            "cheap_news_v": round(float(cheap_news_v[i]), 6),
            "heavy_v": round(float(heavy_v_dev[i]), 6),
            "vader_v": round(float(vader_v_dev[i]), 6),
            "kw_v": round(float(kw_v_dev[i]), 6),
            "cascade_fpb_label": int(cascade_fpb_code[i]), "cascade_fpb_tier": cascade_fpb_tiers[i],
            "cascade_news_label": int(cascade_news_code[i]), "cascade_news_tier": cascade_news_tiers[i],
        })
    for i in range(len(neut)):
        pred_rows.append({
            "set": "neutral", "id": int(neut["id"].iloc[i]), "true": 1,
            "cheap_fpb_v": round(float(neut_cheap_fpb[i]), 6),
            "cheap_news_v": round(float(neut_cheap_news[i]), 6),
            "heavy_v": round(float(neut_heavy[i]), 6),
            "vader_v": round(float(neut_vader[i]), 6),
            "kw_v": round(float(neut_kw[i]), 6),
            "cascade_fpb_label": int(neut_fpb_code[i]), "cascade_fpb_tier": neut_fpb_tiers[i],
            "cascade_news_label": int(neut_news_code[i]), "cascade_news_tier": neut_news_tiers[i],
        })
    pd.DataFrame(pred_rows).to_csv(RESULTS_DIR / "general_news_predictions.csv", index=False)

    print()
    print("Saved results/general_news_benchmark.json and results/general_news_predictions.csv")
    print("Best sweep config:", report["threshold_sweep_best"])


if __name__ == "__main__":
    main()
