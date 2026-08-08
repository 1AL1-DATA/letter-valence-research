# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and this project adheres to
[Semantic Versioning](https://semver.org/).

## [1.2.0] - 2026-08-07

### Added
- **`src/benchmark_general.py`** — cross-domain generalisation benchmark: the same
  2-tier cascade (cheap word tier → heavy transformer) evaluated on **NewsMTSC**
  (EACL 2021) general-news sentiment, held-out `devtest_rw` split (n = 1,067).
  - Cheap tier trained only on FinancialPhraseBank (cross-domain transfer) AND
    retrained on NewsMTSC train (in-domain reference).
  - Two fixed heavy tiers: FinancialBERT (finance-tuned) and a general-domain
    transformer (`cardiffnlp/twitter-roberta-base-sentiment-latest`).
  - Same metrics as the FPB benchmark: Wilson CI, macro-F1, exact McNemar,
    tier routing, threshold sweep, borderline false-polarity rates.
- **`src/figures_general.py`** — 4-panel general-news figure
  (`figures/general_news_eval.png`).
- **`results/general_news_benchmark.json`** + **`results/general_news_predictions.csv`**.
- **`data/newsmtsc/`** — NewsMTSC train + devtest_rw JSONL with the dataset readme.
- Updated README (TL;DR, tree, reproduction steps, "Does the cascade generalise
  beyond finance?" section, citation), `results/SUMMARY.md`, `METHODOLOGY.md`,
  `blog_post.md`, `lit_digest.md`, `linkedin_post.md`, `research_report.md`,
  `docs/architecture.md`, `data/README.md`, `CHANGELOG.md`, `LICENSE`
  (NewsMTSC attribution).

### Findings (cross-domain)
- The cheap word tier transfers out of finance: trained only on FPB it beats the
  finance-tuned FinancialBERT on general news (0.4209 vs 0.3041).
- The finance-tuned heavy is the domain-locked part: FinancialBERT lands *below
  chance* on general news (0.3041; predicts neutral on 65.3% of clear sentences).
- With a domain-appropriate heavy the cascade again beats heavy-only
  (0.6190 vs 0.5760) while the cheap tier absorbs ~24% of calls at 91.8% accuracy.
- The cascade *approach* is a general feature; the heavy model must match the domain.

## [1.2.1] - 2026-08-08

### Fixed
- **Borderline false-polarity numbers corrected** in `README.md` and
  `results/SUMMARY.md` (general-news section): the `_gen` variants were reported
  with stale values. Corrected to the values in
  `results/general_news_benchmark.json`: cascade (news cheap → gen heavy) 18.8% →
  **26.4%**, cascade (FPB cheap → gen heavy) 22.4% → **27.4%**, general BERT
  18.5% → **23.3%**.
- **`heavy_fin` neutral rate on the clear set corrected** from 68.7% → **65.3%**
  in README, SUMMARY, `research_report.md`, `METHODOLOGY.md`, `CHANGELOG.md`,
  `linkedin_post.md` (matches `neutral_predicted` in the JSON).
- **CSV misalignment bug in `src/benchmark_general.py`**: the clear-set `_v`
  columns indexed the full dev-length valence arrays without applying
  `clear_mask`, so those columns were shifted whenever neutral rows are
  interspersed. Cascade labels and neutral-set columns were unaffected. Fixed,
  re-ran the benchmark (JSON byte-identical), regenerated
  `results/general_news_predictions.csv` and `figures/general_news_eval.png`
  (its misclassification panel reads those columns).
- **`[your-org]` placeholders replaced** with the real
  `1AL1-DATA/letter-valence-research` URL in `CHANGELOG.md`, `research_report.md`,
  `arxiv_paper.tex`.

### Added
- **`arxiv_paper.tex` / `arxiv_paper.pdf` updated** with a new
  "Application: a two-tier sentiment cascade" section: the retired 3-tier design,
  the word-level cheap tier, FPB results (0.9512 vs 0.9558, McNemar p = 0.15;
  cheap tier 36.6% @ 97.2%), the cross-domain NewsMTSC evaluation
  (clear-polarity table, borderline false-polarity table, McNemar p = 10⁻⁶,
  cheap tier 24.3% @ 91.8%), and full reproduction commands. Abstract updated;
  NewsMTSC citation added.
- **`CITATION.cff`** bumped to v1.2.0 (2026-08-07) with the NewsMTSC reference.

## [1.1.0] - 2026-08-06

### Added
- **`models/letter_sentiment_rf.pkl`** — trained Random Forest saved to disk
  by `python -m src.train_final` (3.7 MB).
- **`src/train_final.py`** — trains the model on all 1,967 FPB binary sentences
  and saves it. Also runs a held-out 20% test for an honest accuracy number.
- **`src/classify.py`** — paragraph classifier with two interfaces:
  - CLI: `python -m src.classify --text "..." --compare` (shows letter + VADER scores)
  - Library: `from src.classify import classify, classify_batch`
- **`src/visualise.py`** — dedicated DFT and SHAP visualisation script:
  - `dft_spectral_heatmap.png` — average log-magnitude DFT spectrum, positive vs negative
  - `dft_word_fingerprints.png` — individual word DFT spectra, positive vs negative
  - `shap_beeswarm.png` — SHAP feature attribution landscape (top 40 features)
  - `shap_bar.png` — SHAP mean |SHAP| per feature (top 30)
  - `shap_waterfall.png` — SHAP waterfall: one positive + one negative prediction
- **`test_paragraphs.py`** — ad-hoc test of 10 hand-written financial sentences,
  comparing letter classifier vs VADER.
- Updated README with: held-out 20% test result (0.751), SHAP section with
  interpretation, paragraph classifier usage section, SHAP figures listed.

### Changed
- `linkedin_post.md` — rewritten to measured, colleague-to-colleague tone.
  Practical use case (fast pre-filter for large-scale document ingestion,
  compute cost reduction) added as a concrete application note.
- `CHANGELOG.md` — re-tagged v1.0.0 as past work; v1.1.0 now tracks current state.

## [1.0.0] - 2026-08-05

### Added
- 68 letter-derived features in 12 families (`src/features.py`).
- Permutation test, learning curve, family ablation, single-family evaluation (`src/evaluate.py`).
- Random Forest, Logistic Regression, and Ridge classifier with cross-validation (`src/train.py`).
- End-to-end pipeline (`src/analyze.py`).
- Unit tests: 33 passing, ~1.7s runtime (`tests/test_features.py`).
- Data download script with provenance documentation (`data/download.sh`).
- Structured literature digest of 5 foundational papers (`lit_digest.md`).
- Full research report (`research_report.md`).
- Blog post draft for general audience (`blog_post.md`).
- LinkedIn short-form post (`linkedin_post.md`).
- arXiv preprint (LaTeX, NeurIPS-style template) (`arxiv_paper.tex`, `arxiv_paper.pdf`).
- Architecture documentation (`docs/architecture.md`).
- Jupyter notebook walkthrough (`notebooks/01_reproduce_main_result.ipynb`).
- Machine-readable summary, timing, and ablation results (`results/summary.json`).
- 8 PNG figures at 300 dpi (`figures/`).
- `TEMPLATE.md` — generic research-artifact structure for future projects.

### Findings
- Random Forest on 68 letter features reaches 0.7377 ± 0.0058 accuracy (F1 = 0.835)
  on FPB binary sentiment, p < 0.0001 vs permutation null (50 perms).
- Spectral (DFT) features are the dominant family; gematria-style modular
  arithmetic contributes marginally.
- Learning curve plateaus around n=1,376, indicating a feature-bound regime.

### Limitations documented
- Validated on financial text only (FinancialPhraseBank binary).
- Does not beat a properly-tuned VADER (0.750) or FinBERT (~0.87) on the same data.
- Gematria hypothesis (mod-9, mod-26, primality) finds no support after Bonferroni.

## [Unreleased] - future work

- Comparison with a character-level CNN baseline.
- Hierarchical classification for long documents: split into sentences, aggregate predictions.
- Ablation on aggregation strategy: mean/max/min/std vs learned attention.

[1.2.1]: https://github.com/1AL1-DATA/letter-valence-research/releases/tag/v1.2.1
[1.2.0]: https://github.com/1AL1-DATA/letter-valence-research/releases/tag/v1.2.0
[1.1.0]: https://github.com/1AL1-DATA/letter-valence-research/releases/tag/v1.1.0
[1.0.0]: https://github.com/1AL1-DATA/letter-valence-research/releases/tag/v1.0.0
