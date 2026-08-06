# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and this project adheres to
[Semantic Versioning](https://semver.org/).

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

- Cross-dataset transfer: train on FPB, test on a different labeled corpus (SST-2, IMDB).
- Comparison with a character-level CNN baseline.
- Hierarchical classification for long documents: split into sentences, aggregate predictions.
- Ablation on aggregation strategy: mean/max/min/std vs learned attention.

[1.1.0]: https://github.com/[your-org]/letter-valence-research/releases/tag/v1.1.0
[1.0.0]: https://github.com/[your-org]/letter-valence-research/releases/tag/v1.0.0
