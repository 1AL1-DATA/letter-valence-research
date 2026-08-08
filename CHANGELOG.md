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

## [1.2.2] - 2026-08-08

### Changed (statistical-robustness hardening)
- **Borderline interpretation corrected** in `README.md`, `results/SUMMARY.md`,
  `METHODOLOGY.md`, `research_report.md`, `arxiv_paper.tex`: on the general-news
  borderline set (n = 416) the cascade is **not** a false-polarity reducer.
  Exact paired McNemar: vs its own cheap tier the rate is flat (26.4 vs 26.7%,
  p = 1.0; 27.4 vs 30.5%, p = 0.25) and vs heavy-only it is slightly but
  significantly *above* (p ≈ 2×10⁻⁴ / <10⁻⁴). The finance borderline set
  (n = 2,879) is the opposite: the cascade cuts cheap-tier false polarity
  19.6% → 7.7% (p ≈ 3×10⁻⁶¹) while paying a small significant cost vs heavy-only
  (p ≈ 4×10⁻²⁵). The previous "reduces false polarity on the FPB-trained
  variants" framing in the paper §6 was removed.
- **Clear-set significance made explicit** (exact McNemar, n = 651): only the
  news-cheap cascade is a supported win over heavy-only (0.6190 vs 0.5760,
  p ≈ 8×10⁻⁷); the FPB-cheap cascade (0.5975) is numerically higher but not
  significant (p = 0.02, above the 0.01 multiple-comparison threshold). Both
  cascade rows were un-bolded/nuanced accordingly. cheap_fpb vs heavy_fin
  p ≈ 4×10⁻⁷; cheap_news vs VADER p = 0.20 (ns); cheap_fpb vs VADER p ≈ 5×10⁻⁴.
- **Effect-size CIs added for the headline win**: +4.3 points (0.6190 vs 0.5760)
  with paired-Wilson CI [2.8, 4.9] and a 20,000-resample bootstrap CI [2.5, 6.3].
- **cascade_news vs cascade_fpb explicitly reported as marginal**: clear-set
  p = 0.049 (above the corrected 0.01 threshold; 95% bootstrap CI on the
  difference [+0.2, +4.2] points sits at the boundary of zero) and borderline
  p = 0.42 — the two cascade variants are not established as different, so
  0.6190 vs 0.5975 is a marginal, not a supported, difference.
- **Power/resolution analysis added**: at n = 416 a paired McNemar test at
  α = 0.01 / 80% power resolves only ~7-point differences, so the borderline
  null gaps (0–3 pts) mean "indistinguishable in this sample", not "equal";
  the significant *worse*-than-heavy gaps (3.1/4.1 pts) rest on 13/0 and 17/0
  discordant pairs, significant only because perfectly one-directional. The
  FPB reduction (19.6% → 7.7%) carries a paired-difference CI [0.107, 0.128].
- **heavy_fin out-of-domain conservatism noted on the borderline set**: it
  labels 84.6% of the general-neutral sentences neutral (same default that
  sinks it on the clear set at 65.3%), so its low 15.4% false-polarity is
  partly the same "never commit" artifact as keyword's.
- **"Adjacent rates not separable" claim corrected** (it was too broad):
  paired tests resolve a staircase, not a flat ordering — keyword is
  significantly below heavy_fin (p ≈ 7×10⁻⁷), heavy_fin below heavy_gen
  (p ≈ 2×10⁻³), heavy_gen below the cascades (13/0 and 17/0 one-directional
  discordant pairs), and cheap_fpb below VADER (p ≈ 4×10⁻³) — while the
  cascade-vs-cheap-tier steps (0.2–3.1 points, p = 1.0 / 0.86 / 0.25) are
  below the ~7-point resolution of those comparisons and are not supported.
  The resolvable difference is comparison-specific (≈ 2.8·√m/n points for m
  discordant pairs: ~7 points for m ≈ 110, only ~2.4–2.8 for m = 13/17), so
  the earlier blanket "pairwise p ≥ 0.25 across the plateau" phrasing was
  dropped as self-contradictory. Corrected in `README.md`,
  `results/SUMMARY.md`, `METHODOLOGY.md`, `research_report.md`,
  `arxiv_paper.tex`.
- **cascade_news vs cascade_fpb corrected to "marginal", not "unresolved"**:
  exact p = 0.049 (above the corrected 0.01 threshold) but the 95%
  bootstrap CI on the difference is [+0.2, +4.2] points (40/40 resamples
  exclude zero) — the difference is at the boundary of zero, not established
  at the strict level, but not flat either.
- **Wilson 95% CIs added** to every accuracy and false-polarity table and the
  key point estimates (24.3% share n = 158, CI [21.1, 27.7]; 91.8% accuracy CI
  [86.4, 95.1]; all borderline rates ±~4–5 pts on n = 416).
- **Threshold sweep corrected to routing-only (band confound removed)**: the
  stored sweep varied `(cheap_threshold, label_band)` and its maxima — 0.699
  (general) and 0.9583 (FPB, at heavy share ≈ 1.0) — were label-band artifacts
  (a narrower band trivially raises accuracy on a clear-only set; the FPB
  "maximum" was literally heavy-only with band 0.05). Re-run with the band held
  fixed at 0.1: best general point 0.654 (news-cheap, ct = 0.4, 53% heavy;
  0.647 for FPB-cheap) vs 0.619 default, and FPB best = 0.9558 = heavy-only
  exactly. All points remain in-sample grid maxima. `benchmark_general.py`,
  `benchmark_cascade.py`, both JSONs, and all docs updated.
- **α = 0.01 now justified as a procedure, not a claim of pre-registration**:
  phrased as a Bonferroni correction over the five cascade-vs-baseline pairwise
  comparisons at the conventional 0.05 level; the word "pre-specified" (which
  implied pre-registration) removed from `README.md`, `METHODOLOGY.md`,
  `results/SUMMARY.md`, `research_report.md`, `arxiv_paper.tex`.
- **Borderline table caption fixed**: CI overlap alone does not settle
  separability (heavy_gen and the cascades have overlapping CIs yet are
  separable by the paired tests).
- **Discordant-pair counts now reported for every headline p-value** (a
  McNemar p-value without its b/c counts is incomplete): the +4.3-point win
  rests on 31 vs 3 discordant pairs (m = 34), the FPB-cheap +2.2 points on
  24 vs 10, the cascade-vs-cascade difference on 29 vs 15 (m = 44), the FPB
  clear cascade-vs-heavy on 20 vs 11 (p = 0.15), and the FPB borderline
  cascade-vs-heavy on 0 vs 82. Added to `arxiv_paper.tex`, `README.md`,
  `METHODOLOGY.md`, `results/SUMMARY.md`, `research_report.md`, and printed by
  `src/robustness_analysis.py` (clear pairs now include b/c; FPB clear pairs
  added).
- **Keyword caveat added**: its low borderline false-polarity (5.3% / 6.6%) is
  trivially conservative — it predicts neutral on 90.5% (72.7%) of the clear set
  (accuracy 0.066 / 0.191).
- **New robustness/limitations notes**: single held-out split, no repeated
  resampling, no independent replication (paper §6 and METHODOLOGY.md).

### Added
- **`src/robustness_analysis.py`** — runnable, reproducible audit that
  recomputes every significance claim from the stored per-instance CSVs:
  exact paired McNemar p-values, Wilson CIs, paired-difference CIs,
  20,000-resample bootstrap CIs, and the per-comparison resolvable difference
  at α = 0.01 / 80% power. Run with
  `/home/a/esg-dashboard/.venv/bin/python -m src.robustness_analysis`.
- **`arxiv_paper.tex` / `arxiv_paper.pdf`** re-rendered: clear-set and
  borderline tables now carry Wilson 95% CI columns; §6 rewritten with the
  corrected borderline analysis and a "Robustness and limits" paragraph.

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

[1.2.2]: https://github.com/1AL1-DATA/letter-valence-research/releases/tag/v1.2.2
[1.2.1]: https://github.com/1AL1-DATA/letter-valence-research/releases/tag/v1.2.1
[1.2.0]: https://github.com/1AL1-DATA/letter-valence-research/releases/tag/v1.2.0
[1.1.0]: https://github.com/1AL1-DATA/letter-valence-research/releases/tag/v1.1.0
[1.0.0]: https://github.com/1AL1-DATA/letter-valence-research/releases/tag/v1.0.0
