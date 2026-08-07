# Architecture

The data flow through the system, from raw input to final result.

```
                                  +-------------------+
                                  |   data/           |
                                  |  - warriner2013   |
                                  |  - articles_binary|
                                  |  - cmudict        |
                                  |  - letter_freqs   |
                                  +---------+---------+
                                            |
                                            v
                              +-------------+-------------+
                              |   src/data.py             |
                              |   - load_warriner()       |
                              |   - load_articles_binary()|
                              |   - tokenize()            |
                              |   - build_letter_freqs()  |
                              +-------------+-------------+
                                            |
                                            v
                            +---------------+---------------+
                            |   src/features.py            |
                            |   - features(word) -> dict   |
                            |   - 68 features, 12 families |
                            |   - get_feature_names()      |
                            |   - aggregate_article()      |
                            +---------------+---------------+
                                            |
                                            v
                            +---------------+---------------+
                            |   src/train.py               |
                            |   - train()                  |
                            |   - predict()                |
                            |   - save() / load()          |
                            |   - FeaturePipeline          |
                            |   - make_model()             |
                            +---------------+---------------+
                                            |
                                            v
                            +---------------+---------------+
                            |   src/evaluate.py            |
                            |   - cross_validate()         |
                            |   - permutation_test()       |
                            |   - learning_curve()         |
                            |   - family_ablation()        |
                            |   - single_family_evaluation|
                            +---------------+---------------+
                                            |
                                            v
                            +---------------+---------------+
                            |   src/analyze.py             |
                            |   - the main entry point     |
                            |   - runs the full pipeline   |
                            |   - writes to results/       |
                            +---------------+---------------+
                                            |
                                            v
                                  +---------+---------+
                                  |   results/         |
                                  |  - cv_*.csv        |
                                  |  - learning_curve  |
                                  |  - permutation_*.json|
                                  |  - family_ablation |
                                  |  - single_family   |
                                  |  - summary.json    |
                                  |  - SUMMARY.md      |
                                  +-------------------+
```

## Module responsibilities

| Module | Responsibility | Inputs | Outputs |
|---|---|---|---|
| `data.py` | Data loading + derivation | raw files in `data/` | `pandas.DataFrame` with standard columns |
| `features.py` | Compute 68 letter-derived features for each word | word string, optional letter-freq dict, optional CMUdict | dict of feature_name → float |
| `train.py` | Train and use a classifier | tokenized articles, labels | fitted model + feature pipeline |
| `evaluate.py` | All evaluation logic (CV, permutation, learning curve, ablation) | articles, labels, model | dict or DataFrame of results |
| `analyze.py` | Orchestrate the full pipeline | none (reads from `data/`) | results in `results/` |

## Data flow for the headline result

1. `data.py` loads the Warriner and FPB datasets from `data/`.
2. `features.py` is imported by `train.py` and `evaluate.py`; the feature extractor is called once per unique word.
3. `train.py::train()` calls `features.py::aggregate_article()` to convert a list of words into a 272-dim vector.
4. `evaluate.py::cross_validate()` does 5-fold stratified CV: for each fold, it calls `train.py::train()` and then `train.py::predict()`.
5. `evaluate.py::permutation_test()` calls `cross_validate()` 50 times with shuffled labels.
6. `evaluate.py::learning_curve()` calls `cross_validate()` at each train-set size.
7. `analyze.py::main()` orchestrates all of the above and writes the results to `results/`.

## The 2-tier cascade evaluation (follow-up)

`src/benchmark_cascade.py` and `src/benchmark_general.py` evaluate a separate,
deployed component: the sentiment cascade used by
[esg-dashboard](https://github.com/1AL1-DATA/esg-dashboard).

- **2 tiers**: a word-level cheap tier (TF-IDF 1–2 grams + VADER + keyword features
  → 3-class logistic regression, v = p_pos − p_neg) decides when |v| ≥ 0.6; otherwise
  a heavy transformer scores the text (VADER as final fallback).
- `benchmark_cascade.py` — FinancialPhraseBank evaluation (clear n=1,967, neutral
  n=2,879): cascade 0.9512 vs heavy-only 0.9558 (McNemar p = 0.15), cheap tier carries
  36.6% at 97.2% accuracy.
- `benchmark_general.py` — cross-domain evaluation on general news (NewsMTSC,
  devtest_rw n=1,067): two cheap-tier variants (trained only on FPB vs on news) × two
  heavy tiers (FinancialBERT vs a general-domain transformer). Findings: the cheap tier
  transfers out of finance (0.42), the finance-tuned heavy collapses on general news
  (0.30), and a cascade with a general heavy beats heavy-only (0.62 vs 0.58).
- Inputs: `data/newsmtsc/*.jsonl` + the `esg-dashboard` sentiment_engine components.
  Outputs: `results/cascade_benchmark.json`, `results/general_news_benchmark.json`,
  `results/*_predictions.csv`, `figures/cascade_sentiment_eval.png`,
  `figures/general_news_eval.png`.

## Extension points

- **New feature family**: add it to `src/features.py::features()` and to `FEATURE_FAMILIES` in `src/evaluate.py`. The ablation, learning curve, and CV will pick it up automatically.
- **New dataset**: add a loader to `src/data.py` with a `load_X()` function returning a DataFrame with `words` and `label` columns. Add a call to `analyze.py::main()`.
- **New model**: add it to `src/train.py::make_model()`. It must implement `.fit(X, y)` and `.predict(X)`.
- **New evaluation metric**: add it to `src/evaluate.py::_score()`.
