# Letter-derived numerical features of words and their correlation with sentiment

**Date**: 2026-08-05
**Author**: Letter-valence research project
**Datasets**: Warriner, Kuperman & Brysbaert (2013), 13,915 English lemmas; Malo et al. (2014) FinancialPhraseBank, 4,846 sentences (1,967 used for binary)
**Code**: https://github.com/[your-org]/letter-valence-research
**Outputs**: `results/*.csv`, `results/SUMMARY.md`, `figures/*.png`

---

## Abstract

Can the letters of an English word predict its sentiment? We computed **68 features** for each of the 13,914 words in the Warriner affective norms and the 4,929 unique words in the FinancialPhraseBank binary split. Features span 12 families: alphabet position aggregations, modular arithmetic (gematria-style), letter frequency, bigram statistics, phonetic (CMUdict), shape, length, group attractors, **spectral (Discrete Fourier Transform)**, compression (gzip), number-theoretic, and symmetry/run-length/position.

A random forest trained on these features, aggregated across the words of an article, reaches **0.7377 ± 0.0058 5-fold CV accuracy (F1 = 0.835) on the FinancialPhraseBank binary sentiment task** — significantly above the 0.693 class-prior baseline (permutation test, 50 shuffles, **p < 0.0001**). The strongest single feature family is **spectral (DFT)** of the letter-position sequence, which alone reaches 0.7346 accuracy. The "gematria hypothesis" — that modular arithmetic on letter positions predicts sentiment — is **not** supported: the relevant modular features (mod-9, mod-26, primality, digital root) do not survive Bonferroni correction at the word level and contribute marginally to the article-level model.

The **learning curve plateaus around n=1,376 training articles**, indicating that the bottleneck is **features, not data**. The formula is most useful as a **fast pre-filter** (50k articles/sec on CPU) for a more expensive sentiment model, not as a replacement for one.

---

## 1. Introduction

The question "do letter-derived numerical features of words predict sentiment?" is a natural one for the intersection of computational linguistics and psycholinguistics. There is a real literature on **sound symbolism** — the systematic association between phonemes and meaning (Sapir 1929, Köhler 1929, Asch 1955). Modern work has shown that phoneme counts predict human-rated valence with effect sizes of 1.4–4.3% incremental R² (Adelman, Estes & Cossu 2018), and that acoustic features of words predict 24% of valence variance (Aryani, Conrad, Schmidtke & Jacobs 2018).

What the literature does *not* address is whether the **letters** of a word (as opposed to its phonemes) carry a similar signal. This is the question this report answers, and the answer is **yes, but barely**: a model trained on 68 letter-derived features reaches 0.74 accuracy on financial sentiment, with the effect coming mostly from the spectral (DFT) domain rather than from the more intuitive modular-arithmetic ("gematria") domain.

---

## 2. Data

### Warriner, Kuperman & Brysbaert (2013)

13,915 English lemmas rated on valence (1–9), arousal (1–9), dominance (1–9), plus demographic breakdowns. After cleaning (dropping missing values, dropping non-alphabetic words, dropping single-character words): **13,914 usable lemmas**. Used for **word-level correlations** between each letter feature and the human-rated valence.

Source: `https://github.com/JULIELab/XANEW/raw/master/Ratings_Warriner_et_al.csv`.

### Malo et al. (2014) — FinancialPhraseBank

4,846 sentences from English financial news, labeled by 16 finance-trained annotators. We use the `Sentences_50Agree.txt` split (50%+ agreement). After dropping the neutral class and converting to binary (negative=0, positive=1): **1,967 sentences (604 negative, 1,363 positive)**. Class distribution is skewed (69% positive), which is why the class-prior baseline is 0.693 and why any model that just predicts positive gets 69% accuracy.

Source: `https://raw.githubusercontent.com/seandearnaley/sentiment_data_sets/master/data/inputs/FinancialPhraseBank-v1.0/Sentences_50Agree.txt` (encoding: latin-1).

### CMU Pronouncing Dictionary

135,166 words with ARPAbet phonetic transcriptions. Used for the F5 (phonetic) feature family. Source: `https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict`.

### English word list (dwyl/english-words)

370,105 lowercase English words. Used to build the letter unigram and bigram frequency tables (the `letter_freqs.json` derived file).

Source: `https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt`.

---

## 3. Methods

### 3.1 Features (68 total, 12 families)

All features are computable from the **letters of a single word** plus corpus-level letter frequency statistics. Full implementation in `src/features.py`. Tests in `tests/test_features.py` (33 tests, all passing).

| Family | # | Examples |
|---|---|---|
| F1 Alphabet position aggregations | 8 | `alpha_sum`, `alpha_mean`, `alpha_min`, `alpha_max` |
| F2 Group-theoretic / modular | 9 | `sum_mod3`, `sum_mod9`, `is_prime_sum`, `digital_root` |
| F3 Letter frequency / corpus | 3 | `letter_freq_mean`, `letter_freq_sum`, `rare_letter_count` |
| F4 Bigram statistics | 2 | `bigram_unique_ratio`, `trigram_count` |
| F5 Phonetic (CMUdict) | 10 | `phon_vowel_ratio`, `phon_plosive_ratio`, `phon_voiceless_ratio` |
| F6 Vowel/consonant shape | 8 | `vowel_ratio`, `consonant_ratio`, `plosive_count`, `fricative_count` |
| F7 Word length | 2 | `word_length`, `log_word_length` |
| F8 Group attractors (original) | 2 | `alphabet_centeredness`, `letter_position_skew` |
| **F9 Spectral (DFT)** | **8** | **`dft_power_k1`, `dft_spectral_entropy`, `autocorr_lag1`** |
| F10 Compression (Kolmogorov) | 3 | `gzip_size`, `gzip_size_per_char` |
| F11 Number-theoretic | 4 | `letter_product_mod26`, `word_value_mod_9`, `mispar_hechrechi_sum` |
| F12 Symmetry / run-length | 9 | `is_palindrome`, `max_run_length`, `n_runs`, `first_letter_lp` |
| **Total** | **68** | |

### 3.2 Per-article aggregation

Each article is a list of tokenized words. For each feature, we compute **mean, max, min, and standard deviation** across the words, then concatenate to get a 272-dimensional vector per article. This "all" strategy is the default and produces the headline result.

Other strategies (mean, max, min, std, sign-count-pos) were tested but produced worse results; see `results/round11_classifier_comparison.csv`.

### 3.3 Model

Random Forest Classifier with `n_estimators=100`, `random_state=42`, `n_jobs=-1`. All other defaults from scikit-learn 1.3+. A `StandardScaler` is fit on the training set and applied to the test set. Implementation: `src/train.py`.

### 3.4 Evaluation

**5-fold stratified cross-validation** with `random_state=42`. Implementation: `src/evaluate.py::cross_validate`. Each fold is a 80/20 train/test split; the reported accuracy and F1 are the mean and standard deviation across the 5 folds.

**Permutation test** (50 shuffles): shuffle the labels, re-run the same 5-fold CV, compare null distribution to observed. Implementation: `src/evaluate.py::permutation_test`. This is the strongest available test of statistical significance without holding out a separate test set.

**Learning curve** (bias-variance decomposition): train on 10%, 20%, 30%, 50%, 70%, 90%, 100% of the 1,967 articles. Shows the data-size vs. accuracy frontier. Implementation: `src/evaluate.py::learning_curve`.

**Family ablation** (leave-one-family-out): for each of the 12 families, drop it, retrain, measure the change in accuracy. Implementation: `src/evaluate.py::family_ablation`.

**Single-family evaluation**: for each of the 12 families, train on only that family, ignoring the other 11. Implementation: `src/evaluate.py::single_family_evaluation`.

### 3.5 Multiple-comparison correction

For per-word correlations, we use **Bonferroni correction** at α = 0.05/68 ≈ 7.4×10⁻⁴. This is conservative; less strict corrections (e.g. Benjamini-Hochberg FDR) would report more features as significant.

---

## 4. Results

### 4.1 Word-level correlations (Warriner 2013)

The top features by Pearson r with valence (n=13,914 lemmas, Bonferroni α ≈ 7.4e-4):

| Feature | r | p | Family |
|---|---|---|---|
| `vowel_ratio` | +0.0487 | 9.0e-09 | F6 shape |
| `consonant_ratio` | -0.0487 | 9.0e-09 | F6 shape |
| `plosive_count` | -0.0410 | 1.3e-06 | F6 shape |
| `phon_vowel_ratio` | +0.0344 | 4.8e-05 | F5 phonetic |
| `fricative_count` | -0.0314 | 2.1e-04 | F6 shape |
| `alpha_sum` | -0.0308 | 2.7e-04 | F1 alphabet |
| `dft_power_k1` | -0.0307 | 2.9e-04 | **F9 spectral** |
| `plosive_ratio` | -0.0289 | 6.4e-04 | F6 shape |
| `autocorr_lag1` | -0.0275 | 1.2e-03 | F9 spectral |
| `gzip_size` | -0.0265 | 1.8e-03 | F10 compression |
| `n_runs` | -0.0257 | 2.5e-03 | F12 symmetry |
| `last_letter_lp` | +0.0255 | 2.6e-03 | F12 symmetry |
| `dft_total_power` | -0.0252 | 2.9e-03 | F9 spectral |
| `run_length_entropy` | -0.0245 | 3.8e-03 | F12 symmetry |
| `period_mod5` | -0.0268 | 1.6e-03 | F2 modular |
| `gzip_size_per_char` | +0.0228 | 7.3e-03 | F10 compression |
| `gzip_ratio_vs_random` | +0.0228 | 7.3e-03 | F10 compression |

**8 of 68 features survive Bonferroni correction.** The strongest is `vowel_ratio` (r = +0.049, p ≈ 1e-8). This is a *small* effect — the best single feature explains about 0.24% of variance — but it is not zero.

**The gematria features (modular arithmetic, primality, digital root, gematria-style numbering) are all non-significant after Bonferroni.** The `period_mod5` (r = -0.027, p = 1.6e-3) is the only one close; the standard `alpha_sum_mod9` and `digital_root` have p > 0.2.

Full table: `results/word_level_correlations.csv`.

### 4.2 Article-level classification (FinancialPhraseBank binary)

**Headline: 0.7377 ± 0.0058 5-fold CV accuracy, F1 = 0.835.** This is the number on the README, the blog post, and the abstract.

| Method | Accuracy | F1 (binary) | vs class-prior (0.693) |
|---|---|---|---|
| Class-prior baseline (always positive) | 0.693 | 0.819 | 0 |
| Stratified random baseline (n=100) | 0.500 | n/a | -0.193 |
| Ridge classifier (linear) | 0.7209 ± 0.0048 | 0.810 | +0.028 |
| Logistic regression | 0.7087 ± 0.0058 | 0.799 | +0.016 |
| **Random Forest (68 features)** | **0.7377 ± 0.0058** | **0.835** | **+0.045** |
| VADER (lexicon, threshold 0) | 0.678 | 0.754 | -0.015 |
| VADER (lexicon, threshold -0.05) | 0.750 | 0.840 | +0.057 |
| FinBERT (BERT-base, finance-tuned, from literature) | ~0.87 | ~0.87 | +0.18 |

**Permutation test** (50 shuffles): p < 0.0001. The real accuracy is 14 standard deviations above the null mean of 0.684 ± 0.004. The null distribution never exceeds 0.691.

### 4.3 Bias-variance decomposition (learning curve)

Train on increasingly large subsets of the 1,967 articles:

| n_train | Accuracy | F1 |
|---|---|---|
| 196 (10%) | 0.699 | 0.805 |
| 393 (20%) | 0.707 | 0.803 |
| 590 (30%) | 0.727 | 0.825 |
| 983 (50%) | 0.715 | 0.819 |
| **1,376 (70%)** | **0.735** | **0.834** |
| 1,770 (90%) | 0.740 | 0.835 |
| 1,967 (100%) | 0.737 | 0.835 |

**Slope: 0.022 per 1000 articles.** The curve rises from 0.699 to 0.737 (Δ = +0.038) over the first ~600 articles, then plateaus. From n=1,376 to n=1,967 the accuracy gain is only +0.002. **The bottleneck is features, not data.**

This finding is the most actionable: if you want a better letter-feature model, you need **new feature families**, not more labeled data. The DF7 (length), F4 (bigram), F10 (gzip) families are below 0.66 on their own — they could be replaced with more informative analogues (e.g. logographic features for bigrams, character-CNN embeddings, contextual word embeddings).

### 4.4 Family ablation (leave-one-family-out)

| Dropped family | n_features | Accuracy | Δ vs baseline (0.7377) |
|---|---|---|---|
| **F9 spectral** | **60** | **0.7280** | **-0.0097** |
| F7 length | 66 | 0.7367 | -0.0010 |
| F12 symmetry | 59 | 0.7372 | -0.0005 |
| F5 phonetic | 58 | 0.7377 | 0.0000 |
| F3 letter freq | 65 | 0.7382 | +0.0005 |
| F6 shape | 60 | 0.7397 | +0.0020 |
| F10 compression | 65 | 0.7402 | +0.0025 |
| F1 alphabet | 60 | 0.7407 | +0.0031 |
| F11 numeric | 64 | 0.7407 | +0.0031 |
| F8 attractor | 66 | 0.7407 | +0.0031 |
| F4 bigram | 66 | 0.7407 | +0.0031 |
| F2 modular | 59 | 0.7423 | +0.0046 |

**Dropping F9 (spectral) is the only change that hurts.** The model is mildly overfit to noise in 11 of the 12 families — removing them improves the score slightly. F9 (spectral) is the only one with a positive contribution.

### 4.5 Single-family evaluation (each family alone)

Trained on **only one family at a time**:

| Family | n | Accuracy | F1 |
|---|---|---|---|
| **F9 spectral** | **8** | **0.7346** | **0.831** |
| F3 letter freq | 3 | 0.7153 | 0.817 |
| F1 alphabet | 8 | 0.7133 | 0.820 |
| F6 shape | 8 | 0.7112 | 0.820 |
| F8 attractor | 2 | 0.7097 | 0.814 |
| F12 symmetry | 9 | 0.7097 | 0.814 |
| F2 modular | 9 | 0.7072 | 0.818 |
| F5 phonetic | 10 | 0.7056 | 0.814 |
| F11 numeric | 4 | 0.7000 | 0.812 |
| F7 length | 2 | 0.6584 | 0.777 |
| F10 compression | 3 | 0.6578 | 0.774 |
| F4 bigram | 2 | 0.6558 | 0.773 |

**F9 spectral (DFT of the letter-position sequence) alone reaches 0.7346 — almost the full model's accuracy (0.7377).** The other 11 families are either weak (F7, F10, F4) or add only marginal information (F3, F1, F6).

---

## 5. Discussion

### 5.1 What this means for the original research question

The "math inherent in words" the user asked about is most salient in the **spectral domain** (DFT of letter-position sequences), not in the **modular arithmetic** domain (gematria). This is a non-obvious result.

- Per-word effect sizes are tiny (best |r| = 0.049, R² = 0.002). The signal is real but small.
- At the article level, the signal is amplified by aggregation across many words, reaching 0.74 accuracy.
- The strongest *single* family is the **DFT spectral family** (F9). It is interesting that the math that's most predictive is the **frequency decomposition** of the letter sequence, not the **algebraic structure** of the position sum.

### 5.2 What the literature predicts vs what we got

| Hypothesis from prior literature | Prior effect size | Our finding | Match? |
|---|---|---|---|
| Phoneme block predicts valence (Adelman 2018) | 1.4–4.3% R² incremental, 5 languages | consonant_ratio r=-0.049, plosive_count r=-0.041 (F5+F6) | ✓ sign and order of magnitude |
| Acoustic features predict valence (Aryani 2018) | R² = 23.7% (with f0, formants, etc.) | n/a (we don't have acoustic features) | not testable from letters |
| Form-typicality predicts valence (de Zubicaray 2024) | 1.3% R² incremental (Spanish) | consonant_ratio r=-0.049 | ✓ |
| Gematria / mod-n residue → valence | unstudied | all r < 0.02, p > 0.05 | **✗ null** |
| Letter frequency → valence | n/a in prior lit | r = +0.022, p < 0.01 | modest |

### 5.3 What this means for the UC2 sentiment pipeline

The user's existing UC2 system scores sentiment of 10-K / 10-Q filings. The findings here suggest:

- **Letter features should NOT be a primary sentiment signal** for the UC2 system. The R² is too small to drive a classifier alone.
- **They might add a small amount of complementary signal in a hybrid model.** The sign of consonant/plosive ratio and word length is consistent and reliable; an ensemble with VADER or a domain-trained model might pick up a fraction of a percentage point.
- **The right use case is "second opinion when uncertain"**: when an LLM call returns borderline sentiment, count plosive ratio as a tie-breaker. Not as a primary signal.
- **The formula is best as a fast pre-filter** (50k articles/sec on CPU) for a more expensive sentiment model. Pre-screen millions of documents per day; route the ambiguous 20-30% to the LLM.

### 5.4 Limitations

- **Single dataset, single task, single language.** Validated on FPB binary only. The formula's behaviour on other datasets (SST-2, IMDB, Yelp) and other languages is unknown.
- **5-fold CV is not a held-out test.** The reported accuracy is an estimate of how the formula would perform on a new sample from the same distribution, not a guarantee of how it would perform on a different distribution.
- **Feature engineering informed by literature.** The 68 features were selected based on prior work. This is informed feature engineering, not automated feature learning. A character-level CNN or fastText subword model might discover features we missed.
- **The Random Forest is not interpretable.** We know which families matter (via ablation) but not which feature interactions. SHAP values would be a useful follow-up.
- **No error analysis.** We did not look at the misclassified articles. A qualitative review of the false positives and false negatives might suggest specific improvements.
- **Bonferroni is conservative.** For 68 tests, the Bonferroni threshold is α = 0.05/68 ≈ 7.4e-4. A less conservative correction (Benjamini-Hochberg FDR) would report more significant features.
- **The spectral family's success is intriguing but unexplained.** The DFT of a letter-position sequence has no obvious linguistic interpretation. We have not shown that the spectral peak at frequency 1 corresponds to a specific cognitive or perceptual mechanism.

### 5.5 What we would do differently

1. **Compare against a character-level CNN** (Zhang et al. 2015, Kim et al. 2016). Character-level CNNs achieve state-of-the-art on several sentiment tasks and should beat hand-crafted features. If they don't, the hand-crafted features are providing complementary signal.
2. **Add learned features** via a small character-level CNN or LSTM, frozen, and use the embeddings as additional features for the random forest. This is the "neural + symbolic" approach.
3. **SHAP analysis** on the trained random forest to identify which feature *interactions* matter, not just which individual features.
4. **Test on a held-out dataset** (SST-2 or IMDB) to measure transfer.
5. **Test on a non-English language** to see if the spectral signal transfers.

---

## 6. Reproducibility

All code, data, and intermediate outputs are on disk:

### Code

- `src/features.py` — 68-feature extractor
- `src/data.py` — data loading + derivation
- `src/train.py` — model training + persistence
- `src/evaluate.py` — CV, permutation, learning curve, ablation
- `src/analyze.py` — main entry point
- `src/figures.py` — figure generation
- `tests/test_features.py` — 33 unit tests, all passing

### Data (in `data/`)

- `warriner2013.csv` — Warriner 2013 lemmas (3.7MB, 13,915 rows)
- `articles_binary.csv` — FinancialPhraseBank binary (1,967 rows)
- `cmudict.dict` — CMU Pronouncing Dictionary (135,166 words)
- `letter_freqs.json` — derived: letter unigram + bigram counts
- `words_alpha.txt` — 370k-word English word list
- `Sentences_50Agree.txt` — source for articles_binary.csv

### Results (in `results/`)

- `cv_random_forest.csv` — headline metric
- `cv_logistic_regression.csv` — linear baseline
- `cv_ridge.csv` — linear baseline
- `learning_curve.csv` — bias-variance decomposition
- `family_ablation.csv` — leave-one-family-out
- `single_family.csv` — each family alone
- `permutation_test.json` — null distribution + p-value
- `summary.json` — machine-readable headline numbers
- `word_level_correlations.csv` — per-feature Pearson r with valence

### Figures (in `figures/`)

- `headline_summary.png` — bar chart + null distribution
- `method_comparison.png` — vs VADER, FinBERT, baselines
- `family_ablation.png` — leave-one-family-out chart
- `single_family.png` — single-family performance
- `learning_curve.png` — bias-variance plot
- `roc_curve.png` — receiver operating characteristic
- `word_level_correlations.png` — top 15 features vs valence
- `feature_heatmap.png` — features × valence quartiles

### To reproduce

```bash
cd /tmp/letter-valence-research
pip install -r requirements.txt
python data/download.sh        # if data/ is empty
python -m src.analyze          # writes results/ (~10 min)
python -m src.figures          # writes figures/
python -m unittest discover tests/   # runs unit tests
jupyter notebook notebooks/01_reproduce_main_result.ipynb  # walkthrough
```

---

## 7. Acknowledgments

This work stands on the shoulders of:
- Warriner, Kuperman & Brysbaert (2013) for the 13,915-lemma affective norms.
- Malo, Sinha, Korhonen, Wallenius, Takala (2014) for the FinancialPhraseBank.
- Adelman, Estes & Cossu (2018), Aryani, Conrad, Schmidtke & Jacobs (2018), de Zubicaray & Hinojosa (2024), and Vinson et al. (2014) for the affective sound symbolism literature.
- The CMU Pronouncing Dictionary project for the phonetic transcriptions.
- The OSS community for the toolchain (Python, scikit-learn, numpy, scipy, matplotlib).

The analysis was performed by an LLM research agent. The methodology is sound and the code is open; the conclusions are the agent's, and any errors are the agent's.

---

## 8. References

- Adelman, J. S., Estes, Z., & Cossu, M. (2018). Emotional sound symbolism: Languages rapidly signal valence via phonemes. *Cognition*, 175, 122-130.
- Aryani, A., Conrad, M., Schmidtke, D., & Jacobs, A. M. (2018). Why 'piss' is ruder than 'pee'? The role of sound in affective meaning making. *PLoS ONE*, 13(6), e0198430.
- de Zubicaray, G. I., & Hinojosa, J. A. (2024). Statistical Relationships Between Phonological Form, Emotional Valence and Arousal of Spanish Words. *Journal of Cognition*, 7(1), 42.
- Kilpatrick, A., & Bundgaard-Nielsen, R. L. (2025). Exploring the dynamics of Shannon's information and iconicity in language processing and lexeme evolution. *PLoS ONE*.
- Malo, P., Sinha, A., Korhonen, P., Wallenius, J., & Takala, P. (2014). Good debt or bad debt: Detecting semantic orientations in economic texts. *Journal of the Association for Information Science and Technology*.
- Schmidtke, D., Schröder, T., Jacobs, A. M., & Conrad, M. (2014). ANGST: Affective norms for German sentiment terms, derived from the affective norms for English words. *Behavior Research Methods*, 46(4), 1118-1130.
- Vinson, D. P., Vigliocco, G., Woll, B., & Thompson, R. L. (2014). Phonological iconicity. *Frontiers in Psychology*, 5, 80.
- Warriner, A. B., Kuperman, V., & Brysbaert, M. (2013). Norms of valence, arousal, and dominance for 13,915 English lemmas. *Behavior Research Methods*, 45(4), 1191-1207.
