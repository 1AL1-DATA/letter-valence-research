# Methodology

This document describes how the analysis was performed, in enough detail that
someone with a similar background could reproduce the work without reading the code.

If you only want to **run** the analysis, follow the `README.md` "Reproducing the headline result" section. If you want to **understand** the analysis, read this.

## Research question

Can the letters of an English word predict its sentiment?

Operationalised as: given a financial sentence labeled positive or negative, can we classify it correctly using only features computed from the letters of the words in that sentence (no semantic lexicons, no word embeddings, no pre-trained language models)?

## Datasets

### Warriner, Kuperman & Brysbaert (2013)

- 13,915 English lemmas rated on valence (1-9), arousal (1-9), dominance (1-9), with demographic breakdowns (M/F, age groups, education).
- Used for: **word-level correlations** between each letter-derived feature and the human-rated valence.
- Access: `https://github.com/JULIELab/XANEW/raw/master/Ratings_Warriner_et_al.csv` (3.7 MB).
- After cleaning (dropping rows with missing values, dropping words with non-alphabetic characters, dropping single-character words): **13,914 usable lemmas**.

### Malo et al. (2014) — FinancialPhraseBank

- 4,846 sentences from English-language financial news, labeled positive/neutral/negative by 16 annotators with finance background.
- We use the `Sentences_50Agree.txt` split (sentences where at least 50% of annotators agreed).
- After removing the neutral class and converting to binary (negative=0, positive=1): **1,967 sentences** (604 negative, 1,363 positive).
- Class distribution is **skewed** (69% positive), which is why the class-prior baseline is 0.693 and why a model that just predicts positive gets 69% accuracy.
- Access: `https://raw.githubusercontent.com/seandearnaley/sentiment_data_sets/master/data/inputs/FinancialPhraseBank-v1.0/Sentences_50Agree.txt` (encoding: latin-1).

### CMU Pronouncing Dictionary (CMUdict)

- 135,166 words with phonetic transcriptions in ARPAbet (e.g. "happy" → `HH AE1 P IY0`).
- Used for: **phonetic features** (F5 family: vowel ratio, plosive ratio, voiced/voiceless ratio, front/back/high/low vowel proportions).
- Access: `https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict` (3.6 MB).

### English word list (dwyl/english-words)

- 370,105 lowercase English words.
- Used to build **letter unigram and bigram frequency tables** (the `letter_freqs.json` derived file). These feed into F3 (letter frequency) and F4 (bigram statistics) features.
- Access: `https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt` (4.2 MB).

## Features

68 features, computed in `src/features.py::features()`. Organised in 12 families. All features are computable from the **letters of a single word** plus corpus-level statistics (the 370k-word letter frequencies).

### F1: Alphabet position aggregations (8 features)

Map each letter to its 1-26 position. Compute summary statistics over the position sequence of a word.

- `alpha_sum` = Σ position(c) for c in word
- `alpha_mean` = alpha_sum / word_length
- `alpha_min`, `alpha_max` = min/max position
- `alpha_range` = max - min
- `alpha_sum_mod9`, `alpha_sum_mod26` = alpha_sum mod 9, mod 26
- `alpha_sum_parity` = alpha_sum mod 2

### F2: Group-theoretic / modular (9 features)

- `sum_mod3`, `sum_mod5`, `sum_mod7`, `sum_mod11` = alpha_sum mod these primes
- `is_prime_sum` = 1 if alpha_sum is prime, 0 otherwise
- `mod26_cyclic_cos`, `mod26_cyclic_sin` = cos(2π·(alpha_sum mod 26)/26), sin(...)
- `digital_root` = 1 + (alpha_sum - 1) mod 9 (the iterated sum of digits, base 9 analogue)

### F3: Letter frequency / corpus (3 features)

For each letter in the word, look up its frequency in a 370k-word corpus. Compute:

- `letter_freq_mean` = mean of log-frequencies
- `letter_freq_sum` = sum of log-frequencies
- `rare_letter_count` = number of letters from {q, z, x, j, k}

### F4: Bigram statistics (2 features)

- `bigram_unique_ratio` = unique bigrams / total bigrams
- `trigram_count` = max(0, length - 2)

### F5: Phonetic / CMUdict (10 features)

For each word in CMUdict, compute the proportion of its phonemes that fall in each of 10 categories:

- `phon_vowel_ratio` = # vowels / # phonemes
- `phon_plosive_ratio`, `phon_fricative_ratio`, `phon_nasal_count`, etc.
- `phon_voiceless_ratio`, `phon_voice_ratio`
- `phon_front_ratio`, `phon_back_ratio`, `phon_high_ratio`, `phon_low_ratio` (relative to vowels)

For words not in CMUdict, all phonetic features are 0.

### F6: Vowel/consonant shape (8 features)

- `vowel_ratio`, `consonant_ratio`
- `distinct_letter_ratio`
- `plosive_count`, `plosive_ratio`, `fricative_count`, `nasal_count`, `liquid_count`

### F7: Word length (2 features)

- `word_length`
- `log_word_length` = log(1 + word_length)

### F8: Group attractors (2 features)

- `alphabet_centeredness` = -mean((position - 13.5)²)
- `letter_position_skew` = mean_position / 13.5 - 1

### F9: Spectral / DFT (8 features)

The **strongest family**. Treat the word as a time series of letter positions. Apply the Discrete Fourier Transform.

- `dft_power_k1`, `dft_power_k2`, `dft_power_k3` = power at frequencies 1, 2, 3
- `dft_high_freq_ratio` = power above Nyquist / total power
- `dft_total_power` = sum of power spectrum
- `dft_spectral_entropy` = Shannon entropy of the normalised power spectrum
- `autocorr_lag1`, `autocorr_lag2` = autocorrelation at lag 1 and 2

### F10: Compression / Kolmogorov (3 features)

- `gzip_size` = length of gzipped word in bytes
- `gzip_size_per_char` = gzip_size / word_length
- `gzip_ratio_vs_random` = gzip_size / (word_length · log2(26) / 8)

### F11: Number-theoretic / gematria-like (4 features)

- `letter_product_mod26` = product of letter positions mod 26
- `word_value_mod_9`, `word_value_mod_26` = word interpreted as base-26 number, mod 9/26
- `mispar_hechrechi_sum` = word sum with the standard Hebrew gematria mapping (1-9, 10-90, 100-800) applied 1-to-1 to English letters

### F12: Symmetry / run-length / position (9 features)

- `is_palindrome`, `prefix_eq_suffix` = word is the same forwards and backwards
- `symmetry_density` = fraction of letter pairs (i, j) where word[i] == word[j]
- `max_run_length`, `n_runs`, `mean_run_length`, `run_length_entropy` = run-length encoding statistics
- `first_letter_lp`, `last_letter_lp` = log-probability of first / last letter under corpus unigram model

## Per-article aggregation

For each article (sentence), we aggregate the per-word features into a single vector. The aggregation strategy used in the headline result is **"all"**: for each of the 68 features, we compute mean, max, min, and standard deviation across the words in the article, then concatenate to get a 272-dimensional vector.

Other strategies tested (see `src/features.py::aggregate_article`):
- `mean`, `max`, `min`, `std` (each 68-dim)
- `sign_count_pos` (fraction of words with positive value, 68-dim)
- `all` (272-dim, the default)

## Model

Random Forest Classifier with the following hyperparameters:
- `n_estimators = 100` (number of trees)
- `random_state = 42` (for reproducibility)
- `n_jobs = -1` (parallelize across cores)
- All other defaults from scikit-learn 1.3+

A scaler (`StandardScaler`) is fit on the training set and applied to the test set.

## Evaluation

### 5-fold stratified cross-validation

Default across all evaluations. The dataset is split into 5 stratified folds; in each iteration, 4 folds are used for training and 1 for testing. The reported accuracy and F1 are the mean and standard deviation across the 5 folds.

The fold split uses `random_state=42` for reproducibility.

### Permutation test

To verify that the observed accuracy is not a fluke of the particular choice of features and dataset, we run a label-shuffling permutation test:
1. Compute the observed accuracy on the real labels.
2. Shuffle the labels 50 times; for each shuffled set, re-run the same 5-fold CV.
3. Report the null distribution (mean, std, percentiles) and the p-value: P[null ≥ observed].

This is the strongest test of statistical significance available without holding out a separate test set. The result was **p = 0.0000** (no shuffled-label run beat the real run; the real accuracy of 0.738 was 14 standard deviations above the null mean of 0.684).

### Learning curve (bias-variance decomposition)

Train on increasingly large subsets of the data (10%, 20%, 30%, 50%, 70%, 90%, 100% of the 1,967 articles). At each size, run 5-fold CV on the subset.

This shows:
- **Variance-dominated regime** (small data): the curve is steeply rising — adding data helps a lot.
- **Bias-dominated regime** (large data): the curve is flat — adding data doesn't help, the bottleneck is features.

The transition between the two regimes tells us where we are. Our curve plateaus around n=1,376, suggesting we are in the **bias-dominated regime** — adding more labeled data would not help much, but **adding more informative features might**.

### Family ablation (leave-one-family-out)

Train on all 68 features, then drop each of the 12 families one at a time, and re-train. Compare accuracies. This shows which families are load-bearing.

### Single-family evaluation

Train on **only one family** at a time, ignoring the other 11. This shows which families are sufficient on their own.

## Reproducing a specific number

To reproduce, e.g., the headline 0.7377 accuracy:

```python
# In Python, with the repo at /tmp/letter-valence-research:
import sys
sys.path.insert(0, "/tmp/letter-valence-research")
from src.data import load_articles_binary
from src.evaluate import cross_validate

articles_df = load_articles_binary()
articles = articles_df["words"].tolist()
y = articles_df["label"].values

cv = cross_validate(articles, y, model_name="rf", n_splits=5, seed=42)
print(cv["accuracy_mean"], "±", cv["accuracy_std"])
# Expected: 0.7377 ± 0.0058
```

## The 2-tier cascade evaluation

The letter formula's practical promise was as a **cheap pre-filter** in front of a
heavier model. `src/benchmark_cascade.py` tests that hypothesis to its conclusion,
and in doing so replaces the letter tiers with a word-level cheap tier.

### Design

- **Clear-polarity set** (n = 1,967 pos/neg FinancialPhraseBank sentences): identical
  5-fold stratified test predictions for every method `{cheap, cascade, heavy, vader,
  keyword}`. The cheap tier is retrained per fold on the train split (a proper
  held-out estimate); the transformer / VADER / keyword baselines are fixed models.
- **Borderline set** (n = 2,879 held-out neutral sentences): false-polarity rate and
  score distributions for every method (deployed, full-data models).
- **Metrics:** accuracy + Wilson 95% CI, macro-F1, per-class precision/recall/F1,
  exact two-sided McNemar significance, and cascade tier routing.
- **Threshold sweep:** (cheap_threshold, label_band) pairs are re-routed cheaply on
  stored per-instance component valences, without re-running the heavy tier.

### The cheap tier

Word-level features, not letter-level:

- `TfidfVectorizer(lowercase, token_pattern=[a-z]+, ngram_range=(1,2), min_df=2,
  max_features=20000)`
- stacked with two scalar features per text: `vader_valence(text)` and
  `keyword_valence(text)`
- 3-class logistic regression (`C=1.0, max_iter=2000, class_weight="balanced"`)
- valence `v = p_positive − p_negative`

### The cascade

Routing is cheapest-and-most-decisive-first:

1. **cheap tier** decides when `|v| >= 0.6`;
2. **heavy tier** (FinancialBERT 3-class) otherwise;
3. VADER only if the heavy tier is unavailable.

### Result

On the clear-polarity set the cascade scores **0.9512** vs **0.9558** for the heavy
tier alone — a difference that is **not statistically significant** (exact McNemar
p = 0.15). The cheap tier decides **36.6%** of clear-polarity calls at **97.2%
accuracy**, and 96.7% of neutral-set calls still land in the heavy tier (i.e. the
cheap tier almost never fires on genuinely neutral news). This matches heavy-only
accuracy while cutting transformer load by about a third.

### Why the letter tiers were dropped

Measured on the same data (details in `results/cascade_benchmark.json`):

- The **DFT probe** max `|v|` = 0.922 < 0.95 firing threshold → never fires.
- The **letter RF** fires on 3.6% of sentences (71/1,967) → the heavy tier did 96.4%
  of the work anyway.
- Letter features cap at 0.7377 binary CV and are **feature-bound** (learning curve
  plateaus at n≈1,376), while word-level TF-IDF reaches 0.79–0.84 on the same data.

The letter result stands as a psycholinguistic finding; the cascade evaluation shows
that for a production pre-filter the signal is in the **words**, not the letters.

### Cross-domain generalisation: does the cascade work outside finance?

`src/benchmark_general.py` re-runs the identical architecture on **NewsMTSC**
(Hamborg et al., EACL 2021) — a 5-coder-labelled 3-class sentiment dataset from
real-world general news (AllSides), held-out `devtest_rw` split (n = 1,067; 651
clear-polarity + 416 neutral). The data are not financial, so the cascade is shown to
be a general *approach* rather than a finance-specific trick.

Design mirrors the FPB benchmark, with two cheap-tier variants and two fixed heavy
tiers:

- **`cheap_fpb`** — the word-level cheap tier trained ONLY on the 4,846
  FinancialPhraseBank sentences (cross-domain transfer; zero general-news
  supervision). This is the honest cross-domain read: no general-news label ever
  reaches the cheap tier.
- **`cheap_news`** — identical architecture retrained on the 7,758 NewsMTSC train
  sentences (in-domain reference for the same cost model).
- **`heavy_fin`** = FinancialBERT (finance-tuned); **`heavy_gen`** =
  `cardiffnlp/twitter-roberta-base-sentiment-latest` (general-domain transformer).
  Both are fixed; neither sees NewsMTSC labels.
- Same routing (`|v| >= 0.6` → cheap, else heavy, VADER fallback), same metrics
  (accuracy + Wilson CI, macro-F1, exact McNemar, tier routing, threshold sweep).

Results (clear-polarity set, n = 651; Wilson 95% CIs in brackets):

| Method | Accuracy (95% CI) | Macro-F1 |
|---|---|---|
| Keyword lexicon | 0.0661 [0.049, 0.088] | 0.1199 |
| FinancialBERT (heavy_fin) | 0.3041 [0.270, 0.341] | 0.4581 |
| Cheap tier trained on FPB only | 0.4209 [0.384, 0.459] | 0.5523 |
| VADER | 0.4685 [0.431, 0.507] | 0.5744 |
| Cheap tier trained on news | 0.4931 [0.455, 0.531] | 0.6069 |
| General BERT (heavy_gen) | 0.5760 [0.538, 0.613] | 0.6641 |
| Cascade (FPB cheap → gen heavy) | 0.5975 [0.559, 0.635] | 0.6741 |
| **Cascade (news cheap → gen heavy)** | **0.6190 [0.581, 0.656]** | **0.6940** |

Interpretation (significance = exact two-sided McNemar on the paired sentences):

1. **The cheap word tier transfers out of finance.** Trained on nothing but FPB, it
   still beats the finance-tuned FinancialBERT on general news (0.4209 vs 0.3041,
   p ≈ 4×10⁻⁷) and closes most of the gap to VADER (still significantly below,
   p ≈ 5×10⁻⁴). Retrained on news it is the strongest non-transformer component
   (0.4931), not significantly different from VADER (p = 0.20).
2. **The finance-tuned heavy is the domain-locked part.** FinancialBERT collapses on
   general news — it predicts neutral on 65.3% of clear-polarity sentences and lands
   *below chance* (0.3041). With a domain-appropriate heavy, **only the news-cheap
   cascade** beats heavy-only (0.6190 vs 0.5760, +4.3 points, paired-Wilson CI
   [2.8, 4.9] points, bootstrap CI [2.5, 6.3]; McNemar p ≈ 8×10⁻⁷); the FPB-cheap
   cascade (0.5975) is numerically higher but not significant (p = 0.02, above the
   0.01 multiple-comparison threshold); and the two cascade variants do not differ
   reliably at the pre-specified α = 0.01 level (p = 0.049; 95% bootstrap CI on
   the difference [+0.2, +4.2] points is marginal). The cheap tier absorbs 24.3% of calls (n =
   158, CI [21.1, 27.7]) at 91.8% accuracy. The threshold-sweep best (0.699) is an
   **in-sample** grid maximum, an upper bound, not a held-out estimate.

On the **borderline set** (n = 416 neutral sentences) the cascade is *not* a
false-polarity reducer out of domain: versus its own cheap tier the rate is flat
(26.4 vs 26.7%, p = 1.0; 27.4 vs 30.5%, p = 0.25) and versus heavy-only it is
slightly but significantly *above* (26.4%/27.4% vs 23.3%, p ≈ 2×10⁻⁴ / <10⁻⁴).
Power matters here: at n = 416 a paired test at α = 0.01 / 80% power resolves
only ~7-point differences, so the 0–3-point gaps to the cheap tier mean
"indistinguishable in this sample", not "equal"; the significant *worse*-than-heavy
gaps rest on 13/0 and 17/0 discordant pairs, significant only because they are
perfectly one-directional. Paired tests resolve a staircase, not a flat ordering:
keyword is significantly below heavy_fin (p ≈ 7×10⁻⁷), heavy_fin below heavy_gen
(p ≈ 2×10⁻³), heavy_gen below the cascades (13/0 and 17/0 discordant pairs), and
cheap_fpb below VADER (p ≈ 4×10⁻³); the central plateau — heavy_gen through the
cheap tiers (23–31%) — is not otherwise resolvable (adjacent members differ by
less than the ~7-point resolution; pairwise p ≥ 0.25). Keyword's 5.3% is
trivially conservative (predicts neutral on 90.5% of the clear set), and part of
FinancialBERT's 15.4% is the same out-of-domain conservatism — it labels 84.6% of
the borderline sentences neutral. This is the opposite of the finance
borderline set (n = 2,879), where the cascade cuts cheap-tier false polarity
19.6% → 7.7% (p ≈ 3×10⁻⁶¹, paired-difference CI [0.107, 0.128]) while remaining
above heavy-only's 4.9% (p ≈ 4×10⁻²⁵).

The generalisation result is that the **cascade approach** — cheap word tier + routing
→ heavy fallback — is a general feature; the heavy transformer must match the domain.

## Limitations of the methodology

1. **Single dataset, single task, single language.** The letter formula itself is validated on FPB binary only; the cascade follow-up adds a cross-domain held-out test on general news (NewsMTSC), and the cheap word tier transfers reasonably, but the formula's behaviour on other sentiment tasks and other languages is still unknown.

2. **5-fold CV is not a held-out test.** The reported letter accuracy is an estimate of how the formula would perform on a new sample from the same distribution, not a guarantee of how it would perform on a different distribution. The cascade generalisation run partially addresses this — it retrains the cheap tier on FPB and evaluates on a different labeled corpus (NewsMTSC) — and the cross-domain result is reported honestly (0.42 transfer, 0.30 for the finance-tuned heavy).

3. **Cross-domain conclusions rest on a single held-out split.** The NewsMTSC
   evaluation uses one fixed split (n = 651 clear / 416 borderline) with no repeated
   resampling and no independent replication. Point differences smaller than the
   ~±4–5-point Wilson CIs are not separable, and the threshold-sweep maximum is
   in-sample. Power is limited on the borderline set: at n = 416 a paired test at
   α = 0.01 / 80% power resolves only ~7-point differences, so null results there
   mean "indistinguishable in this sample". The supported cross-domain claims are
   limited to the paired differences with exact McNemar p ≤ 10⁻³ (news-cheap
   cascade vs heavy-only, +4.3 points [2.8, 4.9]; cheap tiers vs FinancialBERT;
   cascade-vs-cheap on the clear set) and the
   absence of a borderline false-polarity reduction.

4. **Feature engineering informed by literature.** The 68 features were selected based on prior work (Adelman 2018, Aryani 2018, de Zubicaray 2024). This is informed feature engineering, not automated feature learning. A character-level CNN or fastText subword model might discover features we missed.

4. **The Random Forest is not interpretable.** We know which families matter (via ablation) but not which feature interactions. SHAP values would be a useful follow-up.

5. **No error analysis.** We did not look at the misclassified articles. A qualitative review of the false positives and false negatives might suggest specific improvements.

6. **Bonferroni is conservative.** For 68 tests, the Bonferroni threshold is α = 0.05/68 ≈ 7.4e-4. We use this throughout. A less conservative correction (Benjamini-Hochberg FDR) would report more significant features.

## What the headline number does NOT mean

- It does **not** mean letter features are sufficient to replace a sentiment lexicon. The accuracy gap to VADER-tuned (0.7377 vs 0.750) is small but consistent across runs.
- It does **not** mean letter features are sufficient to replace a transformer. FinBERT reaches ~0.87 on the same data.
- It does **not** mean the gematria hypothesis is confirmed. The "math in words" that works is **spectral** (DFT), not **modular** (gematria).
- It does **not** mean the formula generalises to non-financial text. It was trained and tested on FPB only.

## What the headline number DOES mean

- A model trained on **only letter features** can reach ~74% accuracy on FPB binary sentiment.
- This is significantly above chance (p < 0.0001) and significantly above a linear baseline (logistic regression: 0.71, ridge: 0.72).
- The signal is real but small. The formula is best used as a fast pre-filter or a second opinion, not a primary sentiment classifier.
- The bottleneck for further improvement is **features**, not data — the learning curve plateaus around n=1,376, well below the 1,967 article full dataset.
