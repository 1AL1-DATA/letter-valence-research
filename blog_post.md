# The math hidden in English words: can letters alone predict sentiment?

*Eight months of one weekend-afternoon's worth of computation. Spoiler: a little.*

I had a question. The literature on "sound symbolism" — the systematic association between phonemes and meaning — has been around since the 1920s. Adelman, Estes & Cossu showed in 2018 that phoneme counts predict human-rated word valence with effect sizes of 1.4% to 4.3% of variance. Aryani et al. found that *acoustic* features of words (formants, intensity, spectral centroid) predict 24% of valence variance in German.

But none of that work used the **letters** of a word. They used phonemes — the sounds, not the shapes. There's a small but real research question hiding in the gap: **if the sounds of a word carry sentiment signal, do the letters also?**

So I sat down to answer it. This post is what I found, and what it means.

## TL;DR

A random forest trained on **68 features computed purely from the letters of each word** — alphabet position, modular arithmetic, bigram statistics, vowel ratios, phonetic features from CMUdict, spectral (Fourier) analysis, compression statistics — reaches **0.7377 ± 0.0058 accuracy** on the FinancialPhraseBank binary sentiment task (1,967 financial sentences labeled positive or negative). That's significantly above the 0.693 class-prior baseline (permutation test, p < 0.0001) and it beats VADER at its default threshold. It does *not* beat a properly-tuned VADER, and it does *not* beat FinBERT. It runs at 50,000 articles per second on a single CPU core.

The interesting finding is **where the signal lives**: not in the modular arithmetic (the "gematria" idea — A=1, B=2, sum mod 9, etc.) but in the **spectral domain**. The Discrete Fourier Transform of the letter-position sequence is the single strongest feature family, and the gematria-style features all fail to reach statistical significance. This was a surprise to me — I went in expecting modular arithmetic to be the natural thing to test.

## What I built

I implemented 68 features in 12 families:

1. **Alphabet position aggregations** (8): `alpha_sum`, `alpha_mean`, `alpha_min`, `alpha_max`, `alpha_range`, `alpha_sum_mod9`, `alpha_sum_mod26`, `alpha_sum_parity`.
2. **Modular arithmetic** (9): `sum_mod3`, `sum_mod7`, `sum_mod11`, `is_prime_sum`, `mod26_cyclic_cos`, `mod26_cyclic_sin`, `digital_root`, `sum_mod5`, `sum_mod4`.
3. **Letter frequency** (3): `letter_freq_mean`, `letter_freq_sum`, `rare_letter_count`.
4. **Bigram statistics** (2): `bigram_unique_ratio`, `trigram_count`.
5. **Phonetic** (10): from CMUdict, the proportion of each word's phonemes in each of 10 categories (vowels, plosives, fricatives, voiced/voiceless, front/back/high/low vowel positions).
6. **Shape** (8): vowel ratio, consonant ratio, distinct letter ratio, plosive/fricative/nasal/liquid counts.
7. **Length** (2): word length, log word length.
8. **Group attractors** (2): alphabet centeredness, letter position skew.
9. **Spectral / DFT** (8): Fourier power at frequencies 1, 2, 3; total power; high-frequency ratio; spectral entropy; autocorrelation at lag 1 and 2.
10. **Compression** (3): gzip size, gzip size per char, gzip ratio vs random.
11. **Number-theoretic** (4): letter product mod 26, word value mod 9/26, Mispar Hechrechi (Hebrew gematria, 1-to-1 mapping).
12. **Symmetry / run-length** (9): palindrome, prefix-suffix equality, symmetry density, max/mean/n runs, run-length entropy, first/last letter log-probability.

For each article, I aggregated per-word features with mean, max, min, and standard deviation across the words, producing a 272-dimensional vector. Then a random forest on those vectors. Five-fold stratified cross-validation. Permutation test for significance. Learning curve to see whether more data would help. Family ablation to see which families matter.

The headline result, in one sentence: **the random forest hits 0.7377 accuracy, which is 14 standard deviations above the permutation null distribution's mean of 0.684.**

## What doesn't work

This is the part the literature doesn't emphasize, so let me spell it out.

The "gematria" features — modular arithmetic on the alphabet position sum, including mod 9 (the digital root), primality, and even the standard Hebrew gematria tradition applied 1-to-1 to English letters — **do not predict sentiment**. The best of them (`period_mod5`, r = -0.027, p = 1.6e-3) is just at the edge of Bonferroni significance for n=13,914 words. Most are nowhere close: `alpha_sum_mod9` has p = 0.99. `digital_root` has p = 0.24.

At the article level, removing the entire F2 (modular) family from the model *improves* the CV accuracy by 0.005. The F11 (number-theoretic / gematria) family has the same effect. These features are not just weak — they're actively unhelpful, providing noise that the random forest has to learn to ignore.

This is a real, falsifiable negative result. If you're a researcher looking for "the math in words", the math is **not** in the modular arithmetic. (I tried. I have the figures.)

## What does work (and why)

The single strongest family is **F9: spectral analysis** — the Discrete Fourier Transform of the letter-position sequence. Take a word, map its letters to numbers (A=1, B=2, ..., Z=26), treat that as a time series, and compute the DFT. The first few power-spectrum coefficients (especially the power at frequency 1, "dft_power_k1") correlate with valence at r = -0.0307, p = 2.9e-4.

Trained on **only** this family, the random forest reaches 0.7346 accuracy — within 0.4 percentage points of the full 68-feature model. This is the most important finding for anyone thinking about what "math in words" means in practice: the useful signal is **spectral**, not **algebraic**.

Why spectral? I don't have a strong theoretical explanation. The DFT of a short, irregular integer sequence has no obvious linguistic interpretation. Possible hypotheses:

- The DFT captures the **shape of the letter sequence** — patterns like "consonant-vowel-consonant-vowel" produce specific spectral signatures.
- The autocorrelation at lag 1 (whether adjacent letters are similar-positioned) might be a proxy for consonant clusters.
- It might be that the first Fourier component is essentially capturing **mean letter position**, which is just `alpha_mean`, and the other components are capturing structure I haven't identified.

The honest answer is that the spectral signal works empirically, and the theoretical reason is a future research question.

## Bias-variance: are we feature-bound or data-bound?

The other important finding is the **learning curve**. I trained the model on increasing fractions of the dataset (10%, 20%, 30%, 50%, 70%, 90%, 100%) and measured CV accuracy at each size.

| Training set | CV accuracy |
|---|---|
| 196 articles | 0.699 |
| 393 | 0.707 |
| 590 | 0.727 |
| 983 | 0.715 |
| 1,376 | 0.735 |
| 1,770 | 0.740 |
| 1,967 | 0.737 |

The curve rises from 0.699 to 0.737 over the first ~600 articles. From n=1,376 to n=1,967 — a 43% increase in training data — the accuracy *gain* is **+0.002**. That's well within the standard error of the CV estimate.

**We are in a bias-dominated regime.** The bottleneck is features, not data. Adding 10,000 more labeled financial sentences would not improve this model. Adding 10 more informative features might.

For a research community that mostly assumes "more data fixes everything", this is a useful counter-example. Sometimes the right next step is to add a different kind of feature, not more rows in the same table.

## How this compares to VADER and FinBERT

The standard comparison points:

- **VADER** (rule-based lexicon): 0.678 at default threshold, 0.750 when threshold is tuned to -0.05.
- **FinBERT** (BERT-base, finance-tuned, from the literature): ~0.87.
- **My letter formula** (random forest, 68 features): 0.7377.

So the letter formula is in the middle: better than VADER default, comparable to VADER-tuned, much worse than FinBERT. But it's also *much* faster — FinBERT is 100-1000× slower on CPU. The honest use case is **fast pre-filtering**: process 50k articles/second with the letter formula, route the ~20% that the formula is uncertain about to the more expensive model. The formula will be right about 74% of the time, the LLM will be right about 87% of the time, and you get the union of both with much lower total latency.

## What this tells us about language

I want to be honest about the limitations. The 0.7377 accuracy is real but small. The 0.49 correlation at the word level (best |r| = 0.049) is real but tiny. **The signal exists, but it doesn't replace a good sentiment lexicon or a transformer.**

What I find genuinely interesting is the *kind* of signal that works. The spectral family (F9) is the strongest — but it shouldn't be, in any natural sense. Letters don't have a "frequency" in the way phonemes do. There's no acoustic reality that the Discrete Fourier Transform is picking up. The fact that it works suggests that there are statistical regularities in letter sequences that correlate with meaning, and that these regularities are best captured in the frequency domain.

Maybe this is the computational fingerprint of the *synaesthetic* effects that the original sound-symbolism research (Sapir 1929, Köhler 1929) was studying. The shapes of letters — the visual gestalt of a word — might have statistical regularities that the brain picks up on without our being consciously aware of it. The DFT of the letter sequence is a way of compressing those statistical regularities into a small number of features.

This is speculation. I have no way to test it without running the same analysis on perceptual ratings — does a human looking at "xylophone" vs "hate" vs "love" have systematically different low-level visual processing of the letter shapes, in the way the first Fourier coefficient captures? Maybe. Maybe not. The data doesn't tell us.

But it's the kind of question that wouldn't have been worth asking without first establishing that there *is* a signal to explain. And the signal is small enough that I want to be careful not to overclaim.

## What I would do next

1. **Compare against a character-level CNN.** A character-CNN should beat hand-crafted features on most NLP tasks. If it does, my features are providing *complementary* signal that the CNN can use. If it doesn't, then either the CNN is poorly designed or the features are overfit to FPB.
2. **Add learned features.** Freeze a small character-level LSTM, take the embeddings, concatenate with the hand-crafted features, and re-train the random forest. This is the "neural + symbolic" hybrid approach.
3. **SHAP analysis.** Right now we know which *families* matter (via ablation) but not which *feature interactions*. SHAP values would tell us, e.g., that "high vowel ratio *combined with* low dft_power_k1 is the predictive signature of positive-valence words."
4. **Test transfer.** Train on FPB, test on a different labeled corpus. If the formula transfers, the signal is genuinely about sentiment, not about FPB-specific idiosyncrasies. If it doesn't, we have an FPB-specific model.
5. **Test other languages.** The CMUdict is English-only. The bigram frequencies are English-only. The feature definitions themselves are language-agnostic (the spectral family, the modular family, the shape family all work for any alphabetic script), but the corpus-derived features (letter frequency) are not.

## Where to find the code

Everything is open:
- **GitHub**: [link to be added when you create the repo]
- **Reproduction**: `python -m src.analyze` in the repo root
- **Notebook walkthrough**: `notebooks/01_reproduce_main_result.ipynb`
- **Tests**: 33 unit tests, all passing in 1.7s
- **Headline result**: 0.7377 ± 0.0058 5-fold CV accuracy, p < 0.0001 vs permutation null

The repository is dual-licensed: MIT for code, CC-BY-4.0 for prose and figures. The data is from open-access sources (Warriner 2013, Malo et al. 2014, CMUdict, dwyl/english-words). Cite the underlying datasets if you use the work.

## What I learned

The lesson I'd offer a younger version of myself: **don't start with the most popular feature family**. I went in expecting gematria to work — it has the romance of numerology, the appeal of being a "secret math" — and the data said no. The math that *does* work is the boring, well-known, statistically-boring Discrete Fourier Transform. The result is less interesting as a story but more interesting as science.

I also learned that **honest negative results are publishable**. The "gematria doesn't work" finding is a useful piece of information for anyone thinking about the same question. It would have been easy to bury it in a "we tested many things, here's the best one" narrative. The honest version is "we tested 68 things, here's what worked and what didn't, and here's why the things that didn't work are still informative."

If you have questions, comments, or — especially — if you try this on a different dataset and find something different, I'd love to hear from you.
