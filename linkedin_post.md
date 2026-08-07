# LinkedIn post (short-form)

**Goal**: ~250 words. Warm, honest, research-sharing. Narrative arc from question → hypothesis → findings → meaning. GitHub URL will be filled in after the repo is pushed.

---

## The version to copy-paste into LinkedIn

---

A question I couldn't stop thinking about.

There's a long literature on "sound symbolism" — the finding that the sounds of a word carry measurable sentiment signal. Sapir in 1929, Köhler in 1947, Adelman and Aryani more recently. But all of it uses phonemes: the sounds. Not the shapes.

That left a gap: if the sounds of a word carry sentiment signal, do the letters also?

I started with a simple hypothesis: what if there is something mathematically regular in the letter sequences of positive and negative words? A=1, B=2, Z=26 — does summing, averaging, or doing modular arithmetic on those numbers tell you anything? This is essentially what gematria traditions have claimed for centuries. I tested it rigorously: modular arithmetic, primality, digital roots, Hebrew gematria applied to English letters. None of it survived statistical correction.

But something else did.

The Discrete Fourier Transform of a word's letter-position sequence — treating A=1, B=2 as a time series and looking at its frequency content — turned out to be the single strongest signal. This was unexpected. A Fourier transform of a short alphabetic sequence shouldn't mean anything linguistically. But the model trained on only this spectral family reaches 0.7346 accuracy, within 0.4 percentage points of the full 68-feature model.

The final result on the FinancialPhraseBank: 0.7377 ± 0.0058 accuracy, F1 = 0.835, permutation p < 0.0001. At ~50,000 classifications per second on CPU, the honest practical use is as a fast pre-filter before a heavier model — reducing compute in large-scale financial document pipelines.

So I built the cascade: the word-level tier decides only when confident (|v| ≥ 0.6), everything else falls through to a BERT. On the FinancialPhraseBank it matches the transformer alone — 0.9512 vs 0.9558 accuracy (McNemar p = 0.15, not significant) — while the cheap tier carries 36.6% of decisions at 97.2% accuracy.

Then I asked whether the cascade is a finance-specific trick. On a general-news benchmark (NewsMTSC, 1,067 held-out news sentences, not finance), the cheap tier trained only on financial text still transfers: 0.42 accuracy, 0.49 when retrained on news. With a general-domain BERT the cascade reaches 0.62, beating the BERT alone (0.58). The honest wrinkle: the finance-tuned heavy collapses out-of-domain (0.30, below chance). The cheap tier and the routing logic are the general part — the heavy model is the domain-locked part.

Code, tests, and all figures are in the first comment.

Questions and discussion welcome.

#NLP #SentimentAnalysis #MachineLearning #ReproducibleResearch

---

## Behind-the-scenes notes

- **No external links in the main post body.** LinkedIn suppresses posts with links. Put the link in the FIRST COMMENT.
- **Hashtags**: 5. More than 6 is noise.
- **Character count**: ~2,610 characters — within LinkedIn's 3,000 limit.
- **Tone**: warmer than the previous version. Personal voice, research narrative arc. "A question I couldn't stop thinking about" opens with the personal motivation. "None of it survived" is honest and a little wry. "This was unexpected" conveys the actual research experience. "An honest practical use" avoids overselling.
- The narrative arc: question → gematria hypothesis → spectral discovery → honest conclusion → cascade follow-up (cheap tier + BERT fallback) → cross-domain generalisation on general news → domain-locked heavy caveat.
- The chain of reasoning is the point: the research is not just a random forest, it is a sequence of hypotheses tested and revised. The cascade section continues that arc — each stage asks the previous result a harder question (does it beat the transformer? is the trick finance-only?).

## First comment

> Full paper, code, 33 unit tests, and all figures: https://github.com/1AL1-DATA/letter-valence-research
>
> To reproduce: `pip install -r requirements.txt && python -m src.analyze && python -m src.visualise && python -m src.train_final`
>
> One-line summary: tested 68 letter-derived features, gematria-style modular arithmetic fails Bonferroni correction, spectral (Fourier) features are the strongest single family, 0.7377 ± 0.0058 accuracy on FPB binary sentiment.
>
> Cascade follow-up: a 2-tier cascade (cheap word tier + FinancialBERT fallback) matches the transformer alone on FPB (0.9512 vs 0.9558, McNemar p = 0.15) while the cheap tier carries 36.6% of decisions at 97.2% accuracy. On general news (NewsMTSC, 1,067 held-out sentences) the cheap tier trained only on finance transfers (0.42), a general-domain BERT cascade reaches 0.62, and the finance-tuned heavy collapses out-of-domain (0.30).
>
> The spectral finding is genuinely interesting: the Discrete Fourier Transform of a letter-position sequence is not supposed to mean anything linguistically. I don't have a clean theoretical explanation. Worth thinking about.
