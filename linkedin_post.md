# LinkedIn post (short-form)

**Goal**: ~250 words. Warm, honest, research-sharing. Narrative arc from question → hypothesis → findings → meaning. GitHub URL will be filled in after the repo is pushed.

---

## The version to copy-paste into LinkedIn

---

A question I couldn't stop thinking about.

Research on "sound symbolism" has shown that the sounds of a word carry measurable sentiment signal. But these studies rely on phonemes — the sounds — rather than the written letters. We asked whether the letters of a word also carry signal.

We tested this on 13,914 words and 1,967 financial sentences from FinancialPhraseBank. First we tested letter-level hypotheses from the gematria tradition — modular arithmetic, primality, digital roots. None survived statistical correction. Then we found one signal that did: the Discrete Fourier Transform of a word's letter-position sequence. A model using only this spectral family reaches 0.7346 accuracy, within 0.4 points of the full 68-feature model (0.7377 \u00b1 0.0058, F1 = 0.835, permutation p < 0.0001).

That accuracy is modest, but the method is fast: ~50,000 classifications per second on CPU. The natural use is as a cheap pre-filter ahead of a more expensive model. So we built a two-tier cascade.

The design decision was what the cheap tier should be. Our letter model ran fast, but it decided only 3.6% of sentences — the expensive tier effectively did all the work. So we rebuilt the cheap tier on words rather than letters: TF-IDF over 1–2 grams, plus VADER and keyword-lexicon signals, through a logistic regression. It decides only when its confidence is high (|v| ≥ 0.6); otherwise the sentence falls through to a BERT.

On FinancialPhraseBank the cascade matched the transformer alone — 0.9512 vs 0.9558 accuracy, McNemar p = 0.15 (not significant) — while the cheap tier carried 36.6% of decisions at 97.2% accuracy, reducing transformer load by roughly a third. In concrete terms, on a one-million-sentence document feed that means roughly 366,000 sentences are decided by the cheap tier at ~50,000 sentences per second on a single CPU — before the transformer is ever invoked — while accuracy is unchanged within statistical noise.

Then we asked whether the cascade was a finance-specific result. On NewsMTSC — 1,067 held-out general-news sentences, no financial text — the cheap tier trained only on finance still transferred (0.42 accuracy; 0.49 when retrained on news). With a general-domain BERT the cascade reached 0.62, ahead of the BERT alone (0.58). One limitation is worth stating plainly: the finance-tuned BERT collapses out-of-domain (0.30, below chance, classifying 65.3% of clear sentences as neutral). The cheap tier and the routing logic generalise; the heavy model is domain-locked.

Code, tests, and all figures are in the first comment.

Questions and discussion welcome.

#NLP #SentimentAnalysis #MachineLearning #ReproducibleResearch

---

## Behind-the-scenes notes

- **No external links in the main post body.** LinkedIn suppresses posts with links. Put the link in the FIRST COMMENT.
- **Hashtags**: 5. More than 6 is noise.
- **Character count**: ~2,670 characters — within LinkedIn's 3,000 limit.
- **Tone**: scientific register, first-person plural, plain-language explanations. Statements are precise ("within statistical noise" rather than "the same"), design decisions are motivated, and limitations are stated plainly ("One limitation is worth stating plainly"). The opening hook ("A question I couldn't stop thinking about") softens the register just enough for social media without sacrificing accuracy.
- The narrative arc: question → sound-symbolism gap → gematria hypothesis (failed) → spectral discovery → cheap pre-filter use case → cascade design decision (rebuild cheap tier from words, not letters, because the letter model was dead weight as a pre-filter) → cascade matches transformer on FPB with a concrete compute-gain scenario → cross-domain generalisation on NewsMTSC → domain-locked heavy caveat.
- The chain of reasoning is the point: the research is not just a random forest, it is a sequence of hypotheses tested and revised. The cascade section continues that arc — letter model ruled out as pre-filter, word-level cheap tier designed, transformer matched with load reduction, domain-generalisation tested honestly, limiting factor identified.

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
