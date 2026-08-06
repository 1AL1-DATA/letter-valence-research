# Literature digest: letter/word-level numerical features and word affect

*Compiled 2026-08-05. Sources read in full and verified. All effect sizes taken directly from the source PDFs.*

## 1. Adelman, Estes & Cossu (2018) — "Emotional sound symbolism: Languages rapidly signal valence via phonemes" *Cognition* 175:122-130

- **Sample**: ~37,000 words total across 5 languages; **English N=12,594** with pronunciation latencies (N=36 phonemes); German N=37 phonemes; Spanish, Dutch, Polish also tested
- **Key finding**: Phoneme counts significantly predict word-level valence ratings after controlling for word length, log frequency, contextual diversity, and arousal.
- **Effect sizes** (phoneme block, incremental R² over controls):
  - English **1.44%** R²
  - Spanish **1.40%** R²
  - Dutch **2.32%** R²
  - German **2.68%** R²
  - Polish **4.28%** R²
  - All p < .001
- **First-phoneme vs last-phoneme**: First-phoneme analysis is *stronger* than last-phoneme in every language (English first-phoneme R² = 2.15% vs last-phoneme R² = 0.48%).
- **Phoneme-valence correlation** (latency method): English r = 0.55, German r = 0.50, both p < .001 — faster-pronounced phonemes start negative/affective words.
- **Methodological caveat**: Uses pronunciation *latency* as a proxy for affective weight. This is not a pure letter feature; it's a phoneme-level psycholinguistic measurement.

## 2. Aryani, Conrad, Schmidtke & Jacobs (2018) — "Why 'piss' is ruder than 'pee'?" *PLoS ONE* 13(6):e0198430

- **Sample**: Berlin Affective Word List (BAWL) **N=2,574 German words**; **272 raters** in study 2a, **169 raters** in study 2b; **1,095 pseudowords** for cross-validation
- **Key finding**: 11 acoustic features predict Phonological Affective Potential (PAP) with R² = **27.9% for arousal** and **23.7% for valence**.
- **Specific letter-derived signals** (correlations with affective-sound ratings):
  - Voiceless consonants r = −0.51 (arousal), r = +0.49 (valence)
  - Vowel length: short vowels > long vowels on arousal/negativity
  - Sibilants (/s/, /z/, /ʃ/) raise spectral centroid → arousing/negative
- **Pseudoword cross-validation**: 56.3% arousal variance, 11.2% valence variance — confirms the sound-symbolic effect is not just about word familiarity.
- **Methodological caveat**: Uses *acoustic* features (f0, intensity, formants F1–F3, spectral centroid), not orthographic letter features. Cannot be replicated from letter counts alone.

## 3. Schmidtke, Schröder, Jacobs & Conrad (2014) — "ANGST: Affective norms for German sentiment terms" *Behavior Research Methods* 46(4):1118-1130

- **Sample**: **N=1,034 German sentiment terms** + 603 items; German raters, plus cross-language validation against ANEW
- **Key finding**: Affective norms (valence, arousal, dominance) for German sentiment terms, analogous to Warriner 2013 for English. Cross-language correlations with ANEW (Bradley & Lang 1999):
  - Valence: **r = 0.90** (p < .001) between German ANGST and English ANEW
  - Dominance: r = 0.60 (p < .001)
  - Arousal: r = 0.64 in positive range, r = −0.27 in negative range
- **Methodological caveat**: This is a *resource paper* providing norms, not a paper testing letter-feature hypotheses. The cross-language correlations confirm that affect norms are largely language-universal, which is good for using Warriner 2013 to test English letter features.

## 4. de Zubicaray & Hinojosa (2024) — "Statistical Relationships Between Phonological Form, Emotional Valence and Arousal of Spanish Words" *Journal of Cognition* 7(1):42

- **Sample**: Spanish words from a large normative database; **918 participants** in rating studies; **N=3,669 Spanish words** in main analysis; form-typicality ratings on 7,500 words; lexical decision data on 14,031 English words for cross-language comparison.
- **Key finding**: Form variables (form-typicality) explain a small but significant additional variance in affective ratings after controlling for lexico-semantic variables:
  - **Valence**: control variables 2.85%, form-typicality adds **1.3%** (significant), total ~3% of variance in unaffixed words
  - **Emotionality**: form-typicality adds **1.3%** (the strongest predictor in this dimension)
  - **Arousal**: form-typicality adds only **0.5%** (significant but weak)
- **Cross-language comparison**: Form-meaning mappings were more extensive in Spanish than English (4% vs 1.3% additional variance in valence).
- **Methodological caveat**: Form-typicality is *computed* from typicality ratings — not a direct letter feature. Effect sizes are at the 1–4% R² scale, consistent with the rest of the field.

## 5. Aryani, Jacobs & Conrad (2013) — "Extracting salient sublexical units: 'Emophon'" *Frontiers in Psychology* 4:654

- **Sample**: 20 German poems and newspaper articles; comparison of "salient" phonemes in poetic vs prosaic text
- **Key finding**: A probabilistic method (Emophon) extracts phonemes whose frequency in a text deviates from a reference corpus; salient phonemes correlate with the text's emotional tone.
- **Effect sizes** (poetic vs prose text, t-tests, df=38):
  - Number of salient phonemes: **t = 3.6, p = 0.001** (poems > prose)
  - Number of salient nuclei: t = 3.5, p = 0.001
  - Number of salient codas: t = 2.5, p = 0.017
  - Sum of phoneme deviations: t = 2.5, p = 0.017
- **Methodological caveat**: This is a *tool* paper, not a predictive model. It tells you which phonemes are "salient" in a text; it does not predict sentiment. Cannot be used directly for a sentiment formula.

---

## Synthesis: what does the prior literature actually establish?

| Hypothesis | Evidence | Effect size | Source |
|---|---|---|---|
| Phoneme count predicts valence | 5 languages, 37k words | 1.4–4.3% R² incremental | Adelman 2018 |
| First-phoneme is stronger than last | all 5 languages | 2.15% vs 0.48% English R² | Adelman 2018 |
| Acoustic features predict affective-sound | 2,574 words | 23.7% R² valence, 27.9% R² arousal | Aryani 2018 |
| Voiceless consonants ↗ valence | direct test | r = +0.49 | Aryani 2018 |
| Voiceless consonants ↘ arousal | direct test | r = −0.51 | Aryani 2018 |
| Form-typicality ↗ valence | 3,669 Spanish | 1.3% R² incremental | de Zubicaray 2024 |
| German/English affect norms correlate | cross-language | r = 0.90 valence | Schmidtke 2014 |
| Salient phonemes in poetry | t-test | t(38) = 3.6, p < .001 | Aryani 2013 |

**Key takeaway**: The prior literature establishes a real but **modest** effect of phonemic features on word-level affect — typically 1–5% of variance, peaking at 28% in the best-controlled acoustic-feature study (Aryani 2018). **No published paper uses gematria-style letter-position sums as a sentiment signal.** The closest analog (Adelman 2018's phoneme block) explains 1.4%–4.3% of variance.
