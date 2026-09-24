# Data

This directory holds the datasets used in the analysis. All data files **are
checked into git** (including the several-MB ones) so a fresh clone reproduces
everything offline; `./download.sh` re-fetches the source files if you ever need
to rebuild them from upstream.

## Files

| File | Source | Size | Used for |
|---|---|---|---|
| `warriner2013.csv` | Warriner, Kuperman & Brysbaert (2013), J. Behavior Research Methods (XANEW mirror, fetched as `Ratings_Warriner_et_al.csv` — `src/data.py::load_warriner` accepts both names) | 3.7 MB | Word-level valence / arousal / dominance for 13,915 English lemmas |
| `articles_binary.csv` | Malo et al. (2014) FinancialPhraseBank, GitHub mirror | 670 KB | Article-level positive/negative sentiment for 1,967 financial sentences |
| `words_alpha.txt` | dwyl/english-words GitHub repo | 4.2 MB | 370k-word English word list — used to build letter unigram + bigram frequency tables |
| `letter_freqs.json` | derived from `words_alpha.txt` | 8 KB | Pre-computed letter unigram + bigram counts and probabilities (input to bigram-surprisal features) |
| `cmudict.dict` | CMU Sphinx project, cmudict.dict | 3.6 MB | CMU Pronouncing Dictionary — 135k words with phonetic transcriptions (input to phonetic features) |
| `newsmtsc/train.jsonl` | Hamborg et al. (2021), NewsMTSC | 3.3 MB | General-news 3-class sentiment, train split (7,758 sentences, 5-coder labelled) |
| `newsmtsc/devtest_rw.jsonl` | Hamborg et al. (2021), NewsMTSC | 0.4 MB | Held-out real-world test split (1,067 sentences) — cascade cross-domain eval |
| `cascade_test/news_test.ndjson` | Google News RSS (12 tickers, 2026-09-24) | 124 KB | 240 live headlines for the cascade evaluation (`src/cascade_eval.py`) |
| `cascade_test/gold_labels.json` | LLM judge, synthetic corpus (2026-09-24) | 5 KB | Gold labels for the 240 headlines: 92 neutral / 89 positive / 59 negative. **Synthetic labels, not independent ground truth; single judge, no adjudication** (meta block in the file) |
| `cascade_test/sentiment_cascade_results.csv` | derived by `src/cascade_eval.py` | 129 KB | Per-headline tier valences, routing and labels |

## How to re-download

```bash
cd data
./download.sh
```

`download.sh` fetches the **4 source downloads** (Warriner mirror, FinancialPhraseBank
50Agree split, english-words list, cmudict) and rebuilds the **2 derived files**
(`letter_freqs.json`, `articles_binary.csv`). The script is idempotent: existing
files are skipped.

Two datasets are **not** fetched by the script and ship committed only:

- `newsmtsc/` — NewsMTSC train + devtest splits (see `data/newsmtsc/readme.md` for sources)
- `cascade_test/` — the live-headline corpus and gold labels (produced by `src/cascade_eval.py`,
  whose RSS cache doubles as the raw data)

## How to regenerate the derived file

`letter_freqs.json` is derived from `words_alpha.txt` by the script
`src/data.py::build_letter_frequencies()`. If you want to rebuild it:

```bash
python -m src.data --rebuild-letter-freqs
```

## Provenance

All sources are open access:

- **Warriner et al. 2013**: Standard affective norms, widely redistributed. We use the JULIELab/XANEW mirror because the original OSF URL has been re-purposed.
- **FinancialPhraseBank**: Created by Malo, Sinha, Korhonen, Wallenius, Takala (2014). The `Sentences_50Agree.txt` split (50% inter-annotator agreement) is the standard benchmark.
- **NewsMTSC**: Created by Hamborg, Breitinger, Schubotz, Gipp (2021), EACL. 5-coder-labelled sentence-level sentiment from real-world news (AllSides). MIT licence, see `data/newsmtsc/readme.md`. Note: NewsMTSC is **target-dependent** — each sentence's primary target polarity is used as the sentence-level label in our cascade evaluation.
- **dwyl/english-words**: Open-source list of ~370k English words.
- **CMU Pronouncing Dictionary**: Public domain, maintained by Carnegie Mellon University.

## Licence of the data

- Warriner: research use, cite the original paper.
- FinancialPhraseBank: research use, cite Malo et al.
- dwyl/english-words: MIT licence.
- CMUdict: public domain.
- Our `letter_freqs.json` (derived): MIT, see `LICENSE` in the repo root.

## Citation

```bibtex
@article{warriner2013norms,
  author = {Warriner, Amy Beth and Kuperman, Victor and Brysbaert, Marc},
  title = {Norms of valence, arousal, and dominance for 13,915 English lemmas},
  journal = {Behavior Research Methods},
  volume = {45},
  number = {4},
  pages = {1191--1207},
  year = {2013}
}

@inproceedings{malo2014good,
  author = {Malo, Pekka and Sinha, Ankur and Korhonen, Pekka and Wallenius, Jyrki and Takala, Pyry},
  title = {Good debt or bad debt: Detecting semantic orientations in economic texts},
  booktitle = {Journal of the Association for Information Science and Technology},
  year = {2014}
}
```
