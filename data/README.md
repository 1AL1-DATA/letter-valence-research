# Data

This directory holds the datasets used in the analysis. The files are **not** checked into git
(because the largest ones are several MB) — they are downloaded by `./download.sh`.

## Files

| File | Source | Size | Used for |
|---|---|---|---|
| `Ratings_Warriner_et_al.csv` | Warriner, Kuperman & Brysbaert (2013), J. Behavior Research Methods | 3.7 MB | Word-level valence / arousal / dominance for 13,915 English lemmas |
| `articles_binary.csv` | Malo et al. (2014) FinancialPhraseBank, GitHub mirror | 670 KB | Article-level positive/negative sentiment for 1,967 financial sentences |
| `words_alpha.txt` | dwyl/english-words GitHub repo | 4.2 MB | 370k-word English word list — used to build letter unigram + bigram frequency tables |
| `letter_freqs.json` | derived from `words_alpha.txt` | 8 KB | Pre-computed letter unigram + bigram counts and probabilities (input to bigram-surprisal features) |
| `cmudict.dict` | CMU Sphinx project, cmudict.dict | 3.6 MB | CMU Pronouncing Dictionary — 135k words with phonetic transcriptions (input to phonetic features) |
| `newsmtsc/train.jsonl` | Hamborg et al. (2021), NewsMTSC | 3.3 MB | General-news 3-class sentiment, train split (7,758 sentences, 5-coder labelled) |
| `newsmtsc/devtest_rw.jsonl` | Hamborg et al. (2021), NewsMTSC | 0.4 MB | Held-out real-world test split (1,067 sentences) — cascade cross-domain eval |

## How to download

```bash
cd data
./download.sh
```

This downloads all 5 files (5 source downloads + 1 derivation) and writes them to this directory.
The script is idempotent: if a file already exists, it is skipped.

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
