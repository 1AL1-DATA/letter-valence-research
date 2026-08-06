"""Data loading and preparation for the letter-valence analysis.

This module handles:
- Loading Warriner 2013 affective norms
- Loading the FinancialPhraseBank (binary version, 50% agreement)
- Building letter unigram + bigram frequency tables from a word list
- Tokenizing article text into words

All paths default to `data/` relative to the repo root, but can be overridden.
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterator

import pandas as pd

# ---- paths ----
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"

# ---- text utilities ----
WORD_RE = re.compile(r"[a-z]+")


def clean_word(word: str) -> str:
    """Strip non-letters, lowercase. Returns the cleaned word."""
    return re.sub(r"[^a-z]", "", word.lower())


def tokenize(text: str) -> list[str]:
    """Tokenize a sentence into lowercase alphabetic words."""
    return WORD_RE.findall(text.lower())


# ---- data loaders ----
def load_warriner(path: Path | str | None = None) -> pd.DataFrame:
    """Load the Warriner 2013 affective norms and add a `word_clean` column.

    Returns a DataFrame with columns:
        word, valence, arousal, dominance, word_clean, n_chars
    Drops rows with missing valence and rows where word_clean is empty.
    """
    if path is None:
        # Try the canonical name first, then the local name
        candidates = [
            DATA_DIR / "Ratings_Warriner_et_al.csv",
            DATA_DIR / "warriner2013.csv",
        ]
        path = next((p for p in candidates if p.exists()), candidates[0])
    df = pd.read_csv(path)
    df = df.dropna(subset=["Word", "V.Mean.Sum"])
    df["word_clean"] = df["Word"].astype(str).apply(clean_word)
    df = df[df["word_clean"].str.len() > 0].copy()
    df["n_chars"] = df["word_clean"].str.len()
    df = df.rename(columns={
        "Word": "word",
        "V.Mean.Sum": "valence",
        "A.Mean.Sum": "arousal",
        "D.Mean.Sum": "dominance",
    })
    return df[["word", "word_clean", "valence", "arousal", "dominance", "n_chars"]].reset_index(drop=True)


def load_articles_binary(path: Path | str | None = None) -> pd.DataFrame:
    """Load the FinancialPhraseBank binary sentiment dataset.

    Source: Sentences_50Agree.txt, latin-1 encoded, '@' separator.
    Drops the neutral class and maps {negative, positive} to {0, 1}.
    Adds a `words` column with the tokenized word list.

    Returns a DataFrame with columns: id, text, label, words
    """
    path = Path(path) if path else DATA_DIR / "articles_binary.csv"
    df = pd.read_csv(path)
    if "words" in df.columns and df["words"].dtype == object:
        # If the words column was stored as a string-repr, fix it
        if isinstance(df["words"].iloc[0], str) and df["words"].iloc[0].startswith("["):
            df["words"] = df["words"].apply(eval)
        else:
            df["words"] = df["text"].apply(tokenize)
    else:
        df["words"] = df["text"].apply(tokenize)
    return df


def load_articles_multiclass(path: Path | str | None = None) -> pd.DataFrame:
    """Load the FinancialPhraseBank 3-class version (negative/neutral/positive).

    Returns a DataFrame with columns: id, text, label, words
    Labels: 0=negative, 1=neutral, 2=positive
    """
    path = Path(path) if path else DATA_DIR / "articles.csv"
    df = pd.read_csv(path)
    if "words" in df.columns and isinstance(df["words"].iloc[0], str):
        df["words"] = df["words"].apply(eval)
    else:
        df["words"] = df["text"].apply(tokenize)
    return df


# ---- derived data builders ----
def build_letter_frequencies(word_list_path: Path | str) -> dict:
    """Build letter unigram and bigram frequency tables from a word list.

    Returns a dict suitable for json.dump:
        {
            "unigrams": {letter: count, ...},
            "bigrams":  {bi: count, ...},
            "total_unigrams": int,
            "total_bigrams":  int,
        }
    """
    word_list_path = Path(word_list_path)
    unigrams: Counter = Counter()
    bigrams: Counter = Counter()
    n_words = 0
    with open(word_list_path) as f:
        for line in f:
            w = line.strip()
            if not w or not w.isalpha():
                continue
            w = w.lower()
            n_words += 1
            for c in w:
                if c.isalpha():
                    unigrams[c] += 1
            for i in range(len(w) - 1):
                if w[i].isalpha() and w[i+1].isalpha():
                    bigrams[w[i] + w[i+1]] += 1
    return {
        "unigrams": dict(unigrams),
        "bigrams":  dict(bigrams),
        "total_unigrams": sum(unigrams.values()),
        "total_bigrams":  sum(bigrams.values()),
        "source": str(word_list_path),
        "n_words": n_words,
    }


def build_articles_binary(source_path: Path | str, output_path: Path | str | None = None) -> pd.DataFrame:
    """Read Sentences_50Agree.txt and write a binary-version CSV.

    Drops the neutral class. Label mapping: negative=0, positive=1.
    Columns: id, text, label
    """
    source_path = Path(source_path)
    if not source_path.exists():
        # Try the local name as a fallback
        local = source_path.parent / "articles_binary.csv"
        if local.exists():
            return pd.read_csv(local)
    rows = []
    label_map = {"negative": 0, "neutral": 1, "positive": 2}
    with open(source_path, encoding="latin-1") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if "@" not in line:
                continue
            text, label = line.rsplit("@", 1)
            label = label.strip()
            if label not in ("negative", "positive"):
                continue
            text = text.strip().strip('"')
            rows.append({"id": i, "text": text, "label": label_map[label]})
    df = pd.DataFrame(rows)
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
    return df


# ---- CLI ----
def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Data utilities for letter-valence analysis")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("rebuild-letter-freqs", help="Rebuild letter_freqs.json from words_alpha.txt")
    p1.add_argument("--words", default=str(DATA_DIR / "words_alpha.txt"))
    p1.add_argument("--out",   default=str(DATA_DIR / "letter_freqs.json"))

    p2 = sub.add_parser("rebuild-articles-binary", help="Rebuild articles_binary.csv from Sentences_50Agree.txt")
    p2.add_argument("--source", default=str(DATA_DIR / "Sentences_50Agree.txt"))
    p2.add_argument("--out",    default=str(DATA_DIR / "articles_binary.csv"))

    args = parser.parse_args()
    if args.cmd == "rebuild-letter-freqs":
        freqs = build_letter_frequencies(args.words)
        with open(args.out, "w") as f:
            json.dump(freqs, f)
        print(f"Wrote {args.out} ({freqs['total_unigrams']} letters, {freqs['total_bigrams']} bigrams)")
    elif args.cmd == "rebuild-articles-binary":
        df = build_articles_binary(args.source, args.out)
        print(f"Wrote {args.out} with {len(df)} rows; label dist: {df['label'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
