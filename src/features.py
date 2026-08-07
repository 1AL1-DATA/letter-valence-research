"""Letter-derived numerical features of words.

This module computes 57 features per word, organised into 11 families:

  F1 Alphabet position aggregations       (7 features)
  F2 Group-theoretic / modular arithmetic (9 features)
  F3 Letter frequency / corpus statistics (3 features)
  F4 Bigram statistics                    (2 features)
  F5 Phonetic / articulatory             (10 features, requires CMUdict)
  F6 Vowel/consonant shape                (3 features)
  F7 Word length                          (2 features)
  F8 Group attractors / original          (2 features)
  F9 Spectral / DFT                       (8 features)
  F10 Compression / Kolmogorov            (3 features)
  F11 Number-theoretic (gematria-like)    (4 features)
  F12 Symmetry / run-length / prefix-suffix (8 features)

The `features(word)` entry point returns a dict; missing features (e.g. phonetic
for OOV words) are returned as 0.0.

Tested on 13,914 Warriner lemmas + 4,929 unique words from the FinancialPhraseBank
binary split. See `tests/test_features.py` for unit tests.
"""
from __future__ import annotations

import gzip
import json
import math
import re
from pathlib import Path

import numpy as np

# ---- constants ----
VOWELS: set[str] = set("aeiouy")
PLOSIVES: set[str] = set("pbtdkg")
FRICATIVES: set[str] = set("fvszh")
NASALS: set[str] = set("mn")
LIQUIDS: set[str] = set("lr")

# CMUdict phoneme categories (stress digits stripped, e.g. AE1 -> AE)
_VOWELS_CMU: set[str] = set("AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW".split())
_PLOSIVES_CMU: set[str] = set("P T K B D G".split())
_FRICATIVES_CMU: set[str] = set("F V TH DH S Z SH ZH HH".split())
_NASALS_CMU: set[str] = set("M N NG".split())
_LIQUIDS_CMU: set[str] = set("L R".split())
_GLIDES_CMU: set[str] = set("W Y".split())
_VOICED: set[str] = set("B D G V DH Z ZH J L R W Y M N NG".split())
_VOICELESS: set[str] = set("P T K F TH S SH CH HH".split())
_FRONT_V: set[str] = set("IY IH EY EH AY AE".split())
_BACK_V: set[str] = set("UW UH OW AO AA".split())
_HIGH_V: set[str] = set("IY IH UW UH".split())
_LOW_V: set[str] = set("AE AA".split())

# Default paths
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

# Lazy-loaded state
_LETTER_FREQS: dict | None = None
_CMU: dict | None = None
_FEATURE_NAMES: list[str] | None = None


# ---- text utilities ----
def clean(word: str) -> str:
    """Strip non-letters, lowercase. Returns the cleaned word."""
    return re.sub(r"[^a-z]", "", word.lower())


# ---- main entry point ----
def features(
    word: str,
    letter_freqs: dict | None = None,
    cmu: dict | None = None,
) -> dict | None:
    """Compute all 57 letter-derived features for a single word.

    Returns None for empty words, otherwise a dict with the feature names
    as keys and float values.

    Parameters
    ----------
    word : str
        The word to compute features for. Can be in mixed case; will be
        lowercased and stripped of non-alphabetic characters.
    letter_freqs : dict, optional
        Letter frequency table (from `src.data.build_letter_frequencies`).
        If None, will be loaded lazily from `data/letter_freqs.json`.
    cmu : dict, optional
        CMUdict mapping word -> list of phoneme sequences. If None,
        will be loaded lazily from `data/cmudict.dict`.
    """
    w = clean(word)
    if not w:
        return None

    if letter_freqs is None:
        letter_freqs = _get_letter_freqs()
    if cmu is None:
        cmu = _get_cmu()

    p1 = _prob_unigrams(letter_freqs)
    positions = [ord(c) - ord('a') + 1 for c in w]
    s = sum(positions)
    n_letters = len(positions)
    letters = list(w)

    feats: dict[str, float] = {}

    # ---- F1: alphabet position aggregations ----
    feats["alpha_sum"] = float(s)
    feats["alpha_mean"] = float(s / n_letters)
    feats["alpha_max"] = float(max(positions))
    feats["alpha_min"] = float(min(positions))
    feats["alpha_range"] = float(max(positions) - min(positions))
    feats["alpha_sum_mod9"] = float(s % 9)
    feats["alpha_sum_mod26"] = float(s % 26)
    feats["alpha_sum_parity"] = float(s % 2)

    # ---- F2: group-theoretic / modular ----
    feats["sum_mod3"] = float(s % 3)
    feats["sum_mod7"] = float(s % 7)
    feats["sum_mod11"] = float(s % 11)
    feats["is_prime_sum"] = float(_is_prime(s))
    feats["mod26_cyclic_cos"] = float(math.cos(2 * math.pi * (s % 26) / 26))
    feats["mod26_cyclic_sin"] = float(math.sin(2 * math.pi * (s % 26) / 26))
    feats["digital_root"] = float(_digital_root(s))
    feats["sum_mod5"] = float(s % 5)
    feats["sum_mod4"] = float(s % 4)

    # ---- F3: letter frequency / corpus stats ----
    lf = [letter_freqs.get("unigrams", {}).get(c, 0) for c in letters]
    feats["letter_freq_mean"] = float(np.mean(lf)) if lf else 0.0
    feats["letter_freq_sum"] = float(np.sum(lf)) if lf else 0.0
    feats["rare_letter_count"] = float(sum(1 for c in letters if c in "qzxjk"))

    # ---- F4: bigram statistics ----
    bigrams = [letters[i] + letters[i+1] for i in range(n_letters-1)]
    feats["bigram_unique_ratio"] = float(len(set(bigrams)) / max(1, len(bigrams)))
    feats["trigram_count"] = float(max(0, n_letters-2))

    # ---- F5: phonetic / CMUdict ----
    feats.update(_phonetic_features(w, cmu))

    # ---- F6: vowel/consonant shape ----
    n_vow = sum(1 for c in letters if c in VOWELS)
    n_con = n_letters - n_vow
    feats["vowel_ratio"] = float(n_vow / n_letters)
    feats["consonant_ratio"] = float(n_con / n_letters)
    feats["distinct_letter_ratio"] = float(len(set(letters)) / n_letters)
    feats["plosive_count"] = float(sum(1 for c in letters if c in PLOSIVES))
    feats["plosive_ratio"] = float(sum(1 for c in letters if c in PLOSIVES) / n_letters)
    feats["fricative_count"] = float(sum(1 for c in letters if c in FRICATIVES))
    feats["nasal_count"] = float(sum(1 for c in letters if c in NASALS))
    feats["liquid_count"] = float(sum(1 for c in letters if c in LIQUIDS))

    # ---- F7: word length ----
    feats["word_length"] = float(n_letters)
    feats["log_word_length"] = float(math.log1p(n_letters))

    # ---- F8: group attractors (original) ----
    feats["alphabet_centeredness"] = float(-sum((p - 13.5) ** 2 for p in positions) / n_letters)
    feats["letter_position_skew"] = float(sum(positions) / (n_letters * 13.5) - 1.0)

    # ---- F9: spectral / DFT ----
    feats.update(_spectral_features(positions))

    # ---- F10: compression / Kolmogorov ----
    gz = len(gzip.compress(w.encode("utf-8")))
    feats["gzip_size"] = float(gz)
    feats["gzip_size_per_char"] = float(gz / n_letters) if n_letters > 0 else 0.0
    expected_random = n_letters * math.log2(26) / 8
    feats["gzip_ratio_vs_random"] = float(gz / expected_random) if expected_random > 0 else 1.0

    # ---- F11: number-theoretic (gematria-like) ----
    prod = 1
    for c in w:
        prod = (prod * (ord(c) - ord('a') + 1)) % 26
    feats["letter_product_mod26"] = float(prod)
    val26 = 0
    for c in w:
        val26 = val26 * 26 + (ord(c) - ord('a'))
    feats["word_value_mod_9"] = float(val26 % 9)
    feats["word_value_mod_26"] = float(val26 % 26)
    # Mispar Hechrechi (standard Hebrew gematria tradition, applied 1-to-1)
    feats["mispar_hechrechi_sum"] = float(sum(_mispar_hechrechi(c) for c in w))

    # ---- F12: symmetry / run-length / prefix-suffix ----
    feats["is_palindrome"] = float(w == w[::-1])
    feats["prefix_eq_suffix"] = (
        float(w[:n_letters//2] == w[-n_letters//2:]) if n_letters >= 2 else 0.0
    )
    matches = sum(1 for i in range(n_letters) for j in range(i+1, n_letters) if w[i] == w[j])
    feats["symmetry_density"] = float(matches / max(1, n_letters*(n_letters-1)/2))
    # Run-length
    runs: list[int] = []
    if n_letters > 0:
        cur = 1
        for i in range(1, n_letters):
            if w[i] == w[i-1]:
                cur += 1
            else:
                runs.append(cur)
                cur = 1
        runs.append(cur)
    feats["max_run_length"] = float(max(runs) if runs else 0)
    feats["n_runs"] = float(len(runs))
    feats["mean_run_length"] = float(np.mean(runs)) if runs else 0.0
    feats["run_length_entropy"] = float(
        _entropy([r / sum(runs) for r in runs]) if len(runs) > 1 else 0.0
    )
    # First/last letter probability under unigram model
    feats["first_letter_lp"] = float(math.log2(p1.get(w[0], 1e-10))) if n_letters > 0 else 0.0
    feats["last_letter_lp"] = float(math.log2(p1.get(w[-1], 1e-10))) if n_letters > 0 else 0.0

    return feats


def get_feature_names() -> list[str]:
    """Return the list of feature names in the order produced by `features()`.

    Computed lazily on first call. This is the canonical way to enumerate features.
    """
    global _FEATURE_NAMES
    if _FEATURE_NAMES is None:
        f = features("a")
        _FEATURE_NAMES = list(f.keys()) if f else []
    return _FEATURE_NAMES


# ---- internal helpers ----
def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.isqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True


def _digital_root(n: int) -> int:
    if n == 0:
        return 0
    return 1 + (n - 1) % 9


def _entropy(p: list[float]) -> float:
    return -sum(pi * math.log2(pi) for pi in p if pi > 0)


def _mispar_hechrechi(c: str) -> int:
    """Standard Hebrew gematria (Mispar Hechrechi), applied 1-to-1 to an English letter.

    Letters 1-9 -> values 1-9, letters 10-18 -> 10-90, letters 19-26 -> 100-800.
    """
    v = ord(c) - ord('a') + 1
    if v <= 9:
        return v
    if v <= 18:
        return (v - 9) * 10
    return (v - 18) * 100


def _phonetic_features(word: str, cmu: dict) -> dict:
    """Compute phonetic features from CMUdict.

    Returns all-zero dict if the word is not in the dictionary.
    """
    feats = {
        "phon_n": 0.0, "phon_vowel_ratio": 0.0, "phon_plosive_ratio": 0.0,
        "phon_fricative_ratio": 0.0, "phon_voiceless_ratio": 0.0, "phon_voice_ratio": 0.0,
        "phon_front_ratio": 0.0, "phon_back_ratio": 0.0, "phon_high_ratio": 0.0,
        "phon_low_ratio": 0.0,
    }
    if not cmu or word not in cmu:
        return feats
    phon = cmu[word][0]
    n_p = len(phon)
    if n_p == 0:
        return feats
    n_vow = sum(1 for p in phon if p in _VOWELS_CMU)
    feats["phon_n"] = float(n_p)
    feats["phon_vowel_ratio"] = n_vow / n_p
    feats["phon_plosive_ratio"] = sum(1 for p in phon if p in _PLOSIVES_CMU) / n_p
    feats["phon_fricative_ratio"] = sum(1 for p in phon if p in _FRICATIVES_CMU) / n_p
    feats["phon_voiceless_ratio"] = sum(1 for p in phon if p in _VOICELESS) / n_p
    feats["phon_voice_ratio"] = sum(1 for p in phon if p in _VOICED) / n_p
    n_front = sum(1 for p in phon if p in _FRONT_V)
    n_back = sum(1 for p in phon if p in _BACK_V)
    n_high = sum(1 for p in phon if p in _HIGH_V)
    n_low = sum(1 for p in phon if p in _LOW_V)
    feats["phon_front_ratio"] = n_front / max(1, n_vow)
    feats["phon_back_ratio"] = n_back / max(1, n_vow)
    feats["phon_high_ratio"] = n_high / max(1, n_vow)
    feats["phon_low_ratio"] = n_low / max(1, n_vow)
    return feats


def _spectral_features(positions: list[int]) -> dict:
    """Compute the DFT and autocorrelation features.

    For very short words (n_letters < 4) all features are 0.
    """
    n_letters = len(positions)
    feats = {
        "dft_power_k1": 0.0, "dft_power_k2": 0.0, "dft_power_k3": 0.0,
        "dft_high_freq_ratio": 0.0, "dft_total_power": 0.0,
        "dft_spectral_entropy": 0.0, "autocorr_lag1": 0.0, "autocorr_lag2": 0.0,
    }
    if n_letters < 4:
        return feats
    sig = np.array(positions, dtype=float)
    sig = sig - sig.mean()  # remove DC component
    f = np.fft.fft(sig)
    psd = np.abs(f) ** 2
    if n_letters > 1:
        feats["dft_power_k1"] = float(psd[1])
    if n_letters > 2:
        feats["dft_power_k2"] = float(psd[2])
    if n_letters > 3:
        feats["dft_power_k3"] = float(psd[3])
    feats["dft_high_freq_ratio"] = float(psd[n_letters//2:].sum() / max(1, psd.sum()))
    feats["dft_total_power"] = float(psd.sum())
    p = psd[1:] / max(1e-12, psd[1:].sum())
    feats["dft_spectral_entropy"] = float(-np.sum(p * np.log2(p + 1e-12)))
    mean = sig.mean()
    var = sig.var()
    if var > 0:
        if n_letters >= 3:
            feats["autocorr_lag1"] = float(np.mean((sig[:-1] - mean) * (sig[1:] - mean)) / var)
        if n_letters >= 4:
            feats["autocorr_lag2"] = float(np.mean((sig[:-2] - mean) * (sig[2:] - mean)) / var)
    return feats


# ---- data loading (lazy) ----
def _get_letter_freqs() -> dict:
    global _LETTER_FREQS
    if _LETTER_FREQS is None:
        path = DATA_DIR / "letter_freqs.json"
        if path.exists():
            with open(path) as f:
                _LETTER_FREQS = json.load(f)
        else:
            _LETTER_FREQS = {"unigrams": {}, "bigrams": {}}
    return _LETTER_FREQS


def _get_cmu() -> dict:
    global _CMU
    if _CMU is None:
        path = DATA_DIR / "cmudict.dict"
        if path.exists():
            _CMU = _parse_cmudict(path)
        else:
            _CMU = {}
    return _CMU


def _parse_cmudict(path: Path) -> dict:
    """Parse the CMU Pronouncing Dictionary.

    Format per line: WORD [stress_marked_phoneme1 stress_marked_phoneme2 ...]
    Stress marks are digits 0/1/2. We strip them.
    """
    out: dict[str, list[list[str]]] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";;;"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            word = parts[0].lower()
            # strip alternative pronunciation markers like (2)
            word = re.sub(r"\(\d+\)$", "", word)
            phonemes = [re.sub(r"\d", "", p) for p in parts[1:]]
            out.setdefault(word, []).append(phonemes)
    return out


def _prob_unigrams(letter_freqs: dict) -> dict:
    """Return a {letter: probability} dict from the unigram counts."""
    total = letter_freqs.get("total_unigrams", sum(letter_freqs.get("unigrams", {}).values())) or 1
    return {c: n / total for c, n in letter_freqs.get("unigrams", {}).items()}


# ---- aggregation strategies ----
def aggregate_article(
    words: list[str],
    features_per_word: dict[str, dict],
    feats: list[str] | None = None,
    strategy: str = "all",
) -> np.ndarray:
    """Aggregate per-word features into a single vector for an article.

    Strategies:
        "mean"  - mean of each feature across words
        "max"   - max
        "min"   - min
        "std"   - standard deviation
        "all"   - concatenate [mean, max, min, std]
        "sign_count_pos" - fraction of words with each feature > 0

    Returns a 1D numpy array. The output size is:
        len(FEATURE_NAMES)             for mean/max/min/std/sign_count_pos
        4 * len(FEATURE_NAMES)         for "all"
    """
    if feats is None:
        feats = get_feature_names()
    per_word = []
    for w in words:
        if w in features_per_word:
            per_word.append([features_per_word[w].get(n, 0.0) for n in feats])
    if not per_word:
        size = len(feats) * 4 if strategy == "all" else len(feats)
        return np.zeros(size)
    arr = np.array(per_word)
    if strategy == "mean":
        return arr.mean(axis=0)
    if strategy == "max":
        return arr.max(axis=0)
    if strategy == "min":
        return arr.min(axis=0)
    if strategy == "std":
        return arr.std(axis=0) if len(arr) > 1 else np.zeros(arr.shape[1])
    if strategy == "sign_count_pos":
        return (arr > 0).mean(axis=0)
    if strategy == "all":
        return np.concatenate([
            arr.mean(axis=0),
            arr.max(axis=0),
            arr.min(axis=0),
            arr.std(axis=0) if len(arr) > 1 else np.zeros(arr.shape[1]),
        ])
    raise ValueError(f"Unknown strategy: {strategy}")


# ---- bulk computation ----
def compute_features_for_words(
    words: list[str],
    letter_freqs: dict | None = None,
    cmu: dict | None = None,
    show_progress: bool = False,
) -> dict[str, dict]:
    """Compute features for a list of words. Returns {word: feature_dict}.

    Skips empty words. Dedupes via set for fast bulk computation.
    """
    if letter_freqs is None:
        letter_freqs = _get_letter_freqs()
    if cmu is None:
        cmu = _get_cmu()
    unique = list(set(w for w in words if w))
    out: dict[str, dict] = {}
    iterator = unique
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(unique, desc="features")
        except ImportError:
            pass
    for w in iterator:
        f = features(w, letter_freqs=letter_freqs, cmu=cmu)
        if f is not None:
            out[w] = f
    return out
