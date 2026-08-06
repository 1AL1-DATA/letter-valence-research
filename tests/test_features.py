"""Unit tests for the feature extractor.

Run with:
    cd /tmp/letter-valence-research
    python -m pytest tests/

Or:
    python -m unittest tests/test_features.py
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

# Add src/ to path so tests can import without installing
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.features import (  # noqa: E402
    aggregate_article,
    clean,
    compute_features_for_words,
    features,
    get_feature_names,
)


class TestClean(unittest.TestCase):
    def test_lowercases(self):
        self.assertEqual(clean("HELLO"), "hello")

    def test_strips_punctuation(self):
        self.assertEqual(clean("don't"), "dont")

    def test_strips_numbers(self):
        self.assertEqual(clean("a1b2"), "ab")

    def test_unicode_is_stripped(self):
        self.assertEqual(clean("café"), "caf")

    def test_empty(self):
        self.assertEqual(clean(""), "")

    def test_punctuation_only(self):
        self.assertEqual(clean("!!!"), "")


class TestFeaturesReturnType(unittest.TestCase):
    def test_returns_dict(self):
        f = features("hello")
        self.assertIsInstance(f, dict)

    def test_returns_none_for_empty(self):
        self.assertIsNone(features(""))
        self.assertIsNone(features("!!!"))

    def test_consistent_keys(self):
        """Every word should produce the same set of keys."""
        names = get_feature_names()
        for w in ["a", "the", "happy", "sad", "love", "hate", "wonderful", "xylophone", "tr"]:
            f = features(w)
            self.assertIsNotNone(f, f"features({w!r}) returned None")
            self.assertEqual(set(f.keys()), set(names), f"Key mismatch for {w!r}")

    def test_deterministic(self):
        f1 = features("wonderful")
        f2 = features("wonderful")
        self.assertEqual(f1, f2)

    def test_letter_order_does_not_matter(self):
        """Permuting letters should change order-sensitive features but not order-insensitive ones."""
        # 'az' vs 'za' should have same alpha_sum but different first/last letter
        fab = features("az")
        fba = features("za")
        # Order-insensitive: same alpha_sum, same alpha_max, same alpha_min
        self.assertEqual(fab["alpha_sum"], fba["alpha_sum"])
        self.assertEqual(fab["alpha_max"], fba["alpha_max"])  # 26 either way
        self.assertEqual(fab["alpha_min"], fba["alpha_min"])  # 1 either way
        # Order-sensitive: first letter probability differs (high for 'a' start vs low for 'z' start)
        self.assertNotEqual(fab["first_letter_lp"], fba["first_letter_lp"])
        self.assertNotEqual(fab["last_letter_lp"], fba["last_letter_lp"])

    def test_drops_to_lowercase(self):
        f_upper = features("HELLO")
        f_lower = features("hello")
        self.assertEqual(f_upper, f_lower)


class TestFeatureValues(unittest.TestCase):
    """Pin specific feature values to known outputs.

    These are regression tests: if a future refactor changes the math,
    these will catch it.
    """

    def setUp(self):
        # Force the letter freqs to be loaded
        from src.features import _get_letter_freqs
        _get_letter_freqs()

    def test_alpha_sum(self):
        # 'happy' = 8+1+16+16+25 = 66
        self.assertAlmostEqual(features("happy")["alpha_sum"], 66.0)

    def test_alpha_sum_simple(self):
        # 'a' = 1
        self.assertAlmostEqual(features("a")["alpha_sum"], 1.0)
        # 'z' = 26
        self.assertAlmostEqual(features("z")["alpha_sum"], 26.0)

    def test_word_length(self):
        self.assertEqual(features("happy")["word_length"], 5.0)
        self.assertEqual(features("a")["word_length"], 1.0)
        self.assertAlmostEqual(features("happy")["log_word_length"], math.log1p(5))

    def test_vowel_consonant(self):
        # 'happy' has 2 vowels (a, y) and 3 consonants
        f = features("happy")
        self.assertAlmostEqual(f["vowel_ratio"], 2/5)
        self.assertAlmostEqual(f["consonant_ratio"], 3/5)

    def test_vowel_all(self):
        # 'a' has 1 vowel, 0 consonants
        f = features("a")
        self.assertAlmostEqual(f["vowel_ratio"], 1.0)
        self.assertAlmostEqual(f["consonant_ratio"], 0.0)

    def test_vowel_none(self):
        # 'bcdfg' has 0 vowels (no a/e/i/o/u/y) and 5 consonants
        f = features("bcdfg")
        self.assertAlmostEqual(f["vowel_ratio"], 0.0)
        self.assertAlmostEqual(f["consonant_ratio"], 1.0)

    def test_rhythm_keeps_y_vowel(self):
        # 'rhythms' has 1 vowel (the y) out of 6 letters
        f = features("rhythms")
        self.assertAlmostEqual(f["vowel_ratio"], 1/7)  # 1/7
        # But the canonical vowels (a, e, i, o, u) are absent
        # The vowel count is just the y
        self.assertGreater(f["vowel_ratio"], 0.0)

    def test_modular(self):
        # 'happy' sum = 66. 66 mod 9 = 3. 66 mod 26 = 14.
        f = features("happy")
        self.assertAlmostEqual(f["alpha_sum_mod9"], 66 % 9)
        self.assertAlmostEqual(f["alpha_sum_mod26"], 66 % 26)
        self.assertAlmostEqual(f["alpha_sum_parity"], 66 % 2)
        self.assertAlmostEqual(f["sum_mod3"], 66 % 3)
        self.assertAlmostEqual(f["sum_mod7"], 66 % 7)
        self.assertAlmostEqual(f["sum_mod11"], 66 % 11)

    def test_is_prime(self):
        self.assertEqual(features("ab")["is_prime_sum"], 1.0)   # 1+2=3 (prime)
        self.assertEqual(features("abc")["is_prime_sum"], 0.0)  # 1+2+3=6 (not prime)
        self.assertEqual(features("a")["is_prime_sum"], 0.0)    # 1 (not prime per our impl)

    def test_digital_root(self):
        # digital_root(66) = 1 + (66-1) % 9 = 1 + 65 % 9 = 1 + 2 = 3
        self.assertAlmostEqual(features("happy")["digital_root"], 3.0)
        self.assertAlmostEqual(features("a")["digital_root"], 1.0)

    def test_palindrome(self):
        self.assertEqual(features("aba")["is_palindrome"], 1.0)
        self.assertEqual(features("abc")["is_palindrome"], 0.0)
        self.assertEqual(features("a")["is_palindrome"], 1.0)
        self.assertEqual(features("aa")["is_palindrome"], 1.0)

    def test_distinct_letters(self):
        self.assertAlmostEqual(features("aabbc")["distinct_letter_ratio"], 3/5)
        self.assertAlmostEqual(features("abcde")["distinct_letter_ratio"], 1.0)

    def test_run_length(self):
        # 'aabbc' has runs [2, 2, 1]
        f = features("aabbc")
        self.assertAlmostEqual(f["max_run_length"], 2.0)
        self.assertAlmostEqual(f["n_runs"], 3.0)
        self.assertAlmostEqual(f["mean_run_length"], 5/3)
        # 'aaaaa' has runs [5]
        f2 = features("aaaaa")
        self.assertAlmostEqual(f2["max_run_length"], 5.0)
        self.assertAlmostEqual(f2["n_runs"], 1.0)
        self.assertAlmostEqual(f2["mean_run_length"], 5.0)

    def test_gzip_size(self):
        # 'happy' compressed should be small
        f = features("happy")
        self.assertGreater(f["gzip_size"], 20)  # at least the header
        self.assertLess(f["gzip_size"], 100)    # but small for a 5-letter word
        # 'aaaaaaaaaa' should be smaller per char (high redundancy)
        f_repeat = features("aaaaaaaaaa")
        self.assertLess(f_repeat["gzip_size_per_char"], f["gzip_size_per_char"])

    def test_alphacenteredness(self):
        # Perfectly centered word (all m's = 13) should have max centeredness = 0
        f_m = features("mmmmm")  # all positions = 13, sum (p-13.5)^2 = 5*0.25 = 1.25
        # Expected: -1.25 / 5 = -0.25
        self.assertAlmostEqual(f_m["alphabet_centeredness"], -0.25, places=5)

    def test_dft_short_word(self):
        """Words with L < 4 should have all spectral features = 0."""
        f = features("abc")
        for k in ["dft_power_k1", "dft_power_k2", "dft_power_k3",
                  "dft_high_freq_ratio", "dft_total_power",
                  "dft_spectral_entropy", "autocorr_lag1", "autocorr_lag2"]:
            self.assertEqual(f[k], 0.0, f"{k} should be 0 for L<4")

    def test_dft_long_word(self):
        """A long word should have non-zero DFT values."""
        f = features("wonderful")  # 9 letters
        self.assertGreater(f["dft_total_power"], 0)
        self.assertNotEqual(f["dft_power_k1"], 0)

    def test_phonetic_unknown_word(self):
        """For words not in CMUdict, phonetic features should be 0."""
        # 'asdfgh' is not in CMUdict
        f = features("asdfgh")
        for k in ["phon_n", "phon_vowel_ratio", "phon_plosive_ratio"]:
            self.assertEqual(f[k], 0.0)


class TestBulkAndAggregation(unittest.TestCase):
    def test_bulk_unique(self):
        out = compute_features_for_words(["a", "b", "a", "c", "b"])
        # Should have 3 unique keys: a, b, c
        self.assertEqual(set(out.keys()), {"a", "b", "c"})

    def test_aggregation_shapes(self):
        words = ["the", "happy", "sad"]
        out = compute_features_for_words(words)
        n = len(get_feature_names())
        for strategy, expected_size in [
            ("mean", n), ("max", n), ("min", n), ("std", n),
            ("sign_count_pos", n), ("all", 4 * n),
        ]:
            v = aggregate_article(words, out, strategy=strategy)
            self.assertEqual(v.shape, (expected_size,),
                             f"strategy={strategy} produced shape {v.shape}, expected {(expected_size,)}")

    def test_aggregation_empty_words(self):
        """Empty word list should return zeros, not crash."""
        out = compute_features_for_words(["the", "happy"])
        v = aggregate_article([], out, strategy="all")
        self.assertEqual(v.shape, (4 * len(get_feature_names()),))
        self.assertTrue((v == 0).all())


if __name__ == "__main__":
    unittest.main()
