"""Demo panels + artifact consistency (no model imports; parses source as text)."""
import hashlib
import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
INDEX = REPO / "api" / "index.py"


class TestInfoPanels(unittest.TestCase):
    def test_three_details_panels_with_formula_and_paper_link(self):
        src = INDEX.read_text()
        panels = re.findall(r"<details class=\"model-info\">(.*?)</details>", src, re.S)
        self.assertEqual(len(panels), 3, "one <details> panel per model tier")
        for p in panels:
            self.assertIn("<code>", p, "each panel shows its formula")
            links = re.findall(r'href="(https://[^"]+)"', p)
            self.assertTrue(any(l.startswith("https://") for l in links), "paper link per panel")
        self.assertIn("doi.org/10.1108/eb026526", src)  # Sparck Jones 1972
        self.assertIn("doi.org/10.1023/A:1010933404324", src)  # Breiman 2001
        self.assertIn("ojs.aaai.org/index.php/ICWSM/article/view/14550", src)  # VADER

    def test_letter_tier_flagged_research_artifact(self):
        self.assertIn("research artifact", INDEX.read_text().lower())

    def test_vader_banded_label_used_by_demo(self):
        from src.classify import vader_score  # ponytail: only import in this test

        self.assertEqual(vader_score("")["label"], "neutral")  # compound 0.0 abstains
        self.assertEqual(vader_score("I love this!")["label"], "positive")
        self.assertEqual(vader_score("I hate this!")["label"], "negative")
        self.assertIn("d.vader.label", INDEX.read_text(), "demo reads the banded label")


class TestArtifactConsistency(unittest.TestCase):
    def test_cheap_tier_bundles_identical(self):
        a = REPO / "models" / "cheap_tier.pkl"
        b = REPO / "src" / "sentiment_engine" / "assets" / "cheap_tier.pkl"
        ha = hashlib.sha256(a.read_bytes()).hexdigest()
        hb = hashlib.sha256(b.read_bytes()).hexdigest()
        self.assertEqual(ha, hb, "models/ and assets/ cheap_tier.pkl must not drift")


if __name__ == "__main__":
    unittest.main()
