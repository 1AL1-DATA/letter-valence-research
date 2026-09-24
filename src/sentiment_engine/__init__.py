"""Sentiment cascade engines.

``engine``         — the shipped 2-tier cascade: cheap word tier → FinancialBERT →
                     VADER → keyword fallback (see engine.py for provenance).
``legacy_engine``  — the deprecated 3-tier letter cascade (DFT probe → letter RF →
                     FinancialBERT), kept for side-by-side evaluation only.
"""
