# Tests

Unit tests for the feature extractor and other modules.

Run with (from the repo root):
```bash
python -m pytest tests/ -v
# or, without pytest:
python -m unittest discover tests/ -v
```

The tests are designed to be runnable without downloading the data
(they test pure-function behaviour). Tests that require data files
are skipped if the data is not present.
