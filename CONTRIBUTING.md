# Contributing

Thanks for your interest in this project. There are several ways to contribute,
ordered from easiest to most involved.

## Reporting a bug

Open an issue on the GitHub issue tracker with:
- A minimal reproducible example (the script, the data subset, the actual output, the expected output).
- Your Python and dependency versions (`python --version`, `pip freeze`).
- The full traceback if the script crashes.

## Fixing a bug

Open a pull request with:
- A test that fails before your fix and passes after.
- A clear commit message in the imperative mood ("fix off-by-one in evaluate.py", not "fixed bug").
- No unrelated formatting changes.

## Adding a feature

1. Open an issue first to discuss. This is a small project; we want to make sure
   new features fit the existing architecture before you spend time on them.
2. Add unit tests in `tests/`.
3. Update `docs/architecture.md` if the feature changes the data flow.
4. Update the `CHANGELOG.md`.

## Adding a new feature family

The most natural extension. To add a new family:

1. Define the features in `src/features.py`. Add a comment block identifying
   the family (F13_*, F14_*, ...). Add the new family to the
   `FEATURE_FAMILIES` dict in `src/evaluate.py`.
2. Add unit tests in `tests/test_features.py`. Each new feature should have at
   least one test with a hand-computed expected value.
3. Re-run `python -m src.analyze`. The `family_ablation.csv` and
   `single_family.csv` results will pick up the new family automatically.
4. Update the README's "What the features are" table.

## Adding a new dataset

1. Add a loader function to `src/data.py` with a clear contract: returns a
   DataFrame with specific columns.
2. Add a download URL to `data/download.sh`.
3. Update the analyze script (`src/analyze.py`) to use the new dataset.
4. Add the dataset to `data/README.md` with citation and license.

## Adding a new model

1. Add a factory in `src/train.py::make_model()`.
2. The model must implement `.fit(X, y)` and `.predict(X)`. Optionally
   `.predict_proba(X)`.
3. Add a CV comparison row to `analyze.py::main()`.

## Coding style

- Type hints on all public functions.
- Docstrings on all public functions. Use the Google or NumPy style — pick
   one and be consistent.
- Tests for all non-trivial functions.
- The `code` field in `src/evaluate.py` should not import directly from
   `src/features.py` — always go through `src/features.get_feature_names()`.

## What we will *not* accept

- Breaking changes to the public API of `src/features.py::features()` (this
  breaks every downstream analysis).
- Refactoring that doesn't change behavior or performance (this is research
  code; readability and reproducibility matter more than DRY).
- New dependencies that aren't already in the standard scientific stack
  (numpy, scipy, pandas, scikit-learn, matplotlib, jupyter). If you need
  something exotic, please discuss in an issue first.

## Code of conduct

Be kind, be patient, and assume good faith. This is a research artifact;
the goal is to share findings honestly, not to win arguments.
