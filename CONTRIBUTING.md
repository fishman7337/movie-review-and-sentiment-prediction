# Contributing

Thank you for improving Movie Sentiment RNN. This project is both an academic submission artifact and an engineering repo, so changes should be careful, traceable, and reproducible.

## Development Setup

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Install notebook dependencies only when running the deep learning notebooks:

```bash
python -m pip install -r requirements-notebook.txt
```

## Workflow

1. Keep `DELE_CA1_B.ipynb` as the preserved original unless an explicit academic correction is required.
2. Put reusable logic in `src/movie_sentiment_rnn/`.
3. Add or update tests in `tests/` for code changes.
4. Run `pytest`, `ruff check .`, and `bandit -r src scripts -ll`.
5. Document changes that affect data paths, model artifacts, metrics, or experiment reproduction.

## Notebook Changes

The split notebooks under `notebooks/chapters/` are generated from the original notebook:

```bash
python scripts/split_notebook.py
```

If the original notebook changes, regenerate the chapter notebooks and review the diff before committing.

## Data and Artifacts

Do not commit raw datasets, processed datasets, trained models, checkpoints, or generated reports. Keep them in the local artifact folders documented in `.env.example` and `docs/mlops.md`.
