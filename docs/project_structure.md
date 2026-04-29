# Project Structure

```text
.
├── DELE_CA1_B.ipynb
├── notebooks/
│   └── chapters/
├── src/
│   └── movie_sentiment_rnn/
├── tests/
├── scripts/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── models/
├── reports/
│   ├── figures/
│   └── metrics/
├── docs/
└── .github/workflows/
```

## Key Areas

`DELE_CA1_B.ipynb` is the preserved original notebook.

`notebooks/chapters/` contains generated notebooks split by original chapter headings. These should be regenerated using `scripts/split_notebook.py`.

`src/movie_sentiment_rnn/` contains reusable code for configuration, data validation, text preprocessing, model builders, metrics, plotting, and CLI commands.

`tests/` contains lightweight pytest coverage for the reusable logic. Tests avoid training deep learning models.

`data/`, `models/`, and `reports/` are local artifact zones. Their contents are ignored by Git except `.gitkeep` placeholders.

`.github/workflows/` contains CI and security workflows.
