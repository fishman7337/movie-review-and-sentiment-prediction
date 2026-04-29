# MLOps Guide

## Goals

The MLOps layer makes the coursework project easier to reproduce, test, and extend without changing the original notebook.

## Lifecycle

1. Place raw data in `data/raw/`.
2. Run data quality checks before preprocessing.
3. Write cleaned data to `data/processed/`.
4. Train models with fixed seeds and recorded hyperparameters.
5. Save model artifacts to `models/`.
6. Save metrics and figures to `reports/`.
7. Compare experiments before selecting the final model.

## Artifact Naming

Use descriptive names:

```text
models/gru_classification_aug.keras
models/lstm_regression_no_aug.keras
reports/metrics/gru_classification_aug.json
reports/figures/gru_classification_confusion_matrix.png
```

## Configuration

Use `.env` for local paths and runtime settings. `.env.example` documents the expected variables without storing secrets.

Important values:

- `DATA_RAW_PATH`
- `DATA_PROCESSED_DIR`
- `MODEL_DIR`
- `REPORT_DIR`
- `RANDOM_SEED`
- `MAX_WORDS`
- `MAX_SEQUENCE_LENGTH`
- `MLFLOW_TRACKING_URI`

## Experiment Tracking

MLflow is optional. If enabled, log:

- Git commit SHA.
- Dataset version or checksum.
- Preprocessing settings.
- Model architecture and hyperparameters.
- Training curves.
- Validation and test metrics.
- Final model artifact path.

## Quality Gates

Before publishing changes:

```bash
ruff check .
bandit -r src scripts -ll
pytest
```

Before accepting a trained model:

- Confirm train, validation, and test splits are reproducible.
- Evaluate classification with accuracy, confusion matrix, precision, recall, and F1.
- Evaluate regression with MAE, RMSE, MAPE, and R2.
- Compare augmented and non-augmented runs.
- Save metrics and plots under `reports/`.

## Data Governance

Do not commit raw data or model artifacts. Keep only documented code, config templates, and reproducible instructions in Git.
