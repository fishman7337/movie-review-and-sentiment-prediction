import pytest

from movie_sentiment_rnn.metrics import (
    classification_metrics,
    regression_metrics,
    regression_report,
)


def test_regression_metrics_contains_expected_values() -> None:
    metrics = regression_metrics([1.0, 2.0, 3.0], [1.0, 2.5, 2.5])

    assert metrics["mae"] == pytest.approx(1 / 3)
    assert metrics["rmse"] == pytest.approx((0.5 / 3) ** 0.5)
    assert set(metrics) == {"mae", "mse", "rmse", "mape", "r2"}


def test_regression_report_returns_dataframe() -> None:
    report = regression_report([1.0, 2.0], [1.0, 2.0])

    assert list(report["Metric"]) == ["MAE", "MSE", "RMSE", "MAPE", "R2"]


def test_classification_metrics_contains_expected_values() -> None:
    metrics = classification_metrics([0, 1, 1, 0], [0, 1, 0, 0])

    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(2 / 3)
