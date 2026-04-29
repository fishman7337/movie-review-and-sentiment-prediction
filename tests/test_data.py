import pandas as pd
import pytest

from movie_sentiment_rnn.data import (
    build_quality_report,
    clean_review_frame,
    train_validation_test_split,
    validate_review_frame,
)


def test_validate_review_frame_requires_review_and_score() -> None:
    frame = pd.DataFrame({"Review": ["great"]})
    with pytest.raises(ValueError, match="Score"):
        validate_review_frame(frame)


def test_build_quality_report_counts_missing_duplicates_and_short_reviews() -> None:
    frame = pd.DataFrame(
        {
            "Review": ["Great film overall", "Great film overall", "ok", None],
            "Score": [0.1, 0.1, None, 0.7],
        }
    )

    report = build_quality_report(frame, min_words=2)

    assert report.row_count == 4
    assert report.missing_by_column["Review"] == 1
    assert report.missing_by_column["Score"] == 1
    assert report.duplicate_rows == 1
    assert report.short_reviews == 2


def test_clean_review_frame_removes_noise_and_adds_labels() -> None:
    frame = pd.DataFrame(
        {
            "Review": ["This movie was excellent", "bad", "This movie was excellent"],
            "Score": [0.1, 0.9, 0.1],
        }
    )

    cleaned = clean_review_frame(frame, min_words=3)

    assert len(cleaned) == 1
    assert cleaned.loc[0, "Score"] == pytest.approx(0.9)
    assert cleaned.loc[0, "Sentiment_Label"] == "Positive"
    assert cleaned.loc[0, "source"] == "original"


def test_train_validation_test_split_is_reproducible() -> None:
    frame = pd.DataFrame(
        {
            "Review": [f"review {idx}" for idx in range(20)],
            "Score": [0.1, 0.9] * 10,
            "Sentiment_Binary": [0, 1] * 10,
        }
    )

    train, validation, test = train_validation_test_split(
        frame,
        target_column="Sentiment_Binary",
        test_size=0.2,
        validation_size=0.2,
        random_state=7,
    )

    assert len(train) == 12
    assert len(validation) == 4
    assert len(test) == 4
