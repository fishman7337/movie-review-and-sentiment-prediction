import json

import pandas as pd

from movie_sentiment_rnn.cli import main
from movie_sentiment_rnn.config import ProjectConfig


def test_project_config_from_env_file(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("RANDOM_SEED=123\nMAX_SEQUENCE_LENGTH=88\n", encoding="utf-8")
    monkeypatch.delenv("RANDOM_SEED", raising=False)
    monkeypatch.delenv("MAX_SEQUENCE_LENGTH", raising=False)

    config = ProjectConfig.from_env(env_file)

    assert config.random_seed == 123
    assert config.max_sequence_length == 88


def test_cli_quality_report_outputs_json(tmp_path, capsys) -> None:
    csv_path = tmp_path / "reviews.csv"
    pd.DataFrame({"Review": ["A very good movie"], "Score": [0.2]}).to_csv(
        csv_path,
        index=False,
    )

    exit_code = main(["quality-report", str(csv_path)])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["row_count"] == 1
