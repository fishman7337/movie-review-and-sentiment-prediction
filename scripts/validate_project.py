from __future__ import annotations

from pathlib import Path

REQUIRED_PATHS = [
    "notebooks/DELE_CA1_B.ipynb",
    "README.md",
    ".env.example",
    "pyproject.toml",
    "src/movie_sentiment_rnn",
    "tests",
    ".github/workflows/ci.yml",
]


def main() -> int:
    missing = [path for path in REQUIRED_PATHS if not Path(path).exists()]
    if missing:
        for path in missing:
            print(f"Missing required project path: {path}")
        return 1
    print("Project structure validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
