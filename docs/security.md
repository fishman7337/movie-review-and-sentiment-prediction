# Security Notes

## Repository Checks

CI runs:

- `ruff check .`
- `bandit -r src scripts -ll`
- `pytest`

Dependabot is configured for Python dependencies and GitHub Actions.

## Notebook Runtime

The original notebook contains Colab installation commands and uses external model downloads for augmentation and visualisation. When running it:

- Use a controlled environment.
- Avoid storing credentials in notebooks.
- Do not commit mounted drive paths or private links.
- Review downloaded model licenses before reuse.

## Secrets

Secrets belong in local environment variables or external secret stores. They must not be committed to Git.
