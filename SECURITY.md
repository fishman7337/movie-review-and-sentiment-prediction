# Security Policy

## Supported Scope

Security checks cover the reusable Python package, scripts, GitHub Actions workflows, and dependency documentation. The original coursework notebook is preserved as an academic artifact and may include Colab-specific installation cells.

## Reporting a Vulnerability

Please report suspected vulnerabilities privately to the repository owner. Include:

- Affected file, dependency, or workflow.
- Steps to reproduce the issue.
- Impact and suggested mitigation, if known.

Do not publish exploit details in public issues before a fix is available.

## Security Checks

The repository includes:

- Bandit static analysis in CI.
- Ruff linting to catch unsafe or fragile patterns.
- Dependabot configuration for dependency and GitHub Actions update alerts.
- Optional `pip-audit` for local dependency audits.

Run locally:

```bash
bandit -r src scripts -ll
pip-audit -r requirements.txt -r requirements-dev.txt
```

## Secrets Handling

Do not commit `.env`, API keys, tokens, model registry credentials, Google Drive credentials, or private dataset links. Use `.env.example` as the public template only.
