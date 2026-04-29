.PHONY: install install-notebook test lint security split-notebooks checks

install:
	python -m pip install --upgrade pip
	python -m pip install -e ".[dev]"

install-notebook:
	python -m pip install -r requirements-notebook.txt

test:
	pytest

lint:
	ruff check .

security:
	bandit -r src scripts -ll

split-notebooks:
	python scripts/split_notebook.py

checks: lint security test
