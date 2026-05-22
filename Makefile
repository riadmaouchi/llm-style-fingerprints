.PHONY: help install results fast test lint clean

PYTHON := .venv/bin/python
PIP    := .venv/bin/pip
PYTEST := .venv/bin/pytest
RUFF   := .venv/bin/ruff

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "  install   Create .venv and install all dependencies"
	@echo "  results   Generate all figures in results/"
	@echo "  fast      Generate figures, skipping slow t-SNE (< 30 s)"
	@echo "  test      Run pytest suite (21 tests)"
	@echo "  lint      Run ruff on src/ and tests/"
	@echo "  clean     Remove __pycache__ and .pyc files"

install:
	python3 -m venv .venv
	$(PIP) install --upgrade pip -q
	$(PIP) install -e ".[dev,notebooks]" -q
	@echo "Done — activate with: source .venv/bin/activate"

results:
	$(PYTHON) generate_results.py

fast:
	$(PYTHON) generate_results.py --fast

test:
	$(PYTEST) tests/ -v --tb=short

lint:
	$(RUFF) check src/ tests/

clean:
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name ".pytest_cache" -type d -exec rm -rf {} +
	find . -name "*.egg-info" -type d -exec rm -rf {} +
