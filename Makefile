.PHONY: setup install run-backend run-frontend run-dev test lint format clean help

PYTHON = .venv/Scripts/python
PIP = .venv/Scripts/pip
ACTIVATE = .venv/Scripts/activate

help:
	@echo "======================================================================="
	@echo "                     Enterprise Hybrid RAG Makefile"
	@echo "======================================================================="
	@echo "setup         - Create virtual environment and install dependencies"
	@echo "install       - Install dependencies in existing environment"
	@echo "run-backend   - Run the FastAPI backend service"
	@echo "run-frontend  - Run the Streamlit frontend interface"
	@echo "run-dev       - Boot backend and frontend in parallel"
	@echo "test          - Execute test suite using pytest"
	@echo "lint          - Perform static analysis checks via ruff"
	@echo "format        - Reformat code using black"
	@echo "clean         - Remove temp files, virtualenv, and build caches"
	@echo "======================================================================="

setup:
	python -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .[dev]

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .[dev]

run-backend:
	$(PYTHON) -m uvicorn src.main:app --reload --port 8000

run-frontend:
	$(PYTHON) run.py

test:
	$(PYTHON) -m pytest tests/ -v --cov=src

lint:
	$(PYTHON) -m ruff check src/ tests/

format:
	$(PYTHON) -m black src/ tests/

clean:
	rm -rf .venv/
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.log" -delete
