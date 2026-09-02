# PHAROS — development tasks.
.DEFAULT_GOAL := help
.PHONY: help install install-dev data corpus index build evaluate redteam test test-cov lint format typecheck check clean docker

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package
	pip install -e .

install-dev:  ## Install with dev and provider extras
	pip install -e ".[dev,anthropic,openai]"
	-pre-commit install

data:  ## Download and verify the corpus
	pharos fetch-data

corpus:  ## Clean, subsample, segment, label
	pharos build-corpus

index:  ## Fit the encoder and build the indices
	pharos build-index

build: data corpus index  ## Full pipeline from nothing to a queryable index

evaluate:  ## Run the ablation study and regenerate docs/RESULTS.md
	pharos evaluate

redteam:  ## Run the safety suite
	pharos redteam

test:  ## Run the test suite
	pytest -q

test-cov:  ## Run tests with a coverage report
	pytest --cov=pharos --cov-report=term-missing

lint:  ## Lint
	ruff check src tests

format:  ## Format and autofix
	ruff format src tests
	ruff check --fix src tests

typecheck:  ## Type-check
	mypy src/pharos

check: lint typecheck test  ## Everything CI runs

docker:  ## Build the container image
	docker build -t pharos-rx:latest .

clean:  ## Remove build and cache artifacts
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
