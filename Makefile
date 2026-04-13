SHELL := /bin/bash

.PHONY: help
help:
	@echo "Makefile targets:"
	@echo ""
	@echo "Setup:"
	@echo "  make setup          - Install all dependencies (uv sync)"
	@echo ""
	@echo "Development:"
	@echo "  make format         - Format and lint code (ruff, auto-fix)"
	@echo "  make lint           - Check formatting and lint (no auto-fix)"
	@echo "  make clean          - Clean all venvs and caches"

# ---------- Setup ----------

.PHONY: setup
setup:
	uv sync --all-packages --all-extras
	@echo "\033[32m✓ Dependencies installed\033[0m"

# ---------- Development ----------

.PHONY: format
format:
	uv run ruff format .
	uv run ruff check --fix .
	@echo "\033[32m✓ Formatting complete\033[0m"

.PHONY: lint
lint:
	uv run ruff format --check .
	uv run ruff check .

.PHONY: clean
clean:
	@echo "Cleaning up..."
	find . -type d -name ".venv" -not -path "*/.git/*" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "venv" -not -path "*/.git/*" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "\033[32m✓ Cleaned up\033[0m"

