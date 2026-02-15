SHELL := /bin/bash

.PHONY: help
help:
	@echo "Makefile targets:"
	@echo ""
	@echo "Development:"
	@echo "  make format        - Format and lint code (ruff)"
	@echo "  make clean         - Clean all venvs and cache"
	@echo ""
	@echo "Notebooks (auto-discovered from directories with pyproject.toml):"
	@grep -E "^notebook-" Makefile | sed 's/:.*//; s/^/  make /'
	@echo "  make notebook MODULE=<name>  - Start Jupyter for specific module"
	@echo ""
	@echo "Example: make notebook MODULE=xray-shapley"

.PHONY: format
format:
	uv run ruff format .
	uv run ruff check --fix .
	@echo "\033[32m✓ Formatting complete\033[0m"

.PHONY: clean
clean:
	@echo "Cleaning up..."
	find . -type d -name "venv" -not -path "*/.git/*" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .venv
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "\033[32m✓ Cleaned up\033[0m"

# Notebook targets for each module with pyproject.toml
.PHONY: notebook
notebook:
	@if [ -z "$(MODULE)" ]; then \
		echo "Error: MODULE not specified"; \
		echo "Usage: make notebook MODULE=<module_name>"; \
		echo ""; \
		echo "Available modules:"; \
		find . -maxdepth 2 -name "pyproject.toml" -not -path "*/.venv/*" -not -path "*/venv/*" | \
		xargs grep -l "name = " | sed 's|^./||; s|/pyproject.toml||' | sort; \
		exit 1; \
	fi
	@if [ ! -d "$(MODULE)/notebooks" ]; then \
		echo "Error: $(MODULE)/notebooks not found"; \
		exit 1; \
	fi
	@if [ ! -d "$(MODULE)/.venv" ] && [ ! -d "$(MODULE)/venv" ]; then \
		echo "Setting up $(MODULE) with uv (dependencies from pyproject.toml)..."; \
		cd $(MODULE) && uv sync; \
	fi
	@VENV_DIR=""; if [ -d "$(MODULE)/.venv" ]; then VENV_DIR=".venv"; elif [ -d "$(MODULE)/venv" ]; then VENV_DIR="venv"; fi; \
	echo "Starting Jupyter for $(MODULE)..."; \
	cd $(MODULE) && source $$VENV_DIR/bin/activate && jupyter notebook notebooks/

# Convenience shortcuts for known modules
.PHONY: notebook-xray-shapley
notebook-xray-shapley:
	$(MAKE) notebook MODULE=xray-shapley

.PHONY: notebook-prototype
notebook-prototype:
	$(MAKE) notebook MODULE=prototype
