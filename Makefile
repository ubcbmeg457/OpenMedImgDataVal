SHELL := /bin/bash

MODULES := $(shell find . -maxdepth 2 -name "pyproject.toml" \
	-not -path "./pyproject.toml" \
	-not -path "*/.venv/*" \
	-not -path "*/venv/*" | \
	sed 's|^./||; s|/pyproject.toml||' | sort)

.PHONY: help
help:
	@echo "Makefile targets:"
	@echo ""
	@echo "Setup:"
	@echo "  make setup          - Install all dependencies (uv sync)"
	@echo "  make kernel         - Register Jupyter kernels for all modules"
	@echo "  make kernel MODULE=<name> - Register kernel for a specific module"
	@echo ""
	@echo "Development:"
	@echo "  make format         - Format and lint code (ruff, auto-fix)"
	@echo "  make lint           - Check formatting and lint (no auto-fix)"
	@echo "  make clean          - Clean all venvs and caches"
	@echo ""
	@echo "Notebooks:"
	@echo "  make notebook MODULE=<name> - Start Jupyter Lab for a module"
	@grep -E "^notebook-" Makefile | sed 's/:.*//; s/^/  make /'
	@echo ""
	@echo "Available modules: $(MODULES)"

# ---------- Setup ----------

.PHONY: setup
setup:
	uv sync --all-packages --all-extras
	@echo "\033[32m✓ Dependencies installed\033[0m"
	@echo "Run 'make kernel' to register Jupyter kernels for JupyterHub."

# Register ipykernel for JupyterHub / remote notebook servers.
# Each workspace member gets its own kernel that uses the workspace venv.
.PHONY: kernel
kernel:
	@if [ -n "$(MODULE)" ]; then \
		echo "Registering kernel for $(MODULE)..."; \
		uv run python -m ipykernel install --user \
			--name "$(MODULE)" \
			--display-name "Python ($(MODULE))"; \
		echo "\033[32m✓ Kernel '$(MODULE)' registered\033[0m"; \
	else \
		for mod in $(MODULES); do \
			echo "Registering kernel for $$mod..."; \
			uv run python -m ipykernel install --user \
				--name "$$mod" \
				--display-name "Python ($$mod)"; \
		done; \
		echo "\033[32m✓ All kernels registered\033[0m"; \
	fi
	@echo "Installed kernels:"
	@uv run jupyter kernelspec list

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

# ---------- Notebooks ----------

.PHONY: notebook
notebook:
	@if [ -z "$(MODULE)" ]; then \
		echo "Error: MODULE not specified"; \
		echo "Usage: make notebook MODULE=<module_name>"; \
		echo ""; \
		echo "Available modules: $(MODULES)"; \
		exit 1; \
	fi
	@if [ ! -f "$(MODULE)/pyproject.toml" ]; then \
		echo "Error: $(MODULE)/pyproject.toml not found"; \
		exit 1; \
	fi
	uv sync --all-packages --all-extras
	@echo "Starting Jupyter Lab for $(MODULE)..."
	cd $(MODULE) && uv run --project .. jupyter lab

# Convenience shortcuts
.PHONY: notebook-xray-shapley
notebook-xray-shapley:
	$(MAKE) notebook MODULE=xray-shapley

.PHONY: notebook-prototype
notebook-prototype:
	$(MAKE) notebook MODULE=prototype
