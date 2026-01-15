.PHONY: setup
setup:
	uv sync --all-extras
	uv run --with pre-commit pre-commit install
	@echo "\033[32m✓ Setup complete\033[0m"

.PHONY: format
format:
	uv run ruff format .
	uv run ruff check --fix .
	@echo "\033[32m✓ Formatting complete\033[0m"
