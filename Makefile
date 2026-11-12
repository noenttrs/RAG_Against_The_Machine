.PHONY: install run debug clean lint lint-strict test

install:
	uv sync

run:
	uv run python -m src index
	uv run python -m src search --query "How do I load a LoRA adapter?" --k 5

debug:
	uv run python -m pdb -m src

clean:
	find . -name "__pycache__" -type d -not -path "./.venv/*" -prune -exec rm -rf {} +
	rm -rf .mypy_cache .pytest_cache

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict

test:
	uv run pytest -q
