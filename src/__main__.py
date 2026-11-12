"""Entry point: ``uv run python -m src <command> [options]``."""

import sys

import fire

from src.cli import RagCLI


def main() -> None:
    """Run the Fire command line interface."""
    try:
        fire.Fire(RagCLI)
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
