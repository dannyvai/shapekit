"""Entry point: `uv run main.py <command> ...` is the same as `uv run python -m shapekit <command> ...`."""
from shapekit.cli import main

if __name__ == "__main__":
    main()
