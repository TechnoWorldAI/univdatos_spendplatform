"""Small delegator so callers can run the sample-data loader from the refactored `src/` tree.

This file re-uses the existing top-level `scripts/load_sample_data.py` implementation
so we don't duplicate logic while allowing `uv run python src/scripts/load_sample_data.py`.
"""

from scripts.load_sample_data import load_sample_data


def main():
    load_sample_data()


if __name__ == "__main__":
    main()
