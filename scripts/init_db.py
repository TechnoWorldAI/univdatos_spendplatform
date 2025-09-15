#!/usr/bin/env python3
"""Delegator to initialize the database from the `src` package.

This keeps the runtime script under `scripts/` while the implementation
remains in `src/utils/init_db.py`.
"""
import sys
import os

# Ensure repo root is importable when running this script directly
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils.init_db import init_database


def main():
    # Allow optional DB path as first arg
    db_path = None
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    init_database(database_path=db_path)


if __name__ == "__main__":
    main()
