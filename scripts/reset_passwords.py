#!/usr/bin/env python3
"""One-off script to reset all user passwords to a pattern (e.g. username1234).

This script will:
 - Back up the configured SQLite DB to a timestamped file
 - Prompt for confirmation (unless --yes is provided)
 - Update every user's password to the pattern (securely hashed with Argon2 via project code)

Usage:
  python scripts/reset_passwords.py [--pattern "{username}1234"] [--db /path/to/db] [--yes]

Be careful: this is destructive. Backups are made automatically.
"""

import argparse
import datetime
import os
import shutil
import sqlite3
import sys

# Ensure repo root is importable when running this script directly
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import config
from src.utils.init_db import reset_all_user_passwords
from src.utils.debug import debug_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Reset all user passwords (destructive)")
    parser.add_argument("--pattern", default="{username}1234", help="Password pattern where {username} will be replaced")
    parser.add_argument("--db", help="Path to sqlite DB file (overrides config)")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt")

    args = parser.parse_args()

    db_path = args.db or config.database.path

    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        sys.exit(1)

    # Backup
    ts = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
    backup_path = f"{db_path}.backup.{ts}"
    print(f"Backing up database: {db_path} -> {backup_path}")
    shutil.copy2(db_path, backup_path)

    if not args.yes:
        confirm = input(f"Are you sure you want to reset ALL user passwords using pattern '{args.pattern}'? Type YES to continue: ")
        if confirm != "YES":
            print("Aborted by user. No changes made.")
            sys.exit(0)

    # Perform reset
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        updated = reset_all_user_passwords(cur, pattern=args.pattern)
        conn.commit()
        print(f"Passwords updated for {updated} users.")
    except Exception as e:
        conn.rollback()
        print("Failed to reset passwords:", e)
        try:
            debug_logger.exception("reset_passwords script failed", e)
        except Exception:
            pass
        sys.exit(2)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
