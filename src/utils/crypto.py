"""Cryptographic helpers for password hashing and verification.

This wrapper uses argon2-cffi for secure password hashing.
"""

from argon2 import PasswordHasher, exceptions as argon2_exceptions
from typing import Optional

# Configure argon2 parameters with a reasonable default (can be tuned later)
_ph = PasswordHasher(time_cost=2, memory_cost=102400, parallelism=8)


def hash_password(password: str) -> str:
    """Hash a plain-text password using Argon2.

    Returns a string safe to store in DB (includes salt & parameters).
    """
    return _ph.hash(password)


def verify_password(stored_hash: str, password: str) -> bool:
    """Verify a plain password against the stored Argon2 hash.

    Returns True if password matches, False otherwise.
    """
    try:
        return _ph.verify(stored_hash, password)
    except argon2_exceptions.VerifyMismatchError:
        return False
    except Exception:
        # On any other error, be conservative and return False
        return False
