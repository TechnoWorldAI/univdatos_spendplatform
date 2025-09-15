"""Package initialization for exceptions module."""
from .base import (
    SpendPlatformError,
    AuthenticationError,
    ValidationError,
    DatabaseError,
    NotFoundError
)

__all__ = [
    'SpendPlatformError',
    'AuthenticationError',
    'ValidationError',
    'DatabaseError',
    'NotFoundError'
]
