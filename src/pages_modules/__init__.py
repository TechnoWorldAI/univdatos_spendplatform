"""Package initialization for pages module."""
from . import (
    login,
    dashboard,
    upload,
    master_data,
    error_management,
    reports,
    user_management
)

__all__ = [
    'login',
    'dashboard',
    'upload',
    'master_data',
    'error_management',
    'reports',
    'user_management'
]
