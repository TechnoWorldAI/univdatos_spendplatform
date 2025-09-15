"""Base exceptions for the Spend Platform application."""
from typing import Optional


class SpendPlatformError(Exception):
    """Base exception class for all application errors."""
    
    def __init__(self, message: str, status_code: int = 500) -> None:
        """Initialize the base exception.
        
        Args:
            message: The error message
            status_code: HTTP status code (default: 500)
        """
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class AuthenticationError(SpendPlatformError):
    """Raised when authentication fails."""
    
    def __init__(self, message: str = "Authentication failed") -> None:
        super().__init__(message, status_code=401)


class ValidationError(SpendPlatformError):
    """Raised when data validation fails."""
    
    def __init__(self, message: str, field: Optional[str] = None) -> None:
        self.field = field
        super().__init__(message, status_code=400)


class DatabaseError(SpendPlatformError):
    """Raised when database operations fail."""
    
    def __init__(self, message: str = "Database operation failed") -> None:
        super().__init__(message, status_code=500)


class NotFoundError(SpendPlatformError):
    """Raised when a requested resource is not found."""
    
    def __init__(self, resource: str, identifier: str) -> None:
        message = f"{resource} not found with identifier: {identifier}"
        super().__init__(message, status_code=404)
