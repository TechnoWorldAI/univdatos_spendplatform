"""User model for authentication and authorization."""
from datetime import datetime
from typing import Optional
from src.utils.crypto import hash_password as crypto_hash_password, verify_password as crypto_verify_password
from src.models.base import BaseModel
from src.exceptions.base import AuthenticationError
from src.utils.db_simple import get_db_connection


class User(BaseModel):
    """User model for authentication and authorization."""
    
    table_name = "users"
    primary_key = "user_id"

    def __init__(
        self,
        username: str,
        password_hash: str,
        role: str,
        user_id: Optional[int] = None,
        created_at: Optional[str] = None,
        last_login: Optional[str] = None,
        is_deleted: Optional[int] = None,
        **kwargs
    ) -> None:
        """Initialize a user instance.
        
        Args:
            username: Unique username
            password_hash: Hashed password
            role: User role (Admin, Spend Manager, Data Analyst)
            user_id: Optional user ID for existing users
            created_at: Optional creation timestamp
            last_login: Optional last login timestamp
            is_deleted: Optional soft delete flag
            **kwargs: Additional keyword arguments (for database compatibility)
        """
        super().__init__()
        self.username = username
        self.password_hash = password_hash
        self.role = role
        if user_id:
            self.user_id = user_id
        self.created_at = created_at or datetime.utcnow().isoformat()
        self.last_login = last_login
        self.is_deleted = is_deleted  # Handle soft delete field

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password for storage.
        
        Args:
            password: Plain text password
            
        Returns:
            Hashed password
        """
        # Use Argon2 wrapper
        return crypto_hash_password(password)

    def verify_password(self, password: str) -> bool:
        """Verify a password against the stored hash.

        Args:
            password: Plain text password to verify

        Returns:
            True if password matches, False otherwise
        """
        # Use Argon2 verification; supports rehashing on next login elsewhere
        return crypto_verify_password(self.password_hash, password)

    def update_last_login(self) -> None:
        """Update the last login timestamp."""
        self.last_login = datetime.utcnow().isoformat()
        self.save()

    @classmethod
    def authenticate(cls, username: str, password: str) -> 'User':
        """Authenticate a user with username and password.
        
        Args:
            username: Username to authenticate
            password: Password to verify
            
        Returns:
            Authenticated User instance
            
        Raises:
            AuthenticationError: If authentication fails
        """
        user = cls.get_by_username(username)
        if not user or not user.verify_password(password):
            raise AuthenticationError()
        
        user.update_last_login()
        return user

    @classmethod
    def get_by_username(cls, username: str) -> Optional['User']:
        """Get a user by username.
        
        Args:
            username: Username to look up
            
        Returns:
            User instance if found, None otherwise
        """
        with get_db_connection() as conn:
            cursor = conn.cursor()
            # Safe query - users table and username column are hardcoded
            cursor.execute(
                "SELECT * FROM users WHERE username = ?",
                (username,)
            )
            row = cursor.fetchone()
            return cls(**dict(row)) if row else None
