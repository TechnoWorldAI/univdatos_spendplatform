"""Authentication service for user management and authentication."""
from typing import Dict, Optional
from datetime import datetime

from src.models.user import User
from src.exceptions.base import AuthenticationError, ValidationError
from src.config import config
from src.utils.debug import debug_logger


class AuthService:
    """Service for handling user authentication and management."""

    @staticmethod
    def authenticate(username: str, password: str) -> Dict[str, str]:
        """Authenticate a user and create a session.
        
        Args:
            username: Username to authenticate
            password: Password to verify
            
        Returns:
            Dictionary with user info and session token
            
        Raises:
            AuthenticationError: If authentication fails
        """
        debug_logger.debug("Authentication attempt", {"username": username})
        
        try:
            user = User.authenticate(username, password)
            debug_logger.info("Authentication successful", {"username": username, "user_id": user.user_id})
            
            # TODO: Generate and store proper session token
            return {
                "user_id": str(user.user_id),
                "username": user.username,
                "role": user.role,
                "session_token": "temp_token"
            }
        except AuthenticationError as e:
            debug_logger.warning("Authentication failed", extra_data={"username": username, "error": str(e)})
            raise
        except Exception as e:
            debug_logger.exception("Unexpected error during authentication", e, {"username": username})
            raise AuthenticationError("Authentication system error")

    @staticmethod
    def create_user(
        username: str,
        password: str,
        role: str,
        created_by: Optional[str] = None
    ) -> User:
        """Create a new user.
        
        Args:
            username: Unique username
            password: Plain text password
            role: User role
            created_by: Username of creating user
            
        Returns:
            Created User instance
            
        Raises:
            ValidationError: If validation fails
        """
        # Validate input
        if len(password) < config.security.password_min_length:
            raise ValidationError(
                f"Password must be at least {config.security.password_min_length} characters",
                "password"
            )
        
        if role not in config.roles:
            raise ValidationError(f"Invalid role: {role}", "role")
        
        # Check if username exists
        if User.get_by_username(username):
            raise ValidationError(f"Username already exists: {username}", "username")
        
        # Create user
        user = User(
            username=username,
            password_hash=User.hash_password(password),
            role=role,
            created_at=datetime.utcnow().isoformat()
        )
        user.save()
        
        return user

    @staticmethod
    def update_user(
        user_id: int,
        updates: Dict[str, str]
    ) -> User:
        """Update a user's information.
        
        Args:
            user_id: ID of user to update
            updates: Dictionary of fields to update
            
        Returns:
            Updated User instance
            
        Raises:
            NotFoundError: If user doesn't exist
            ValidationError: If validation fails
        """
        user = User.get_by_id(user_id)
        if not user:
            raise ValidationError(f"User not found: {user_id}")
        
        # Update allowed fields
        if 'username' in updates and updates['username'] != user.username:
            if User.get_by_username(updates['username']):
                raise ValidationError("Username already exists", "username")
            user.username = updates['username']
        
        if 'role' in updates:
            if updates['role'] not in config.roles:
                raise ValidationError(f"Invalid role: {updates['role']}", "role")
            user.role = updates['role']
        
        if 'password' in updates:
            if len(updates['password']) < config.security.password_min_length:
                raise ValidationError(
                    f"Password must be at least {config.security.password_min_length} characters",
                    "password"
                )
            user.password_hash = User.hash_password(updates['password'])
        
        user.save()
        return user
    
    @staticmethod
    def logout():
        """Log out the current user by clearing session state."""
        import streamlit as st
        # Clear non-internal session state keys to avoid leaking user-specific state
        keys = list(st.session_state.keys())
        for key in keys:
            # Preserve Streamlit internal keys (start with underscore)
            if key.startswith('_'):
                continue
            try:
                del st.session_state[key]
            except Exception:
                # ignore inability to delete certain keys
                pass

        # Set safe defaults after clearing
        st.session_state.authenticated = False
        st.session_state.user_exists = False
        st.session_state.selected_page = "dashboard"
