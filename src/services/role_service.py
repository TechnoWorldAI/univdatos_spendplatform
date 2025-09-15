"""
Role Service - Handles user roles and permissions
"""
from typing import Dict, Any
from src.config import config


class RoleService:
    """Service for managing user roles and permissions."""
    
    def get_role_permissions(self, role: str) -> Dict[str, Any]:
        """Get permissions for a specific role.
        
        Args:
            role: The user role
            
        Returns:
            Dictionary of permissions for the role
        """
        return config.roles.get(role, {})
    
    def get_available_roles(self) -> list:
        """Get list of available roles.
        
        Returns:
            List of available role names
        """
        return list(config.roles.keys())
    
    def has_permission(self, role: str, permission: str) -> bool:
        """Check if a role has a specific permission.
        
        Args:
            role: The user role
            permission: The permission to check
            
        Returns:
            True if the role has the permission, False otherwise
        """
        role_permissions = self.get_role_permissions(role)
        return role_permissions.get(permission, False)
