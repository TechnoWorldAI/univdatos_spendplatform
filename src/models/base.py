"""Base model class for database entities."""
from datetime import datetime
from typing import Any, Dict, Optional
from src.utils.db_simple import get_db_connection
from src.exceptions.base import DatabaseError, NotFoundError


class BaseModel:
    """Base class for all database models."""
    
    table_name: str = ""
    primary_key: str = "id"

    def __init__(self, **kwargs: Any) -> None:
        """Initialize model with attribute values."""
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def get_by_id(cls, id_value: Any) -> Optional['BaseModel']:
        """Get a record by its primary key.
        
        Args:
            id_value: The primary key value
            
        Returns:
            Model instance if found, None otherwise
            
        Raises:
            DatabaseError: If database operation fails
        """
        # Validate table and column names to prevent SQL injection
        allowed_tables = {
            'users', 'vendors', 'categories', 'spend_transactions', 
            'error_logs', 'rules', 'user_settings'
        }
        
        if cls.table_name not in allowed_tables:
            raise ValueError(f"Invalid table name: {cls.table_name}")
            
        if not cls.primary_key.replace('_', '').isalnum():
            raise ValueError(f"Invalid primary key: {cls.primary_key}")
            
        with get_db_connection() as conn:
            cursor = conn.cursor()
            query = f"SELECT * FROM {cls.table_name} WHERE {cls.primary_key} = ?"
            cursor.execute(query, (id_value,))
            row = cursor.fetchone()
            return cls(**dict(row)) if row else None

    def save(self) -> None:
        """Save the current instance to the database.
        
        Raises:
            DatabaseError: If save operation fails
        """
        # Validate table name
        allowed_tables = {
            'users', 'vendors', 'categories', 'spend_transactions', 
            'error_logs', 'rules', 'user_settings'
        }
        
        if self.table_name not in allowed_tables:
            raise ValueError(f"Invalid table name: {self.table_name}")
            
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Get all instance attributes that are not None
            attrs = {k: v for k, v in self.__dict__.items() 
                    if not k.startswith('_') and v is not None}
            
            if hasattr(self, self.primary_key):
                # Update existing record
                set_clause = ", ".join(f"{k} = ?" for k in attrs.keys())
                values = tuple(attrs.values())
                query = f"UPDATE {self.table_name} SET {set_clause} WHERE {self.primary_key} = ?"
                cursor.execute(query, values + (getattr(self, self.primary_key),))
            else:
                # Insert new record
                columns = ", ".join(attrs.keys())
                placeholders = ", ".join("?" * len(attrs))
                values = tuple(attrs.values())
                query = f"INSERT INTO {self.table_name} ({columns}) VALUES ({placeholders})"
                cursor.execute(query, values)
            
            conn.commit()

    def delete(self) -> None:
        """Delete the current instance from the database.
        
        Raises:
            NotFoundError: If record doesn't exist
            DatabaseError: If delete operation fails
        """
        if not hasattr(self, self.primary_key):
            raise NotFoundError(self.table_name, "No primary key")
        
        # Validate table and column names
        allowed_tables = {
            'users', 'vendors', 'categories', 'spend_transactions', 
            'error_logs', 'rules', 'user_settings'
        }
        
        if self.table_name not in allowed_tables:
            raise ValueError(f"Invalid table name: {self.table_name}")
            
        if not self.primary_key.replace('_', '').isalnum():
            raise ValueError(f"Invalid primary key: {self.primary_key}")
            
        with get_db_connection() as conn:
            cursor = conn.cursor()
            query = f"DELETE FROM {self.table_name} WHERE {self.primary_key} = ?"
            cursor.execute(query, (getattr(self, self.primary_key),))
            if cursor.rowcount == 0:
                raise NotFoundError(
                    self.table_name,
                    str(getattr(self, self.primary_key))
                )
            conn.commit()

    def to_dict(self) -> Dict[str, Any]:
        """Convert model instance to dictionary."""
        return {k: v for k, v in self.__dict__.items() 
                if not k.startswith('_')}
