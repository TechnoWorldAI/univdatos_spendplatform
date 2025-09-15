"""Data service for managing business data operations."""
from typing import List, Dict, Any, Optional
import pandas as pd
from src.utils.db_simple import get_db_connection
from src.exceptions.base import ValidationError, DatabaseError


class DataService:
    """Service for managing spend transactions and related operations."""

    @staticmethod
    def get_transactions(
        limit: int = 100,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Get spend transactions with pagination and filtering.
        
        Args:
            limit: Maximum number of records to return
            offset: Number of records to skip
            filters: Optional filters to apply
            
        Returns:
            List of transaction dictionaries
            
        Raises:
            DatabaseError: If database operation fails
        """
        with get_db_connection() as conn:
            query = "SELECT * FROM spend_transactions"
            params = []
            
            if filters:
                conditions = []
                if 'region' in filters:
                    conditions.append("region = ?")
                    params.append(filters['region'])
                if 'supplier_name' in filters:
                    conditions.append("supplier_name LIKE ?")
                    params.append(f"%{filters['supplier_name']}%")
                
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
            
            query += f" ORDER BY created_at DESC LIMIT {limit} OFFSET {offset}"
            
            cursor = conn.cursor()
            cursor.execute(query, params)
            
            columns = [desc[0] for desc in cursor.description]
            results = []
            for row in cursor.fetchall():
                results.append(dict(zip(columns, row)))
            
            return results

    @staticmethod
    def create_transaction(data: Dict[str, Any]) -> int:
        """Create a new spend transaction.
        
        Args:
            data: Transaction data dictionary
            
        Returns:
            ID of created transaction
            
        Raises:
            ValidationError: If data validation fails
            DatabaseError: If database operation fails
        """
        # Validate required fields
        required_fields = ['supplier_name', 'item_invoice_value']
        for field in required_fields:
            if field not in data or data[field] is None:
                raise ValidationError(f"Required field missing: {field}", field)
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Build insert query dynamically
            columns = list(data.keys())
            placeholders = ', '.join(['?' for _ in columns])
            values = list(data.values())
            
            query = f"""
                INSERT INTO spend_transactions ({', '.join(columns)})
                VALUES ({placeholders})
            """
            
            cursor.execute(query, values)
            conn.commit()
            return cursor.lastrowid or 0

    @staticmethod
    def update_transaction(transaction_id: int, updates: Dict[str, Any]) -> bool:
        """Update an existing transaction.
        
        Args:
            transaction_id: ID of transaction to update
            updates: Dictionary of fields to update
            
        Returns:
            True if update successful, False otherwise
            
        Raises:
            DatabaseError: If database operation fails
        """
        if not updates:
            return False
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
            values = list(updates.values()) + [transaction_id]
            
            query = f"""
                UPDATE spend_transactions 
                SET {set_clause} 
                WHERE transaction_id = ?
            """
            
            cursor.execute(query, values)
            conn.commit()
            return cursor.rowcount > 0

    @staticmethod
    def delete_transaction(transaction_id: int) -> bool:
        """Delete a transaction.
        
        Args:
            transaction_id: ID of transaction to delete
            
        Returns:
            True if deletion successful, False otherwise
            
        Raises:
            DatabaseError: If database operation fails
        """
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM spend_transactions WHERE transaction_id = ?",
                (transaction_id,)
            )
            conn.commit()
            return cursor.rowcount > 0
