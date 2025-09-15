"""Database connection management module."""
from contextlib import contextmanager
from typing import Generator, Optional
import sqlite3
from src.exceptions.base import DatabaseError


class DatabasePool:
    """Manages a pool of database connections."""

    def __init__(self, database_path: str, max_connections: int = 5) -> None:
        """Initialize the database pool.
        
        Args:
            database_path: Path to the SQLite database file
            max_connections: Maximum number of concurrent connections
        """
        self.database_path = database_path
        self.max_connections = max_connections
        self._pool: list[sqlite3.Connection] = []

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection from the pool.
        
        Yields:
            A SQLite connection object
            
        Raises:
            DatabaseError: If connection cannot be established
        """
        connection = self._get_connection()
        try:
            yield connection
        except sqlite3.Error as e:
            raise DatabaseError(f"Database operation failed: {str(e)}")
        finally:
            self._return_connection(connection)

    def _get_connection(self) -> sqlite3.Connection:
        """Get an available connection or create a new one."""
        # Don't use connection pooling with SQLite due to threading issues
        # Create a fresh connection each time with thread safety
        try:
            connection = sqlite3.connect(
                self.database_path,
                check_same_thread=False  # Allow connections across threads
            )
            connection.row_factory = sqlite3.Row
            return connection
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to establish database connection: {str(e)}")

    def _return_connection(self, connection: sqlite3.Connection) -> None:
        """Return a connection to the pool."""
        # Don't pool SQLite connections due to threading issues
        # Just close the connection
        connection.close()


# Global database pool instance
_db_pool: Optional[DatabasePool] = None


def get_db_pool(database_path: str) -> DatabasePool:
    """Get or create the global database pool instance.
    
    Args:
        database_path: Path to the SQLite database file
        
    Returns:
        DatabasePool instance
    """
    global _db_pool
    if _db_pool is None:
        _db_pool = DatabasePool(database_path)
    return _db_pool
