"""Simple database utilities for thread-safe SQLite operations."""
import sqlite3
from contextlib import contextmanager
from typing import Generator, Optional
import time
from src.config import config
from src.exceptions.base import DatabaseError
from src.utils.debug import debug_logger


@contextmanager
def get_db_connection() -> Generator[sqlite3.Connection, None, None]:
    """Get a thread-safe database connection.
    
    Yields:
        A SQLite connection object
        
    Raises:
        DatabaseError: If connection cannot be established
    """
    connection = None
    try:
        debug_logger.debug("Opening database connection", {"path": config.database.path})
        connection = sqlite3.connect(
            config.database.path,
            check_same_thread=False  # Allow connections across threads
        )
        connection.row_factory = sqlite3.Row
        debug_logger.debug("Database connection established successfully")
        yield connection
    except sqlite3.Error as e:
        debug_logger.exception("Database operation failed", e)
        if connection:
            connection.rollback()
        raise DatabaseError(f"Database operation failed: {str(e)}")
    finally:
        if connection:
            connection.close()


def execute_query_with_timing(query: str, params: Optional[tuple] = None, connection: Optional[sqlite3.Connection] = None) -> sqlite3.Cursor:
    """Execute a query with performance timing and logging.
    
    Args:
        query: SQL query string
        params: Query parameters
        connection: Optional database connection (will create if None)
        
    Returns:
        SQLite cursor with query results
        
    Raises:
        DatabaseError: If query execution fails
    """
    should_close_connection = connection is None
    
    try:
        if connection is None:
            connection = sqlite3.connect(
                config.database.path,
                check_same_thread=False
            )
            connection.row_factory = sqlite3.Row
            
        start_time = time.time()
        cursor = connection.cursor()
        cursor.execute(query, params or ())
        execution_time = time.time() - start_time
        
        # Log slow queries (>100ms)
        if execution_time > 0.1:
            debug_logger.warning("Slow query detected", {
                "query": query[:200] + "..." if len(query) > 200 else query,
                "execution_time": f"{execution_time:.3f}s",
                "params_count": len(params) if params else 0
            })
        else:
            debug_logger.debug("Query executed", {
                "execution_time": f"{execution_time:.3f}s",
                "query_type": query.strip().split()[0].upper()
            })
            
        return cursor
        
    except sqlite3.Error as e:
        debug_logger.exception("Query execution failed", e, {
            "query": query[:200] + "..." if len(query) > 200 else query,
            "error": str(e)
        })
        raise DatabaseError(f"Query execution failed: {str(e)}")
    finally:
        if should_close_connection and connection:
            connection.close()
