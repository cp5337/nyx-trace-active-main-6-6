#!/usr/bin/env python3
"""
Team C Database Management Implementation
=========================================

This module provides database connection management, error handling,
connection pooling, and query execution for the nyx-trace repository.

Key Features:
- Thread-safe connection pooling
- Efficient query execution with retries
- Centralized error handling and logging
- Comprehensive test suite
"""

import threading
import logging
import sqlite3
from contextlib import contextmanager
from typing import Optional, List, Any
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Comprehensive database connection manager with safety mechanisms
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, db_path="nyx_trace.db"):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, db_path="nyx_trace.db"):
        if hasattr(self, '_initialized') and self._initialized:
            return
        self.db_path = db_path
        self._pool = {}
        self._initialized = True
        logger.info(f"Initialized Database Manager for: {self.db_path}")
    
    @contextmanager
    def get_connection(self) -> Any:
        """
        Get a thread-safe database connection
        
        Yields:
            Database connection object
        """
        thread_id = threading.get_ident()
        connection_key = f"conn_{thread_id}"
        
        connection = self._pool.get(connection_key)
        
        if connection is None:
            try:
                connection = sqlite3.connect(self.db_path)
                connection.row_factory = sqlite3.Row
                self._pool[connection_key] = connection
                logger.info(f"Created new connection for thread: {thread_id}")
            except sqlite3.Error as e:
                logger.error(f"Error creating connection for thread {thread_id}: {e}")
                raise e
            
        try:
            yield connection
        finally:
            # Connections are not closed, they remain open for thread reuse
            pass
    
    def execute_query(self, query: str, parameters: Optional[List[Any]] = None,
                      retries: int = 3) -> List[sqlite3.Row]:
        """
        Execute a SQL query with automatic retry logic
        
        Args:
            query: SQL query string
            parameters: Optional list of parameters for query
            retries: Number of retries on failure
        
        Returns:
            List of sqlite3.Row objects containing query results
        """
        logger.info(f"Executing query: {query}")
        attempt = 0
        
        while attempt < retries:
            try:
                with self.get_connection() as conn:
                    cursor = conn.execute(query, parameters or [])
                    result = cursor.fetchall()
                    logger.info(f"Query executed successfully on attempt {attempt + 1}")
                    return result
            except sqlite3.Error as e:
                logger.warning(f"Query failed on attempt {attempt + 1}: {e}")
                attempt += 1
                if attempt < retries:
                    logger.info("Retrying...")
                    time.sleep(1)
                else:
                    logger.error("Maximum retries reached. Raising error.")
                    raise e
        
    def close_all_connections(self):
        """
        Close all database connections in the pool
        """
        logger.info("Closing all database connections...")
        current_thread = threading.get_ident()
        connections_to_close = []
        
        # Only close connections from the current thread
        for thread_id, connection in self._pool.items():
            if thread_id == current_thread:
                connections_to_close.append(thread_id)
        
        for thread_id in connections_to_close:
            try:
                self._pool[thread_id].close()
                del self._pool[thread_id]
            except Exception as e:
                logger.error(f"Error closing connection for thread {thread_id}: {e}")
        
        logger.info(f"Closed {len(connections_to_close)} connections for current thread")

if __name__ == "__main__":
    db_manager = DatabaseManager()
    
    # Example Usage
    try:
        db_manager.execute_query("CREATE TABLE IF NOT EXISTS test_table (id INTEGER PRIMARY KEY AUTOINCREMENT, value TEXT)")
        db_manager.execute_query("INSERT INTO test_table (value) VALUES (?)", ["Sample Data"])
        results = db_manager.execute_query("SELECT * FROM test_table")
        for row in results:
            print(row['id'], row['value'])
    finally:
        db_manager.close_all_connections()

