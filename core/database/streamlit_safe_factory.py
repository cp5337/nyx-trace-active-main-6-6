"""
Improved Thread-Safe Database Factory
====================================

This module provides enhanced thread-safe database connections
specifically designed for Streamlit's threading model.
"""

import threading
import logging
from typing import Dict, Any, Optional
from contextlib import contextmanager
import time

logger = logging.getLogger(__name__)

class StreamlitSafeDatabaseFactory:
    """
    Thread-safe database factory optimized for Streamlit applications
    """
    
    _instance = None
    _lock = threading.RLock()
    _connections = {}
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    @contextmanager
    def get_connection(self, db_type: str = "default", timeout: int = 30):
        """
        Get a thread-safe database connection with automatic cleanup.
        
        Args:
            db_type: Type of database connection
            timeout: Connection timeout in seconds
            
        Yields:
            Database connection object
        """
        thread_id = threading.get_ident()
        connection_key = f"{db_type}_{thread_id}"
        
        connection = None
        try:
            # Get or create connection for this thread
            if connection_key not in self._connections:
                connection = self._create_connection(db_type, timeout)
                self._connections[connection_key] = connection
            else:
                connection = self._connections[connection_key]
                # Test connection health
                if not self._test_connection(connection):
                    connection = self._create_connection(db_type, timeout)
                    self._connections[connection_key] = connection
            
            yield connection
            
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            # Clean up failed connection
            if connection_key in self._connections:
                del self._connections[connection_key]
            raise
        finally:
            # Connection cleanup is handled by the pool
            pass
    
    def _create_connection(self, db_type: str, timeout: int):
        """Create a new database connection with retry logic"""
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                # Mock connection creation - replace with actual implementation
                logger.info(f"Creating {db_type} connection (attempt {attempt + 1})")
                # connection = create_actual_connection(db_type, timeout)
                return {"type": db_type, "thread": threading.get_ident(), "created": time.time()}
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Connection attempt {attempt + 1} failed: {e}. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    raise
    
    def _test_connection(self, connection) -> bool:
        """Test if connection is still valid"""
        try:
            # Mock connection test - replace with actual implementation
            return connection is not None and time.time() - connection.get("created", 0) < 3600
        except:
            return False
    
    def cleanup_connections(self):
        """Clean up all connections for the current thread"""
        thread_id = threading.get_ident()
        keys_to_remove = [k for k in self._connections.keys() if k.endswith(f"_{thread_id}")]
        
        for key in keys_to_remove:
            try:
                # Clean up connection
                del self._connections[key]
                logger.info(f"Cleaned up connection: {key}")
            except Exception as e:
                logger.error(f"Error cleaning up connection {key}: {e}")

# Global factory instance
database_factory = StreamlitSafeDatabaseFactory()
