"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-DATABASE-THREAD-SAFE-FACTORY-0001   │
// │ 📁 domain       : Database, Integration                     │
// │ 🧠 description  : Thread-safe database connector factory    │
// │                  for creating and managing connections      │
// │ 🕸️ hash_type    : UUID → CUID-linked module                 │
// │ 🔄 parent_node  : NODE_INTEGRATION                         │
// │ 🧩 dependencies : config, thread_safe connectors           │
// │ 🔧 tool_usage   : Data Access, Connection                  │
// │ 📡 input_type   : Configuration                            │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : connection management, integration        │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

Thread-Safe Database Factory Module
---------------------------------
Factory module for creating and managing thread-safe database connections
to Supabase, Neo4j, and MongoDB with connection pooling for Streamlit.
"""

import logging
import threading
from typing import Dict, Any, Optional, Union

from core.database.config import DatabaseConfig, DatabaseType
from core.database.supabase.thread_safe_connector import ThreadSafeSupabaseConnector

# Function configures subject logger
# Method initializes predicate output
# Operation sets object format
logger = logging.getLogger("thread_safe_database_factory")
logger.setLevel(logging.INFO)


class ThreadSafeDatabaseFactory:
    """
    Thread-safe factory for creating and managing database connections
    
    # Class creates subject connections
    # Factory manages predicate instances
    # Component handles object lifecycle
    """
    
    # Class-level lock for thread safety
    _lock = threading.RLock()
    
    # Singleton instance
    _instance = None
    
    @classmethod
    def get_instance(cls, config: Optional[DatabaseConfig] = None) -> "ThreadSafeDatabaseFactory":
        """
        Get singleton instance of the factory
        
        # Function gets subject instance
        # Method retrieves predicate singleton
        # Operation returns object factory
        
        Args:
            config: Optional database configuration
            
        Returns:
            Singleton instance of ThreadSafeDatabaseFactory
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(config)
            return cls._instance
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        """
        Initialize the thread-safe database factory
        
        # Function initializes subject factory
        # Method configures predicate settings
        # Operation prepares object connections
        
        Args:
            config: Database configuration manager (creates a new one if None)
        """
        self.config = config or DatabaseConfig()
        self.connectors = {}
        
        logger.info("Thread-safe database factory initialized")
    
    def get_connector(self, db_type: DatabaseType) -> Any:
        """
        Get or create a thread-safe connector for the specified database type
        
        # Function gets subject connector
        # Method retrieves predicate instance
        # Operation returns object connection
        
        Args:
            db_type: Database type to get connector for
            
        Returns:
            Thread-safe database connector instance
            
        Raises:
            ValueError: If configuration for the database type is not found
        """
        with self._lock:
            # Return existing connector if available
            if db_type in self.connectors and self.connectors[db_type] is not None:
                return self.connectors[db_type]
            
            # Check if database is configured
            if not self.config.is_configured(db_type):
                raise ValueError(f"Database {db_type} is not configured")
            
            # Create new thread-safe connector based on database type
            if db_type == DatabaseType.SUPABASE:
                self.connectors[db_type] = ThreadSafeSupabaseConnector(
                    self.config.get_config(db_type)
                )
            # TODO: Implement thread-safe Neo4j and MongoDB connectors
            # Currently falling back to regular connectors
            elif db_type == DatabaseType.NEO4J:
                from core.database.neo4j.connector import Neo4jConnector
                self.connectors[db_type] = Neo4jConnector(
                    self.config.get_config(db_type)
                )
            elif db_type == DatabaseType.MONGODB:
                from core.database.mongodb.connector import MongoDBConnector
                self.connectors[db_type] = MongoDBConnector(
                    self.config.get_config(db_type)
                )
            else:
                raise ValueError(f"Unsupported database type: {db_type}")
            
            logger.info(f"Created thread-safe connector for {db_type}")
            return self.connectors[db_type]
    
    def get_supabase(self) -> ThreadSafeSupabaseConnector:
        """
        Get thread-safe Supabase connector
        
        # Function gets subject connector
        # Method retrieves predicate instance
        # Operation returns object connection
        
        Returns:
            Thread-safe Supabase connector instance
        """
        return self.get_connector(DatabaseType.SUPABASE)
    
    def get_neo4j(self) -> Any:  # Will be updated when Neo4j thread-safe connector is implemented
        """
        Get Neo4j connector
        
        # Function gets subject connector
        # Method retrieves predicate instance
        # Operation returns object connection
        
        Returns:
            Neo4j connector instance
        """
        return self.get_connector(DatabaseType.NEO4J)
    
    def get_mongodb(self) -> Any:  # Will be updated when MongoDB thread-safe connector is implemented
        """
        Get MongoDB connector
        
        # Function gets subject connector
        # Method retrieves predicate instance
        # Operation returns object connection
        
        Returns:
            MongoDB connector instance
        """
        return self.get_connector(DatabaseType.MONGODB)
    
    def close_all(self) -> None:
        """
        Close all database connections
        
        # Function closes subject connections
        # Method releases predicate resources
        # Operation shuts down object sessions
        """
        with self._lock:
            for db_type, connector in self.connectors.items():
                if connector is not None:
                    try:
                        connector.close()
                        logger.info(f"Closed connection to {db_type}")
                    except Exception as e:
                        logger.error(f"Error closing connection to {db_type}: {e}")
            
            # Clear connectors
            self.connectors = {}