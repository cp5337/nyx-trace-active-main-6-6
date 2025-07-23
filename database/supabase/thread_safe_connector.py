"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-SUPABASE-THREAD-SAFE-CONNECTOR-0001 │
// │ 📁 domain       : Database, Supabase                        │
// │ 🧠 description  : Thread-safe Supabase connector            │
// │                  with connection pooling for Streamlit      │
// │ 🕸️ hash_type    : UUID → CUID-linked module                 │
// │ 🔄 parent_node  : NODE_INTEGRATION                         │
// │ 🧩 dependencies : sqlalchemy, psycopg2, supabase, threading │
// │ 🔧 tool_usage   : Data Access, SQL                         │
// │ 📡 input_type   : SQL queries, data objects                │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : data persistence, querying               │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

Thread-Safe Supabase Connector Module
------------------------------------
Provides a thread-safe connector for interacting with Supabase PostgreSQL
database using SQLAlchemy connection pooling for safe use in Streamlit.
"""

import logging
import json
import threading
from typing import Any, Dict, List, Optional, Tuple, Union

from sqlalchemy import create_engine, text, pool
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.engine import Engine, Connection
from sqlalchemy.orm import scoped_session, sessionmaker, Session

from core.database.config import SupabaseConfig

# Function configures subject logger
# Method initializes predicate output
# Operation sets object format
logger = logging.getLogger("thread_safe_supabase")
logger.setLevel(logging.INFO)


class ThreadSafeSupabaseConnector:
    """
    Thread-safe connector for Supabase PostgreSQL database
    
    # Class connects subject database
    # Connector manages predicate connection
    # Component handles object persistence
    """
    
    # Thread-local storage for database sessions
    _thread_local = threading.local()
    
    # Class-level lock for thread safety when initializing the engine
    _lock = threading.RLock()
    
    def __init__(self, config: SupabaseConfig):
        """
        Initialize thread-safe Supabase connector
        
        # Function initializes subject connector
        # Method configures predicate connection
        # Operation establishes object pool
        
        Args:
            config: Supabase configuration
        """
        self.config = config
        self.engine = None
        self.session_factory = None
        
        # Initialize the connection pool
        self._initialize_engine()
        
        logger.info("Thread-safe Supabase connector initialized with connection pool")
    
    def _initialize_engine(self) -> None:
        """
        Initialize the SQLAlchemy engine with connection pooling
        
        # Function initializes subject engine
        # Method creates predicate pool
        # Operation configures object settings
        """
        with self._lock:
            if self.engine is None:
                try:
                    # Create SQLAlchemy engine with connection pooling
                    self.engine = create_engine(
                        self.config.database_url,
                        poolclass=pool.QueuePool,
                        pool_size=5,
                        max_overflow=10,
                        pool_timeout=30,
                        pool_recycle=1800,  # Recycle connections after 30 minutes
                        pool_pre_ping=True  # Verify connections before use
                    )
                    
                    # Create thread-safe session factory
                    self.session_factory = scoped_session(
                        sessionmaker(bind=self.engine)
                    )
                    
                    # Test connection
                    with self.engine.connect() as conn:
                        conn.execute(text("SELECT 1"))
                    
                    logger.info("Connected to Supabase PostgreSQL with connection pool")
                except Exception as e:
                    logger.error(f"Failed to initialize connection pool: {e}")
                    raise
    
    def get_session(self) -> Session:
        """
        Get a thread-local session from the connection pool
        
        # Function gets subject session
        # Method retrieves predicate connection
        # Operation returns object instance
        
        Returns:
            SQLAlchemy session from the connection pool
        """
        if not hasattr(self._thread_local, 'session'):
            if self.session_factory is None:
                raise ValueError("Database connection not initialized")
            self._thread_local.session = self.session_factory()
        
        return self._thread_local.session
    
    def release_session(self) -> None:
        """
        Release the thread-local session back to the pool
        
        # Function releases subject session
        # Method returns predicate connection
        # Operation cleans object resources
        """
        if hasattr(self._thread_local, 'session'):
            self._thread_local.session.close()
            delattr(self._thread_local, 'session')
    
    def execute_query(
        self, query: str, params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a SQL query and return results
        
        # Function executes subject query
        # Method runs predicate SQL
        # Operation returns object results
        
        Args:
            query: SQL query to execute
            params: Query parameters
            
        Returns:
            List of dictionaries representing rows
        """
        session = self.get_session()
        try:
            result = session.execute(text(query), params or {})
            
            # Convert result to list of dictionaries
            columns = result.keys()
            return [dict(zip(columns, row)) for row in result.fetchall()]
        except SQLAlchemyError as e:
            logger.error(f"Error executing query: {e}")
            session.rollback()
            raise
        finally:
            session.close()
    
    def execute_with_json_result(
        self, query: str, params: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Execute a SQL query and return results as JSON string
        
        # Function executes subject query
        # Method formats predicate results
        # Operation returns object JSON
        
        Args:
            query: SQL query to execute
            params: Query parameters
            
        Returns:
            JSON string representing results
        """
        results = self.execute_query(query, params)
        return json.dumps(results, default=str)
    
    def insert(self, table: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Insert a row into a table
        
        # Function inserts subject data
        # Method stores predicate record
        # Operation creates object row
        
        Args:
            table: Table name
            data: Dictionary of column:value pairs
            
        Returns:
            Dictionary representing the inserted row
        """
        columns = ", ".join(data.keys())
        placeholders = ", ".join([f":{k}" for k in data.keys()])
        
        query = f"""
        INSERT INTO {self.config.schema}.{table} ({columns})
        VALUES ({placeholders})
        RETURNING *;
        """
        
        session = self.get_session()
        try:
            result = session.execute(text(query), data)
            row = result.fetchone()
            session.commit()
            
            # Convert row to dictionary
            if row:
                return dict(zip(result.keys(), row))
            return {}
        except SQLAlchemyError as e:
            logger.error(f"Error inserting data: {e}")
            session.rollback()
            raise
        finally:
            session.close()
    
    def update(
        self,
        table: str,
        data: Dict[str, Any],
        condition: str,
        condition_params: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Update rows in a table
        
        # Function updates subject data
        # Method modifies predicate records
        # Operation changes object rows
        
        Args:
            table: Table name
            data: Dictionary of column:value pairs to update
            condition: WHERE clause condition
            condition_params: Parameters for condition
            
        Returns:
            List of dictionaries representing the updated rows
        """
        set_clause = ", ".join([f"{k} = :{k}" for k in data.keys()])
        
        query = f"""
        UPDATE {self.config.schema}.{table}
        SET {set_clause}
        WHERE {condition}
        RETURNING *;
        """
        
        # Combine data and condition parameters
        params = {**data, **condition_params}
        
        session = self.get_session()
        try:
            result = session.execute(text(query), params)
            rows = result.fetchall()
            session.commit()
            
            # Convert rows to dictionaries
            if rows:
                return [dict(zip(result.keys(), row)) for row in rows]
            return []
        except SQLAlchemyError as e:
            logger.error(f"Error updating data: {e}")
            session.rollback()
            raise
        finally:
            session.close()
    
    def delete(self, table: str, condition: str, params: Dict[str, Any]) -> int:
        """
        Delete rows from a table
        
        # Function deletes subject data
        # Method removes predicate records
        # Operation eliminates object rows
        
        Args:
            table: Table name
            condition: WHERE clause condition
            params: Parameters for condition
            
        Returns:
            Number of rows deleted
        """
        query = f"""
        DELETE FROM {self.config.schema}.{table}
        WHERE {condition}
        RETURNING *;
        """
        
        session = self.get_session()
        try:
            result = session.execute(text(query), params)
            # Count the number of rows in the RETURNING result
            rows = result.fetchall()
            deleted_count = len(rows)
            session.commit()
            return deleted_count
        except SQLAlchemyError as e:
            logger.error(f"Error deleting data: {e}")
            session.rollback()
            raise
        finally:
            session.close()
    
    def execute_select(
        self,
        table: str,
        columns: str = "*",
        condition: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute a SELECT query
        
        # Function selects subject data
        # Method retrieves predicate records
        # Operation fetches object rows
        
        Args:
            table: Table name
            columns: Columns to select
            condition: WHERE clause condition
            params: Parameters for condition
            order_by: ORDER BY clause
            limit: LIMIT clause
            
        Returns:
            List of dictionaries representing rows
        """
        query = f"SELECT {columns} FROM {self.config.schema}.{table}"
        
        if condition:
            query += f" WHERE {condition}"
        
        if order_by:
            query += f" ORDER BY {order_by}"
        
        if limit:
            query += f" LIMIT {limit}"
        
        return self.execute_query(query, params or {})
    
    def get_by_id(
        self, table: str, id_value: Union[str, int], id_column: str = "id"
    ) -> Dict[str, Any]:
        """
        Get a row by ID
        
        # Function gets subject row
        # Method retrieves predicate record
        # Operation fetches object by id
        
        Args:
            table: Table name
            id_value: ID value
            id_column: ID column name
            
        Returns:
            Dictionary representing the row
        """
        results = self.execute_select(
            table,
            condition=f"{id_column} = :{id_column}",
            params={id_column: id_value},
            limit=1,
        )
        
        return results[0] if results else {}
    
    def close(self) -> None:
        """
        Close the database connection pool
        
        # Function closes subject pool
        # Method releases predicate resources
        # Operation disposes object engine
        """
        if self.engine:
            self.engine.dispose()
            logger.info("Closed Supabase PostgreSQL connection pool")