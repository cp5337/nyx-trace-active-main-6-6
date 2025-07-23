"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-SUPABASE-CONNECTOR-0001             │
// │ 📁 domain       : Database, Supabase                        │
// │ 🧠 description  : Supabase connector for PostgreSQL database│
// │                  integration through SQLAlchemy             │
// │ 🕸️ hash_type    : UUID → CUID-linked module                 │
// │ 🔄 parent_node  : NODE_INTEGRATION                         │
// │ 🧩 dependencies : sqlalchemy, psycopg2, supabase           │
// │ 🔧 tool_usage   : Data Access, SQL                         │
// │ 📡 input_type   : SQL queries, data objects                │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : data persistence, querying               │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

Supabase Connector Module
-----------------------
Provides a connector for interacting with Supabase PostgreSQL
database using SQLAlchemy for direct SQL access and operations.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import json

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session

from core.database.config import SupabaseConfig

# Function configures subject logger
# Method initializes predicate output
# Operation sets object format
logger = logging.getLogger("supabase_connector")
logger.setLevel(logging.INFO)


class SupabaseConnector:
    """
    Connector for Supabase PostgreSQL database

    # Class connects subject database
    # Connector manages predicate connection
    # Component handles object persistence
    """

    def __init__(self, config: SupabaseConfig):
        """
        Initialize Supabase connector

        # Function initializes subject connector
        # Method configures predicate connection
        # Operation establishes object session

        Args:
            config: Supabase configuration
        """
        self.config = config
        self.engine = None
        self.Session = None

        # Connect to database
        self._connect()

        logger.info("Supabase connector initialized")

    def _connect(self) -> None:
        """
        Establish connection to Supabase PostgreSQL database

        # Function connects subject database
        # Method establishes predicate connection
        # Operation creates object engine
        """
        try:
            # Create SQLAlchemy engine
            self.engine = create_engine(self.config.database_url)

            # Create sessionmaker
            self.Session = sessionmaker(bind=self.engine)

            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))

            logger.info("Connected to Supabase PostgreSQL database")
        except Exception as e:
            logger.error(
                f"Failed to connect to Supabase PostgreSQL database: {e}"
            )
            raise

    def get_session(self) -> Session:
        """
        Get a new SQLAlchemy session

        # Function gets subject session
        # Method creates predicate connection
        # Operation returns object instance

        Returns:
            New SQLAlchemy session
        """
        if self.Session is None:
            raise ValueError("Not connected to database")

        return self.Session()

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
        with self.get_session() as session:
            try:
                result = session.execute(text(query), params or {})

                # Convert result to list of dictionaries
                columns = result.keys()
                return [dict(zip(columns, row)) for row in result.fetchall()]
            except SQLAlchemyError as e:
                logger.error(f"Error executing query: {e}")
                session.rollback()
                raise

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

        with self.get_session() as session:
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

    def update(
        self,
        table: str,
        data: Dict[str, Any],
        condition: str,
        condition_params: Dict[str, Any],
    ) -> Dict[str, Any]:
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
            Dictionary representing the updated row(s)
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

        with self.get_session() as session:
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

        with self.get_session() as session:
            try:
                result = session.execute(text(query), params)
                deleted_rows = result.rowcount
                session.commit()
                return deleted_rows
            except SQLAlchemyError as e:
                logger.error(f"Error deleting data: {e}")
                session.rollback()
                raise

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
        Close database connection

        # Function closes subject connection
        # Method releases predicate resources
        # Operation disposes object engine
        """
        if self.engine:
            self.engine.dispose()
            logger.info("Closed connection to Supabase PostgreSQL database")
