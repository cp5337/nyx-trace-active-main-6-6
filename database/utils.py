"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-DATABASE-UTILS-0001                 │
// │ 📁 domain       : Database, Utilities                       │
// │ 🧠 description  : Database utilities for multi-database     │
// │                  operations and common patterns             │
// │ 🕸️ hash_type    : UUID → CUID-linked module                 │
// │ 🔄 parent_node  : NODE_INTEGRATION                         │
// │ 🧩 dependencies : config, factory                          │
// │ 🔧 tool_usage   : Data Access, Integration                 │
// │ 📡 input_type   : Database operations                      │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : data integration, cross-database         │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

Database Utilities Module
-----------------------
Provides utility functions and patterns for working with
multiple databases in the NyxTrace system, including examples
of cross-database operations and data migration.
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import json
import uuid
from datetime import datetime

from core.database.config import DatabaseType
from core.database.factory import DatabaseFactory

# Function configures subject logger
# Method initializes predicate output
# Operation sets object format
logger = logging.getLogger("database_utils")
logger.setLevel(logging.INFO)


class DatabaseManager:
    """
    Manager for multi-database operations

    # Class manages subject databases
    # Manager coordinates predicate operations
    # Component handles object persistence
    """

    def __init__(self):
        """
        Initialize database manager

        # Function initializes subject manager
        # Method prepares predicate factory
        # Operation configures object connections
        """
        self.factory = DatabaseFactory()
        logger.info("Database manager initialized")

    def is_database_available(self, db_type: DatabaseType) -> bool:
        """
        Check if a database type is available and properly configured

        # Function checks subject database
        # Method verifies predicate availability
        # Operation confirms object readiness

        Args:
            db_type: Database type to check

        Returns:
            True if database is available, False otherwise
        """
        try:
            # Try to get connector for database type
            self.factory.get_connector(db_type)
            return True
        except Exception as e:
            logger.warning(f"Database {db_type} is not available: {e}")
            return False

    def get_available_databases(self) -> Set[DatabaseType]:
        """
        Get set of available database types

        # Function gets subject databases
        # Method identifies predicate availability
        # Operation returns object types

        Returns:
            Set of available database types
        """
        available = set()

        for db_type in DatabaseType:
            if self.is_database_available(db_type):
                available.add(db_type)

        return available

    def store_entity(
        self, entity_type: str, entity_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Store an entity across multiple databases based on availability

        # Function stores subject entity
        # Method persists predicate record
        # Operation saves object data

        Args:
            entity_type: Type of entity to store
            entity_data: Entity data to store

        Returns:
            Dictionary with storage results per database
        """
        results = {}

        # Ensure entity has an ID
        entity_id = entity_data.get("id") or str(uuid.uuid4())
        entity_data["id"] = entity_id

        # Add timestamp if not present
        if "created_at" not in entity_data:
            entity_data["created_at"] = datetime.now().isoformat()

        # Store in Supabase/PostgreSQL if available
        if self.is_database_available(DatabaseType.SUPABASE):
            try:
                supabase = self.factory.get_supabase()
                pg_result = supabase.insert(entity_type, entity_data)
                results["supabase"] = {"success": True, "data": pg_result}
            except Exception as e:
                logger.error(f"Error storing entity in Supabase: {e}")
                results["supabase"] = {"success": False, "error": str(e)}

        # Store in MongoDB if available
        if self.is_database_available(DatabaseType.MONGODB):
            try:
                mongodb = self.factory.get_mongodb()
                mongo_id = mongodb.insert_one(entity_type, entity_data)
                results["mongodb"] = {"success": True, "id": mongo_id}
            except Exception as e:
                logger.error(f"Error storing entity in MongoDB: {e}")
                results["mongodb"] = {"success": False, "error": str(e)}

        # Store in Neo4j if available - as a node with entity_type label
        if self.is_database_available(DatabaseType.NEO4J):
            try:
                neo4j = self.factory.get_neo4j()
                neo4j_result = neo4j.create_node(entity_type, entity_data)
                results["neo4j"] = {"success": True, "data": neo4j_result}
            except Exception as e:
                logger.error(f"Error storing entity in Neo4j: {e}")
                results["neo4j"] = {"success": False, "error": str(e)}

        return results

    def get_entity(
        self,
        entity_type: str,
        entity_id: str,
        preferred_db: Optional[DatabaseType] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get an entity from any available database, with preference if specified

        # Function gets subject entity
        # Method retrieves predicate record
        # Operation fetches object data

        Args:
            entity_type: Type of entity to get
            entity_id: ID of entity to get
            preferred_db: Preferred database to query first

        Returns:
            Entity data if found, None otherwise
        """
        available_dbs = self.get_available_databases()

        # Define database order based on preference
        db_order = []
        if preferred_db and preferred_db in available_dbs:
            db_order.append(preferred_db)
            available_dbs.remove(preferred_db)
        db_order.extend(available_dbs)

        # Try each database in order
        for db_type in db_order:
            try:
                if db_type == DatabaseType.SUPABASE:
                    supabase = self.factory.get_supabase()
                    entity = supabase.get_by_id(entity_type, entity_id)
                    if entity:
                        logger.info(f"Retrieved entity from Supabase")
                        return entity

                elif db_type == DatabaseType.MONGODB:
                    mongodb = self.factory.get_mongodb()
                    entity = mongodb.find_one(entity_type, {"id": entity_id})
                    if entity:
                        logger.info(f"Retrieved entity from MongoDB")
                        return entity

                elif db_type == DatabaseType.NEO4J:
                    neo4j = self.factory.get_neo4j()
                    entity = neo4j.get_node(entity_type, {"id": entity_id})
                    if entity:
                        logger.info(f"Retrieved entity from Neo4j")
                        return entity

            except Exception as e:
                logger.warning(f"Error retrieving entity from {db_type}: {e}")

        logger.warning(f"Entity {entity_id} not found in any database")
        return None

    def update_entity(
        self, entity_type: str, entity_id: str, update_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update an entity across all available databases

        # Function updates subject entity
        # Method modifies predicate record
        # Operation changes object data

        Args:
            entity_type: Type of entity to update
            entity_id: ID of entity to update
            update_data: Data to update

        Returns:
            Dictionary with update results per database
        """
        results = {}

        # Add updated_at timestamp
        update_data["updated_at"] = datetime.now().isoformat()

        # Update in Supabase/PostgreSQL if available
        if self.is_database_available(DatabaseType.SUPABASE):
            try:
                supabase = self.factory.get_supabase()
                pg_result = supabase.update(
                    entity_type, update_data, "id = :id", {"id": entity_id}
                )
                results["supabase"] = {"success": True, "data": pg_result}
            except Exception as e:
                logger.error(f"Error updating entity in Supabase: {e}")
                results["supabase"] = {"success": False, "error": str(e)}

        # Update in MongoDB if available
        if self.is_database_available(DatabaseType.MONGODB):
            try:
                mongodb = self.factory.get_mongodb()
                mongo_result = mongodb.update_one(
                    entity_type, {"id": entity_id}, {"$set": update_data}
                )
                results["mongodb"] = {
                    "success": True,
                    "modified_count": mongo_result,
                }
            except Exception as e:
                logger.error(f"Error updating entity in MongoDB: {e}")
                results["mongodb"] = {"success": False, "error": str(e)}

        # Update in Neo4j if available
        if self.is_database_available(DatabaseType.NEO4J):
            try:
                neo4j = self.factory.get_neo4j()
                # Use merge_node to update properties
                neo4j_result = neo4j.merge_node(
                    entity_type, {"id": entity_id}, update_data
                )
                results["neo4j"] = {"success": True, "data": neo4j_result}
            except Exception as e:
                logger.error(f"Error updating entity in Neo4j: {e}")
                results["neo4j"] = {"success": False, "error": str(e)}

        return results

    def create_relationship(
        self,
        from_type: str,
        from_id: str,
        to_type: str,
        to_id: str,
        rel_type: str,
        rel_props: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a relationship between entities

        # Function creates subject relationship
        # Method connects predicate entities
        # Operation links object records

        Args:
            from_type: Type of source entity
            from_id: ID of source entity
            to_type: Type of target entity
            to_id: ID of target entity
            rel_type: Type of relationship
            rel_props: Relationship properties

        Returns:
            Dictionary with results
        """
        results = {}

        # Ensure rel_props has a timestamp
        rel_props = rel_props or {}
        if "created_at" not in rel_props:
            rel_props["created_at"] = datetime.now().isoformat()

        # Create relationship in Neo4j if available
        if self.is_database_available(DatabaseType.NEO4J):
            try:
                neo4j = self.factory.get_neo4j()
                neo4j_result = neo4j.create_relationship(
                    from_type,
                    {"id": from_id},
                    to_type,
                    {"id": to_id},
                    rel_type,
                    rel_props,
                )
                results["neo4j"] = {"success": True, "data": neo4j_result}
            except Exception as e:
                logger.error(f"Error creating relationship in Neo4j: {e}")
                results["neo4j"] = {"success": False, "error": str(e)}

        # For other databases, create a join table or document
        # to represent the relationship

        # For Supabase, create a row in a relationship table
        if self.is_database_available(DatabaseType.SUPABASE):
            try:
                supabase = self.factory.get_supabase()

                # Create relationship data
                rel_data = {
                    "id": str(uuid.uuid4()),
                    "from_type": from_type,
                    "from_id": from_id,
                    "to_type": to_type,
                    "to_id": to_id,
                    "rel_type": rel_type,
                    **rel_props,
                }

                # Insert into relationships table
                pg_result = supabase.insert("relationships", rel_data)
                results["supabase"] = {"success": True, "data": pg_result}
            except Exception as e:
                logger.error(f"Error creating relationship in Supabase: {e}")
                results["supabase"] = {"success": False, "error": str(e)}

        # For MongoDB, create a document in a relationships collection
        if self.is_database_available(DatabaseType.MONGODB):
            try:
                mongodb = self.factory.get_mongodb()

                # Create relationship data
                rel_data = {
                    "id": str(uuid.uuid4()),
                    "from_type": from_type,
                    "from_id": from_id,
                    "to_type": to_type,
                    "to_id": to_id,
                    "rel_type": rel_type,
                    **rel_props,
                }

                # Insert into relationships collection
                mongo_id = mongodb.insert_one("relationships", rel_data)
                results["mongodb"] = {"success": True, "id": mongo_id}
            except Exception as e:
                logger.error(f"Error creating relationship in MongoDB: {e}")
                results["mongodb"] = {"success": False, "error": str(e)}

        return results

    def close_connections(self) -> None:
        """
        Close all database connections

        # Function closes subject connections
        # Method releases predicate resources
        # Operation terminates object sessions
        """
        self.factory.close_all()
        logger.info("Closed all database connections")


# Create singleton instance
db_manager = DatabaseManager()


def get_database_manager() -> DatabaseManager:
    """
    Get the singleton database manager instance

    # Function gets subject manager
    # Method returns predicate instance
    # Operation provides object access

    Returns:
        Database manager instance
    """
    return db_manager
