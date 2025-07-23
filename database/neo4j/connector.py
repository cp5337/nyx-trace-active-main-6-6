"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-NEO4J-CONNECTOR-0001                │
// │ 📁 domain       : Database, Neo4j                           │
// │ 🧠 description  : Neo4j connector for graph database        │
// │                  integration                                │
// │ 🕸️ hash_type    : UUID → CUID-linked module                 │
// │ 🔄 parent_node  : NODE_INTEGRATION                         │
// │ 🧩 dependencies : neo4j                                    │
// │ 🔧 tool_usage   : Data Access, Cypher                      │
// │ 📡 input_type   : Cypher queries, data objects             │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : data persistence, graph operations        │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

Neo4j Connector Module
--------------------
Provides a connector for interacting with Neo4j graph database
using the official Neo4j Python driver for Cypher query execution.
"""

import logging
from typing import Any, Dict, List, Optional, Union
import json

from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError

from core.database.config import Neo4jConfig

# Function configures subject logger
# Method initializes predicate output
# Operation sets object format
logger = logging.getLogger("neo4j_connector")
logger.setLevel(logging.INFO)


class Neo4jConnector:
    """
    Connector for Neo4j graph database

    # Class connects subject database
    # Connector manages predicate connection
    # Component handles object persistence
    """

    def __init__(self, config: Neo4jConfig):
        """
        Initialize Neo4j connector

        # Function initializes subject connector
        # Method configures predicate connection
        # Operation establishes object session

        Args:
            config: Neo4j configuration
        """
        self.config = config
        self.driver = None

        # Connect to database
        self._connect()

        logger.info("Neo4j connector initialized")

    def _connect(self) -> None:
        """
        Establish connection to Neo4j database

        # Function connects subject database
        # Method establishes predicate connection
        # Operation creates object driver
        """
        try:
            # Create Neo4j driver
            self.driver = GraphDatabase.driver(
                self.config.uri,
                auth=(self.config.username, self.config.password),
            )

            # Test connection
            with self.driver.session(database=self.config.database) as session:
                session.run("RETURN 1")

            logger.info("Connected to Neo4j database")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j database: {e}")
            raise

    def execute_query(
        self, query: str, params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query and return results

        # Function executes subject query
        # Method runs predicate Cypher
        # Operation returns object results

        Args:
            query: Cypher query to execute
            params: Query parameters

        Returns:
            List of dictionaries representing records
        """
        if self.driver is None:
            raise ValueError("Not connected to database")

        with self.driver.session(database=self.config.database) as session:
            try:
                result = session.run(query, params or {})

                # Convert result to list of dictionaries
                return [dict(record) for record in result]
            except Neo4jError as e:
                logger.error(f"Error executing query: {e}")
                raise

    def execute_with_json_result(
        self, query: str, params: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Execute a Cypher query and return results as JSON string

        # Function executes subject query
        # Method formats predicate results
        # Operation returns object JSON

        Args:
            query: Cypher query to execute
            params: Query parameters

        Returns:
            JSON string representing results
        """
        results = self.execute_query(query, params)
        return json.dumps(results, default=str)

    def create_node(
        self, label: str, properties: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a node with label and properties

        # Function creates subject node
        # Method stores predicate entity
        # Operation creates object vertex

        Args:
            label: Node label
            properties: Node properties

        Returns:
            Dictionary representing created node
        """
        query = f"""
        CREATE (n:{label} $props)
        RETURN n
        """

        results = self.execute_query(query, {"props": properties})
        return results[0]["n"] if results else {}

    def merge_node(
        self,
        label: str,
        match_properties: Dict[str, Any],
        update_properties: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Merge a node (create if not exists, update if exists)

        # Function merges subject node
        # Method upserts predicate entity
        # Operation updates object vertex

        Args:
            label: Node label
            match_properties: Properties to match existing node
            update_properties: Properties to update/set on the node

        Returns:
            Dictionary representing merged node
        """
        # Build match properties string
        match_props = ", ".join(
            [f"n.{k} = ${k}" for k in match_properties.keys()]
        )

        # Prepare parameters
        params = {**match_properties}

        if update_properties:
            # Add SET clause
            set_clause = "SET " + ", ".join(
                [f"n.{k} = ${k}_update" for k in update_properties.keys()]
            )

            # Add update properties to parameters with different keys
            for k, v in update_properties.items():
                params[f"{k}_update"] = v

            query = f"""
            MERGE (n:{label})
            ON MATCH {set_clause}
            ON CREATE {set_clause}
            WHERE {match_props}
            RETURN n
            """
        else:
            query = f"""
            MERGE (n:{label})
            WHERE {match_props}
            RETURN n
            """

        results = self.execute_query(query, params)
        return results[0]["n"] if results else {}

    def delete_node(self, label: str, match_properties: Dict[str, Any]) -> int:
        """
        Delete node(s) matching properties

        # Function deletes subject node
        # Method removes predicate entity
        # Operation eliminates object vertex

        Args:
            label: Node label
            match_properties: Properties to match nodes for deletion

        Returns:
            Number of nodes deleted
        """
        # Build match properties string
        match_props = " AND ".join(
            [f"n.{k} = ${k}" for k in match_properties.keys()]
        )

        query = f"""
        MATCH (n:{label})
        WHERE {match_props}
        DETACH DELETE n
        RETURN count(n) AS deleted_count
        """

        results = self.execute_query(query, match_properties)
        return results[0]["deleted_count"] if results else 0

    def create_relationship(
        self,
        from_label: str,
        from_props: Dict[str, Any],
        to_label: str,
        to_props: Dict[str, Any],
        rel_type: str,
        rel_props: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a relationship between two nodes

        # Function creates subject relationship
        # Method connects predicate nodes
        # Operation links object vertices

        Args:
            from_label: Label of source node
            from_props: Properties to match source node
            to_label: Label of target node
            to_props: Properties to match target node
            rel_type: Type of relationship
            rel_props: Properties for relationship

        Returns:
            Dictionary representing created relationship
        """
        # Build match properties strings
        from_match = " AND ".join(
            [f"n1.{k} = ${k}_from" for k in from_props.keys()]
        )
        to_match = " AND ".join([f"n2.{k} = ${k}_to" for k in to_props.keys()])

        # Prepare parameters
        params = {}
        for k, v in from_props.items():
            params[f"{k}_from"] = v
        for k, v in to_props.items():
            params[f"{k}_to"] = v

        # Add relationship properties
        if rel_props:
            params["rel_props"] = rel_props
            rel_props_str = " $rel_props"
        else:
            rel_props_str = ""

        query = f"""
        MATCH (n1:{from_label}), (n2:{to_label})
        WHERE {from_match} AND {to_match}
        CREATE (n1)-[r:{rel_type}{rel_props_str}]->(n2)
        RETURN r
        """

        results = self.execute_query(query, params)
        return results[0]["r"] if results else {}

    def get_node(
        self, label: str, properties: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get a node by label and properties

        # Function gets subject node
        # Method retrieves predicate entity
        # Operation fetches object vertex

        Args:
            label: Node label
            properties: Properties to match node

        Returns:
            Dictionary representing node
        """
        # Build match properties string
        match_props = " AND ".join([f"n.{k} = ${k}" for k in properties.keys()])

        query = f"""
        MATCH (n:{label})
        WHERE {match_props}
        RETURN n
        LIMIT 1
        """

        results = self.execute_query(query, properties)
        return results[0]["n"] if results else {}

    def get_nodes_by_label(
        self, label: str, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all nodes with a specific label

        # Function gets subject nodes
        # Method retrieves predicate entities
        # Operation fetches object vertices

        Args:
            label: Node label
            limit: Maximum number of nodes to return

        Returns:
            List of dictionaries representing nodes
        """
        query = f"""
        MATCH (n:{label})
        RETURN n
        """

        if limit is not None:
            query += f" LIMIT {limit}"

        results = self.execute_query(query)
        return [record["n"] for record in results]

    def execute_graph_algorithm(
        self, algorithm: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a graph algorithm

        # Function executes subject algorithm
        # Method runs predicate analysis
        # Operation computes object metrics

        Args:
            algorithm: Name of graph algorithm
            params: Algorithm parameters

        Returns:
            Dictionary representing algorithm results
        """
        # Example implementation, needs to be customized based on Neo4j Graph Data Science library
        query = f"""
        CALL gds.{algorithm}($params)
        YIELD *
        """

        results = self.execute_query(query, {"params": params})
        return results[0] if results else {}

    def close(self) -> None:
        """
        Close database connection

        # Function closes subject connection
        # Method releases predicate resources
        # Operation disposes object driver
        """
        if self.driver:
            self.driver.close()
            logger.info("Closed connection to Neo4j database")
