"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-PERIODIC-REGISTRY-0001              │
// │ 📁 domain       : Classification, Registry                  │
// │ 🧠 description  : Element registry for the                  │
// │                  CTAS Periodic Table of Nodes               │
// │ 🕸️ hash_type    : UUID → CUID-linked module                 │
// │ 🔄 parent_node  : NODE_CLASSIFICATION                      │
// │ 🧩 dependencies : uuid, dataclasses, sqlite3               │
// │ 🔧 tool_usage   : Storage, Retrieval                       │
// │ 📡 input_type   : Element metadata                         │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : persistence, lookup                       │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

Registry Implementation
-------------------
This module provides a registry for storing and retrieving elements
in the CTAS Periodic Table of Nodes.
"""

import uuid
import json
import sqlite3
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass, field
import os
import logging
from datetime import datetime
from pathlib import Path

from core.periodic_table.element import Element, ElementProperty
from core.periodic_table.group import Group, Period, Category
from core.periodic_table.relationships import RelationshipManager, Relationship

# Function configures subject logger
# Method initializes predicate output
# Operation sets object format
logger = logging.getLogger("periodic_table.registry")
logger.setLevel(logging.INFO)


class PeriodicTableRegistry:
    """
    Registry for elements in the CTAS Periodic Table.

    # Class manages subject registry
    # Registry handles predicate storage
    # Component controls object persistence
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize registry with optional database path.

        # Function initializes subject registry
        # Method prepares predicate storage
        # Operation configures object database

        Args:
            db_path: Path to SQLite database (default: in-memory)
        """
        self.db_path = db_path or ":memory:"
        self.conn = None
        self.relationship_manager = RelationshipManager()

        # Connect to database
        self._connect()

        # Initialize database schema
        self._initialize_schema()

        logger.info(
            f"Periodic Table Registry initialized with database: {self.db_path}"
        )

    def _connect(self):
        """
        Connect to the SQLite database.

        # Function connects subject database
        # Method establishes predicate connection
        # Operation initializes object storage
        """
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
        except sqlite3.Error as e:
            logger.error(f"Database connection error: {e}")
            raise

    def _initialize_schema(self):
        """
        Initialize database schema if it doesn't exist.

        # Function initializes subject schema
        # Method creates predicate tables
        # Operation sets object structure
        """
        if not self.conn:
            self._connect()

        cursor = self.conn.cursor()

        # Create elements table
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS elements (
            id TEXT PRIMARY KEY,
            symbol TEXT NOT NULL,
            name TEXT NOT NULL,
            atomic_number INTEGER NOT NULL,
            group_id TEXT,
            period_id TEXT,
            category_id TEXT,
            properties TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
        )

        # Create groups table
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS groups (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            symbol TEXT NOT NULL,
            number INTEGER NOT NULL,
            description TEXT,
            type TEXT NOT NULL,
            properties TEXT NOT NULL
        )
        """
        )

        # Create periods table
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS periods (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            number INTEGER NOT NULL,
            description TEXT,
            properties TEXT NOT NULL
        )
        """
        )

        # Create categories table
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS categories (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            symbol TEXT NOT NULL,
            color TEXT NOT NULL,
            description TEXT,
            parent_id TEXT,
            properties TEXT NOT NULL
        )
        """
        )

        # Create relationships table
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS relationships (
            id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            type TEXT NOT NULL,
            name TEXT NOT NULL,
            description TEXT,
            weight REAL NOT NULL,
            bidirectional INTEGER NOT NULL,
            temporal_validity TEXT,
            confidence REAL NOT NULL,
            properties TEXT NOT NULL
        )
        """
        )

        # Create indices
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_elements_symbol ON elements(symbol)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_elements_atomic_number ON elements(atomic_number)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_elements_group_id ON elements(group_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_elements_period_id ON elements(period_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_elements_category_id ON elements(category_id)"
        )

        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_relationships_source ON relationships(source_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_relationships_target ON relationships(target_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_relationships_type ON relationships(type)"
        )

        self.conn.commit()

    # Element methods
    def add_element(self, element: Element) -> None:
        """
        Add an element to the registry.

        # Function adds subject element
        # Method stores predicate data
        # Operation inserts object record

        Args:
            element: Element to add
        """
        if not self.conn:
            self._connect()

        cursor = self.conn.cursor()

        # Prepare data
        now = datetime.now().isoformat()

        # Convert ElementProperty enum keys to strings for JSON serialization
        properties_dict = {
            str(key.name): value for key, value in element.properties.items()
        }
        properties_json = json.dumps(properties_dict, default=str)

        # Insert element
        cursor.execute(
            """
        INSERT OR REPLACE INTO elements 
        (id, symbol, name, atomic_number, group_id, period_id, category_id, 
         properties, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                str(element.id),
                element.symbol,
                element.name,
                element.atomic_number,
                str(element.group_id) if element.group_id else None,
                str(element.period_id) if element.period_id else None,
                str(element.category_id) if element.category_id else None,
                properties_json,
                now,
                now,
            ),
        )

        self.conn.commit()
        logger.info(f"Added element: {element.symbol} ({element.name})")

    def get_element(self, element_id: uuid.UUID) -> Optional[Element]:
        """
        Get an element by ID.

        # Function gets subject element
        # Method retrieves predicate data
        # Operation fetches object record

        Args:
            element_id: Element ID

        Returns:
            Element if found, None otherwise
        """
        if not self.conn:
            self._connect()

        cursor = self.conn.cursor()

        # Query element
        cursor.execute(
            """
        SELECT * FROM elements WHERE id = ?
        """,
            (str(element_id),),
        )

        row = cursor.fetchone()
        if not row:
            return None

        # Convert properties from JSON
        properties = json.loads(row["properties"])

        # Reconstruct property enum keys
        enum_properties = {}
        for key_str, value in properties.items():
            try:
                key = ElementProperty[key_str]
                enum_properties[key] = value
            except KeyError:
                # Skip unknown properties
                pass

        # Create element instance
        element = Element(
            id=uuid.UUID(row["id"]),
            symbol=row["symbol"],
            name=row["name"],
            atomic_number=row["atomic_number"],
            group_id=uuid.UUID(row["group_id"]) if row["group_id"] else None,
            period_id=uuid.UUID(row["period_id"]) if row["period_id"] else None,
            category_id=(
                uuid.UUID(row["category_id"]) if row["category_id"] else None
            ),
        )

        # Set properties
        element.properties = enum_properties

        return element

    def get_element_by_symbol(self, symbol: str) -> Optional[Element]:
        """
        Get an element by symbol.

        # Function gets subject element
        # Method retrieves predicate data
        # Operation fetches object record

        Args:
            symbol: Element symbol

        Returns:
            Element if found, None otherwise
        """
        if not self.conn:
            self._connect()

        cursor = self.conn.cursor()

        # Query element
        cursor.execute(
            """
        SELECT id FROM elements WHERE symbol = ?
        """,
            (symbol,),
        )

        row = cursor.fetchone()
        if not row:
            return None

        return self.get_element(uuid.UUID(row["id"]))

    def get_element_by_atomic_number(
        self, atomic_number: int
    ) -> Optional[Element]:
        """
        Get an element by atomic number.

        # Function gets subject element
        # Method retrieves predicate data
        # Operation fetches object record

        Args:
            atomic_number: Element atomic number

        Returns:
            Element if found, None otherwise
        """
        if not self.conn:
            self._connect()

        cursor = self.conn.cursor()

        # Query element
        cursor.execute(
            """
        SELECT id FROM elements WHERE atomic_number = ?
        """,
            (atomic_number,),
        )

        row = cursor.fetchone()
        if not row:
            return None

        return self.get_element(uuid.UUID(row["id"]))

    def get_all_elements(self) -> List[Element]:
        """
        Get all elements in the registry.

        # Function gets subject elements
        # Method retrieves predicate data
        # Operation fetches object records

        Returns:
            List of all elements
        """
        if not self.conn:
            self._connect()

        cursor = self.conn.cursor()

        # Query all elements
        cursor.execute(
            """
        SELECT id FROM elements
        """
        )

        elements = []
        for row in cursor.fetchall():
            element = self.get_element(uuid.UUID(row["id"]))
            if element:
                elements.append(element)

        return elements

    def remove_element(self, element_id: uuid.UUID) -> bool:
        """
        Remove an element from the registry.

        # Function removes subject element
        # Method deletes predicate data
        # Operation removes object record

        Args:
            element_id: Element ID

        Returns:
            True if removed, False if not found
        """
        if not self.conn:
            self._connect()

        cursor = self.conn.cursor()

        # Query element first to check if it exists
        cursor.execute(
            """
        SELECT id FROM elements WHERE id = ?
        """,
            (str(element_id),),
        )

        if not cursor.fetchone():
            return False

        # Delete element
        cursor.execute(
            """
        DELETE FROM elements WHERE id = ?
        """,
            (str(element_id),),
        )

        self.conn.commit()
        logger.info(f"Removed element with ID: {element_id}")

        return True

    # Group methods
    def add_group(self, group: Group) -> None:
        """
        Add a group to the registry.

        # Function adds subject group
        # Method stores predicate data
        # Operation inserts object record

        Args:
            group: Group to add
        """
        if not self.conn:
            self._connect()

        cursor = self.conn.cursor()

        # Prepare data
        properties_json = json.dumps(group.properties, default=str)

        # Insert group
        cursor.execute(
            """
        INSERT OR REPLACE INTO groups 
        (id, name, symbol, number, description, type, properties)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                str(group.id),
                group.name,
                group.symbol,
                group.number,
                group.description,
                group.type.name,
                properties_json,
            ),
        )

        self.conn.commit()
        logger.info(f"Added group: {group.symbol} ({group.name})")

    def get_group(self, group_id: uuid.UUID) -> Optional[Group]:
        """
        Get a group by ID.

        # Function gets subject group
        # Method retrieves predicate data
        # Operation fetches object record

        Args:
            group_id: Group ID

        Returns:
            Group if found, None otherwise
        """
        if not self.conn:
            self._connect()

        cursor = self.conn.cursor()

        # Query group
        cursor.execute(
            """
        SELECT * FROM groups WHERE id = ?
        """,
            (str(group_id),),
        )

        row = cursor.fetchone()
        if not row:
            return None

        # Create group instance
        return Group.from_dict(
            {
                "id": row["id"],
                "name": row["name"],
                "symbol": row["symbol"],
                "number": row["number"],
                "description": row["description"],
                "type": row["type"],
                "properties": json.loads(row["properties"]),
            }
        )

    def get_all_groups(self) -> List[Group]:
        """
        Get all groups in the registry.

        # Function gets subject groups
        # Method retrieves predicate data
        # Operation fetches object records

        Returns:
            List of all groups
        """
        if not self.conn:
            self._connect()

        cursor = self.conn.cursor()

        # Query all groups
        cursor.execute(
            """
        SELECT * FROM groups
        """
        )

        groups = []
        for row in cursor.fetchall():
            group = Group.from_dict(
                {
                    "id": row["id"],
                    "name": row["name"],
                    "symbol": row["symbol"],
                    "number": row["number"],
                    "description": row["description"],
                    "type": row["type"],
                    "properties": json.loads(row["properties"]),
                }
            )
            groups.append(group)

        return groups

    # Period methods
    def add_period(self, period: Period) -> None:
        """
        Add a period to the registry.

        # Function adds subject period
        # Method stores predicate data
        # Operation inserts object record

        Args:
            period: Period to add
        """
        if not self.conn:
            self._connect()

        cursor = self.conn.cursor()

        # Prepare data
        properties_json = json.dumps(period.properties, default=str)

        # Insert period
        cursor.execute(
            """
        INSERT OR REPLACE INTO periods 
        (id, name, number, description, properties)
        VALUES (?, ?, ?, ?, ?)
        """,
            (
                str(period.id),
                period.name,
                period.number,
                period.description,
                properties_json,
            ),
        )

        self.conn.commit()
        logger.info(f"Added period: {period.name} (#{period.number})")

    def get_period(self, period_id: uuid.UUID) -> Optional[Period]:
        """
        Get a period by ID.

        # Function gets subject period
        # Method retrieves predicate data
        # Operation fetches object record

        Args:
            period_id: Period ID

        Returns:
            Period if found, None otherwise
        """
        if not self.conn:
            self._connect()

        cursor = self.conn.cursor()

        # Query period
        cursor.execute(
            """
        SELECT * FROM periods WHERE id = ?
        """,
            (str(period_id),),
        )

        row = cursor.fetchone()
        if not row:
            return None

        # Create period instance
        return Period.from_dict(
            {
                "id": row["id"],
                "name": row["name"],
                "number": row["number"],
                "description": row["description"],
                "properties": json.loads(row["properties"]),
            }
        )

    def get_all_periods(self) -> List[Period]:
        """
        Get all periods in the registry.

        # Function gets subject periods
        # Method retrieves predicate data
        # Operation fetches object records

        Returns:
            List of all periods
        """
        if not self.conn:
            self._connect()

        cursor = self.conn.cursor()

        # Query all periods
        cursor.execute(
            """
        SELECT * FROM periods
        """
        )

        periods = []
        for row in cursor.fetchall():
            period = Period.from_dict(
                {
                    "id": row["id"],
                    "name": row["name"],
                    "number": row["number"],
                    "description": row["description"],
                    "properties": json.loads(row["properties"]),
                }
            )
            periods.append(period)

        return periods

    # Category methods
    def add_category(self, category: Category) -> None:
        """
        Add a category to the registry.

        # Function adds subject category
        # Method stores predicate data
        # Operation inserts object record

        Args:
            category: Category to add
        """
        if not self.conn:
            self._connect()

        cursor = self.conn.cursor()

        # Prepare data
        properties_json = json.dumps(category.properties, default=str)

        # Insert category
        cursor.execute(
            """
        INSERT OR REPLACE INTO categories 
        (id, name, symbol, color, description, parent_id, properties)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                str(category.id),
                category.name,
                category.symbol,
                category.color,
                category.description,
                str(category.parent_id) if category.parent_id else None,
                properties_json,
            ),
        )

        self.conn.commit()
        logger.info(f"Added category: {category.symbol} ({category.name})")

    def get_category(self, category_id: uuid.UUID) -> Optional[Category]:
        """
        Get a category by ID.

        # Function gets subject category
        # Method retrieves predicate data
        # Operation fetches object record

        Args:
            category_id: Category ID

        Returns:
            Category if found, None otherwise
        """
        if not self.conn:
            self._connect()

        cursor = self.conn.cursor()

        # Query category
        cursor.execute(
            """
        SELECT * FROM categories WHERE id = ?
        """,
            (str(category_id),),
        )

        row = cursor.fetchone()
        if not row:
            return None

        # Create category instance
        return Category.from_dict(
            {
                "id": row["id"],
                "name": row["name"],
                "symbol": row["symbol"],
                "color": row["color"],
                "description": row["description"],
                "parent_id": row["parent_id"],
                "properties": json.loads(row["properties"]),
            }
        )

    def get_all_categories(self) -> List[Category]:
        """
        Get all categories in the registry.

        # Function gets subject categories
        # Method retrieves predicate data
        # Operation fetches object records

        Returns:
            List of all categories
        """
        if not self.conn:
            self._connect()

        cursor = self.conn.cursor()

        # Query all categories
        cursor.execute(
            """
        SELECT * FROM categories
        """
        )

        categories = []
        for row in cursor.fetchall():
            category = Category.from_dict(
                {
                    "id": row["id"],
                    "name": row["name"],
                    "symbol": row["symbol"],
                    "color": row["color"],
                    "description": row["description"],
                    "parent_id": row["parent_id"],
                    "properties": json.loads(row["properties"]),
                }
            )
            categories.append(category)

        return categories

    # Relationship methods
    def add_relationship(self, relationship: Relationship) -> None:
        """
        Add a relationship to the registry.

        # Function adds subject relationship
        # Method stores predicate data
        # Operation inserts object record

        Args:
            relationship: Relationship to add
        """
        if not self.conn:
            self._connect()

        cursor = self.conn.cursor()

        # Prepare data
        properties_json = json.dumps(relationship.properties, default=str)
        temporal_validity_json = (
            json.dumps(relationship.temporal_validity)
            if relationship.temporal_validity
            else None
        )

        # Insert relationship
        cursor.execute(
            """
        INSERT OR REPLACE INTO relationships 
        (id, source_id, target_id, type, name, description, weight, 
         bidirectional, temporal_validity, confidence, properties)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                str(relationship.id),
                str(relationship.source_id),
                str(relationship.target_id),
                relationship.type.name,
                relationship.name,
                relationship.description,
                relationship.weight,
                1 if relationship.bidirectional else 0,
                temporal_validity_json,
                relationship.confidence,
                properties_json,
            ),
        )

        self.conn.commit()

        # Add to relationship manager
        self.relationship_manager.add_relationship(relationship)

        logger.info(
            f"Added relationship: {relationship.name} ({relationship.type.name})"
        )

    def get_relationship(
        self, relationship_id: uuid.UUID
    ) -> Optional[Relationship]:
        """
        Get a relationship by ID.

        # Function gets subject relationship
        # Method retrieves predicate data
        # Operation fetches object record

        Args:
            relationship_id: Relationship ID

        Returns:
            Relationship if found, None otherwise
        """
        return self.relationship_manager.get_relationship(relationship_id)

    def load_relationships_from_db(self) -> None:
        """
        Load all relationships from the database into the relationship manager.

        # Function loads subject relationships
        # Method retrieves predicate data
        # Operation populates object manager
        """
        if not self.conn:
            self._connect()

        cursor = self.conn.cursor()

        # Query all relationships
        cursor.execute(
            """
        SELECT * FROM relationships
        """
        )

        for row in cursor.fetchall():
            # Create relationship instance
            relationship = Relationship.from_dict(
                {
                    "id": row["id"],
                    "source_id": row["source_id"],
                    "target_id": row["target_id"],
                    "type": row["type"],
                    "name": row["name"],
                    "description": row["description"],
                    "weight": row["weight"],
                    "bidirectional": bool(row["bidirectional"]),
                    "temporal_validity": (
                        json.loads(row["temporal_validity"])
                        if row["temporal_validity"]
                        else None
                    ),
                    "confidence": row["confidence"],
                    "properties": json.loads(row["properties"]),
                }
            )

            # Add to relationship manager
            self.relationship_manager.add_relationship(relationship)

        logger.info(
            f"Loaded {len(self.relationship_manager.relationships)} relationships from database"
        )

    def get_element_relationships(
        self, element_id: uuid.UUID
    ) -> Dict[str, List[Tuple[Relationship, uuid.UUID]]]:
        """
        Get all relationships for an element, grouped by type.

        # Function gets subject relationships
        # Method retrieves predicate connections
        # Operation returns object links

        Args:
            element_id: Element ID

        Returns:
            Dictionary mapping relationship types to lists of (relationship, connected_element_id) tuples
        """
        # Get all connections for the element
        connections = self.relationship_manager.get_connections(element_id)

        # Group by type
        grouped = {}
        for relationship, connected_id in connections:
            rel_type = relationship.type.name
            if rel_type not in grouped:
                grouped[rel_type] = []
            grouped[rel_type].append((relationship, connected_id))

        return grouped

    def save_to_file(self, file_path: str) -> None:
        """
        Save the registry to a file.

        # Function saves subject registry
        # Method serializes predicate data
        # Operation writes object file

        Args:
            file_path: Path to save file
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

        # Get all data
        elements = self.get_all_elements()
        groups = self.get_all_groups()
        periods = self.get_all_periods()
        categories = self.get_all_categories()

        # Load relationships
        self.load_relationships_from_db()

        # Create data structure
        data = {
            "elements": [element.to_dict() for element in elements],
            "groups": [group.to_dict() for group in groups],
            "periods": [period.to_dict() for period in periods],
            "categories": [category.to_dict() for category in categories],
            "relationships": self.relationship_manager.to_dict(),
        }

        # Save to file
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Saved registry to file: {file_path}")

    def load_from_file(self, file_path: str) -> None:
        """
        Load the registry from a file.

        # Function loads subject registry
        # Method deserializes predicate data
        # Operation reads object file

        Args:
            file_path: Path to load file
        """
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return

        try:
            # Read file
            with open(file_path, "r") as f:
                data = json.load(f)

            # Clear existing data
            if self.conn:
                cursor = self.conn.cursor()
                cursor.execute("DELETE FROM elements")
                cursor.execute("DELETE FROM groups")
                cursor.execute("DELETE FROM periods")
                cursor.execute("DELETE FROM categories")
                cursor.execute("DELETE FROM relationships")
                self.conn.commit()

            # Load groups
            for group_data in data.get("groups", []):
                group = Group.from_dict(group_data)
                self.add_group(group)

            # Load periods
            for period_data in data.get("periods", []):
                period = Period.from_dict(period_data)
                self.add_period(period)

            # Load categories
            for category_data in data.get("categories", []):
                category = Category.from_dict(category_data)
                self.add_category(category)

            # Load elements
            for element_data in data.get("elements", []):
                element = Element.from_dict(element_data)
                self.add_element(element)

            # Load relationships
            relationship_data = data.get("relationships", {})
            relationship_manager = RelationshipManager.from_dict(
                relationship_data
            )

            # Add relationships to database and manager
            for rel_id, rel in relationship_manager.relationships.items():
                self.add_relationship(rel)

            logger.info(f"Loaded registry from file: {file_path}")

        except Exception as e:
            logger.error(f"Error loading registry from file: {e}")
            raise

    def close(self):
        """
        Close the database connection.

        # Function closes subject connection
        # Method releases predicate resources
        # Operation cleans object state
        """
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("Database connection closed")
