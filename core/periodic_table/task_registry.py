"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-TASK-REGISTRY-0001                  │
// │ 📁 domain       : Classification, Registry                  │
// │ 🧠 description  : Task registry for the                     │
// │                  CTAS Periodic Table of Nodes               │
// │ 🕸️ hash_type    : UUID → CUID-linked module                 │
// │ 🔄 parent_node  : NODE_CLASSIFICATION                      │
// │ 🧩 dependencies : uuid, dataclasses, sqlite3               │
// │ 🔧 tool_usage   : Storage, Retrieval                       │
// │ 📡 input_type   : Task metadata                            │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : persistence, lookup                       │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

Task Registry Implementation
-------------------
This module provides a registry for storing and retrieving adversary tasks
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

from core.periodic_table.adversary_task import AdversaryTask
from core.periodic_table.simple_relationships import (
    RelationshipManager,
    Relationship,
)
from core.periodic_table.task_loader import (
    load_tasks_from_directory,
    create_demo_tasks,
)

# Configure logger
logger = logging.getLogger("periodic_table.task_registry")
logger.setLevel(logging.INFO)


class TaskRegistry:
    """
    Registry for adversary tasks in the CTAS Periodic Table.

    # Class manages subject registry
    # Registry handles predicate storage
    # Component controls object persistence
    """

    def __init__(
        self, db_path: Optional[str] = None, data_dir: Optional[str] = None
    ):
        """
        Initialize registry with optional database path.

        # Function initializes subject registry
        # Method prepares predicate storage
        # Operation configures object database

        Args:
            db_path: Path to SQLite database (default: in-memory)
            data_dir: Directory containing task JSON files (default: None)
        """
        self.db_path = db_path or ":memory:"
        self.data_dir = data_dir
        self.conn = None
        self.relationship_manager = RelationshipManager()
        self.tasks = {}  # In-memory cache of tasks by ID

        # Connect to database
        self._connect()

        # Initialize database schema
        self._initialize_schema()

        # Load tasks if data directory is provided
        if data_dir:
            self.load_tasks_from_directory(data_dir)
        else:
            logger.warning(
                "No data directory provided. Registry initialized empty."
            )

        # Load relationships
        self.load_relationships_from_db()

        logger.info(f"Task Registry initialized with database: {self.db_path}")

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
        try:
            if not self.conn:
                self._connect()

            cursor = self.conn.cursor()

            # Create tasks table
            cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                hash_id TEXT NOT NULL,
                task_name TEXT NOT NULL,
                description TEXT NOT NULL,
                data TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
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
                "CREATE INDEX IF NOT EXISTS idx_tasks_task_id ON tasks(task_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_tasks_hash_id ON tasks(hash_id)"
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
        except Exception as e:
            logger.error(f"Error initializing schema: {str(e)}")
            raise

    def load_tasks_from_directory(self, directory_path: str) -> None:
        """
        Load tasks from JSON files in the given directory.

        Args:
            directory_path: Path to directory containing task JSON files
        """
        tasks = load_tasks_from_directory(directory_path)
        for task in tasks:
            self.add_task(task)

        logger.info(
            f"Loaded {len(tasks)} tasks from directory: {directory_path}"
        )

    def initialize_demo_data(self) -> None:
        """
        Create and add demo tasks.
        """
        tasks = create_demo_tasks()
        for task in tasks:
            self.add_task(task)

        logger.info(f"Created {len(tasks)} demo tasks")

    # Task methods
    def add_task(self, task: AdversaryTask) -> None:
        """
        Add a task to the registry.

        # Function adds subject task
        # Method stores predicate data
        # Operation inserts object record

        Args:
            task: Task to add
        """
        if not self.conn:
            self._connect()

        cursor = self.conn.cursor()

        # Prepare data
        now = datetime.now().isoformat()

        # Convert task to dictionary for JSON serialization
        task_dict = task.to_dict()
        task_data = json.dumps(task_dict, default=str)

        # Insert task
        cursor.execute(
            """
        INSERT OR REPLACE INTO tasks 
        (id, task_id, hash_id, task_name, description, data, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                str(task.id),
                task.task_id,
                task.hash_id,
                task.task_name,
                task.description,
                task_data,
                now,
                now,
            ),
        )

        self.conn.commit()

        # Add to in-memory cache
        self.tasks[task.id] = task

        logger.info(f"Added task: {task.hash_id} ({task.task_name})")

    def get_task(self, task_id: uuid.UUID) -> Optional[AdversaryTask]:
        """
        Get a task by ID.

        # Function gets subject task
        # Method retrieves predicate data
        # Operation fetches object record

        Args:
            task_id: Task ID

        Returns:
            Task if found, None otherwise
        """
        # Check in-memory cache first
        if task_id in self.tasks:
            return self.tasks[task_id]

        if not self.conn:
            self._connect()

        cursor = self.conn.cursor()

        # Query task
        cursor.execute(
            """
        SELECT * FROM tasks WHERE id = ?
        """,
            (str(task_id),),
        )

        row = cursor.fetchone()
        if not row:
            return None

        # Parse task data
        task_data = json.loads(row["data"])
        task = AdversaryTask.from_dict(task_data)

        # Add to in-memory cache
        self.tasks[task.id] = task

        return task

    def get_task_by_hash_id(self, hash_id: str) -> Optional[AdversaryTask]:
        """
        Get a task by hash_id.

        # Function gets subject task
        # Method retrieves predicate data
        # Operation fetches object record

        Args:
            hash_id: Task hash_id (e.g., "SCH001-000")

        Returns:
            Task if found, None otherwise
        """
        # Check in-memory cache first
        for task in self.tasks.values():
            if task.hash_id == hash_id:
                return task

        if not self.conn:
            self._connect()

        cursor = self.conn.cursor()

        # Query task
        cursor.execute(
            """
        SELECT id FROM tasks WHERE hash_id = ?
        """,
            (hash_id,),
        )

        row = cursor.fetchone()
        if not row:
            return None

        return self.get_task(uuid.UUID(row["id"]))

    def get_task_by_task_id(self, task_id_str: str) -> Optional[AdversaryTask]:
        """
        Get a task by task_id.

        # Function gets subject task
        # Method retrieves predicate data
        # Operation fetches object record

        Args:
            task_id_str: Task ID string (e.g., "uuid-001-000-000")

        Returns:
            Task if found, None otherwise
        """
        # Check in-memory cache first
        for task in self.tasks.values():
            if task.task_id == task_id_str:
                return task

        if not self.conn:
            self._connect()

        cursor = self.conn.cursor()

        # Query task
        cursor.execute(
            """
        SELECT id FROM tasks WHERE task_id = ?
        """,
            (task_id_str,),
        )

        row = cursor.fetchone()
        if not row:
            return None

        return self.get_task(uuid.UUID(row["id"]))

    def get_all_tasks(self) -> List[AdversaryTask]:
        """
        Get all tasks in the registry.

        # Function gets subject tasks
        # Method retrieves predicate data
        # Operation fetches object records

        Returns:
            List of all tasks
        """
        # If in-memory cache is populated, use it
        if self.tasks:
            return list(self.tasks.values())

        if not self.conn:
            self._connect()

        cursor = self.conn.cursor()

        # Query all tasks
        cursor.execute(
            """
        SELECT id FROM tasks
        """
        )

        tasks = []
        for row in cursor.fetchall():
            task = self.get_task(uuid.UUID(row["id"]))
            if task:
                tasks.append(task)

        return tasks

    def remove_task(self, task_id: uuid.UUID) -> bool:
        """
        Remove a task from the registry.

        # Function removes subject task
        # Method deletes predicate data
        # Operation removes object record

        Args:
            task_id: Task ID

        Returns:
            True if removed, False if not found
        """
        if not self.conn:
            self._connect()

        cursor = self.conn.cursor()

        # Query task first to check if it exists
        cursor.execute(
            """
        SELECT id FROM tasks WHERE id = ?
        """,
            (str(task_id),),
        )

        if not cursor.fetchone():
            return False

        # Remove from in-memory cache
        if task_id in self.tasks:
            del self.tasks[task_id]

        # Delete task
        cursor.execute(
            """
        DELETE FROM tasks WHERE id = ?
        """,
            (str(task_id),),
        )

        self.conn.commit()
        logger.info(f"Removed task with ID: {task_id}")

        return True

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
        # Add to relationship manager
        if self.relationship_manager.add_relationship(relationship):
            if not self.conn:
                self._connect()

            cursor = self.conn.cursor()

            # Convert temporal_validity to JSON
            temporal_validity_json = None
            if relationship.temporal_validity:
                temporal_validity_json = json.dumps(
                    relationship.temporal_validity, default=str
                )

            # Prepare properties
            properties_json = json.dumps(relationship.properties, default=str)

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
                    relationship.type,
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
            logger.info(
                f"Added relationship: {relationship.type} from {relationship.source_id} to {relationship.target_id}"
            )

    def load_relationships_from_db(self) -> None:
        """
        Load relationships from the database into the relationship manager.

        # Function loads subject relationships
        # Method populates predicate manager
        # Operation fetches object records
        """
        if not self.conn:
            self._connect()

        cursor = self.conn.cursor()

        # Clear current relationships
        self.relationship_manager.clear()

        # Query all relationships
        cursor.execute(
            """
        SELECT * FROM relationships
        """
        )

        for row in cursor.fetchall():
            relationship = Relationship(
                id=uuid.UUID(row["id"]),
                source_id=uuid.UUID(row["source_id"]),
                target_id=uuid.UUID(row["target_id"]),
                type=row["type"],
                name=row["name"],
                description=row["description"],
                weight=row["weight"],
                bidirectional=bool(row["bidirectional"]),
                temporal_validity=(
                    json.loads(row["temporal_validity"])
                    if row["temporal_validity"]
                    else None
                ),
                confidence=row["confidence"],
                properties=json.loads(row["properties"]),
            )

            # Add to relationship manager (skip DB insert by using skip_save=True)
            self.relationship_manager.add_relationship(
                relationship, skip_save=True
            )

        logger.info(
            f"Loaded {len(self.relationship_manager.relationships)} relationships from database"
        )

    def get_relationships_for_task(
        self, task_id: uuid.UUID
    ) -> List[Relationship]:
        """
        Get all relationships for a task.

        # Function gets subject relationships
        # Method retrieves predicate connections
        # Operation fetches object connections

        Args:
            task_id: Task ID

        Returns:
            List of relationships where the task is source or target
        """
        return self.relationship_manager.get_relationships_for_node(task_id)

    def get_all_relationships(self) -> List[Relationship]:
        """
        Get all relationships in the registry.

        # Function gets all subject relationships
        # Method retrieves all predicate connections
        # Operation fetches all object connections

        Returns:
            List of all relationships
        """
        return self.relationship_manager.get_all_relationships()

    # Methods for compatibility with PeriodicTableRegistry
    def get_all_categories(self) -> List[Dict[str, Any]]:
        """
        Get all categories (for compatibility with PeriodicTableRegistry).

        Returns:
            List of category dictionaries
        """
        # Create synthetic categories based on task schema codes
        categories = {}

        for task in self.get_all_tasks():
            # Extract the main schema code (e.g., SCH001 from SCH001-000)
            if not task.hash_id or "-" not in task.hash_id:
                continue

            schema_code = task.hash_id.split("-")[0]

            if schema_code not in categories:
                category_id = str(uuid.uuid4())
                categories[schema_code] = {
                    "id": category_id,
                    "name": f"Category {schema_code}",
                    "description": f"Tasks in schema {schema_code}",
                    "color": "#4287f5",  # Default blue color
                    "tasks": [],
                }

            categories[schema_code]["tasks"].append(str(task.id))

        return list(categories.values())

    def get_all_groups(self) -> List[Dict[str, Any]]:
        """
        Get all groups (for compatibility with PeriodicTableRegistry).

        Returns:
            List of group dictionaries
        """
        # Create a default group for all tasks
        group_id = str(uuid.uuid4())
        default_group = {
            "id": group_id,
            "name": "All Tasks",
            "description": "All tasks in the registry",
            "tasks": [str(task.id) for task in self.get_all_tasks()],
        }

        return [default_group]

    def get_all_periods(self) -> List[Dict[str, Any]]:
        """
        Get all periods (for compatibility with PeriodicTableRegistry).

        Returns:
            List of period dictionaries
        """
        # Create a default period for all tasks
        period_id = str(uuid.uuid4())
        default_period = {
            "id": period_id,
            "name": "Current Period",
            "description": "Current operational period",
            "tasks": [str(task.id) for task in self.get_all_tasks()],
        }

        return [default_period]

    def get_task_by_id(self, task_id: str) -> Optional[AdversaryTask]:
        """
        Get a task by its UUID string (for compatibility with PeriodicTableRegistry).

        Args:
            task_id: Task ID as string

        Returns:
            Task if found, None otherwise
        """
        try:
            return self.get_task(uuid.UUID(task_id))
        except ValueError:
            return None

    def get_all_elements(self) -> List[Dict[str, Any]]:
        """
        Get all tasks formatted as elements (for compatibility with PeriodicTableRegistry).

        Returns:
            List of task dictionaries formatted as elements
        """
        elements = []

        for task in self.get_all_tasks():
            element = {
                "id": str(task.id),
                "name": task.task_name,
                "short_name": (
                    task.hash_id.split("-")[-1]
                    if task.hash_id and "-" in task.hash_id
                    else "?"
                ),
                "description": task.description,
                "task_id": task.task_id,
                "hash_id": task.hash_id,
                "properties": task.to_dict(),
                "group_id": None,
                "period_id": None,
                "category_id": None,
            }

            elements.append(element)

        return elements
