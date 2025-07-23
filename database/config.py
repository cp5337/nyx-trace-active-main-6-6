"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-DATABASE-CONFIG-0001                │
// │ 📁 domain       : Database, Configuration                   │
// │ 🧠 description  : Database configuration management for     │
// │                  multi-database architecture                │
// │ 🕸️ hash_type    : UUID → CUID-linked module                 │
// │ 🔄 parent_node  : NODE_INTEGRATION                         │
// │ 🧩 dependencies : dotenv, pydantic                         │
// │ 🔧 tool_usage   : Configuration, Management                │
// │ 📡 input_type   : Environment variables                    │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : configuration, management                │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

Database Configuration Module
---------------------------
Provides configuration management for multi-database architecture,
supporting Supabase, Neo4j, and MongoDB connections.
"""

import os
import logging
from typing import Dict, Optional, Any
from enum import Enum
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Function configures subject logger
# Method initializes predicate output
# Operation sets object format
logger = logging.getLogger("database_config")
logger.setLevel(logging.INFO)

# Load environment variables
load_dotenv()


class DatabaseType(str, Enum):
    """
    Enumeration of supported database types

    # Class defines subject types
    # Enumeration lists predicate options
    # Type specifies object choices
    """

    SUPABASE = "supabase"
    NEO4J = "neo4j"
    MONGODB = "mongodb"


class SupabaseConfig(BaseModel):
    """
    Supabase configuration model

    # Class defines subject config
    # Model specifies predicate parameters
    # Type documents object fields
    """

    url: str = Field(..., description="Supabase project URL")
    key: str = Field(..., description="Supabase API key")
    database_url: str = Field(..., description="PostgreSQL connection string")
    schema: str = Field("public", description="PostgreSQL schema")

    @classmethod
    def from_env(cls) -> "SupabaseConfig":
        """
        Create configuration from environment variables

        # Function creates subject config
        # Method loads predicate variables
        # Operation builds object instance
        """
        return cls(
            url=os.getenv("SUPABASE_URL", ""),
            key=os.getenv("SUPABASE_KEY", ""),
            database_url=os.getenv("DATABASE_URL", ""),
            schema=os.getenv("SUPABASE_SCHEMA", "public"),
        )


class Neo4jConfig(BaseModel):
    """
    Neo4j configuration model

    # Class defines subject config
    # Model specifies predicate parameters
    # Type documents object fields
    """

    uri: str = Field(..., description="Neo4j connection URI")
    username: str = Field(..., description="Neo4j username")
    password: str = Field(..., description="Neo4j password")
    database: Optional[str] = Field(None, description="Neo4j database name")

    @classmethod
    def from_env(cls) -> "Neo4jConfig":
        """
        Create configuration from environment variables

        # Function creates subject config
        # Method loads predicate variables
        # Operation builds object instance
        """
        return cls(
            uri=os.getenv("NEO4J_URI", ""),
            username=os.getenv("NEO4J_USERNAME", ""),
            password=os.getenv("NEO4J_PASSWORD", ""),
            database=os.getenv("NEO4J_DATABASE"),
        )


class MongoDBConfig(BaseModel):
    """
    MongoDB configuration model

    # Class defines subject config
    # Model specifies predicate parameters
    # Type documents object fields
    """

    connection_string: str = Field(..., description="MongoDB connection string")
    database: str = Field(..., description="MongoDB database name")

    @classmethod
    def from_env(cls) -> "MongoDBConfig":
        """
        Create configuration from environment variables

        # Function creates subject config
        # Method loads predicate variables
        # Operation builds object instance
        """
        return cls(
            connection_string=os.getenv("MONGODB_URI", ""),
            database=os.getenv("MONGODB_DATABASE", "nyxtrace"),
        )


class DatabaseConfig:
    """
    Database configuration manager for multi-database setup

    # Class manages subject config
    # Manager handles predicate settings
    # Component controls object parameters
    """

    def __init__(self):
        """
        Initialize database configuration manager

        # Function initializes subject manager
        # Method prepares predicate configs
        # Operation sets object defaults
        """
        self.configs = {}

        # Load configurations from environment
        self._load_from_env()

        logger.info("Database configuration initialized")

    def _load_from_env(self) -> None:
        """
        Load all database configurations from environment variables

        # Function loads subject configs
        # Method reads predicate variables
        # Operation populates object settings
        """
        # Supabase config
        try:
            self.configs[DatabaseType.SUPABASE] = SupabaseConfig.from_env()
            logger.info("Supabase configuration loaded")
        except Exception as e:
            logger.warning(f"Failed to load Supabase configuration: {e}")

        # Neo4j config
        try:
            self.configs[DatabaseType.NEO4J] = Neo4jConfig.from_env()
            logger.info("Neo4j configuration loaded")
        except Exception as e:
            logger.warning(f"Failed to load Neo4j configuration: {e}")

        # MongoDB config
        try:
            self.configs[DatabaseType.MONGODB] = MongoDBConfig.from_env()
            logger.info("MongoDB configuration loaded")
        except Exception as e:
            logger.warning(f"Failed to load MongoDB configuration: {e}")

    def get_config(self, db_type: DatabaseType) -> Any:
        """
        Get configuration for a specific database type

        # Function gets subject config
        # Method retrieves predicate settings
        # Operation returns object parameters

        Args:
            db_type: Database type to get configuration for

        Returns:
            Configuration model for the specified database type

        Raises:
            ValueError: If configuration for the specified database type is not found
        """
        if db_type not in self.configs:
            raise ValueError(f"Configuration for {db_type} not found")

        return self.configs[db_type]

    def is_configured(self, db_type: DatabaseType) -> bool:
        """
        Check if configuration for a database type exists and is valid

        # Function checks subject config
        # Method validates predicate existence
        # Operation verifies object parameters

        Args:
            db_type: Database type to check configuration for

        Returns:
            True if configuration exists, False otherwise
        """
        return db_type in self.configs

    def get_all_configs(self) -> Dict[DatabaseType, Any]:
        """
        Get all database configurations

        # Function gets subject configs
        # Method retrieves predicate settings
        # Operation returns object parameters

        Returns:
            Dictionary of all database configurations
        """
        return self.configs.copy()
