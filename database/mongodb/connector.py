"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-MONGODB-CONNECTOR-0001              │
// │ 📁 domain       : Database, MongoDB                         │
// │ 🧠 description  : MongoDB connector for document database   │
// │                  integration                                │
// │ 🕸️ hash_type    : UUID → CUID-linked module                 │
// │ 🔄 parent_node  : NODE_INTEGRATION                         │
// │ 🧩 dependencies : pymongo                                  │
// │ 🔧 tool_usage   : Data Access, NoSQL                       │
// │ 📡 input_type   : MongoDB operations, data objects         │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : data persistence, document operations     │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

MongoDB Connector Module
----------------------
Provides a connector for interacting with MongoDB document database
using PyMongo for storing and retrieving unstructured data.
"""

import logging
from typing import Any, Dict, List, Optional, Union
import json
from bson import ObjectId, json_util
from datetime import datetime

import pymongo
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import PyMongoError

from core.database.config import MongoDBConfig

# Function configures subject logger
# Method initializes predicate output
# Operation sets object format
logger = logging.getLogger("mongodb_connector")
logger.setLevel(logging.INFO)


class JSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to handle MongoDB specific types

    # Class encodes subject JSON
    # Encoder converts predicate BSON
    # Component serializes object values
    """

    def default(self, o: Any) -> Any:
        """
        Convert MongoDB-specific types to JSON serializable types

        # Function converts subject types
        # Method transforms predicate values
        # Operation serializes object data

        Args:
            o: Object to encode

        Returns:
            JSON serializable value
        """
        if isinstance(o, ObjectId):
            return str(o)
        elif isinstance(o, datetime):
            return o.isoformat()
        return super().default(o)


class MongoDBConnector:
    """
    Connector for MongoDB document database

    # Class connects subject database
    # Connector manages predicate connection
    # Component handles object persistence
    """

    def __init__(self, config: MongoDBConfig):
        """
        Initialize MongoDB connector

        # Function initializes subject connector
        # Method configures predicate connection
        # Operation establishes object client

        Args:
            config: MongoDB configuration
        """
        self.config = config
        self.client = None
        self.db = None

        # Connect to database
        self._connect()

        logger.info("MongoDB connector initialized")

    def _connect(self) -> None:
        """
        Establish connection to MongoDB database

        # Function connects subject database
        # Method establishes predicate connection
        # Operation creates object client
        """
        try:
            # Create MongoDB client
            self.client = MongoClient(self.config.connection_string)

            # Get database
            self.db = self.client[self.config.database]

            # Test connection
            self.client.admin.command("ping")

            logger.info(
                f"Connected to MongoDB database: {self.config.database}"
            )
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB database: {e}")
            raise

    def get_collection(self, collection_name: str) -> Collection:
        """
        Get a MongoDB collection

        # Function gets subject collection
        # Method retrieves predicate container
        # Operation returns object reference

        Args:
            collection_name: Name of the collection

        Returns:
            MongoDB collection
        """
        if self.db is None:
            raise ValueError("Not connected to database")

        return self.db[collection_name]

    def insert_one(self, collection_name: str, document: Dict[str, Any]) -> str:
        """
        Insert a document into a collection

        # Function inserts subject document
        # Method stores predicate record
        # Operation creates object data

        Args:
            collection_name: Name of the collection
            document: Document to insert

        Returns:
            ID of inserted document
        """
        collection = self.get_collection(collection_name)

        try:
            result = collection.insert_one(document)
            logger.info(
                f"Inserted document in {collection_name} with id: {result.inserted_id}"
            )
            return str(result.inserted_id)
        except PyMongoError as e:
            logger.error(f"Error inserting document: {e}")
            raise

    def insert_many(
        self, collection_name: str, documents: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Insert multiple documents into a collection

        # Function inserts subject documents
        # Method stores predicate records
        # Operation creates object dataset

        Args:
            collection_name: Name of the collection
            documents: List of documents to insert

        Returns:
            List of inserted document IDs
        """
        collection = self.get_collection(collection_name)

        try:
            result = collection.insert_many(documents)
            inserted_ids = [str(id) for id in result.inserted_ids]
            logger.info(
                f"Inserted {len(inserted_ids)} documents in {collection_name}"
            )
            return inserted_ids
        except PyMongoError as e:
            logger.error(f"Error inserting documents: {e}")
            raise

    def find_one(
        self, collection_name: str, query: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Find a single document in a collection

        # Function finds subject document
        # Method retrieves predicate record
        # Operation fetches object data

        Args:
            collection_name: Name of the collection
            query: Query to find document

        Returns:
            Document if found, None otherwise
        """
        collection = self.get_collection(collection_name)

        try:
            document = collection.find_one(query)
            return document
        except PyMongoError as e:
            logger.error(f"Error finding document: {e}")
            raise

    def find(
        self,
        collection_name: str,
        query: Dict[str, Any],
        limit: Optional[int] = None,
        skip: Optional[int] = None,
        sort: Optional[List[tuple]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find documents in a collection

        # Function finds subject documents
        # Method retrieves predicate records
        # Operation fetches object dataset

        Args:
            collection_name: Name of the collection
            query: Query to find documents
            limit: Maximum number of documents to return
            skip: Number of documents to skip
            sort: List of (field, direction) tuples to sort by

        Returns:
            List of documents
        """
        collection = self.get_collection(collection_name)

        try:
            cursor = collection.find(query)

            if skip is not None:
                cursor = cursor.skip(skip)

            if limit is not None:
                cursor = cursor.limit(limit)

            if sort is not None:
                cursor = cursor.sort(sort)

            return list(cursor)
        except PyMongoError as e:
            logger.error(f"Error finding documents: {e}")
            raise

    def update_one(
        self,
        collection_name: str,
        query: Dict[str, Any],
        update: Dict[str, Any],
        upsert: bool = False,
    ) -> int:
        """
        Update a single document in a collection

        # Function updates subject document
        # Method modifies predicate record
        # Operation changes object data

        Args:
            collection_name: Name of the collection
            query: Query to find document to update
            update: Update operations to apply
            upsert: Create document if not found

        Returns:
            Number of documents updated
        """
        collection = self.get_collection(collection_name)

        try:
            result = collection.update_one(query, update, upsert=upsert)
            logger.info(
                f"Updated {result.modified_count} document in {collection_name}"
            )
            return result.modified_count
        except PyMongoError as e:
            logger.error(f"Error updating document: {e}")
            raise

    def update_many(
        self,
        collection_name: str,
        query: Dict[str, Any],
        update: Dict[str, Any],
        upsert: bool = False,
    ) -> int:
        """
        Update multiple documents in a collection

        # Function updates subject documents
        # Method modifies predicate records
        # Operation changes object dataset

        Args:
            collection_name: Name of the collection
            query: Query to find documents to update
            update: Update operations to apply
            upsert: Create documents if not found

        Returns:
            Number of documents updated
        """
        collection = self.get_collection(collection_name)

        try:
            result = collection.update_many(query, update, upsert=upsert)
            logger.info(
                f"Updated {result.modified_count} documents in {collection_name}"
            )
            return result.modified_count
        except PyMongoError as e:
            logger.error(f"Error updating documents: {e}")
            raise

    def delete_one(self, collection_name: str, query: Dict[str, Any]) -> int:
        """
        Delete a single document from a collection

        # Function deletes subject document
        # Method removes predicate record
        # Operation eliminates object data

        Args:
            collection_name: Name of the collection
            query: Query to find document to delete

        Returns:
            Number of documents deleted
        """
        collection = self.get_collection(collection_name)

        try:
            result = collection.delete_one(query)
            logger.info(
                f"Deleted {result.deleted_count} document from {collection_name}"
            )
            return result.deleted_count
        except PyMongoError as e:
            logger.error(f"Error deleting document: {e}")
            raise

    def delete_many(self, collection_name: str, query: Dict[str, Any]) -> int:
        """
        Delete multiple documents from a collection

        # Function deletes subject documents
        # Method removes predicate records
        # Operation eliminates object dataset

        Args:
            collection_name: Name of the collection
            query: Query to find documents to delete

        Returns:
            Number of documents deleted
        """
        collection = self.get_collection(collection_name)

        try:
            result = collection.delete_many(query)
            logger.info(
                f"Deleted {result.deleted_count} documents from {collection_name}"
            )
            return result.deleted_count
        except PyMongoError as e:
            logger.error(f"Error deleting documents: {e}")
            raise

    def aggregate(
        self, collection_name: str, pipeline: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Execute an aggregation pipeline on a collection

        # Function aggregates subject data
        # Method processes predicate pipeline
        # Operation analyzes object collection

        Args:
            collection_name: Name of the collection
            pipeline: Aggregation pipeline stages

        Returns:
            List of documents resulting from aggregation
        """
        collection = self.get_collection(collection_name)

        try:
            return list(collection.aggregate(pipeline))
        except PyMongoError as e:
            logger.error(f"Error executing aggregation: {e}")
            raise

    def count_documents(
        self, collection_name: str, query: Dict[str, Any]
    ) -> int:
        """
        Count documents in a collection matching a query

        # Function counts subject documents
        # Method quantifies predicate records
        # Operation measures object dataset

        Args:
            collection_name: Name of the collection
            query: Query to match documents

        Returns:
            Number of matching documents
        """
        collection = self.get_collection(collection_name)

        try:
            return collection.count_documents(query)
        except PyMongoError as e:
            logger.error(f"Error counting documents: {e}")
            raise

    def create_index(
        self,
        collection_name: str,
        keys: Union[str, List[tuple]],
        unique: bool = False,
    ) -> str:
        """
        Create an index on a collection

        # Function creates subject index
        # Method optimizes predicate access
        # Operation accelerates object queries

        Args:
            collection_name: Name of the collection
            keys: Index keys (either a string or a list of (field, direction) tuples)
            unique: Whether the index should enforce uniqueness

        Returns:
            Name of created index
        """
        collection = self.get_collection(collection_name)

        try:
            result = collection.create_index(keys, unique=unique)
            logger.info(f"Created index {result} on {collection_name}")
            return result
        except PyMongoError as e:
            logger.error(f"Error creating index: {e}")
            raise

    def drop_collection(self, collection_name: str) -> None:
        """
        Drop a collection

        # Function drops subject collection
        # Method removes predicate container
        # Operation deletes object dataset

        Args:
            collection_name: Name of the collection to drop
        """
        if self.db is None:
            raise ValueError("Not connected to database")

        try:
            self.db.drop_collection(collection_name)
            logger.info(f"Dropped collection {collection_name}")
        except PyMongoError as e:
            logger.error(f"Error dropping collection: {e}")
            raise

    def to_json(self, data: Any) -> str:
        """
        Convert MongoDB data to JSON string

        # Function converts subject data
        # Method transforms predicate BSON
        # Operation serializes object values

        Args:
            data: MongoDB data to convert

        Returns:
            JSON string
        """
        return json.dumps(data, cls=JSONEncoder)

    def close(self) -> None:
        """
        Close database connection

        # Function closes subject connection
        # Method releases predicate resources
        # Operation terminates object client
        """
        if self.client:
            self.client.close()
            logger.info("Closed connection to MongoDB database")
