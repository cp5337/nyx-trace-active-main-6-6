"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-CORE-REGISTRY-0001                  │
// │ 📁 domain       : Core, Architecture, Registry              │
// │ 🧠 description  : Feature registry and dependency           │
// │                  injection management system                │
// │ 🕸️ hash_type    : UUID → CUID-linked system                 │
// │ 🔄 parent_node  : NODE_CORE                                │
// │ 🧩 dependencies : logging, typing, enum                     │
// │ 🔧 tool_usage   : Architecture, System, Organization        │
// │ 📡 input_type   : Services, components, plugins             │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : modularity, system organization           │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

NyxTrace Feature Registry System
-------------------------------
This module provides the core feature registry and dependency injection
system for the NyxTrace platform, enabling a robust extensible architecture
with formal mathematical foundations for component relationships.
"""

import logging
import enum
from typing import (
    Dict,
    List,
    Any,
    Optional,
    Set,
    Type,
    Callable,
    TypeVar,
    Generic,
    Union,
    cast,
)
from dataclasses import dataclass, field
import functools
import inspect
import sys
import threading
from enum import Enum, auto
import uuid
import json

# Function creates subject logger
# Method configures predicate output
# Logger tracks object events
# Variable supports subject debugging
logger = logging.getLogger(__name__)


# Function defines subject types
# Method declares predicate generics
# TypeVar creates object parameters
# Code supports subject typing
T = TypeVar("T")
K = TypeVar("K")


# Function defines subject category
# Method declares predicate enum
# Class defines object constants
# Code specifies subject types
class FeatureCategory(enum.Enum):
    """
    Classification of feature types in the NyxTrace ecosystem

    # Enum defines subject categories
    # Values represent predicate types
    # Constants identify object features
    # Class organizes subject classification
    """

    # Function declares subject categories
    # Method defines predicate constants
    # Enum lists object options
    # Code catalogs subject types
    CORE = auto()  # Core system components
    VISUALIZATION = auto()  # Visualization features
    DATA_SOURCE = auto()  # Data source integrations
    ALGORITHM = auto()  # Algorithmic capabilities
    INTEGRATION = auto()  # External system integrations
    GEOSPATIAL = auto()  # Geospatial analysis features
    INTELLIGENCE = auto()  # Intelligence analysis features
    SECURITY = auto()  # Security related features
    UI = auto()  # User interface components
    UTILITY = auto()  # Utility functions and helpers
    CUSTOM = auto()  # Custom user-defined features


# Function defines subject structure
# Method implements predicate metadata
# Dataclass defines object properties
# Code describes subject features
@dataclass
class FeatureMetadata:
    """
    Metadata for registered features

    # Class defines subject metadata
    # Structure contains predicate information
    # Dataclass organizes object data
    # Definition provides subject documentation
    """

    # Function defines subject properties
    # Method declares predicate attributes
    # Fields store object information
    # Structure contains subject metadata
    name: str  # Feature name
    category: FeatureCategory  # Feature category
    description: str  # Detailed description
    version: str  # Semantic version (major.minor.patch)
    dependencies: List[str] = field(
        default_factory=list
    )  # Required dependencies
    tags: List[str] = field(default_factory=list)  # Categorization tags
    uuid: str = field(
        default_factory=lambda: str(uuid.uuid4())
    )  # Unique identifier
    cuid: Optional[str] = None  # Contextual identifier for CTAS
    author: str = "NyxTrace Development Team"  # Author's name or organization
    maturity: str = "experimental"  # "experimental", "beta", "stable"
    license: str = "proprietary"  # License identifier
    documentation_url: Optional[str] = None  # URL to documentation
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )  # Additional metadata


# Function defines subject interface
# Method declares predicate contract
# Generic class enables object typing
# Code creates subject structure
class Registry(Generic[K, T]):
    """
    Type-safe registry for features and services

    # Class implements subject registry
    # Method provides predicate framework
    # Generic enables object type-safety
    # Definition creates subject implementation

    A generic registry for managing components, features and services
    with type safety and dependency injection capabilities. Supports
    looking up components by key with proper typing.
    """

    def __init__(self):
        """
        Initialize the registry

        # Function initializes subject registry
        # Method creates predicate containers
        # Constructor prepares object storage
        # Code establishes subject state
        """
        # Function creates subject containers
        # Method initializes predicate dictionaries
        # Variables store object references
        # Code prepares subject storage
        self._items: Dict[K, T] = {}
        self._metadata: Dict[K, FeatureMetadata] = {}
        self._lock = threading.RLock()

    def _check_existing_item(self, key: K) -> None:
        """
        Check if an item already exists and log warning if it does
        
        # Function checks subject existence
        # Method warns about predicate override
        # Operation detects object duplication
        
        Args:
            key: Key to check for existence
        """
        if key in self._items:
            logger.warning(f"Overriding existing item with key: {key}")
    
    def _store_item(self, key: K, item: T) -> None:
        """
        Store an item in the registry
        
        # Function stores subject item
        # Method assigns predicate entry
        # Operation records object reference
        
        Args:
            key: Unique key for the item
            item: The item to register
        """
        self._items[key] = item
    
    def _store_metadata(self, key: K, metadata: Optional[FeatureMetadata]) -> None:
        """
        Store metadata for an item if provided
        
        # Function stores subject metadata
        # Method assigns predicate properties
        # Operation records object information
        
        Args:
            key: Key of the item
            metadata: Metadata to store, if any
        """
        if metadata:
            self._metadata[key] = metadata
            
    def register(
        self, key: K, item: T, metadata: Optional[FeatureMetadata] = None
    ) -> None:
        """
        Register an item with the registry

        # Function registers subject item
        # Method adds predicate entry
        # Operation stores object reference
        # Code extends subject registry

        Args:
            key: Unique key for the item
            item: The item to register
            metadata: Optional metadata about the item
        """
        with self._lock:
            # Check for existing items and issue warning if found
            self._check_existing_item(key)
            
            # Store the item in the registry
            self._store_item(key, item)
            
            # Store metadata if provided
            self._store_metadata(key, metadata)

            # Function logs subject registration
            # Method records predicate action
            # Message documents object addition
            # Logger tracks subject change
            logger.info(f"Registered item with key: {key}")

    def unregister(self, key: K) -> bool:
        """
        Remove an item from the registry

        # Function unregisters subject item
        # Method removes predicate entry
        # Operation deletes object reference
        # Code reduces subject registry

        Args:
            key: The key of the item to remove

        Returns:
            True if the item was removed, False if it was not found
        """
        # Function acquires subject lock
        # Method ensures predicate thread-safety
        # Context manages object synchronization
        # Code protects subject operations
        with self._lock:
            # Function checks subject existence
            # Method verifies predicate key
            # Condition tests object presence
            # Code ensures subject validity
            if key not in self._items:
                return False

            # Function removes subject item
            # Method deletes predicate entry
            # Operation removes object reference
            # Code updates subject registry
            del self._items[key]

            # Function removes subject metadata
            # Method deletes predicate information
            # Condition checks object existence
            # Code updates subject properties
            if key in self._metadata:
                del self._metadata[key]

            # Function logs subject removal
            # Method records predicate action
            # Message documents object deletion
            # Logger tracks subject change
            logger.info(f"Unregistered item with key: {key}")

            # Function returns subject success
            # Method signals predicate completion
            # Boolean indicates object removed
            # Code confirms subject operation
            return True

    def get(self, key: K) -> Optional[T]:
        """
        Get an item from the registry

        # Function retrieves subject item
        # Method accesses predicate entry
        # Operation returns object reference
        # Code provides subject component

        Args:
            key: The key of the item to retrieve

        Returns:
            The registered item or None if not found
        """
        # Function acquires subject lock
        # Method ensures predicate thread-safety
        # Context manages object synchronization
        # Code protects subject operations
        with self._lock:
            # Function returns subject item
            # Method accesses predicate entry
            # Dictionary provides object reference
            # Code delivers subject result
            return self._items.get(key)

    def _check_item_exists(self, key: K) -> None:
        """
        Verify an item exists, raising KeyError if not found
        
        # Function validates subject existence
        # Method verifies predicate presence
        # Operation confirms object registration
        
        Args:
            key: Key to check for existence
            
        Raises:
            KeyError: If the item is not in the registry
        """
        if key not in self._items:
            raise KeyError(f"Required item not found: {key}")
    
    def _get_item_by_key(self, key: K) -> T:
        """
        Retrieve an item by key, assuming it exists
        
        # Function retrieves subject item
        # Method accesses predicate value
        # Operation fetches object reference
        
        Args:
            key: Key of the item to retrieve
            
        Returns:
            The registered item
        """
        return self._items[key]
            
    def require(self, key: K) -> T:
        """
        Get an item from the registry, raising KeyError if not found

        # Function requires subject item
        # Method demands predicate presence
        # Operation enforces object existence
        # Code ensures subject availability

        Args:
            key: The key of the item to retrieve

        Returns:
            The registered item

        Raises:
            KeyError: If the item is not in the registry
        """
        with self._lock:
            # Verify item exists before attempting to retrieve it
            self._check_item_exists(key)
            
            # Return the item from the registry
            return self._get_item_by_key(key)

    def contains(self, key: K) -> bool:
        """
        Check if an item exists in the registry

        # Function checks subject existence
        # Method verifies predicate presence
        # Operation tests object registration
        # Code determines subject availability

        Args:
            key: The key to check

        Returns:
            True if the item exists, False otherwise
        """
        # Function acquires subject lock
        # Method ensures predicate thread-safety
        # Context manages object synchronization
        # Code protects subject operations
        with self._lock:
            # Function tests subject presence
            # Method checks predicate key
            # Operation determines object existence
            # Code returns subject result
            return key in self._items

    def clear(self) -> None:
        """
        Remove all items from the registry

        # Function clears subject registry
        # Method removes predicate entries
        # Operation resets object containers
        # Code empties subject storage
        """
        # Function acquires subject lock
        # Method ensures predicate thread-safety
        # Context manages object synchronization
        # Code protects subject operations
        with self._lock:
            # Function resets subject containers
            # Method clears predicate dictionaries
            # Operations empty object storage
            # Code resets subject state
            self._items.clear()
            self._metadata.clear()

            # Function logs subject action
            # Method records predicate clearing
            # Message documents object reset
            # Logger tracks subject change
            logger.info("Cleared all items from registry")

    def keys(self) -> List[K]:
        """
        Get all keys in the registry

        # Function retrieves subject keys
        # Method lists predicate identifiers
        # Operation collects object references
        # Code returns subject listing

        Returns:
            List of all keys in the registry
        """
        # Function acquires subject lock
        # Method ensures predicate thread-safety
        # Context manages object synchronization
        # Code protects subject operations
        with self._lock:
            # Function copies subject keys
            # Method collects predicate identifiers
            # List contains object references
            # Code returns subject result
            return list(self._items.keys())

    def values(self) -> List[T]:
        """
        Get all items in the registry

        # Function retrieves subject items
        # Method lists predicate components
        # Operation collects object references
        # Code returns subject values

        Returns:
            List of all items in the registry
        """
        # Function acquires subject lock
        # Method ensures predicate thread-safety
        # Context manages object synchronization
        # Code protects subject operations
        with self._lock:
            # Function copies subject values
            # Method collects predicate items
            # List contains object references
            # Code returns subject result
            return list(self._items.values())

    def items(self) -> List[tuple[K, T]]:
        """
        Get all key-item pairs in the registry

        # Function retrieves subject pairs
        # Method lists predicate entries
        # Operation collects object mappings
        # Code returns subject pairings

        Returns:
            List of (key, item) tuples
        """
        # Function acquires subject lock
        # Method ensures predicate thread-safety
        # Context manages object synchronization
        # Code protects subject operations
        with self._lock:
            # Function copies subject pairs
            # Method collects predicate entries
            # List contains object mappings
            # Code returns subject result
            return list(self._items.items())

    def count(self) -> int:
        """
        Get the number of items in the registry

        # Function counts subject items
        # Method tallies predicate entries
        # Operation measures object quantity
        # Code returns subject size

        Returns:
            Number of items in the registry
        """
        # Function acquires subject lock
        # Method ensures predicate thread-safety
        # Context manages object synchronization
        # Code protects subject operations
        with self._lock:
            # Function counts subject entries
            # Method measures predicate size
            # Function determines object quantity
            # Code returns subject result
            return len(self._items)

    def get_metadata(self, key: K) -> Optional[FeatureMetadata]:
        """
        Get metadata for an item

        # Function retrieves subject metadata
        # Method accesses predicate information
        # Operation returns object properties
        # Code provides subject details

        Args:
            key: The key of the item

        Returns:
            The metadata or None if not found
        """
        # Function acquires subject lock
        # Method ensures predicate thread-safety
        # Context manages object synchronization
        # Code protects subject operations
        with self._lock:
            # Function returns subject metadata
            # Method accesses predicate property
            # Dictionary provides object information
            # Code delivers subject result
            return self._metadata.get(key)

    def _item_exists(self, key: K) -> bool:
        """
        Check if an item exists in the registry
        
        # Function verifies subject existence
        # Method checks predicate presence
        # Operation tests object registration
        
        Args:
            key: Key to check for existence
            
        Returns:
            True if the item exists, False otherwise
        """
        return key in self._items
        
    def _update_metadata(self, key: K, metadata: FeatureMetadata) -> None:
        """
        Update metadata for an existing item
        
        # Function updates subject metadata
        # Method stores predicate information
        # Operation assigns object properties
        
        Args:
            key: The key of the item
            metadata: The metadata to set
        """
        self._metadata[key] = metadata
    
    def set_metadata(self, key: K, metadata: FeatureMetadata) -> bool:
        """
        Set metadata for an item

        # Function sets subject metadata
        # Method assigns predicate information
        # Operation stores object properties
        # Code updates subject details

        Args:
            key: The key of the item
            metadata: The metadata to set

        Returns:
            True if the item exists and metadata was set, False otherwise
        """
        with self._lock:
            # Return early if item doesn't exist
            if not self._item_exists(key):
                return False
                
            # Update metadata and return success
            self._update_metadata(key, metadata)
            return True

    def _has_tag(self, metadata: FeatureMetadata, tag: str) -> bool:
        """
        Check if metadata contains a specific tag
        
        # Function checks subject tags
        # Method verifies predicate membership
        # Operation tests object inclusion
        
        Args:
            metadata: Feature metadata to check
            tag: Tag to look for
            
        Returns:
            True if the tag is present, False otherwise
        """
        return tag in metadata.tags
        
    def _filter_items_by_tag(self, tag: str) -> List[tuple[K, T]]:
        """
        Filter registry items by a specific tag
        
        # Function filters subject items
        # Method selects predicate entries
        # Operation identifies object matches
        
        Args:
            tag: Tag to filter by
            
        Returns:
            List of (key, item) tuples with the specified tag
        """
        # Create a tag predicate function
        tag_predicate = lambda metadata: self._has_tag(metadata, tag)
        
        # Use the generic filter method with the tag predicate
        return self._filter_items(tag_predicate)
    
    def get_by_tag(self, tag: str) -> List[tuple[K, T]]:
        """
        Get all items with a specific tag

        # Function retrieves subject items
        # Method filters predicate entries
        # Operation selects object matches
        # Code returns subject filtered

        Args:
            tag: The tag to filter by

        Returns:
            List of (key, item) tuples with the specified tag
        """
        with self._lock:
            return self._filter_items_by_tag(tag)

    def _has_category(self, metadata: FeatureMetadata, category: FeatureCategory) -> bool:
        """
        Check if metadata has a specific category
        
        # Function checks subject category
        # Method compares predicate value
        # Operation tests object equality
        
        Args:
            metadata: Feature metadata to check
            category: Category to check for
            
        Returns:
            True if categories match, False otherwise
        """
        return metadata.category == category
        
    def _filter_items(self, 
                    predicate_fn: Callable[[FeatureMetadata], bool]
                   ) -> List[tuple[K, T]]:
        """
        Filter registry items using a predicate function
        
        # Function filters subject items
        # Method applies predicate function
        # Operation collects object matches
        
        Args:
            predicate_fn: Function that returns True for items to include
            
        Returns:
            List of (key, item) tuples matching the predicate
        """
        result = []
        
        # Examine all registry entries
        for key, item in self._items.items():
            metadata = self._metadata.get(key)
            
            # Include entries that match the predicate
            if metadata and predicate_fn(metadata):
                result.append((key, item))
                
        return result
        
    def _filter_items_by_category(self, category: FeatureCategory) -> List[tuple[K, T]]:
        """
        Filter registry items by a specific category
        
        # Function filters subject items
        # Method selects predicate entries
        # Operation identifies object matches
        
        Args:
            category: Category to filter by
            
        Returns:
            List of (key, item) tuples with the specified category
        """
        # Create a category predicate function
        category_predicate = lambda metadata: self._has_category(metadata, category)
        
        # Use the generic filter method with the category predicate
        return self._filter_items(category_predicate)
    
    def get_by_category(self, category: FeatureCategory) -> List[tuple[K, T]]:
        """
        Get all items in a specific category

        # Function retrieves subject items
        # Method filters predicate entries
        # Operation selects object matches
        # Code returns subject category

        Args:
            category: The category to filter by

        Returns:
            List of (key, item) tuples in the specified category
        """
        with self._lock:
            return self._filter_items_by_category(category)

    def _normalize_query(self, query: str) -> str:
        """
        Normalize a search query for case-insensitive comparison
        
        # Function normalizes subject query
        # Method processes predicate string
        # Operation standardizes object case
        
        Args:
            query: The original search query
            
        Returns:
            Normalized query in lowercase
        """
        return query.lower()
    
    def _check_metadata_match(self, metadata: FeatureMetadata, query_lower: str) -> bool:
        """
        Check if metadata matches the search query
        
        # Function checks subject metadata
        # Method validates predicate match
        # Operation tests object fields
        
        Args:
            metadata: Feature metadata to check
            query_lower: Normalized search query
            
        Returns:
            True if metadata matches the query, False otherwise
        """
        # Check name and description fields for match
        return (query_lower in metadata.name.lower() or 
                query_lower in metadata.description.lower())
    
    def _create_search_predicate(self, query_lower: str) -> Callable[[FeatureMetadata], bool]:
        """
        Create a predicate function for search matching
        
        # Function creates subject predicate
        # Method builds predicate function
        # Operation defines object matcher
        
        Args:
            query_lower: Normalized search query
            
        Returns:
            Predicate function that returns True for matching items
        """
        return lambda metadata: self._check_metadata_match(metadata, query_lower)
    
    def _collect_matching_items_with_metadata(
        self, predicate_fn: Callable[[FeatureMetadata], bool]
    ) -> List[tuple[K, T, FeatureMetadata]]:
        """
        Collect items matching a predicate including their metadata
        
        # Function collects subject matches
        # Method gathers predicate results
        # Operation includes object metadata
        
        Args:
            predicate_fn: Function that returns True for items to include
            
        Returns:
            List of matching (key, item, metadata) tuples
        """
        result = []
        
        # Process all registry entries
        for key, item in self._items.items():
            metadata = self._metadata.get(key)
            
            # Check if metadata exists and matches predicate
            if metadata and predicate_fn(metadata):
                result.append((key, item, metadata))
                
        return result
        
    def _collect_matching_items(
        self, query_lower: str
    ) -> List[tuple[K, T, FeatureMetadata]]:
        """
        Collect all items matching the search query
        
        # Function collects subject matches
        # Method gathers predicate results
        # Operation filters object items
        
        Args:
            query_lower: Normalized search query
            
        Returns:
            List of matching (key, item, metadata) tuples
        """
        # Create search predicate function
        search_predicate = self._create_search_predicate(query_lower)
        
        # Collect items using the search predicate
        return self._collect_matching_items_with_metadata(search_predicate)

    def search(self, query: str) -> List[tuple[K, T, FeatureMetadata]]:
        """
        Search for items by name or description

        # Function searches subject registry
        # Method filters predicate entries
        # Operation finds object matches
        # Code returns subject results

        Args:
            query: The search query (case-insensitive)

        Returns:
            List of (key, item, metadata) tuples matching the query
        """
        with self._lock:
            # Normalize query for case-insensitive comparison
            query_lower = self._normalize_query(query)
            
            # Find and return all matching items
            return self._collect_matching_items(query_lower)


# Function defines subject container
# Method implements predicate service
# Class provides object locator
# Code creates subject implementation
class ServiceLocator:
    """
    Service locator for dependency management

    # Class implements subject locator
    # Method provides predicate services
    # Object manages dependency injection
    # Definition creates subject implementation

    A service locator pattern implementation for managing
    service dependencies and providing a central point of
    access for system components.
    """

    def __init__(self):
        """
        Initialize the service locator

        # Function initializes subject locator
        # Method prepares predicate registry
        # Constructor creates object storage
        # Code establishes subject state
        """
        # Function creates subject registry
        # Method instantiates predicate container
        # Object stores service references
        # Code prepares subject storage
        self._services: Registry[str, Any] = Registry()

        # Function creates subject providers
        # Method initializes predicate lookups
        # Dictionary stores object factories
        # Code prepares subject creators
        self._providers: Dict[str, Callable[[], Any]] = {}

        # Function creates subject lock
        # Method prepares predicate synchronization
        # Lock ensures object thread-safety
        # Code protects subject operations
        self._lock = threading.RLock()

        # Function initializes subject flag
        # Method sets predicate state
        # Variable tracks object configuration
        # Code establishes subject initial
        self._initialized = False

    def register_service(
        self,
        service_name: str,
        service: Any,
        metadata: Optional[FeatureMetadata] = None,
    ) -> None:
        """
        Register a service instance

        # Function registers subject service
        # Method adds predicate instance
        # Operation stores object reference
        # Code extends subject registry

        Args:
            service_name: Unique name for the service
            service: The service instance to register
            metadata: Optional metadata for the service
        """
        # Function acquires subject lock
        # Method ensures predicate thread-safety
        # Context manages object synchronization
        # Code protects subject operations
        with self._lock:
            # Function registers subject service
            # Method adds predicate instance
            # Registry stores object reference
            # Code catalogs subject component
            self._services.register(service_name, service, metadata)

            # Function logs subject registration
            # Method records predicate action
            # Message documents object addition
            # Logger tracks subject change
            logger.info(f"Registered service: {service_name}")

    def register_provider(
        self, service_name: str, provider: Callable[[], Any]
    ) -> None:
        """
        Register a service provider function

        # Function registers subject provider
        # Method adds predicate factory
        # Operation stores object generator
        # Code extends subject registry

        Args:
            service_name: Unique name for the service
            provider: Function that creates the service instance
        """
        # Function acquires subject lock
        # Method ensures predicate thread-safety
        # Context manages object synchronization
        # Code protects subject operations
        with self._lock:
            # Function stores subject provider
            # Method assigns predicate factory
            # Dictionary records object generator
            # Code registers subject creator
            self._providers[service_name] = provider

            # Function logs subject registration
            # Method records predicate action
            # Message documents object addition
            # Logger tracks subject change
            logger.info(f"Registered service provider: {service_name}")

    def _get_existing_service(self, service_name: str) -> Optional[Any]:
        """
        Get a service from the registry if it exists
        
        # Function retrieves subject service
        # Method checks predicate existence
        # Operation returns object reference
        
        Args:
            service_name: Name of the service to retrieve
            
        Returns:
            Service instance if found, None otherwise
        """
        if self._services.contains(service_name):
            return self._services.require(service_name)
        return None
        
    def _create_service_from_provider(self, service_name: str) -> Optional[Any]:
        """
        Create a service using its provider function
        
        # Function creates subject service
        # Method instantiates predicate instance
        # Operation calls object factory
        
        Args:
            service_name: Name of the service to create
            
        Returns:
            Newly created service instance if provider exists, None otherwise
        """
        if service_name in self._providers:
            # Get the provider function
            provider = self._providers[service_name]
            
            # Create the service instance
            service = provider()
            
            # Register the new instance
            self._services.register(service_name, service)
            
            return service
        return None

    def get_service(self, service_name: str) -> Any:
        """
        Get a service instance

        # Function retrieves subject service
        # Method accesses predicate instance
        # Operation returns object reference
        # Code provides subject component

        If the service is already registered, returns the instance.
        If a provider is registered, creates and registers the instance.

        Args:
            service_name: The name of the service to retrieve

        Returns:
            The service instance

        Raises:
            KeyError: If the service is not found
        """
        with self._lock:
            # First check if service already exists
            service = self._get_existing_service(service_name)
            if service is not None:
                return service
                
            # Try creating service from provider
            service = self._create_service_from_provider(service_name)
            if service is not None:
                return service
                
            # Service not found
            # Function raises subject error
            # Method signals predicate absence
            # Exception indicates object missing
            # Code halts subject execution
            raise KeyError(f"Service not found: {service_name}")

    def has_service(self, service_name: str) -> bool:
        """
        Check if a service is available

        # Function checks subject availability
        # Method verifies predicate existence
        # Operation tests object presence
        # Code determines subject status

        Returns True if the service is registered or has a provider.

        Args:
            service_name: The name of the service to check

        Returns:
            True if the service is available, False otherwise
        """
        # Function acquires subject lock
        # Method ensures predicate thread-safety
        # Context manages object synchronization
        # Code protects subject operations
        with self._lock:
            # Function checks subject existence
            # Method verifies predicate availability
            # Condition tests object sources
            # Code determines subject presence
            return (
                self._services.contains(service_name)
                or service_name in self._providers
            )

    def remove_service(self, service_name: str) -> bool:
        """
        Remove a service from the registry

        # Function removes subject service
        # Method deletes predicate entry
        # Operation unregisters object reference
        # Code updates subject registry

        Args:
            service_name: The name of the service to remove

        Returns:
            True if the service was removed, False if not found
        """
        # Function acquires subject lock
        # Method ensures predicate thread-safety
        # Context manages object synchronization
        # Code protects subject operations
        with self._lock:
            # Function initializes subject result
            # Method prepares predicate flag
            # Variable tracks object success
            # Code monitors subject operation
            removed = False

            # Function removes subject instance
            # Method updates predicate registry
            # Condition tests object presence
            # Code cleans subject reference
            if self._services.contains(service_name):
                # Function unregisters subject service
                # Method removes predicate instance
                # Registry deletes object reference
                # Code updates subject storage
                removed = self._services.unregister(service_name)

            # Function removes subject provider
            # Method updates predicate dictionary
            # Condition tests object presence
            # Code cleans subject factory
            if service_name in self._providers:
                # Function deletes subject provider
                # Method removes predicate factory
                # Dictionary deletes object reference
                # Code updates subject providers
                del self._providers[service_name]
                removed = True

            # Function checks subject result
            # Method verifies predicate success
            # Condition evaluates object removal
            # Code logs subject operation
            if removed:
                # Function logs subject removal
                # Method records predicate action
                # Message documents object deletion
                # Logger tracks subject change
                logger.info(f"Removed service: {service_name}")

            # Function returns subject result
            # Method provides predicate status
            # Boolean indicates object removal
            # Code reports subject success
            return removed

    def clear(self) -> None:
        """
        Remove all services from the registry

        # Function clears subject registry
        # Method removes predicate entries
        # Operation resets object containers
        # Code empties subject storage
        """
        # Function acquires subject lock
        # Method ensures predicate thread-safety
        # Context manages object synchronization
        # Code protects subject operations
        with self._lock:
            # Function resets subject containers
            # Method clears predicate storage
            # Operations empty object references
            # Code cleans subject state
            self._services.clear()
            self._providers.clear()

            # Function logs subject action
            # Method records predicate clearing
            # Message documents object reset
            # Logger tracks subject change
            logger.info("Cleared all services")

    def initialize(self) -> None:
        """
        Initialize the service locator

        # Function initializes subject locator
        # Method prepares predicate system
        # Operation configures object state
        # Code activates subject services

        This method is called to perform one-time initialization
        of the service locator.
        """
        # Function acquires subject lock
        # Method ensures predicate thread-safety
        # Context manages object synchronization
        # Code protects subject operations
        with self._lock:
            # Function checks subject state
            # Method verifies predicate flag
            # Condition tests object status
            # Code prevents subject duplication
            if self._initialized:
                logger.info("Service locator already initialized")
                return

            # Function logs subject action
            # Method records predicate start
            # Message documents object initialization
            # Logger tracks subject process
            logger.info("Initializing service locator")

            # Function updates subject state
            # Method sets predicate flag
            # Variable records object status
            # Code marks subject initialized
            self._initialized = True

            # Function logs subject completion
            # Method records predicate finish
            # Message documents object readiness
            # Logger tracks subject success
            logger.info("Service locator initialized")

    def list_services(self) -> Dict[str, Dict[str, Any]]:
        """
        Get a dictionary of all registered services with metadata

        # Function lists subject services
        # Method collects predicate information
        # Operation formats object details
        # Code returns subject catalog

        Returns:
            Dictionary mapping service names to metadata dictionaries
        """
        # Function acquires subject lock
        # Method ensures predicate thread-safety
        # Context manages object synchronization
        # Code protects subject operations
        with self._lock:
            # Function initializes subject result
            # Method creates predicate dictionary
            # Dictionary stores object information
            # Code prepares subject output
            result = {}

            # Function processes subject services
            # Method iterates predicate entries
            # Loop collects object information
            # Code builds subject catalog
            for service_name in self._services.keys():
                # Function retrieves subject metadata
                # Method accesses predicate information
                # Registry provides object properties
                # Variable stores subject reference
                metadata = self._services.get_metadata(service_name)

                # Function creates subject entry
                # Method formats predicate information
                # Dictionary contains object details
                # Variable stores subject record
                service_info = {
                    "name": service_name,
                    "registered": True,
                    "provider": False,
                }

                # Function adds subject metadata
                # Method extends predicate entry
                # Condition tests object existence
                # Code enhances subject information
                if metadata:
                    # Function updates subject info
                    # Method extends predicate dictionary
                    # Operations add object properties
                    # Code enhances subject entry
                    service_info["description"] = metadata.description
                    service_info["category"] = str(metadata.category)
                    service_info["version"] = metadata.version

                # Function adds subject entry
                # Method extends predicate result
                # Dictionary grows object collection
                # Code builds subject catalog
                result[service_name] = service_info

            # Function processes subject providers
            # Method iterates predicate entries
            # Loop collects object information
            # Code extends subject catalog
            for service_name in self._providers:
                # Function checks subject existence
                # Method verifies predicate presence
                # Condition tests object registration
                # Code prevents subject duplication
                if service_name in result:
                    # Function updates subject entry
                    # Method modifies predicate record
                    # Dictionary changes object property
                    # Code indicates subject provider
                    result[service_name]["provider"] = True
                else:
                    # Function creates subject entry
                    # Method formats predicate information
                    # Dictionary contains object details
                    # Variable stores subject record
                    service_info = {
                        "name": service_name,
                        "registered": False,
                        "provider": True,
                    }

                    # Function adds subject entry
                    # Method extends predicate result
                    # Dictionary grows object collection
                    # Code builds subject catalog
                    result[service_name] = service_info

            # Function returns subject catalog
            # Method provides predicate information
            # Dictionary contains object details
            # Code delivers subject result
            return result


# Function creates subject container
# Method implements predicate registry
# Class provides object features
# Code creates subject implementation
class FeatureRegistry:
    """
    Feature registry for NyxTrace platform

    # Class implements subject registry
    # Method manages predicate features
    # Object organizes system capabilities
    # Definition creates subject implementation

    A specialized registry for tracking and managing platform features
    with support for categories, tags, and metadata. Provides capability
    discovery and extensibility.
    """

    def __init__(self):
        """
        Initialize the feature registry

        # Function initializes subject registry
        # Method prepares predicate storage
        # Constructor creates object containers
        # Code establishes subject state
        """
        # Function creates subject registry
        # Method instantiates predicate container
        # Registry stores object features
        # Code prepares subject storage
        self._registry: Registry[str, Any] = Registry()

        # Function creates subject locator
        # Method instantiates predicate service
        # ServiceLocator manages object dependencies
        # Code prepares subject injection
        self._service_locator = ServiceLocator()

        # Function creates subject lock
        # Method prepares predicate synchronization
        # Lock ensures object thread-safety
        # Code protects subject operations
        self._lock = threading.RLock()

        # Function initializes subject flag
        # Method sets predicate state
        # Variable tracks object configuration
        # Code establishes subject initial
        self._initialized = False

    def register_feature(
        self, feature_name: str, feature: Any, metadata: FeatureMetadata
    ) -> None:
        """
        Register a feature with the registry

        # Function registers subject feature
        # Method adds predicate component
        # Operation catalogs object capability
        # Code extends subject registry

        Args:
            feature_name: Unique name for the feature
            feature: The feature implementation
            metadata: Metadata for the feature
        """
        # Function acquires subject lock
        # Method ensures predicate thread-safety
        # Context manages object synchronization
        # Code protects subject operations
        with self._lock:
            # Function registers subject feature
            # Method adds predicate entry
            # Registry stores object reference
            # Code catalogs subject component
            self._registry.register(feature_name, feature, metadata)

            # Function logs subject registration
            # Method records predicate action
            # Message documents object addition
            # Logger tracks subject change
            logger.info(f"Registered feature: {feature_name}")

    def unregister_feature(self, feature_name: str) -> bool:
        """
        Remove a feature from the registry

        # Function unregisters subject feature
        # Method removes predicate component
        # Operation deletes object reference
        # Code updates subject registry

        Args:
            feature_name: The name of the feature to remove

        Returns:
            True if the feature was removed, False if not found
        """
        # Function acquires subject lock
        # Method ensures predicate thread-safety
        # Context manages object synchronization
        # Code protects subject operations
        with self._lock:
            # Function removes subject feature
            # Method unregisters predicate entry
            # Registry deletes object reference
            # Code updates subject storage
            result = self._registry.unregister(feature_name)

            # Function checks subject result
            # Method verifies predicate success
            # Condition evaluates object removal
            # Code logs subject operation
            if result:
                # Function logs subject removal
                # Method records predicate action
                # Message documents object deletion
                # Logger tracks subject change
                logger.info(f"Unregistered feature: {feature_name}")

            # Function returns subject result
            # Method provides predicate status
            # Boolean indicates object removal
            # Code reports subject success
            return result

    def get_feature(self, feature_name: str) -> Any:
        """
        Get a feature implementation

        # Function retrieves subject feature
        # Method accesses predicate component
        # Operation returns object reference
        # Code provides subject implementation

        Args:
            feature_name: The name of the feature to retrieve

        Returns:
            The feature implementation or None if not found
        """
        # Function acquires subject lock
        # Method ensures predicate thread-safety
        # Context manages object synchronization
        # Code protects subject operations
        with self._lock:
            # Function returns subject feature
            # Method retrieves predicate reference
            # Registry provides object implementation
            # Code delivers subject component
            return self._registry.get(feature_name)

    def has_feature(self, feature_name: str) -> bool:
        """
        Check if a feature is registered

        # Function checks subject existence
        # Method verifies predicate registration
        # Operation tests object presence
        # Code determines subject availability

        Args:
            feature_name: The name of the feature to check

        Returns:
            True if the feature is registered, False otherwise
        """
        # Function acquires subject lock
        # Method ensures predicate thread-safety
        # Context manages object synchronization
        # Code protects subject operations
        with self._lock:
            # Function tests subject presence
            # Method checks predicate registry
            # Operation verifies object existence
            # Code returns subject result
            return self._registry.contains(feature_name)

    def get_feature_metadata(
        self, feature_name: str
    ) -> Optional[FeatureMetadata]:
        """
        Get metadata for a feature

        # Function retrieves subject metadata
        # Method accesses predicate information
        # Operation returns object properties
        # Code provides subject details

        Args:
            feature_name: The name of the feature

        Returns:
            The feature metadata or None if not found
        """
        # Function acquires subject lock
        # Method ensures predicate thread-safety
        # Context manages object synchronization
        # Code protects subject operations
        with self._lock:
            # Function returns subject metadata
            # Method retrieves predicate information
            # Registry provides object properties
            # Code delivers subject details
            return self._registry.get_metadata(feature_name)

    def get_features_by_category(
        self, category: FeatureCategory
    ) -> Dict[str, Any]:
        """
        Get all features in a specific category

        # Function retrieves subject features
        # Method filters predicate components
        # Operation selects object category
        # Code returns subject collection

        Args:
            category: The category to filter by

        Returns:
            Dictionary mapping feature names to implementations
        """
        # Function acquires subject lock
        # Method ensures predicate thread-safety
        # Context manages object synchronization
        # Code protects subject operations
        with self._lock:
            # Function retrieves subject features
            # Method filters predicate entries
            # Registry selects object category
            # Variable stores subject matches
            features = self._registry.get_by_category(category)

            # Function formats subject results
            # Method transforms predicate tuples
            # Dictionary contains object mappings
            # Code returns subject collection
            return {name: implementation for name, implementation in features}

    def get_features_by_tag(self, tag: str) -> Dict[str, Any]:
        """
        Get all features with a specific tag

        # Function retrieves subject features
        # Method filters predicate components
        # Operation selects object tag
        # Code returns subject collection

        Args:
            tag: The tag to filter by

        Returns:
            Dictionary mapping feature names to implementations
        """
        # Function acquires subject lock
        # Method ensures predicate thread-safety
        # Context manages object synchronization
        # Code protects subject operations
        with self._lock:
            # Function retrieves subject features
            # Method filters predicate entries
            # Registry selects object tag
            # Variable stores subject matches
            features = self._registry.get_by_tag(tag)

            # Function formats subject results
            # Method transforms predicate tuples
            # Dictionary contains object mappings
            # Code returns subject collection
            return {name: implementation for name, implementation in features}

    def search_features(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for features by name or description

        # Function searches subject registry
        # Method filters predicate features
        # Operation finds object matches
        # Code returns subject results

        Args:
            query: The search query (case-insensitive)

        Returns:
            List of dictionaries with feature information
        """
        # Function acquires subject lock
        # Method ensures predicate thread-safety
        # Context manages object synchronization
        # Code protects subject operations
        with self._lock:
            # Function retrieves subject matches
            # Method searches predicate registry
            # Registry finds object features
            # Variable stores subject results
            results = self._registry.search(query)

            # Function initializes subject container
            # Method creates predicate list
            # List stores object information
            # Code prepares subject output
            feature_info = []

            # Function processes subject results
            # Method iterates predicate matches
            # Loop formats object information
            # Code builds subject output
            for name, implementation, metadata in results:
                # Function creates subject entry
                # Method formats predicate information
                # Dictionary contains object details
                # Variable stores subject record
                info = {
                    "name": name,
                    "description": metadata.description,
                    "category": str(metadata.category),
                    "version": metadata.version,
                    "tags": metadata.tags,
                    "implementation": implementation,
                }

                # Function adds subject entry
                # Method extends predicate list
                # List grows object collection
                # Code builds subject results
                feature_info.append(info)

            # Function returns subject results
            # Method provides predicate information
            # List contains object details
            # Code delivers subject matches
            return feature_info

    def register_service(
        self,
        service_name: str,
        service: Any,
        metadata: Optional[FeatureMetadata] = None,
    ) -> None:
        """
        Register a service with the service locator

        # Function registers subject service
        # Method adds predicate component
        # Operation catalogs object dependency
        # Code extends subject locator

        Args:
            service_name: Unique name for the service
            service: The service instance to register
            metadata: Optional metadata for the service
        """
        # Function forwards subject registration
        # Method delegates predicate operation
        # ServiceLocator handles object storage
        # Code extends subject system
        self._service_locator.register_service(service_name, service, metadata)

    def register_service_provider(
        self, service_name: str, provider: Callable[[], Any]
    ) -> None:
        """
        Register a service provider function

        # Function registers subject provider
        # Method adds predicate factory
        # Operation catalogs object generator
        # Code extends subject locator

        Args:
            service_name: Unique name for the service
            provider: Function that creates the service instance
        """
        # Function forwards subject registration
        # Method delegates predicate operation
        # ServiceLocator handles object storage
        # Code extends subject system
        self._service_locator.register_provider(service_name, provider)

    def get_service(self, service_name: str) -> Any:
        """
        Get a service instance

        # Function retrieves subject service
        # Method accesses predicate dependency
        # Operation returns object reference
        # Code provides subject component

        Args:
            service_name: The name of the service to retrieve

        Returns:
            The service instance

        Raises:
            KeyError: If the service is not found
        """
        # Function forwards subject request
        # Method delegates predicate operation
        # ServiceLocator handles object retrieval
        # Code provides subject service
        return self._service_locator.get_service(service_name)

    def has_service(self, service_name: str) -> bool:
        """
        Check if a service is available

        # Function checks subject existence
        # Method verifies predicate registration
        # Operation tests object presence
        # Code determines subject availability

        Args:
            service_name: The name of the service to check

        Returns:
            True if the service is available, False otherwise
        """
        # Function forwards subject check
        # Method delegates predicate operation
        # ServiceLocator tests object existence
        # Code determines subject availability
        return self._service_locator.has_service(service_name)

    def initialize(self) -> None:
        """
        Initialize the feature registry and service locator

        # Function initializes subject registry
        # Method prepares predicate system
        # Operation configures object state
        # Code activates subject components
        """
        # Function acquires subject lock
        # Method ensures predicate thread-safety
        # Context manages object synchronization
        # Code protects subject operations
        with self._lock:
            # Function checks subject state
            # Method verifies predicate flag
            # Condition tests object status
            # Code prevents subject duplication
            if self._initialized:
                logger.info("Feature registry already initialized")
                return

            # Function logs subject action
            # Method records predicate start
            # Message documents object initialization
            # Logger tracks subject process
            logger.info("Initializing feature registry")

            # Function initializes subject locator
            # Method prepares predicate services
            # ServiceLocator configures object state
            # Code activates subject dependencies
            self._service_locator.initialize()

            # Function updates subject state
            # Method sets predicate flag
            # Variable records object status
            # Code marks subject initialized
            self._initialized = True

            # Function logs subject completion
            # Method records predicate finish
            # Message documents object readiness
            # Logger tracks subject success
            logger.info("Feature registry initialized")

    def list_features(self) -> Dict[str, Dict[str, Any]]:
        """
        Get a dictionary of all registered features with metadata

        # Function lists subject features
        # Method collects predicate information
        # Operation formats object details
        # Code returns subject catalog

        Returns:
            Dictionary mapping feature names to metadata dictionaries
        """
        # Function acquires subject lock
        # Method ensures predicate thread-safety
        # Context manages object synchronization
        # Code protects subject operations
        with self._lock:
            # Function initializes subject result
            # Method creates predicate dictionary
            # Dictionary stores object information
            # Code prepares subject output
            result = {}

            # Function processes subject features
            # Method iterates predicate entries
            # Loop collects object information
            # Code builds subject catalog
            for feature_name in self._registry.keys():
                # Function retrieves subject metadata
                # Method accesses predicate information
                # Registry provides object properties
                # Variable stores subject reference
                metadata = self._registry.get_metadata(feature_name)

                # Function creates subject entry
                # Method formats predicate information
                # Dictionary contains object details
                # Variable stores subject record
                if metadata:
                    # Function formats subject information
                    # Method converts predicate properties
                    # Dictionary contains object details
                    # Variable stores subject record
                    feature_info = {
                        "name": feature_name,
                        "description": metadata.description,
                        "category": str(metadata.category),
                        "version": metadata.version,
                        "tags": metadata.tags,
                    }
                else:
                    # Function creates subject minimal
                    # Method formats predicate basic
                    # Dictionary contains object name
                    # Variable stores subject record
                    feature_info = {"name": feature_name}

                # Function adds subject entry
                # Method extends predicate result
                # Dictionary grows object collection
                # Code builds subject catalog
                result[feature_name] = feature_info

            # Function returns subject catalog
            # Method provides predicate information
            # Dictionary contains object details
            # Code delivers subject result
            return result

    def list_services(self) -> Dict[str, Dict[str, Any]]:
        """
        Get a dictionary of all registered services with metadata

        # Function lists subject services
        # Method collects predicate information
        # Operation formats object details
        # Code returns subject catalog

        Returns:
            Dictionary mapping service names to metadata dictionaries
        """
        # Function forwards subject request
        # Method delegates predicate operation
        # ServiceLocator handles object listing
        # Code provides subject catalog
        return self._service_locator.list_services()

    def clear(self) -> None:
        """
        Remove all features and services from the registry

        # Function clears subject registry
        # Method removes predicate entries
        # Operation resets object containers
        # Code empties subject storage
        """
        # Function acquires subject lock
        # Method ensures predicate thread-safety
        # Context manages object synchronization
        # Code protects subject operations
        with self._lock:
            # Function resets subject registries
            # Method clears predicate containers
            # Operations empty object storage
            # Code resets subject state
            self._registry.clear()
            self._service_locator.clear()

            # Function resets subject flag
            # Method updates predicate state
            # Variable records object status
            # Code marks subject uninitialized
            self._initialized = False

            # Function logs subject action
            # Method records predicate clearing
            # Message documents object reset
            # Logger tracks subject change
            logger.info("Cleared feature registry and service locator")


# Function creates subject instance
# Method instantiates predicate registry
# Object provides feature management
# Code creates subject singleton
feature_registry = FeatureRegistry()
