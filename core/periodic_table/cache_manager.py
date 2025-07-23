"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-PT-CACHE-MANAGER-0001               │
// │ 📁 domain       : Core, Periodic Table, Threading           │
// │ 🧠 description  : Thread-safe cache manager for periodic    │
// │                  table data to avoid SQLite thread issues   │
// │ 🕸️ hash_type    : UUID → CUID-linked module                │
// │ 🔄 parent_node  : NODE_PERIODIC_TABLE                      │
// │ 🧩 dependencies : None                                     │
// │ 🔧 tool_usage   : Core Infrastructure                      │
// │ 📡 input_type   : Elements, Categories, Relationships       │
// │ 🧪 test_status  : experimental                             │
// │ 🧠 cognitive_fn : thread safety, data caching               │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

Thread-Safe Cache Manager for Periodic Table
-------------------------------------------
This module implements a thread-safe caching system for periodic table data.
It addresses SQLite thread safety issues with Streamlit by providing a layer
that can load data from the database once and then serve it across threads
without requiring additional database access.
"""

import threading
import logging
from typing import Dict, List, Any, Optional, TypeVar, Generic, Set, Tuple
import uuid
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Generic type for cache items
T = TypeVar("T")


class CacheManager:
    """
    Thread-safe cache management for periodic table data.

    This class provides a centralized cache to store element, category,
    relationship, and other periodic table data to avoid SQLite threading issues
    with Streamlit. It uses a thread-local storage approach for connection management
    while keeping the actual data in a shared structure.
    """

    def __init__(self):
        """Initialize the cache manager with empty storage."""
        self._lock = threading.RLock()
        self._elements_cache: Dict[uuid.UUID, Any] = {}
        self._categories_cache: Dict[str, Any] = {}
        self._groups_cache: Dict[str, Any] = {}
        self._periods_cache: Dict[int, Any] = {}
        self._relationships_cache: List[Any] = []
        self._relationship_map: Dict[uuid.UUID, Set[uuid.UUID]] = {}
        self._initialized = False
        self._last_update = 0.0

    def is_initialized(self) -> bool:
        """Check if the cache has been initialized."""
        return self._initialized

    def mark_initialized(self) -> None:
        """Mark the cache as initialized."""
        with self._lock:
            self._initialized = True
            self._last_update = time.time()
            logger.info("Cache marked as initialized")

    def cache_elements(self, elements: List[Any]) -> None:
        """
        Cache a list of elements for thread-safe access.

        Args:
            elements: List of Element objects to cache
        """
        with self._lock:
            for element in elements:
                element_id = getattr(element, "id", None)
                if element_id:
                    self._elements_cache[element_id] = element
            self._last_update = time.time()
            logger.debug(
                f"Cached {len(elements)} elements, total: {len(self._elements_cache)}"
            )

    def cache_categories(self, categories: List[Any]) -> None:
        """
        Cache a list of categories for thread-safe access.

        Args:
            categories: List of Category objects to cache
        """
        with self._lock:
            for category in categories:
                category_name = getattr(category, "name", None)
                if category_name:
                    self._categories_cache[category_name] = category
            self._last_update = time.time()
            logger.debug(
                f"Cached {len(categories)} categories, total: {len(self._categories_cache)}"
            )

    def cache_groups(self, groups: List[Any]) -> None:
        """
        Cache a list of groups for thread-safe access.

        Args:
            groups: List of Group objects to cache
        """
        with self._lock:
            for group in groups:
                group_name = getattr(group, "name", None)
                if group_name:
                    self._groups_cache[group_name] = group
            self._last_update = time.time()
            logger.debug(
                f"Cached {len(groups)} groups, total: {len(self._groups_cache)}"
            )

    def cache_periods(self, periods: List[Any]) -> None:
        """
        Cache a list of periods for thread-safe access.

        Args:
            periods: List of Period objects to cache
        """
        with self._lock:
            for period in periods:
                period_number = getattr(period, "number", None)
                if period_number is not None:
                    self._periods_cache[period_number] = period
            self._last_update = time.time()
            logger.debug(
                f"Cached {len(periods)} periods, total: {len(self._periods_cache)}"
            )

    def cache_relationships(self, relationships: List[Any]) -> None:
        """
        Cache a list of relationships for thread-safe access and build relationship map.

        Args:
            relationships: List of Relationship objects to cache
        """
        with self._lock:
            self._relationships_cache = relationships

            # Build relationship map for fast lookup
            self._relationship_map.clear()
            for rel in relationships:
                source_id = getattr(rel, "source_id", None)
                target_id = getattr(rel, "target_id", None)

                if source_id and target_id:
                    if source_id not in self._relationship_map:
                        self._relationship_map[source_id] = set()
                    self._relationship_map[source_id].add(target_id)

            self._last_update = time.time()
            logger.debug(f"Cached {len(relationships)} relationships")

    def get_all_elements(self) -> List[Any]:
        """
        Get all cached elements.

        Returns:
            List of all elements in the cache
        """
        with self._lock:
            return list(self._elements_cache.values())

    def get_element(self, element_id: uuid.UUID) -> Optional[Any]:
        """
        Get an element by its ID.

        Args:
            element_id: UUID of the element to retrieve

        Returns:
            Element if found, None otherwise
        """
        with self._lock:
            return self._elements_cache.get(element_id)

    def get_elements_by_category(self, category_name: str) -> List[Any]:
        """
        Get all elements belonging to a specific category.

        Args:
            category_name: Name of the category to filter by

        Returns:
            List of elements in the specified category
        """
        with self._lock:
            return [
                e
                for e in self._elements_cache.values()
                if getattr(e, "category", None) == category_name
            ]

    def get_elements_by_group(self, group_name: str) -> List[Any]:
        """
        Get all elements belonging to a specific group.

        Args:
            group_name: Name of the group to filter by

        Returns:
            List of elements in the specified group
        """
        with self._lock:
            return [
                e
                for e in self._elements_cache.values()
                if getattr(e, "group", None) == group_name
            ]

    def get_elements_by_period(self, period: int) -> List[Any]:
        """
        Get all elements belonging to a specific period.

        Args:
            period: Number of the period to filter by

        Returns:
            List of elements in the specified period
        """
        with self._lock:
            return [
                e
                for e in self._elements_cache.values()
                if getattr(e, "period", None) == period
            ]

    def get_all_categories(self) -> List[Any]:
        """
        Get all cached categories.

        Returns:
            List of all categories in the cache
        """
        with self._lock:
            return list(self._categories_cache.values())

    def get_category(self, name: str) -> Optional[Any]:
        """
        Get a category by name.

        Args:
            name: Name of the category to retrieve

        Returns:
            Category if found, None otherwise
        """
        with self._lock:
            return self._categories_cache.get(name)

    def get_all_groups(self) -> List[Any]:
        """
        Get all cached groups.

        Returns:
            List of all groups in the cache
        """
        with self._lock:
            return list(self._groups_cache.values())

    def get_group(self, name: str) -> Optional[Any]:
        """
        Get a group by name.

        Args:
            name: Name of the group to retrieve

        Returns:
            Group if found, None otherwise
        """
        with self._lock:
            return self._groups_cache.get(name)

    def get_all_periods(self) -> List[Any]:
        """
        Get all cached periods.

        Returns:
            List of all periods in the cache
        """
        with self._lock:
            return list(self._periods_cache.values())

    def get_period(self, number: int) -> Optional[Any]:
        """
        Get a period by number.

        Args:
            number: Number of the period to retrieve

        Returns:
            Period if found, None otherwise
        """
        with self._lock:
            return self._periods_cache.get(number)

    def get_all_relationships(self) -> List[Any]:
        """
        Get all cached relationships.

        Returns:
            List of all relationships in the cache
        """
        with self._lock:
            return self._relationships_cache.copy()

    def get_relationships_for_element(self, element_id: uuid.UUID) -> List[Any]:
        """
        Get all relationships where the specified element is the source.

        Args:
            element_id: UUID of the element to find relationships for

        Returns:
            List of relationships with the element as source
        """
        with self._lock:
            return [
                r
                for r in self._relationships_cache
                if getattr(r, "source_id", None) == element_id
            ]

    def has_relationship(
        self, source_id: uuid.UUID, target_id: uuid.UUID
    ) -> bool:
        """
        Check if a relationship exists between two elements.

        Args:
            source_id: UUID of the source element
            target_id: UUID of the target element

        Returns:
            True if a relationship exists, False otherwise
        """
        with self._lock:
            if source_id in self._relationship_map:
                return target_id in self._relationship_map[source_id]
            return False

    def clear_cache(self) -> None:
        """Clear all cached data."""
        with self._lock:
            self._elements_cache.clear()
            self._categories_cache.clear()
            self._groups_cache.clear()
            self._periods_cache.clear()
            self._relationships_cache.clear()
            self._relationship_map.clear()
            self._initialized = False
            logger.info("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the cache.

        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            return {
                "elements": len(self._elements_cache),
                "categories": len(self._categories_cache),
                "groups": len(self._groups_cache),
                "periods": len(self._periods_cache),
                "relationships": len(self._relationships_cache),
                "initialized": self._initialized,
                "last_update": self._last_update,
            }


# Global singleton instance
_cache_manager = CacheManager()


def get_cache_manager() -> CacheManager:
    """
    Get the global cache manager instance.

    Returns:
        Global CacheManager instance
    """
    return _cache_manager
