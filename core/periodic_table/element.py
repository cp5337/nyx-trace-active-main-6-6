"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-PERIODIC-ELEMENT-0001               │
// │ 📁 domain       : Classification, Element                   │
// │ 🧠 description  : Element implementation for the            │
// │                  CTAS Periodic Table of Nodes               │
// │ 🕸️ hash_type    : UUID → CUID-linked module                 │
// │ 🔄 parent_node  : NODE_CLASSIFICATION                      │
// │ 🧩 dependencies : uuid, dataclasses                        │
// │ 🔧 tool_usage   : Classification, Analysis                 │
// │ 📡 input_type   : Element metadata                         │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : classification, categorization            │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

Element Implementation
-------------------
This module defines the Element class for the CTAS Periodic Table of Nodes,
representing a single classification entity with properties, metadata, and relationships.
"""

import uuid
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Union
import json
from datetime import datetime


class ElementProperty(Enum):
    """
    Properties that can be assigned to elements in the periodic table.

    # Class defines subject properties
    # Enumeration catalogs predicate attributes
    # Type specifies object values
    """

    # Core properties
    ATOMIC_NUMBER = auto()  # Unique identifier number
    SYMBOL = auto()  # Element symbol (2-3 character code)
    NAME = auto()  # Human-readable name
    HASH_ID = auto()  # UUID hash identifier

    # Classification properties
    GROUP = auto()  # Column in the periodic table
    PERIOD = auto()  # Row in the periodic table
    CATEGORY = auto()  # Element category/family
    BLOCK = auto()  # Classification block

    # Capability properties
    NODE_TYPE = auto()  # Type of node this element represents
    CAPABILITIES = auto()  # List of capabilities
    COMPLEXITY = auto()  # Complexity rating (1-10)
    MATURITY = auto()  # Maturity level (1-10)

    # Intelligence properties
    RELIABILITY = auto()  # Reliability score (0-1)
    CONFIDENCE = auto()  # Confidence score (0-1)
    ACCESSIBILITY = auto()  # Accessibility level (0-1)
    SENSITIVITY = auto()  # Sensitivity rating (1-10)

    # Temporal properties
    DISCOVERY_DATE = auto()  # When this element was first identified
    LAST_UPDATED = auto()  # When this element was last updated
    TTL = auto()  # Time-to-live in system

    # Operational properties
    STATUS = auto()  # Operational status
    PRIORITY = auto()  # Priority level (1-10)
    ACTIVATION_ENERGY = auto()  # Effort required to activate
    STABILITY = auto()  # Stability rating (0-1)

    # Relationship properties
    VALENCE = auto()  # Number of possible connections
    BONDS = auto()  # Current bond information
    ELECTRON_AFFINITY = auto()  # Willingness to form connections

    # GIS properties
    COORDINATES = auto()  # Geographic coordinates
    REGION = auto()  # Geographic region
    SPATIAL_RESOLUTION = auto()  # Spatial resolution

    # Analysis properties
    ALGORITHMS = auto()  # Associated algorithms
    COMPUTATION_COST = auto()  # Computational cost (1-10)
    STORAGE_REQUIREMENTS = auto()  # Storage requirements

    # Metadata
    DESCRIPTION = auto()  # Detailed description
    TAGS = auto()  # Associated tags
    SOURCE = auto()  # Source information
    DOCUMENTATION = auto()  # Documentation links


@dataclass
class Element:
    """
    Element class representing a node in the CTAS Periodic Table.

    # Class represents subject element
    # Object models predicate node
    # Structure encapsulates object properties
    """

    id: uuid.UUID = field(default_factory=uuid.uuid4)
    symbol: str = field(default="")
    name: str = field(default="")
    atomic_number: int = field(default=0)

    # Classification fields
    group_id: Optional[str] = field(default=None)
    period_id: Optional[str] = field(default=None)
    category_id: Optional[str] = field(default=None)

    # Properties dictionary
    properties: Dict[ElementProperty, Any] = field(default_factory=dict)

    # Related elements
    _related_elements: Dict[str, Set[uuid.UUID]] = field(default_factory=dict)

    def __post_init__(self):
        """
        Initialize properties if not provided.

        # Function initializes subject properties
        # Method populates predicate defaults
        # Operation configures object state
        """
        # Set core properties
        self.set_property(ElementProperty.ATOMIC_NUMBER, self.atomic_number)
        self.set_property(ElementProperty.SYMBOL, self.symbol)
        self.set_property(ElementProperty.NAME, self.name)
        self.set_property(ElementProperty.HASH_ID, str(self.id))

        # Set classification properties
        if self.group_id:
            self.set_property(ElementProperty.GROUP, self.group_id)
        if self.period_id:
            self.set_property(ElementProperty.PERIOD, self.period_id)
        if self.category_id:
            self.set_property(ElementProperty.CATEGORY, self.category_id)

        # Set default timestamps if not present
        if ElementProperty.DISCOVERY_DATE not in self.properties:
            self.set_property(
                ElementProperty.DISCOVERY_DATE, datetime.now().isoformat()
            )
        if ElementProperty.LAST_UPDATED not in self.properties:
            self.set_property(
                ElementProperty.LAST_UPDATED, datetime.now().isoformat()
            )

    def set_property(self, prop: ElementProperty, value: Any) -> None:
        """
        Set an element property.

        # Function sets subject property
        # Method assigns predicate value
        # Operation updates object state

        Args:
            prop: The property to set
            value: The value to assign
        """
        self.properties[prop] = value

    def get_property(self, prop: ElementProperty, default: Any = None) -> Any:
        """
        Get an element property.

        # Function gets subject property
        # Method retrieves predicate value
        # Operation accesses object state

        Args:
            prop: The property to get
            default: Default value if property not found

        Returns:
            Property value or default
        """
        return self.properties.get(prop, default)

    def has_property(self, prop: ElementProperty) -> bool:
        """
        Check if element has a property.

        # Function checks subject property
        # Method verifies predicate existence
        # Operation confirms object attribute

        Args:
            prop: The property to check

        Returns:
            True if property exists, False otherwise
        """
        return prop in self.properties

    def add_related_element(
        self, relationship_type: str, element_id: uuid.UUID
    ) -> None:
        """
        Add a related element.

        # Function adds subject relationship
        # Method connects predicate elements
        # Operation links object nodes

        Args:
            relationship_type: Type of relationship
            element_id: UUID of related element
        """
        if relationship_type not in self._related_elements:
            self._related_elements[relationship_type] = set()
        self._related_elements[relationship_type].add(element_id)

    def remove_related_element(
        self, relationship_type: str, element_id: uuid.UUID
    ) -> bool:
        """
        Remove a related element.

        # Function removes subject relationship
        # Method disconnects predicate elements
        # Operation unlinks object nodes

        Args:
            relationship_type: Type of relationship
            element_id: UUID of related element

        Returns:
            True if removed, False if not found
        """
        if relationship_type in self._related_elements:
            if element_id in self._related_elements[relationship_type]:
                self._related_elements[relationship_type].remove(element_id)
                return True
        return False

    def get_related_elements(
        self, relationship_type: Optional[str] = None
    ) -> Dict[str, Set[uuid.UUID]]:
        """
        Get related elements.

        # Function gets subject relationships
        # Method retrieves predicate connections
        # Operation returns object links

        Args:
            relationship_type: Optional type to filter by

        Returns:
            Dictionary of relationship types to sets of element IDs
        """
        if relationship_type:
            return {
                relationship_type: self._related_elements.get(
                    relationship_type, set()
                )
            }
        return self._related_elements

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert element to dictionary.

        # Function converts subject element
        # Method transforms predicate object
        # Operation serializes object state

        Returns:
            Dictionary representation of element
        """
        # Convert properties to string keys for serialization
        properties_dict = {
            prop.name: value for prop, value in self.properties.items()
        }

        # Convert related elements to lists for serialization
        related_elements = {
            rel_type: list(map(str, element_ids))
            for rel_type, element_ids in self._related_elements.items()
        }

        return {
            "id": str(self.id),
            "symbol": self.symbol,
            "name": self.name,
            "atomic_number": self.atomic_number,
            "group_id": self.group_id,
            "period_id": self.period_id,
            "category_id": self.category_id,
            "properties": properties_dict,
            "related_elements": related_elements,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Element":
        """
        Create element from dictionary.

        # Function creates subject element
        # Method parses predicate dictionary
        # Operation deserializes object state

        Args:
            data: Dictionary representation of element

        Returns:
            Element instance
        """
        # Extract core fields
        element = cls(
            id=uuid.UUID(data["id"]),
            symbol=data["symbol"],
            name=data["name"],
            atomic_number=data["atomic_number"],
            group_id=data.get("group_id"),
            period_id=data.get("period_id"),
            category_id=data.get("category_id"),
        )

        # Convert string keys back to ElementProperty enum
        properties = {}
        for key, value in data.get("properties", {}).items():
            try:
                prop = ElementProperty[key]
                properties[prop] = value
            except KeyError:
                # Skip unknown properties
                pass

        element.properties = properties

        # Convert related elements back to UUIDs
        related_elements = {}
        for rel_type, element_ids in data.get("related_elements", {}).items():
            related_elements[rel_type] = {uuid.UUID(eid) for eid in element_ids}

        element._related_elements = related_elements

        return element

    def __eq__(self, other: Any) -> bool:
        """
        Compare elements for equality.

        # Function compares subject elements
        # Method checks predicate equality
        # Operation determines object equivalence

        Args:
            other: Another element to compare with

        Returns:
            True if elements are equal
        """
        if not isinstance(other, Element):
            return False
        return self.id == other.id

    def __hash__(self) -> int:
        """
        Hash function for element.

        # Function hashes subject element
        # Method computes predicate value
        # Operation generates object identifier

        Returns:
            Hash value
        """
        return hash(self.id)
