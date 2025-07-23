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
from dataclasses import dataclass, field
from typing import Dict, Set, Any, Optional
from datetime import datetime

from core.new.models.element_property import ElementProperty


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

    def __post_init__(self) -> None:
        """
        Initialize properties if not provided.

        # Function initializes subject properties
        # Method populates predicate defaults
        # Operation configures object state
        """
        self._initialize_core_properties()
        self._initialize_classification_properties()
        self._initialize_timestamps()

    def _initialize_core_properties(self) -> None:
        """
        Initialize core element properties.
        
        # Function sets subject core properties
        # Method initializes predicate identifiers
        # Operation configures object basics
        """
        self.set_property(ElementProperty.ATOMIC_NUMBER, self.atomic_number)
        self.set_property(ElementProperty.SYMBOL, self.symbol)
        self.set_property(ElementProperty.NAME, self.name)
        self.set_property(ElementProperty.HASH_ID, str(self.id))

    def _initialize_classification_properties(self) -> None:
        """
        Initialize classification properties if provided.
        
        # Function sets subject classification
        # Method initializes predicate categories
        # Operation configures object grouping
        """
        if self.group_id:
            self.set_property(ElementProperty.GROUP, self.group_id)
        if self.period_id:
            self.set_property(ElementProperty.PERIOD, self.period_id)
        if self.category_id:
            self.set_property(ElementProperty.CATEGORY, self.category_id)

    def _initialize_timestamps(self) -> None:
        """
        Initialize timestamp properties if not present.
        
        # Function sets subject timestamps
        # Method initializes predicate dates
        # Operation configures object timeline
        """
        current_time = datetime.now().isoformat()
        
        if ElementProperty.DISCOVERY_DATE not in self.properties:
            self.set_property(ElementProperty.DISCOVERY_DATE, current_time)
            
        if ElementProperty.LAST_UPDATED not in self.properties:
            self.set_property(ElementProperty.LAST_UPDATED, current_time)

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