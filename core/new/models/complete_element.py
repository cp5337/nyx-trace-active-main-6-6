"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-PERIODIC-ELEMENT-COMPLETE-0001      │
// │ 📁 domain       : Classification, Element                   │
// │ 🧠 description  : Complete Element implementation for the   │
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

Complete Element Implementation
-------------------
This module combines all Element functionality into a single class
for the CTAS Periodic Table of Nodes.
"""

import uuid
from dataclasses import dataclass, field
from typing import Dict, Set, List, Any, Optional, ClassVar

from core.new.models.element import Element
from core.new.models.element_property import ElementProperty
from core.new.models.element_relationships import ElementRelationshipManager
from core.new.models.element_serialization import ElementSerializationManager


@dataclass
class CompleteElement(Element):
    """
    Complete Element class with all functionality for CTAS Periodic Table.

    # Class represents subject element
    # Object models predicate node
    # Structure encapsulates object properties
    """
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CompleteElement":
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

        # Parse properties
        element.properties = ElementSerializationManager._parse_properties(data)
        
        # Parse related elements
        element._related_elements = (
            ElementSerializationManager._parse_related_elements(data)
        )

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
        if not isinstance(other, CompleteElement):
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


# Add relationship methods to CompleteElement
for method_name in dir(ElementRelationshipManager):
    # Skip dunder methods and non-callable attributes
    if (
        not method_name.startswith("_") 
        or method_name in ("__init__", "__hash__", "__eq__") 
        or not callable(getattr(ElementRelationshipManager, method_name))
    ):
        continue
    
    # Add method to CompleteElement
    setattr(
        CompleteElement, 
        method_name, 
        getattr(ElementRelationshipManager, method_name)
    )

# Add serialization methods to CompleteElement
for method_name in dir(ElementSerializationManager):
    # Skip dunder methods, class methods, and non-callable attributes  
    if (
        method_name.startswith("_") 
        and method_name not in ("_parse_properties", "_parse_related_elements")
        or not callable(getattr(ElementSerializationManager, method_name))
    ):
        continue
    
    # Don't override methods that already exist
    if method_name in dir(CompleteElement) and method_name != "from_dict":
        continue
        
    # Add method to CompleteElement
    setattr(
        CompleteElement, 
        method_name, 
        getattr(ElementSerializationManager, method_name)
    )