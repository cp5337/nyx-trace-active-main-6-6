"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-ELEMENT-SERIALIZATION-0001          │
// │ 📁 domain       : Classification, Element, Serialization    │
// │ 🧠 description  : Element serialization for the             │
// │                  CTAS Periodic Table of Nodes               │
// │ 🕸️ hash_type    : UUID → CUID-linked module                 │
// │ 🔄 parent_node  : NODE_CLASSIFICATION                      │
// │ 🧩 dependencies : uuid, typing                             │
// │ 🔧 tool_usage   : Classification, Analysis                 │
// │ 📡 input_type   : Element metadata                         │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : classification, categorization            │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

Element Serialization
-------------------
This module extends the Element class with serialization functionality
for converting elements to and from dictionary format for storage and transmission.
"""

import uuid
from typing import Dict, Set, Any

from core.new.models.element_property import ElementProperty


class ElementSerializationManager:
    """
    Mixin class for element serialization.
    
    # Class manages subject serialization
    # Mixin provides predicate conversion
    # Component handles object persistence
    """
    
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
    def _parse_properties(cls, data: Dict[str, Any]) -> Dict[ElementProperty, Any]:
        """
        Parse properties from dictionary data.
        
        # Function parses subject properties
        # Method converts predicate keys
        # Operation transforms object format
        
        Args:
            data: Dictionary containing property data
            
        Returns:
            Parsed properties dictionary
        """
        properties = {}
        for key, value in data.get("properties", {}).items():
            try:
                prop = ElementProperty[key]
                properties[prop] = value
            except KeyError:
                # Skip unknown properties
                pass
        return properties
    
    @classmethod
    def _parse_related_elements(cls, data: Dict[str, Any]) -> Dict[str, Set[uuid.UUID]]:
        """
        Parse related elements from dictionary data.
        
        # Function parses subject relationships
        # Method converts predicate references
        # Operation transforms object format
        
        Args:
            data: Dictionary containing related elements data
            
        Returns:
            Parsed related elements dictionary
        """
        related_elements = {}
        for rel_type, element_ids in data.get("related_elements", {}).items():
            related_elements[rel_type] = {uuid.UUID(eid) for eid in element_ids}
        return related_elements