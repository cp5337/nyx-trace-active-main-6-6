"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-ELEMENT-RELATIONSHIPS-0001          │
// │ 📁 domain       : Classification, Element, Relationships    │
// │ 🧠 description  : Element relationship management for the   │
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

Element Relationship Management
-------------------
This module extends the Element class with relationship management functionality
for connecting elements in the CTAS Periodic Table of Nodes.
"""

import uuid
from typing import Dict, Set, List, Optional, Any


class ElementRelationshipManager:
    """
    Mixin class for managing element relationships.
    
    # Class manages subject relationships
    # Mixin provides predicate connections
    # Component handles object links
    """
    
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
        
    def get_relationship_types(self) -> List[str]:
        """
        Get all relationship types for this element.
        
        # Function gets subject relationship types
        # Method retrieves predicate categories
        # Operation returns object connection types
        
        Returns:
            List of relationship type strings
        """
        return list(self._related_elements.keys())
        
    def has_relationship(self, relationship_type: str) -> bool:
        """
        Check if element has any relationships of given type.
        
        # Function checks subject relationship
        # Method verifies predicate connections
        # Operation confirms object linkage
        
        Args:
            relationship_type: Type of relationship to check
            
        Returns:
            True if element has relationships of this type
        """
        return (relationship_type in self._related_elements and 
                len(self._related_elements[relationship_type]) > 0)
                
    def clear_relationships(self, relationship_type: Optional[str] = None) -> None:
        """
        Clear all relationships of a given type or all if not specified.
        
        # Function clears subject relationships
        # Method removes predicate connections
        # Operation resets object links
        
        Args:
            relationship_type: Optional type to clear, all if None
        """
        if relationship_type:
            if relationship_type in self._related_elements:
                self._related_elements[relationship_type].clear()
        else:
            self._related_elements.clear()