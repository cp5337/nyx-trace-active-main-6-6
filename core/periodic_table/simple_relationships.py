"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-SIMPLE-RELATIONSHIPS-0001            │
// │ 📁 domain       : Classification, Relationships            │
// │ 🧠 description  : Simplified relationship manager for       │
// │                  adversary tasks visualization             │
// │ 🕸️ hash_type    : UUID → CUID-linked module                 │
// │ 🔄 parent_node  : NODE_CLASSIFICATION                      │
// │ 🧩 dependencies : uuid, dataclasses                        │
// │ 🔧 tool_usage   : Classification, Relationships            │
// │ 📡 input_type   : Relationship metadata                    │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : relationship modeling                     │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

Simple Relationship Implementation
-----------------------
This module provides a simpler implementation of relationships for the CTAS
Adversary Task model, focusing on string-based relationship types for flexibility.
"""

import uuid
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Tuple

# Configure logger
logger = logging.getLogger("periodic_table.simple_relationships")
logger.setLevel(logging.INFO)


@dataclass
class Relationship:
    """
    Simple relationship class representing a connection between tasks.

    # Class represents subject relationship
    # Object models predicate connection
    # Structure encapsulates object properties
    """

    id: uuid.UUID = field(default_factory=uuid.uuid4)
    source_id: uuid.UUID = field(default=None)
    target_id: uuid.UUID = field(default=None)
    type: str = field(default="CONNECTED_TO")
    name: str = field(default="")
    description: str = field(default="")
    weight: float = field(default=1.0)  # Relationship strength/weight
    bidirectional: bool = field(
        default=False
    )  # Whether relationship goes both ways
    temporal_validity: Optional[Dict[str, str]] = field(
        default=None
    )  # Valid time range
    confidence: float = field(default=1.0)  # Confidence score (0-1)
    properties: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """
        Initialize with default values if needed.
        """
        if not self.name:
            self.name = self.type.replace("_", " ").title()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert relationship to dictionary.

        Returns:
            Dictionary representation of relationship
        """
        return {
            "id": str(self.id),
            "source_id": str(self.source_id) if self.source_id else None,
            "target_id": str(self.target_id) if self.target_id else None,
            "type": self.type,
            "name": self.name,
            "description": self.description,
            "weight": self.weight,
            "bidirectional": self.bidirectional,
            "temporal_validity": self.temporal_validity,
            "confidence": self.confidence,
            "properties": self.properties,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Relationship":
        """
        Create relationship from dictionary.

        Args:
            data: Dictionary representation of relationship

        Returns:
            Relationship instance
        """
        return cls(
            id=uuid.UUID(data["id"]) if "id" in data else uuid.uuid4(),
            source_id=(
                uuid.UUID(data["source_id"]) if data.get("source_id") else None
            ),
            target_id=(
                uuid.UUID(data["target_id"]) if data.get("target_id") else None
            ),
            type=data.get("type", "CONNECTED_TO"),
            name=data.get("name", ""),
            description=data.get("description", ""),
            weight=data.get("weight", 1.0),
            bidirectional=data.get("bidirectional", False),
            temporal_validity=data.get("temporal_validity"),
            confidence=data.get("confidence", 1.0),
            properties=data.get("properties", {}),
        )


class RelationshipManager:
    """
    Manager for relationships between tasks.
    """

    def __init__(self):
        """Initialize relationship manager."""
        self.relationships = {}  # Dictionary of relationships by ID
        self.relationships_by_source = (
            {}
        )  # Dictionary of relationships by source ID
        self.relationships_by_target = (
            {}
        )  # Dictionary of relationships by target ID
        self.relationships_by_type = {}  # Dictionary of relationships by type

    def clear(self):
        """Clear all relationships."""
        self.relationships = {}
        self.relationships_by_source = {}
        self.relationships_by_target = {}
        self.relationships_by_type = {}

    def add_relationship(
        self,
        relationship: Relationship,
        skip_inverse: bool = False,
        skip_save: bool = False,
    ) -> bool:
        """
        Add a relationship to the manager.

        Args:
            relationship: Relationship to add
            skip_inverse: Whether to skip creating the inverse relationship for bidirectional
                          relationships (to prevent infinite recursion)
            skip_save: Whether to skip saving to database (for bulk loading)

        Returns:
            True if added, False if duplicate
        """
        # Check for duplicates
        for existing_rel in self.get_relationships(
            source_id=relationship.source_id,
            target_id=relationship.target_id,
            relationship_type=relationship.type,
        ):
            logger.info(
                f"Skipping duplicate relationship: {relationship.type} from {relationship.source_id} to {relationship.target_id}"
            )
            return False

        # Add to main dictionary
        self.relationships[str(relationship.id)] = relationship

        # Add to indices
        source_id_str = str(relationship.source_id)
        if source_id_str not in self.relationships_by_source:
            self.relationships_by_source[source_id_str] = []
        self.relationships_by_source[source_id_str].append(relationship)

        target_id_str = str(relationship.target_id)
        if target_id_str not in self.relationships_by_target:
            self.relationships_by_target[target_id_str] = []
        self.relationships_by_target[target_id_str].append(relationship)

        if relationship.type not in self.relationships_by_type:
            self.relationships_by_type[relationship.type] = []
        self.relationships_by_type[relationship.type].append(relationship)

        # If bidirectional, add inverse relationship
        if relationship.bidirectional and not skip_inverse:
            inverse_type = self._get_inverse_type(relationship.type)

            # Create inverse relationship
            inverse_relationship = Relationship(
                source_id=relationship.target_id,
                target_id=relationship.source_id,
                type=inverse_type,
                name=inverse_type.replace("_", " ").title(),
                description=relationship.description,
                weight=relationship.weight,
                bidirectional=True,
                temporal_validity=relationship.temporal_validity,
                confidence=relationship.confidence,
                properties=(
                    relationship.properties.copy()
                    if relationship.properties
                    else {}
                ),
            )

            # Add inverse relationship with skip_inverse to prevent infinite recursion
            self.add_relationship(
                inverse_relationship, skip_inverse=True, skip_save=skip_save
            )

        return True

    def _get_inverse_type(self, rel_type: str) -> str:
        """Get the inverse relationship type."""
        # Define inverse mappings
        inverse_mappings = {
            "PARENT_OF": "CHILD_OF",
            "CHILD_OF": "PARENT_OF",
            "CONNECTED_TO": "CONNECTED_TO",
            "ANALYZES": "ANALYZED_BY",
            "ANALYZED_BY": "ANALYZES",
            "DERIVED_FROM": "DERIVED_INTO",
            "DERIVED_INTO": "DERIVED_FROM",
            "PRECEDES": "FOLLOWS",
            "FOLLOWS": "PRECEDES",
            "ENABLES": "ENABLED_BY",
            "ENABLED_BY": "ENABLES",
            "INHIBITS": "INHIBITED_BY",
            "INHIBITED_BY": "INHIBITS",
            "LOCATED_AT": "LOCATION_OF",
            "LOCATION_OF": "LOCATED_AT",
            "CONTAINS": "CONTAINED_BY",
            "CONTAINED_BY": "CONTAINS",
            "CAUSES": "CAUSED_BY",
            "CAUSED_BY": "CAUSES",
            "AFFECTS": "AFFECTED_BY",
            "AFFECTED_BY": "AFFECTS",
        }

        # Return inverse type if defined, otherwise return same type
        return inverse_mappings.get(rel_type, rel_type)

    def get_relationships(
        self, source_id=None, target_id=None, relationship_type=None
    ) -> List[Relationship]:
        """
        Get relationships by source ID, target ID, and/or type.

        Args:
            source_id: Optional source ID filter
            target_id: Optional target ID filter
            relationship_type: Optional relationship type filter

        Returns:
            List of matching relationships
        """
        result = []

        # Get all relationships if no filters provided
        if (
            source_id is None
            and target_id is None
            and relationship_type is None
        ):
            return list(self.relationships.values())

        # Filter by source ID
        if source_id is not None:
            source_id_str = str(source_id)
            if source_id_str in self.relationships_by_source:
                if not result:
                    result = self.relationships_by_source[source_id_str]
                else:
                    result = [
                        r for r in result if str(r.source_id) == source_id_str
                    ]
            else:
                return []  # No relationships with this source ID

        # Filter by target ID
        if target_id is not None:
            target_id_str = str(target_id)
            if target_id_str in self.relationships_by_target:
                if not result:
                    result = self.relationships_by_target[target_id_str]
                else:
                    result = [
                        r for r in result if str(r.target_id) == target_id_str
                    ]
            else:
                return []  # No relationships with this target ID

        # Filter by relationship type
        if relationship_type is not None:
            if relationship_type in self.relationships_by_type:
                if not result:
                    result = self.relationships_by_type[relationship_type]
                else:
                    result = [r for r in result if r.type == relationship_type]
            else:
                return []  # No relationships with this type

        return result

    def get_relationships_for_node(
        self, node_id: uuid.UUID
    ) -> List[Relationship]:
        """
        Get all relationships for a node (source or target).

        Args:
            node_id: Node ID

        Returns:
            List of relationships
        """
        relationships = []
        node_id_str = str(node_id)

        # Get relationships where node is source
        if node_id_str in self.relationships_by_source:
            relationships.extend(self.relationships_by_source[node_id_str])

        # Get relationships where node is target
        if node_id_str in self.relationships_by_target:
            relationships.extend(self.relationships_by_target[node_id_str])

        return relationships

    def get_all_relationships(self) -> List[Relationship]:
        """
        Get all relationships.

        Returns:
            List of all relationships
        """
        return list(self.relationships.values())
