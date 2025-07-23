"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-PERIODIC-RELATIONSHIPS-0001         │
// │ 📁 domain       : Classification, Relationships            │
// │ 🧠 description  : Relationship definitions for the         │
// │                  CTAS Periodic Table of Nodes              │
// │ 🕸️ hash_type    : UUID → CUID-linked module                 │
// │ 🔄 parent_node  : NODE_CLASSIFICATION                      │
// │ 🧩 dependencies : uuid, dataclasses                        │
// │ 🔧 tool_usage   : Classification, Relationships            │
// │ 📡 input_type   : Relationship metadata                    │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : relationship modeling                     │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

Relationship Implementation
-----------------------
This module defines the relationship types and classes for the CTAS Periodic Table,
enabling the representation of connections between elements.
"""

import uuid
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Tuple


class RelationshipType(Enum):
    """
    Standard relationship types between elements in the periodic table.

    # Class defines subject relationships
    # Enumeration catalogs predicate connections
    # Type specifies object variants
    """

    # Core relationships
    PARENT_OF = auto()  # Hierarchical parent
    CHILD_OF = auto()  # Hierarchical child
    CONNECTED_TO = auto()  # Generic connection

    # Intelligence relationships
    ANALYZES = auto()  # Analysis relationship
    ANALYZED_BY = auto()  # Target of analysis
    DERIVED_FROM = auto()  # Derivation relationship
    DERIVED_INTO = auto()  # Target of derivation
    CORROBORATES = auto()  # Corroboration relationship
    CONTRADICTS = auto()  # Contradiction relationship

    # Operational relationships
    PRECEDES = auto()  # Temporal precedence
    FOLLOWS = auto()  # Temporal follow-up
    ENABLES = auto()  # Enablement relationship
    ENABLED_BY = auto()  # Target of enablement
    INHIBITS = auto()  # Inhibition relationship
    INHIBITED_BY = auto()  # Target of inhibition

    # Spatial relationships
    LOCATED_AT = auto()  # Location relationship
    LOCATION_OF = auto()  # Target of location
    CONTAINS = auto()  # Containment relationship
    CONTAINED_BY = auto()  # Target of containment
    PROXIMATE_TO = auto()  # Proximity relationship

    # Causal relationships
    CAUSES = auto()  # Causal relationship
    CAUSED_BY = auto()  # Target of causation
    AFFECTS = auto()  # Effect relationship
    AFFECTED_BY = auto()  # Target of effect
    CORRELATED_WITH = auto()  # Correlation relationship

    # Cognitive relationships
    SIMILAR_TO = auto()  # Similarity relationship
    COMPLEMENTARY_TO = auto()  # Complementary relationship
    ANALOGOUS_TO = auto()  # Analogy relationship

    # Communication relationships
    COMMUNICATES_WITH = auto()  # Communication relationship
    INFORMS = auto()  # Information transfer
    INFORMED_BY = auto()  # Target of information
    REPORTS_TO = auto()  # Reporting relationship
    RECEIVES_REPORTS_FROM = auto()  # Target of reporting

    # Meta relationships
    REFERENCES = auto()  # Reference relationship
    REFERENCED_BY = auto()  # Target of reference
    CLASSIFIES = auto()  # Classification relationship
    CLASSIFIED_BY = auto()  # Target of classification
    INSTANTIATES = auto()  # Instantiation relationship
    INSTANTIATED_BY = auto()  # Target of instantiation


@dataclass
class Relationship:
    """
    Relationship class representing a connection between elements.

    # Class represents subject relationship
    # Object models predicate connection
    # Structure encapsulates object properties
    """

    id: uuid.UUID = field(default_factory=uuid.uuid4)
    source_id: uuid.UUID = field(default=None)
    target_id: uuid.UUID = field(default=None)
    type: RelationshipType = field(default=RelationshipType.CONNECTED_TO)
    name: str = field(default="")
    description: str = field(default="")
    weight: float = field(default=1.0)  # Relationship strength/weight
    bidirectional: bool = field(
        default=False
    )  # Whether relationship goes both ways
    temporal_validity: Optional[Tuple[str, str]] = field(
        default=None
    )  # Valid time range
    confidence: float = field(default=1.0)  # Confidence score (0-1)
    properties: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """
        Initialize with default values if needed.

        # Function initializes subject relationship
        # Method populates predicate defaults
        # Operation configures object state
        """
        if not self.name:
            self.name = self.type.name.replace("_", " ").title()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert relationship to dictionary.

        # Function converts subject relationship
        # Method transforms predicate object
        # Operation serializes object state

        Returns:
            Dictionary representation of relationship
        """
        return {
            "id": str(self.id),
            "source_id": str(self.source_id) if self.source_id else None,
            "target_id": str(self.target_id) if self.target_id else None,
            "type": self.type.name,
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

        # Function creates subject relationship
        # Method parses predicate dictionary
        # Operation deserializes object state

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
            type=RelationshipType[data["type"]],
            name=data.get("name", ""),
            description=data.get("description", ""),
            weight=data.get("weight", 1.0),
            bidirectional=data.get("bidirectional", False),
            temporal_validity=data.get("temporal_validity"),
            confidence=data.get("confidence", 1.0),
            properties=data.get("properties", {}),
        )

    def get_inverse_type(self) -> RelationshipType:
        """
        Get the inverse relationship type.

        # Function gets subject inverse
        # Method determines predicate opposite
        # Operation returns object counterpart

        Returns:
            Inverse relationship type
        """
        # Define inverse mappings
        inverse_mappings = {
            # Core relationships
            RelationshipType.PARENT_OF: RelationshipType.CHILD_OF,
            RelationshipType.CHILD_OF: RelationshipType.PARENT_OF,
            RelationshipType.CONNECTED_TO: RelationshipType.CONNECTED_TO,
            # Intelligence relationships
            RelationshipType.ANALYZES: RelationshipType.ANALYZED_BY,
            RelationshipType.ANALYZED_BY: RelationshipType.ANALYZES,
            RelationshipType.DERIVED_FROM: RelationshipType.DERIVED_INTO,
            RelationshipType.DERIVED_INTO: RelationshipType.DERIVED_FROM,
            # Temporal relationships
            RelationshipType.PRECEDES: RelationshipType.FOLLOWS,
            RelationshipType.FOLLOWS: RelationshipType.PRECEDES,
            # Operational relationships
            RelationshipType.ENABLES: RelationshipType.ENABLED_BY,
            RelationshipType.INHIBITS: RelationshipType.INHIBITED_BY,
            RelationshipType.LOCATED_AT: RelationshipType.LOCATION_OF,
            RelationshipType.CONTAINS: RelationshipType.CONTAINED_BY,
            RelationshipType.CAUSES: RelationshipType.CAUSED_BY,
            RelationshipType.CAUSED_BY: RelationshipType.CAUSES,
            RelationshipType.AFFECTS: RelationshipType.AFFECTED_BY,
            RelationshipType.AFFECTED_BY: RelationshipType.AFFECTS,
            RelationshipType.COMMUNICATES_WITH: RelationshipType.COMMUNICATES_WITH,
            RelationshipType.INFORMS: RelationshipType.INFORMED_BY,
            RelationshipType.REPORTS_TO: RelationshipType.RECEIVES_REPORTS_FROM,
            RelationshipType.REFERENCES: RelationshipType.REFERENCED_BY,
            RelationshipType.CLASSIFIES: RelationshipType.CLASSIFIED_BY,
            RelationshipType.INSTANTIATES: RelationshipType.INSTANTIATED_BY,
        }

        # Return inverse type if defined, otherwise return same type
        return inverse_mappings.get(self.type, self.type)

    def create_inverse(self) -> "Relationship":
        """
        Create inverse relationship.

        # Function creates subject inverse
        # Method generates predicate opposite
        # Operation returns object counterpart

        Returns:
            Inverse relationship
        """
        inverse_type = self.get_inverse_type()

        return Relationship(
            source_id=self.target_id,
            target_id=self.source_id,
            type=inverse_type,
            name=inverse_type.name.replace("_", " ").title(),
            description=self.description,
            weight=self.weight,
            bidirectional=self.bidirectional,
            temporal_validity=self.temporal_validity,
            confidence=self.confidence,
            properties=self.properties.copy(),
        )


class RelationshipManager:
    """
    Manager for creating and tracking relationships between elements.

    # Class manages subject relationships
    # Manager handles predicate connections
    # Component tracks object links
    """

    def __init__(self):
        """
        Initialize relationship manager.

        # Function initializes subject manager
        # Method prepares predicate storage
        # Operation configures object state
        """
        self.relationships: Dict[uuid.UUID, Relationship] = {}
        self.source_index: Dict[uuid.UUID, Set[uuid.UUID]] = {}
        self.target_index: Dict[uuid.UUID, Set[uuid.UUID]] = {}
        self.type_index: Dict[RelationshipType, Set[uuid.UUID]] = {}

    def add_relationship(
        self, relationship: Relationship, skip_inverse: bool = False
    ) -> None:
        """
        Add a relationship to the manager.

        # Function adds subject relationship
        # Method stores predicate connection
        # Operation indexes object link

        Args:
            relationship: Relationship to add
            skip_inverse: Flag to prevent creating inverse relationships (prevents recursion)
        """
        # Check if relationship already exists
        for existing_rel in self.relationships.values():
            if (
                existing_rel.source_id == relationship.source_id
                and existing_rel.target_id == relationship.target_id
                and existing_rel.type == relationship.type
            ):
                print(
                    f"Skipping duplicate relationship: {relationship.type.name} from {relationship.source_id} to {relationship.target_id}"
                )
                return

        # Store relationship
        self.relationships[relationship.id] = relationship
        print(
            f"Added relationship: {relationship.type.name} from {relationship.source_id} to {relationship.target_id}"
        )

        # Update source index
        if relationship.source_id not in self.source_index:
            self.source_index[relationship.source_id] = set()
        self.source_index[relationship.source_id].add(relationship.id)

        # Update target index
        if relationship.target_id not in self.target_index:
            self.target_index[relationship.target_id] = set()
        self.target_index[relationship.target_id].add(relationship.id)

        # Update type index
        if relationship.type not in self.type_index:
            self.type_index[relationship.type] = set()
        self.type_index[relationship.type].add(relationship.id)

        # Add inverse relationship if bidirectional and not skipping inverses
        if relationship.bidirectional and not skip_inverse:
            inverse = relationship.create_inverse()
            print(
                f"Adding inverse relationship: {inverse.type.name} from {inverse.source_id} to {inverse.target_id}"
            )
            # Pass skip_inverse=True to prevent infinite recursion
            self.add_relationship(inverse, skip_inverse=True)

    def remove_relationship(self, relationship_id: uuid.UUID) -> bool:
        """
        Remove a relationship from the manager.

        # Function removes subject relationship
        # Method deletes predicate connection
        # Operation unindexes object link

        Args:
            relationship_id: ID of relationship to remove

        Returns:
            True if removed, False if not found
        """
        if relationship_id not in self.relationships:
            return False

        relationship = self.relationships[relationship_id]

        # Remove from source index
        if relationship.source_id in self.source_index:
            self.source_index[relationship.source_id].discard(relationship_id)
            if not self.source_index[relationship.source_id]:
                del self.source_index[relationship.source_id]

        # Remove from target index
        if relationship.target_id in self.target_index:
            self.target_index[relationship.target_id].discard(relationship_id)
            if not self.target_index[relationship.target_id]:
                del self.target_index[relationship.target_id]

        # Remove from type index
        if relationship.type in self.type_index:
            self.type_index[relationship.type].discard(relationship_id)
            if not self.type_index[relationship.type]:
                del self.type_index[relationship.type]

        # Remove relationship
        del self.relationships[relationship_id]

        return True

    def get_relationship(
        self, relationship_id: uuid.UUID
    ) -> Optional[Relationship]:
        """
        Get a relationship by ID.

        # Function gets subject relationship
        # Method retrieves predicate connection
        # Operation returns object link

        Args:
            relationship_id: ID of relationship to get

        Returns:
            Relationship if found, None otherwise
        """
        return self.relationships.get(relationship_id)

    def get_relationships_by_source(
        self, source_id: uuid.UUID
    ) -> List[Relationship]:
        """
        Get relationships by source element ID.

        # Function gets subject relationships
        # Method retrieves predicate outgoing
        # Operation returns object links

        Args:
            source_id: Source element ID

        Returns:
            List of relationships with the given source
        """
        if source_id not in self.source_index:
            return []

        return [
            self.relationships[rel_id]
            for rel_id in self.source_index[source_id]
        ]

    def get_relationships_by_target(
        self, target_id: uuid.UUID
    ) -> List[Relationship]:
        """
        Get relationships by target element ID.

        # Function gets subject relationships
        # Method retrieves predicate incoming
        # Operation returns object links

        Args:
            target_id: Target element ID

        Returns:
            List of relationships with the given target
        """
        if target_id not in self.target_index:
            return []

        return [
            self.relationships[rel_id]
            for rel_id in self.target_index[target_id]
        ]

    def get_relationships_by_type(
        self, rel_type: RelationshipType
    ) -> List[Relationship]:
        """
        Get relationships by type.

        # Function gets subject relationships
        # Method retrieves predicate typed
        # Operation returns object links

        Args:
            rel_type: Relationship type

        Returns:
            List of relationships with the given type
        """
        if rel_type not in self.type_index:
            return []

        return [
            self.relationships[rel_id] for rel_id in self.type_index[rel_type]
        ]

    def get_connections(
        self, element_id: uuid.UUID
    ) -> List[Tuple[Relationship, uuid.UUID]]:
        """
        Get all connections for an element.

        # Function gets subject connections
        # Method retrieves predicate links
        # Operation returns object relationships

        Args:
            element_id: Element ID

        Returns:
            List of (relationship, connected_element_id) tuples
        """
        connections = []

        # Get outgoing relationships
        for rel in self.get_relationships_by_source(element_id):
            connections.append((rel, rel.target_id))

        # Get incoming relationships
        for rel in self.get_relationships_by_target(element_id):
            connections.append((rel, rel.source_id))

        return connections

    def get_path(
        self, source_id: uuid.UUID, target_id: uuid.UUID, max_depth: int = 5
    ) -> List[Relationship]:
        """
        Find a path between two elements.

        # Function finds subject path
        # Method discovers predicate route
        # Operation returns object connections

        Args:
            source_id: Source element ID
            target_id: Target element ID
            max_depth: Maximum path depth

        Returns:
            List of relationships forming a path, or empty list if no path found
        """
        # Implement breadth-first search
        visited = set()
        queue = [(source_id, [])]

        while queue and len(visited) < max_depth:
            current_id, path = queue.pop(0)

            if current_id == target_id:
                return path

            if current_id in visited:
                continue

            visited.add(current_id)

            # Process outgoing connections
            for rel in self.get_relationships_by_source(current_id):
                if rel.target_id not in visited:
                    queue.append((rel.target_id, path + [rel]))

            # Process incoming connections
            for rel in self.get_relationships_by_target(current_id):
                if rel.source_id not in visited:
                    queue.append((rel.source_id, path + [rel]))

        return []  # No path found

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert manager to dictionary.

        # Function converts subject manager
        # Method transforms predicate object
        # Operation serializes object state

        Returns:
            Dictionary representation of relationship manager
        """
        return {
            "relationships": {
                str(rel_id): rel.to_dict()
                for rel_id, rel in self.relationships.items()
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RelationshipManager":
        """
        Create manager from dictionary.

        # Function creates subject manager
        # Method parses predicate dictionary
        # Operation deserializes object state

        Args:
            data: Dictionary representation of relationship manager

        Returns:
            RelationshipManager instance
        """
        manager = cls()

        for rel_id_str, rel_data in data.get("relationships", {}).items():
            rel = Relationship.from_dict(rel_data)
            manager.relationships[rel.id] = rel

            # Update source index
            if rel.source_id not in manager.source_index:
                manager.source_index[rel.source_id] = set()
            manager.source_index[rel.source_id].add(rel.id)

            # Update target index
            if rel.target_id not in manager.target_index:
                manager.target_index[rel.target_id] = set()
            manager.target_index[rel.target_id].add(rel.id)

            # Update type index
            if rel.type not in manager.type_index:
                manager.type_index[rel.type] = set()
            manager.type_index[rel.type].add(rel.id)

        return manager
