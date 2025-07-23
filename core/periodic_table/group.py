"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-PERIODIC-GROUP-0001                 │
// │ 📁 domain       : Classification, Groups                    │
// │ 🧠 description  : Group, Period, and Category definitions   │
// │                  for the CTAS Periodic Table                │
// │ 🕸️ hash_type    : UUID → CUID-linked module                 │
// │ 🔄 parent_node  : NODE_CLASSIFICATION                      │
// │ 🧩 dependencies : uuid, dataclasses                        │
// │ 🔧 tool_usage   : Classification, Organization             │
// │ 📡 input_type   : Classification metadata                  │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : classification, organization              │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

Group Implementation
-----------------
This module defines the Group, Period, and Category classes for the CTAS Periodic Table,
allowing for structured organization of elements based on common properties.
"""

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any
from enum import Enum, auto


class GroupType(Enum):
    """
    Types of groups in the periodic table.

    # Class defines subject types
    # Enumeration catalogs predicate groups
    # Type specifies object variants
    """

    VERTICAL = auto()  # Traditional column (Group)
    HORIZONTAL = auto()  # Traditional row (Period)
    CATEGORY = auto()  # Element category/family
    BLOCK = auto()  # Classification block
    CUSTOM = auto()  # Custom grouping


@dataclass
class Group:
    """
    Group class representing a column in the periodic table.

    # Class represents subject group
    # Object models predicate column
    # Structure defines object classification
    """

    id: uuid.UUID = field(default_factory=uuid.uuid4)
    name: str = field(default="")
    symbol: str = field(default="")
    number: int = field(default=0)
    description: str = field(default="")
    type: GroupType = field(default=GroupType.VERTICAL)
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert group to dictionary.

        # Function converts subject group
        # Method transforms predicate object
        # Operation serializes object state

        Returns:
            Dictionary representation of group
        """
        return {
            "id": str(self.id),
            "name": self.name,
            "symbol": self.symbol,
            "number": self.number,
            "description": self.description,
            "type": self.type.name,
            "properties": self.properties,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Group":
        """
        Create group from dictionary.

        # Function creates subject group
        # Method parses predicate dictionary
        # Operation deserializes object state

        Args:
            data: Dictionary representation of group

        Returns:
            Group instance
        """
        return cls(
            id=uuid.UUID(data["id"]),
            name=data["name"],
            symbol=data["symbol"],
            number=data["number"],
            description=data["description"],
            type=GroupType[data["type"]],
            properties=data.get("properties", {}),
        )


@dataclass
class Period:
    """
    Period class representing a row in the periodic table.

    # Class represents subject period
    # Object models predicate row
    # Structure defines object classification
    """

    id: uuid.UUID = field(default_factory=uuid.uuid4)
    name: str = field(default="")
    number: int = field(default=0)
    description: str = field(default="")
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert period to dictionary.

        # Function converts subject period
        # Method transforms predicate object
        # Operation serializes object state

        Returns:
            Dictionary representation of period
        """
        return {
            "id": str(self.id),
            "name": self.name,
            "number": self.number,
            "description": self.description,
            "properties": self.properties,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Period":
        """
        Create period from dictionary.

        # Function creates subject period
        # Method parses predicate dictionary
        # Operation deserializes object state

        Args:
            data: Dictionary representation of period

        Returns:
            Period instance
        """
        return cls(
            id=uuid.UUID(data["id"]),
            name=data["name"],
            number=data["number"],
            description=data["description"],
            properties=data.get("properties", {}),
        )


@dataclass
class Category:
    """
    Category class representing an element family or classification.

    # Class represents subject category
    # Object models predicate family
    # Structure defines object classification
    """

    id: uuid.UUID = field(default_factory=uuid.uuid4)
    name: str = field(default="")
    symbol: str = field(default="")
    color: str = field(default="#CCCCCC")  # Default color
    description: str = field(default="")
    parent_id: Optional[uuid.UUID] = field(default=None)
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert category to dictionary.

        # Function converts subject category
        # Method transforms predicate object
        # Operation serializes object state

        Returns:
            Dictionary representation of category
        """
        return {
            "id": str(self.id),
            "name": self.name,
            "symbol": self.symbol,
            "color": self.color,
            "description": self.description,
            "parent_id": str(self.parent_id) if self.parent_id else None,
            "properties": self.properties,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Category":
        """
        Create category from dictionary.

        # Function creates subject category
        # Method parses predicate dictionary
        # Operation deserializes object state

        Args:
            data: Dictionary representation of category

        Returns:
            Category instance
        """
        return cls(
            id=uuid.UUID(data["id"]),
            name=data["name"],
            symbol=data["symbol"],
            color=data["color"],
            description=data["description"],
            parent_id=(
                uuid.UUID(data["parent_id"]) if data.get("parent_id") else None
            ),
            properties=data.get("properties", {}),
        )


# Define some standard CTAS category colors for visualization
CATEGORY_COLORS = {
    "ENTITY": "#1f77b4",  # Blue
    "INFRASTRUCTURE": "#ff7f0e",  # Orange
    "CAPABILITY": "#2ca02c",  # Green
    "THREAT": "#d62728",  # Red
    "ACTOR": "#9467bd",  # Purple
    "LOCATION": "#8c564b",  # Brown
    "EVENT": "#e377c2",  # Pink
    "INTELLIGENCE": "#7f7f7f",  # Gray
    "ALGORITHM": "#bcbd22",  # Olive
    "RESOURCE": "#17becf",  # Cyan
    "RELATIONSHIP": "#aec7e8",  # Light blue
    "ATTRIBUTE": "#ffbb78",  # Light orange
    "PERCEPTION": "#98df8a",  # Light green
    "RESPONSE": "#ff9896",  # Light red
    "INTERACTION": "#c5b0d5",  # Light purple
    "OPERATION": "#c49c94",  # Light brown
}
