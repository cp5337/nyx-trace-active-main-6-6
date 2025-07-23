"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-STORYTELLER-ELEMENTS-0001           │
// │ 📁 domain       : Storytelling, Data Model, Elements        │
// │ 🧠 description  : Story elements, milestones, and timelines │
// │                  for operational workflow tracking          │
// │ 🕸️ hash_type    : UUID → CUID-linked data model             │
// │ 🔄 parent_node  : NODE_INTERFACE                           │
// │ 🧩 dependencies : dataclasses, enum, datetime, uuid, typing │
// │ 🔧 tool_usage   : Data Model, Storytelling, Tracking        │
// │ 📡 input_type   : Application data, user-defined content    │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : data representation, narrative structuring │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

CTAS Story Elements
-----------------
This module provides data models for representing story elements,
milestones, and timelines for the Interactive Workflow Progress
Storyteller. It defines the core data structures used throughout
the storyteller module for representing workflow elements.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime
import uuid
from typing import List, Dict, Optional, Any, Union


# Function defines subject status
# Method specifies predicate states
# Enumeration categorizes object conditions
class ElementStatus(Enum):
    """
    Status values for story elements

    # Enumeration defines subject states
    # Class specifies predicate statuses
    # Type categorizes object conditions
    """

    PLANNED = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    BLOCKED = auto()
    SKIPPED = auto()
    FAILED = auto()


# Function defines subject types
# Method specifies predicate categories
# Enumeration categorizes object classifications
class StoryElementType(Enum):
    """
    Types of story elements

    # Enumeration defines subject types
    # Class specifies predicate categories
    # Type categorizes object classifications
    """

    EVENT = auto()
    MILESTONE = auto()
    DISCOVERY = auto()
    DECISION = auto()
    OBSTACLE = auto()
    INSIGHT = auto()
    ACTION = auto()
    RESOURCE = auto()


# Function defines subject element
# Method specifies predicate structure
# Class represents object component
@dataclass
class StoryElement:
    """
    Base class for story elements

    # Class defines subject element
    # Data structure specifies predicate component
    # Model represents object entity
    """

    title: str
    description: str
    element_type: StoryElementType
    status: ElementStatus = ElementStatus.PLANNED
    timestamp: datetime = field(default_factory=datetime.now)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert element to dictionary

        # Function converts subject element
        # Method transforms predicate data
        # Operation serializes object state

        Returns:
            Dictionary representation of the element
        """
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "element_type": self.element_type.name,
            "status": self.status.name,
            "timestamp": self.timestamp.isoformat(),
            "parent_id": self.parent_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StoryElement":
        """
        Create element from dictionary

        # Function creates subject element
        # Method constructs predicate instance
        # Operation deserializes object data

        Args:
            data: Dictionary with element data

        Returns:
            New StoryElement instance
        """
        # Convert string representations back to enum values
        element_type = StoryElementType[data["element_type"]]
        status = ElementStatus[data["status"]]

        # Parse timestamp
        if isinstance(data["timestamp"], str):
            timestamp = datetime.fromisoformat(data["timestamp"])
        else:
            timestamp = data["timestamp"]

        return cls(
            id=data["id"],
            title=data["title"],
            description=data["description"],
            element_type=element_type,
            status=status,
            timestamp=timestamp,
            parent_id=data.get("parent_id"),
            metadata=data.get("metadata", {}),
        )


# Function defines subject milestone
# Method specifies predicate marker
# Class represents object achievement
@dataclass
class StoryMilestone(StoryElement):
    """
    A milestone in a story timeline

    # Class defines subject milestone
    # Data structure specifies predicate marker
    # Model represents object achievement
    """

    def __post_init__(self):
        """
        Post-initialization hook

        # Function initializes subject milestone
        # Method completes predicate initialization
        # Operation finalizes object state
        """
        if self.element_type != StoryElementType.MILESTONE:
            self.element_type = StoryElementType.MILESTONE


# Function defines subject timeline
# Method specifies predicate sequence
# Class represents object narrative
@dataclass
class StoryTimeline:
    """
    A timeline containing story elements

    # Class defines subject timeline
    # Data structure specifies predicate sequence
    # Model represents object narrative
    """

    title: str
    description: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    elements: List[Union[StoryElement, StoryMilestone]] = field(
        default_factory=list
    )
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_element(self, element: Union[StoryElement, StoryMilestone]) -> None:
        """
        Add an element to the timeline

        # Function adds subject element
        # Method extends predicate timeline
        # Operation appends object item

        Args:
            element: Element to add
        """
        self.elements.append(element)
        self.updated_at = datetime.now()

    def get_element(
        self, element_id: str
    ) -> Optional[Union[StoryElement, StoryMilestone]]:
        """
        Get an element by ID

        # Function gets subject element
        # Method retrieves predicate item
        # Operation finds object component

        Args:
            element_id: ID of the element to find

        Returns:
            Element with matching ID or None
        """
        for element in self.elements:
            if element.id == element_id:
                return element
        return None

    def update_element(self, element_id: str, **kwargs) -> bool:
        """
        Update an element's attributes

        # Function updates subject element
        # Method modifies predicate attributes
        # Operation changes object state

        Args:
            element_id: ID of the element to update
            **kwargs: Attributes to update

        Returns:
            True if successful, False otherwise
        """
        element = self.get_element(element_id)
        if element is None:
            return False

        for key, value in kwargs.items():
            if hasattr(element, key):
                setattr(element, key, value)

        self.updated_at = datetime.now()
        return True

    def remove_element(self, element_id: str) -> bool:
        """
        Remove an element from the timeline

        # Function removes subject element
        # Method deletes predicate item
        # Operation discards object component

        Args:
            element_id: ID of the element to remove

        Returns:
            True if successful, False otherwise
        """
        element = self.get_element(element_id)
        if element is None:
            return False

        self.elements.remove(element)
        self.updated_at = datetime.now()
        return True

    def get_child_elements(
        self, parent_id: str
    ) -> List[Union[StoryElement, StoryMilestone]]:
        """
        Get all elements with a specific parent

        # Function gets subject children
        # Method retrieves predicate descendants
        # Operation finds object related

        Args:
            parent_id: ID of the parent element

        Returns:
            List of child elements
        """
        return [e for e in self.elements if e.parent_id == parent_id]

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert timeline to dictionary

        # Function converts subject timeline
        # Method transforms predicate data
        # Operation serializes object state

        Returns:
            Dictionary representation of the timeline
        """
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "elements": [e.to_dict() for e in self.elements],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StoryTimeline":
        """
        Create timeline from dictionary

        # Function creates subject timeline
        # Method constructs predicate instance
        # Operation deserializes object data

        Args:
            data: Dictionary with timeline data

        Returns:
            New StoryTimeline instance
        """
        # Parse timestamps
        if isinstance(data["created_at"], str):
            created_at = datetime.fromisoformat(data["created_at"])
        else:
            created_at = data["created_at"]

        if isinstance(data["updated_at"], str):
            updated_at = datetime.fromisoformat(data["updated_at"])
        else:
            updated_at = data["updated_at"]

        # Create timeline
        timeline = cls(
            id=data["id"],
            title=data["title"],
            description=data["description"],
            created_at=created_at,
            updated_at=updated_at,
            metadata=data.get("metadata", {}),
        )

        # Add elements
        for element_data in data["elements"]:
            if element_data.get("element_type") == "MILESTONE":
                element = StoryMilestone.from_dict(element_data)
            else:
                element = StoryElement.from_dict(element_data)
            timeline.elements.append(element)

        return timeline


# CTAS Module Line Count:
# USIM Header: 17 lines
# Comments: 113 lines
# Code: 246 lines
# Total: 376 lines
