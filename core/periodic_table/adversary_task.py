"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-ADVERSARY-TASK-0001                 │
// │ 📁 domain       : Classification, Task Modeling             │
// │ 🧠 description  : Adversary task model for CTAS             │
// │                  periodic table visualization               │
// │ 🕸️ hash_type    : UUID → CUID-linked module                 │
// │ 🔄 parent_node  : NODE_CLASSIFICATION                      │
// │ 🧩 dependencies : uuid, dataclasses, typing                │
// │ 🔧 tool_usage   : Modeling, Analysis                       │
// │ 📡 input_type   : Task metadata                            │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : classification, analysis                  │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

Adversary Task Model
-------------------
This module provides a data model for representing adversary tasks
in the CTAS Periodic Table of Nodes.
"""

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any
from datetime import datetime


@dataclass
class AdversaryTask:
    """
    Represents an adversary task in the CTAS framework.

    # Class models subject task
    # Model represents predicate adversary
    # Object defines operational behavior
    """

    id: uuid.UUID = field(default_factory=uuid.uuid4)
    task_id: str = ""  # Format: uuid-XXX-XXX-XXX
    hash_id: str = ""  # Format: SCHXXX-XXX
    task_name: str = ""
    description: str = ""
    capabilities: str = ""
    limitations: str = ""
    ttps: List[str] = field(
        default_factory=list
    )  # Tactics, Techniques, Procedures
    relationships: str = ""
    indicators: List[str] = field(default_factory=list)
    historical_reference: str = ""
    toolchain_refs: List[str] = field(default_factory=list)
    eei_priority: List[str] = field(
        default_factory=list
    )  # Essential Elements of Information

    # Optional advanced fields
    contextual_embedding: str = ""
    usim_meta: Dict[str, Any] = field(default_factory=dict)
    atomic_properties: Dict[str, Any] = field(default_factory=dict)
    neural_lattice: Dict[str, Any] = field(default_factory=dict)
    validation_rules: Dict[str, Any] = field(default_factory=dict)

    # Display properties
    color: str = "#4285F4"  # Default blue
    symbol: str = ""  # Derived from hash_id
    atomic_number: int = 0  # Derived from task_id

    def __post_init__(self):
        """Initialize derived properties after initialization."""
        # Set symbol from hash_id if not provided (e.g., SCH001 -> SC1)
        if not self.symbol and self.hash_id:
            if self.hash_id.startswith("SCH"):
                number = "".join(
                    filter(str.isdigit, self.hash_id.split("-")[0])
                )
                self.symbol = f"S{number}"

        # Set atomic number from task_id if not provided
        if self.atomic_number == 0 and self.task_id:
            parts = self.task_id.split("-")
            if len(parts) > 1 and parts[1].isdigit():
                self.atomic_number = int(parts[1])

    def get_property(self, property_name: str) -> Any:
        """Get a property value by name."""
        if hasattr(self, property_name):
            return getattr(self, property_name)

        # Check in atomic_properties
        if property_name in self.atomic_properties:
            return self.atomic_properties[property_name]

        # Check in usim_meta
        if property_name in self.usim_meta:
            return self.usim_meta[property_name]

        return None

    def set_property(self, property_name: str, value: Any) -> None:
        """Set a property value by name."""
        if hasattr(self, property_name):
            setattr(self, property_name, value)
        else:
            # Store in atomic_properties
            self.atomic_properties[property_name] = value

    def get_category(self) -> str:
        """Get the category based on the hash_id."""
        if not self.hash_id:
            return "Unknown"

        # Extract category from SCH prefix
        if self.hash_id.startswith("SCH"):
            category_num = self.hash_id.split("-")[0].replace("SCH", "")
            categories = {
                "001": "Planning",
                "002": "Reconnaissance",
                "003": "Security",
                "004": "Resources",
                "005": "Access",
                "006": "Network",
                "007": "Cyber",
                "008": "Physical",
                "009": "Execution",
                "010": "Evasion",
                "011": "Transnational",
            }
            return categories.get(category_num, "Unknown")

        return "Unknown"

    def get_valence(self) -> int:
        """Get the valence (connection capacity) of the task."""
        # Check atomic properties first
        if "valence" in self.atomic_properties:
            return self.atomic_properties["valence"]

        # Default based on category
        category = self.get_category()
        default_valence = {
            "Planning": 3,
            "Reconnaissance": 5,
            "Security": 4,
            "Resources": 2,
            "Access": 3,
            "Network": 6,
            "Cyber": 4,
            "Physical": 3,
            "Execution": 4,
            "Evasion": 4,
            "Transnational": 7,
        }
        return default_valence.get(category, 4)

    def get_reliability(self) -> float:
        """Get the reliability score."""
        return self.atomic_properties.get("reliability", 0.7)

    def get_confidence(self) -> float:
        """Get the confidence score."""
        return self.atomic_properties.get("confidence", 0.7)

    def get_maturity(self) -> float:
        """Get the maturity score."""
        return self.atomic_properties.get("maturity", 0.7)

    def get_complexity(self) -> float:
        """Get the complexity score."""
        return self.atomic_properties.get("complexity", 0.5)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": str(self.id),
            "task_id": self.task_id,
            "hash_id": self.hash_id,
            "task_name": self.task_name,
            "description": self.description,
            "capabilities": self.capabilities,
            "limitations": self.limitations,
            "ttps": self.ttps,
            "relationships": self.relationships,
            "indicators": self.indicators,
            "historical_reference": self.historical_reference,
            "toolchain_refs": self.toolchain_refs,
            "eei_priority": self.eei_priority,
            "contextual_embedding": self.contextual_embedding,
            "usim_meta": self.usim_meta,
            "atomic_properties": self.atomic_properties,
            "symbol": self.symbol,
            "atomic_number": self.atomic_number,
            "category": self.get_category(),
            "valence": self.get_valence(),
            "reliability": self.get_reliability(),
            "confidence": self.get_confidence(),
            "maturity": self.get_maturity(),
            "complexity": self.get_complexity(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AdversaryTask":
        """Create an AdversaryTask from a dictionary."""
        # Handle ID conversion
        if "id" in data and isinstance(data["id"], str):
            data["id"] = uuid.UUID(data["id"])

        # Extract atomic properties
        atomic_props = data.pop("atomic_properties", {})
        # Extract usim_meta
        usim_meta = data.pop("usim_meta", {})
        # Extract neural_lattice
        neural_lattice = data.pop("neural_lattice", {})
        # Extract validation_rules
        validation_rules = data.pop("validation_rules", {})

        # Create instance with basic properties
        task = cls(
            **{
                k: v
                for k, v in data.items()
                if k
                not in [
                    "atomic_properties",
                    "usim_meta",
                    "neural_lattice",
                    "validation_rules",
                ]
            }
        )

        # Set additional properties
        task.atomic_properties = atomic_props
        task.usim_meta = usim_meta
        task.neural_lattice = neural_lattice
        task.validation_rules = validation_rules

        return task
