"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-PERIODIC-TABLE-0001                 │
// │ 📁 domain       : Classification, Ontology                  │
// │ 🧠 description  : Periodic Table of Nodes implementation    │
// │                  for CTAS classification system             │
// │ 🕸️ hash_type    : UUID → CUID-linked module                 │
// │ 🔄 parent_node  : NODE_CLASSIFICATION                      │
// │ 🧩 dependencies : numpy, pandas, sqlalchemy                │
// │ 🔧 tool_usage   : Classification, Analysis                 │
// │ 📡 input_type   : Node metadata                            │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : classification, categorization            │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

Periodic Table of Nodes Package
-----------------------------
This package implements the CTAS Periodic Table of Nodes classification system,
enabling the structured organization and traversal of intelligence entities.
"""

from core.periodic_table.table import PeriodicTable, ElementSymbol
from core.periodic_table.element import Element, ElementProperty
from core.periodic_table.group import Group, Period, Category
from core.periodic_table.registry import PeriodicTableRegistry
from core.periodic_table.relationships import Relationship, RelationshipType

__all__ = [
    "PeriodicTable",
    "ElementSymbol",
    "Element",
    "ElementProperty",
    "Group",
    "Period",
    "Category",
    "PeriodicTableRegistry",
    "Relationship",
    "RelationshipType",
]
