"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-NEO4J-PACKAGE-0001                  │
// │ 📁 domain       : Database, Neo4j                           │
// │ 🧠 description  : Neo4j connector package                   │
// │ 🕸️ hash_type    : UUID → CUID-linked module                 │
// │ 🔄 parent_node  : NODE_INTEGRATION                         │
// │ 🧩 dependencies : neo4j                                    │
// │ 🔧 tool_usage   : Package                                  │
// │ 📡 input_type   : N/A                                      │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : package management                       │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

Neo4j Integration Package
------------------------
Package for Neo4j graph database integration.
"""

from core.database.neo4j.connector import Neo4jConnector

__all__ = ["Neo4jConnector"]
