"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-DATABASE-INTEGRATION-0001           │
// │ 📁 domain       : Database, Integration                     │
// │ 🧠 description  : Database integration package for NyxTrace │
// │                  multi-database storage and analytics       │
// │ 🕸️ hash_type    : UUID → CUID-linked module                 │
// │ 🔄 parent_node  : NODE_INTEGRATION                         │
// │ 🧩 dependencies : supabase, neo4j, mongodb                 │
// │ 🔧 tool_usage   : Data Storage, Integration                │
// │ 📡 input_type   : Application data                         │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : data persistence, retrieval              │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

Database Integration Package
--------------------------
Core database integration package for NyxTrace multi-database
architecture, providing interfaces to Supabase, Neo4j, and MongoDB.
"""

from core.database.config import DatabaseConfig
from core.database.factory import DatabaseFactory

__all__ = ["DatabaseConfig", "DatabaseFactory"]
