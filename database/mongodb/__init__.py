"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-MONGODB-PACKAGE-0001                │
// │ 📁 domain       : Database, MongoDB                         │
// │ 🧠 description  : MongoDB connector package                 │
// │ 🕸️ hash_type    : UUID → CUID-linked module                 │
// │ 🔄 parent_node  : NODE_INTEGRATION                         │
// │ 🧩 dependencies : pymongo                                  │
// │ 🔧 tool_usage   : Package                                  │
// │ 📡 input_type   : N/A                                      │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : package management                       │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

MongoDB Integration Package
-------------------------
Package for MongoDB document database integration.
"""

from core.database.mongodb.connector import MongoDBConnector

__all__ = ["MongoDBConnector"]
