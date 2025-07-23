"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-SUPABASE-PACKAGE-0001               │
// │ 📁 domain       : Database, Supabase                        │
// │ 🧠 description  : Supabase connector package                │
// │ 🕸️ hash_type    : UUID → CUID-linked module                 │
// │ 🔄 parent_node  : NODE_INTEGRATION                         │
// │ 🧩 dependencies : sqlalchemy, psycopg2, supabase           │
// │ 🔧 tool_usage   : Package                                  │
// │ 📡 input_type   : N/A                                      │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : package management                       │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

Supabase Integration Package
--------------------------
Package for Supabase PostgreSQL database integration.
"""

from core.database.supabase.connector import SupabaseConnector

__all__ = ["SupabaseConnector"]
