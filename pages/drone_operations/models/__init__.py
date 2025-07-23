"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-DRONE-MODELS-INIT-0001              │
// │ 📁 domain       : Drone, Package, Initialization            │
// │ 🧠 description  : Package initialization for drone           │
// │                  operations models                           │
// │ 🕸️ hash_type    : UUID → CUID-linked module                 │
// │ 🔄 parent_node  : NODE_DRONE_OPERATIONS                     │
// │ 🧩 dependencies : N/A                                       │
// │ 🔧 tool_usage   : Initialization                            │
// │ 📡 input_type   : N/A                                       │
// │ 🧪 test_status  : stable                                    │
// │ 🧠 cognitive_fn : package organization                       │
// │ ⌛ TTL Policy   : 6.5 Persistent                            │
// └─────────────────────────────────────────────────────────────┘

Drone Operations Models Package
-----------------------------
This package contains data models and type definitions for drone operations.
Models are designed to be immutable and follow Rust-like principles for
data integrity and type safety.
"""

# Export model classes and types
from pages.drone_operations.models.telemetry import DroneLocation, TelemetryReading, ResourceStatus
from pages.drone_operations.models.mission import MissionParameters, SearchGridPattern, WaypointRoute

# CTAS Module Line Count:
# USIM Header: 17 lines
# Comments: 6 lines
# Code: 3 lines
# Total: 26 lines