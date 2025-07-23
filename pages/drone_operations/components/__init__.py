"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-DRONE-COMPONENTS-INIT-0001          │
// │ 📁 domain       : Drone, Package, Initialization            │
// │ 🧠 description  : Package initialization for drone           │
// │                  operations components                       │
// │ 🕸️ hash_type    : UUID → CUID-linked module                 │
// │ 🔄 parent_node  : NODE_DRONE_OPERATIONS                     │
// │ 🧩 dependencies : N/A                                       │
// │ 🔧 tool_usage   : Initialization                            │
// │ 📡 input_type   : N/A                                       │
// │ 🧪 test_status  : stable                                    │
// │ 🧠 cognitive_fn : package organization                       │
// │ ⌛ TTL Policy   : 6.5 Persistent                            │
// └─────────────────────────────────────────────────────────────┘

Drone Operations Components Package
---------------------------------
This package contains UI components for drone operations monitoring and control.
Each component is designed to be modular and reusable across different applications.
"""

# Export component rendering functions
from pages.drone_operations.components.squadron_monitor import render_squadron_monitor
from pages.drone_operations.components.telemetry_feed import render_telemetry_feed
from pages.drone_operations.components.mission_planning import render_mission_planning
from pages.drone_operations.components.traffic_camera import render_traffic_camera_feed
from pages.drone_operations.components.airspace_monitor import render_airspace_monitoring

# CTAS Module Line Count:
# USIM Header: 17 lines
# Comments: 6 lines
# Code: 6 lines
# Total: 29 lines