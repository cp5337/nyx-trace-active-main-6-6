"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-DRONE-OPS-INIT-0001                 │
// │ 📁 domain       : Drone, Package, Initialization            │
// │ 🧠 description  : Package initialization for drone           │
// │                  operations component                        │
// │ 🕸️ hash_type    : UUID → CUID-linked module                 │
// │ 🔄 parent_node  : NODE_DRONE_OPERATIONS                     │
// │ 🧩 dependencies : N/A                                       │
// │ 🔧 tool_usage   : Initialization                            │
// │ 📡 input_type   : N/A                                       │
// │ 🧪 test_status  : stable                                    │
// │ 🧠 cognitive_fn : package organization                       │
// │ ⌛ TTL Policy   : 6.5 Persistent                            │
// └─────────────────────────────────────────────────────────────┘

Drone Operations Package
-----------------------
This package contains modular components for drone operations monitoring,
telemetry visualization, and mission planning. The components are designed
to be reusable for various applications including geospatial intelligence
and digital crop assay.
"""

# Import all component modules for easy access
from pages.drone_operations.components.squadron_monitor import render_squadron_monitor
from pages.drone_operations.components.telemetry_feed import render_telemetry_feed
from pages.drone_operations.components.mission_planning import render_mission_planning
from pages.drone_operations.components.traffic_camera import render_traffic_camera_feed
from pages.drone_operations.components.airspace_monitor import render_airspace_monitoring
from pages.drone_operations.utils.map_utils import get_tile_with_attribution

# Expose the main dashboard render function
from pages.drone_operations.dashboard import render_drone_operations_dashboard

# CTAS Module Line Count:
# USIM Header: 17 lines
# Comments: 7 lines
# Code: 9 lines
# Total: 33 lines