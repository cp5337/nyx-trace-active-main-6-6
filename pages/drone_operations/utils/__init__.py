"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-DRONE-UTILS-INIT-0001               │
// │ 📁 domain       : Drone, Package, Initialization            │
// │ 🧠 description  : Package initialization for drone           │
// │                  operations utilities                        │
// │ 🕸️ hash_type    : UUID → CUID-linked module                 │
// │ 🔄 parent_node  : NODE_DRONE_OPERATIONS                     │
// │ 🧩 dependencies : N/A                                       │
// │ 🔧 tool_usage   : Initialization                            │
// │ 📡 input_type   : N/A                                       │
// │ 🧪 test_status  : stable                                    │
// │ 🧠 cognitive_fn : package organization                       │
// │ ⌛ TTL Policy   : 6.5 Persistent                            │
// └─────────────────────────────────────────────────────────────┘

Drone Operations Utilities Package
--------------------------------
This package contains utility functions for drone operations components,
including map utilities and mission pattern generators.
"""

# Export utility functions
from pages.drone_operations.utils.map_utils import get_tile_with_attribution, create_drone_icon

# Export mission pattern utilities
from pages.drone_operations.utils.mission_patterns import (
    generate_search_grid_pattern, add_search_grid_to_map,
    generate_direct_attack_vector, add_direct_attack_to_map,
    generate_orbit_pattern, add_orbit_to_map,
    generate_reconnaissance_path, add_reconnaissance_to_map
)

# CTAS Module Line Count:
# USIM Header: 17 lines
# Comments: 6 lines
# Code: 11 lines
# Total: 34 lines