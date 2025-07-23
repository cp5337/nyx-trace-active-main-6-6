"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-DRONE-OPERATIONS-0001                │
// │ 📁 domain       : Drone, Operations, Monitoring              │
// │ 🧠 description  : Drone operations monitor dashboard with    │
// │                  live telemetry visualization and control    │
// │ 🕸️ hash_type    : UUID → CUID-linked module                 │
// │ 🔄 parent_node  : NODE_DRONE_OPERATIONS                     │
// │ 🧩 dependencies : streamlit, folium, json, pandas, math      │
// │ 🔧 tool_usage   : Visualization, Monitoring                 │
// │ 📡 input_type   : Telemetry data, simulation parameters      │
// │ 🧪 test_status  : stable                                    │
// │ 🧠 cognitive_fn : monitoring, visualization                  │
// │ ⌛ TTL Policy   : 6.5 Persistent                            │
// └─────────────────────────────────────────────────────────────┘

CTAS Drone Operations Monitor
----------------------------
This module provides a complete operational dashboard for drone squadron
monitoring with live telemetry visualization, mission planning, and
status monitoring capabilities.
"""

import streamlit as st
from datetime import datetime

# Import refactored components
from pages.drone_operations.dashboard import render_drone_operations_dashboard

# Main entry point for the page
if __name__ == "__main__":
    render_drone_operations_dashboard()

# CTAS Module Line Count:
# USIM Header: 17 lines
# Comments: 7 lines
# Code: 9 lines
# Total: 33 lines