"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-DRONE-AIRSPACE-MONITOR-0001         │
// │ 📁 domain       : Drone, Airspace, Monitoring               │
// │ 🧠 description  : Airspace monitoring component for         │
// │                  tracking nearby aircraft and alerts         │
// │ 🕸️ hash_type    : UUID → CUID-linked module                 │
// │ 🔄 parent_node  : NODE_DRONE_OPERATIONS                     │
// │ 🧩 dependencies : streamlit, folium, datetime, random       │
// │ 🔧 tool_usage   : Monitoring, Visualization                 │
// │ 📡 input_type   : Airspace data                            │
// │ 🧪 test_status  : stable                                    │
// │ 🧠 cognitive_fn : monitoring, visualization                  │
// │ ⌛ TTL Policy   : 6.5 Persistent                            │
// └─────────────────────────────────────────────────────────────┘

Airspace Monitoring Component
---------------------------
This component provides a simulated interface for monitoring nearby
aircraft and airspace restrictions. It visualizes aircraft positions
on a map and displays relevant alerts and NOTAMs.
"""

import streamlit as st
import folium
from streamlit_folium import st_folium
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any

from pages.drone_operations.utils.map_utils import get_tile_with_attribution

def render_airspace_monitoring():
    """
    Render the airspace monitoring panel
    
    # Function monitors subject airspace
    # Method tracks predicate aircraft
    # Interface assesses object conflicts
    """
    st.header("Airspace Monitoring")
    
    # Create columns for the display
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Airspace map with nearby aircraft
        st.subheader("Airspace Map")
        
        # Create a map centered on the drone squadron (if any)
        if len(st.session_state.drone_ids) > 0:
            # Calculate center of drone positions
            center_lat = 0.0
            center_lon = 0.0
            for drone_id in st.session_state.drone_ids:
                if drone_id in st.session_state.drone_simulator.drones:
                    drone = st.session_state.drone_simulator.drones[drone_id]
                    center_lat += drone["position"]["latitude"]
                    center_lon += drone["position"]["longitude"]
            
            if len(st.session_state.drone_ids) > 0:
                center_lat /= len(st.session_state.drone_ids)
                center_lon /= len(st.session_state.drone_ids)
        else:
            # Default center if no drones
            center_lat = 39.8283
            center_lon = -98.5795
        
        # Create a map
        tile, attr = get_tile_with_attribution("CartoDB positron")
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=8,
            tiles=tile,
            attr=attr
        )
        
        # Add our drones to the map
        for drone_id in st.session_state.drone_ids:
            if drone_id in st.session_state.drone_simulator.drones:
                drone = st.session_state.drone_simulator.drones[drone_id]
                folium.Marker(
                    [drone["position"]["latitude"], drone["position"]["longitude"]],
                    tooltip=f"{drone['name']} ({drone['type']})<br>Alt: {drone['position']['altitude']}m",
                    icon=folium.Icon(color="blue", icon="plane", prefix="fa")
                ).add_to(m)
        
        # Add simulated nearby aircraft
        # In a real implementation, this would use actual ADS-B data
        random.seed(42)  # Fixed seed for consistent demo
        for i in range(5):
            aircraft_lat = center_lat + random.uniform(-0.5, 0.5)
            aircraft_lon = center_lon + random.uniform(-0.5, 0.5)
            aircraft_alt = random.uniform(5000, 35000)
            aircraft_type = random.choice(["B737", "A320", "E175", "CRJ9", "B777"])
            aircraft_id = f"ADS-B{random.randint(10000, 99999)}"
            
            folium.Marker(
                [aircraft_lat, aircraft_lon],
                tooltip=f"Flight: {aircraft_id}<br>Type: {aircraft_type}<br>Alt: {aircraft_alt:.0f}ft",
                icon=folium.Icon(color="green", icon="plane", prefix="fa")
            ).add_to(m)
        
        # Display the map
        st_folium(m, width=800, height=500)
    
    with col2:
        # Airspace alerts and notifications
        st.subheader("Airspace Alerts")
        
        # Simulated alerts
        alert_types = ["Proximity Warning", "Restricted Airspace", "Weather Advisory"]
        for i in range(3):
            alert_type = random.choice(alert_types)
            alert_time = (datetime.now() - timedelta(minutes=random.randint(0, 60))).strftime("%H:%M:%S")
            st.warning(f"[{alert_time}] {alert_type}")
        
        # Display NOTAMs (Notice to Airmen)
        st.subheader("NOTAMs")
        notams = [
            "Temporary flight restrictions in effect within 5nm radius",
            "Military operations in progress - coordinate with ATC",
            "Uncontrolled airspace beyond 15nm - exercise caution"
        ]
        for notam in notams:
            st.info(notam)

# CTAS Module Line Count:
# USIM Header: 17 lines
# Comments: 13 lines
# Code: 89 lines
# Total: 119 lines