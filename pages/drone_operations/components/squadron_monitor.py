"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-DRONE-SQUADRON-MONITOR-0001         │
// │ 📁 domain       : Drone, Squadron, Monitoring               │
// │ 🧠 description  : Squadron monitoring component with        │
// │                  interactive map and status overview        │
// │ 🕸️ hash_type    : UUID → CUID-linked module                 │
// │ 🔄 parent_node  : NODE_DRONE_OPERATIONS                     │
// │ 🧩 dependencies : streamlit, folium, pandas                 │
// │ 🔧 tool_usage   : Visualization, Monitoring                 │
// │ 📡 input_type   : Drone telemetry data                      │
// │ 🧪 test_status  : stable                                    │
// │ 🧠 cognitive_fn : monitoring, visualization                  │
// │ ⌛ TTL Policy   : 6.5 Persistent                            │
// └─────────────────────────────────────────────────────────────┘

Squadron Monitoring Component
---------------------------
This component provides an interactive map visualization and controls for
drone squadron monitoring. It includes squadron creation, simulation controls,
and a status overview table.
"""

import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

from core.drone.simulation import DroneSimulator
from pages.drone_operations.utils.map_utils import get_tile_with_attribution

def render_squadron_monitor():
    """
    Render the squadron monitor panel with drone status and map
    
    # Function displays subject squadron
    # Method visualizes predicate status
    # Interface monitors object locations
    """
    st.header("Squadron Monitor")
    
    # Create columns for controls and summary
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("Squadron Controls")
        
        # Squadron creation controls
        with st.expander("Create Squadron", expanded=False):
            squadron_size = st.slider("Squadron Size", 1, 10, 3)
            drone_type_options = list(DroneSimulator.DRONE_TYPES.keys())
            drone_type = st.selectbox("Drone Type", drone_type_options, index=1)
            
            # Base location for squadron
            st.subheader("Base Location")
            base_lat = st.number_input("Latitude", value=39.8283, format="%.4f")
            base_lon = st.number_input("Longitude", value=-98.5795, format="%.4f")
            base_altitude = st.number_input("Altitude (m)", value=100, step=50)
            
            if st.button("Create Squadron"):
                # Clear existing drones if any
                st.session_state.drone_simulator = DroneSimulator()
                
                # Create a new squadron
                st.session_state.drone_ids = st.session_state.drone_simulator.create_squadron(
                    size=squadron_size,
                    drone_type=drone_type,
                    base_position=(base_lat, base_lon),
                    altitude=base_altitude
                )
                st.success(f"Created squadron with {squadron_size} drones")
        
        # Simulation controls
        with st.expander("Simulation Controls", expanded=True):
            time_acceleration = st.slider("Time Acceleration", 1.0, 10.0, 1.0, step=0.5)
            
            # Start/Stop simulation
            if st.session_state.simulator_running:
                if st.button("Stop Simulation"):
                    st.session_state.simulator_running = False
                    st.info("Simulation stopped")
            else:
                if st.button("Start Simulation"):
                    st.session_state.simulator_running = True
                    st.info("Simulation started")
            
            # If simulation is running, update drone states
            if st.session_state.simulator_running and len(st.session_state.drone_ids) > 0:
                # Update the simulation
                current_time = datetime.now()
                time_delta = (current_time - st.session_state.last_update).total_seconds()
                st.session_state.drone_simulator.update_simulation(
                    time_step=time_delta * time_acceleration
                )
                st.session_state.last_update = current_time

    with col1:
        # Create a map to display drone positions
        if len(st.session_state.drone_ids) > 0:
            try:
                # Get drone data
                drones = []
                for drone_id in st.session_state.drone_ids:
                    if drone_id in st.session_state.drone_simulator.drones:
                        drone = st.session_state.drone_simulator.drones[drone_id]
                        drones.append({
                            "drone_id": drone["name"],
                            "lat": drone["position"]["latitude"],
                            "lon": drone["position"]["longitude"],
                            "alt": drone["position"]["altitude"],
                            "heading": drone["position"]["heading"],
                            "model": drone["type"],
                            "status": drone["status"]["state"]
                        })
                
                # Create a map
                if drones:
                    # Find the center of the drones
                    center_lat = sum(d["lat"] for d in drones) / len(drones)
                    center_lon = sum(d["lon"] for d in drones) / len(drones)
                    
                    # Create a Folium map
                    tile, attr = get_tile_with_attribution("CartoDB positron")
                    m = folium.Map(
                        location=[center_lat, center_lon],
                        zoom_start=6,
                        tiles=tile,
                        attr=attr
                    )
                    
                    # Add drone markers
                    for d in drones:
                        # Choose icon color based on status
                        color = "blue"
                        if d["status"] == "mission":
                            color = "green"
                        elif d["status"] == "returning":
                            color = "orange"
                        elif d["status"] == "emergency":
                            color = "red"
                        
                        # Add drone marker
                        folium.Marker(
                            [d["lat"], d["lon"]],
                            tooltip=f"{d['drone_id']} ({d['model']})<br>Alt: {d['alt']}m<br>Status: {d['status']}",
                            icon=folium.Icon(color=color, icon="plane", prefix="fa")
                        ).add_to(m)
                    
                    # Display the map
                    st_folium(m, width=800, height=500)
                else:
                    st.warning("No drone data available yet")
            except Exception as e:
                st.error(f"Error rendering drone map: {e}")
        else:
            st.info("Create a squadron to see the map")
        
        # Display squadron overview
        if len(st.session_state.drone_ids) > 0:
            st.subheader("Squadron Overview")
            
            # Create a dataframe of drone status
            drone_data = []
            for drone_id in st.session_state.drone_ids:
                if drone_id in st.session_state.drone_simulator.drones:
                    drone = st.session_state.drone_simulator.drones[drone_id]
                    drone_data.append({
                        "ID": drone_id[:8],
                        "Name": drone["name"],
                        "Type": drone["type"],
                        "Status": drone["status"]["state"],
                        "Battery": f"{drone['status']['battery']:.1f}%",
                        "Signal": f"{drone['status']['signal_strength']:.1f}%",
                        "Position": f"{drone['position']['latitude']:.4f}, {drone['position']['longitude']:.4f}",
                        "Altitude": f"{drone['position']['altitude']:.1f}m",
                        "Speed": f"{drone['position']['speed']:.1f}m/s"
                    })
            
            if drone_data:
                df = pd.DataFrame(drone_data)
                st.dataframe(df, hide_index=True)

# CTAS Module Line Count:
# USIM Header: 17 lines
# Comments: 13 lines
# Code: 149 lines
# Total: 179 lines