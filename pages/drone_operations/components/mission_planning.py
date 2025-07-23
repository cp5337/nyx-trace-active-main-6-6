"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-DRONE-MISSION-PLANNING-0001         │
// │ 📁 domain       : Drone, Mission, Planning                  │
// │ 🧠 description  : Mission planning component for drone       │
// │                  operations dashboard                        │
// │ 🕸️ hash_type    : UUID → CUID-linked module                 │
// │ 🔄 parent_node  : NODE_DRONE_OPERATIONS                     │
// │ 🧩 dependencies : streamlit, folium, datetime, math         │
// │ 🔧 tool_usage   : UI, Planning                             │
// │ 📡 input_type   : User input, drone data                    │
// │ 🧪 test_status  : stable                                    │
// │ 🧠 cognitive_fn : planning, visualization                    │
// │ ⌛ TTL Policy   : 6.5 Persistent                            │
// └─────────────────────────────────────────────────────────────┘

Mission Planning Component
------------------------
This component provides a comprehensive interface for planning drone
missions with different mission types and parameters. It visualizes
mission paths on a map and handles mission assignment to drones.
"""

import streamlit as st
import folium
from streamlit_folium import st_folium
import math
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

from core.drone.simulation import DroneSimulator
from pages.drone_operations.utils.map_utils import get_tile_with_attribution
from pages.drone_operations.utils.mission_patterns import (
    add_search_grid_to_map,
    add_direct_attack_to_map,
    add_orbit_to_map,
    add_reconnaissance_to_map
)

def render_mission_config_panel() -> Dict[str, Any]:
    """
    Render the mission configuration panel and collect parameters
    
    # Function renders subject panel
    # Method collects predicate parameters
    # UI gathers object inputs
    
    Returns:
        Dictionary of mission configuration parameters
    """
    st.subheader("Mission Configuration")
    
    # Mission type selector
    mission_type = st.selectbox(
        "Mission Type",
        ["Search Grid", "Direct Attack", "Surveillance", "Reconnaissance"],
        index=0
    )
    
    # Target selection
    st.subheader("Target Location")
    target_lat = st.number_input("Target Latitude", value=39.7, format="%.4f", key="target_lat")
    target_lon = st.number_input("Target Longitude", value=-98.2, format="%.4f", key="target_lon")
    
    # Mission-specific parameters based on mission type
    mission_params = {}
    
    if mission_type == "Search Grid":
        st.subheader("Search Grid Parameters")
        grid_size = st.slider("Grid Size (km)", 1.0, 10.0, 3.0, 0.5)
        grid_spacing = st.slider("Grid Spacing (km)", 0.2, 2.0, 0.5, 0.1)
        altitude = st.number_input("Mission Altitude (m)", 100, 500, 200, 50)
        
        mission_params = {
            "grid_size": grid_size, 
            "grid_spacing": grid_spacing,
            "altitude": altitude
        }
        
    elif mission_type == "Direct Attack":
        st.subheader("Attack Parameters")
        approach_vector = st.slider("Approach Vector (degrees)", 0, 359, 45, 5)
        attack_speed = st.slider("Attack Speed (%)", 50, 100, 80, 5)
        altitude = st.number_input("Mission Altitude (m)", 50, 300, 100, 25)
        
        mission_params = {
            "approach_vector": approach_vector,
            "attack_speed": attack_speed,
            "altitude": altitude
        }
        
    elif mission_type == "Surveillance":
        st.subheader("Surveillance Parameters")
        orbit_radius = st.slider("Orbit Radius (km)", 0.5, 5.0, 1.0, 0.1)
        orbit_time = st.slider("Orbit Time (min)", 5, 60, 15, 5)
        altitude = st.number_input("Mission Altitude (m)", 100, 1000, 300, 50)
        
        mission_params = {
            "orbit_radius": orbit_radius,
            "orbit_time": orbit_time,
            "altitude": altitude
        }
        
    else:  # Reconnaissance
        st.subheader("Reconnaissance Parameters")
        path_type = st.selectbox("Path Type", ["Linear", "S-Pattern", "Circular"])
        path_length = st.slider("Path Length (km)", 1.0, 20.0, 5.0, 0.5)
        altitude = st.number_input("Mission Altitude (m)", 200, 1500, 500, 100)
        
        mission_params = {
            "path_type": path_type,
            "path_length": path_length,
            "altitude": altitude
        }
    
    return {
        "mission_type": mission_type,
        "target_position": (target_lat, target_lon),
        "mission_params": mission_params
    }

def render_drone_selection_panel() -> List[str]:
    """
    Render the drone selection panel and return selected drone IDs
    
    # Function renders subject panel
    # Method selects predicate drones
    # UI chooses object resources
    
    Returns:
        List of selected drone IDs
    """
    st.subheader("Drone Selection")
    
    # Get available drones
    available_drones = []
    for idx, drone_id in enumerate(st.session_state.drone_ids):
        drone = st.session_state.drone_simulator.drones.get(drone_id, {})
        if drone.get("status", {}).get("state", "") == "standby":
            available_drones.append((idx, drone_id, drone.get("name", "")))
    
    if not available_drones:
        st.warning("No available drones in standby mode")
        return []
    
    options = [f"Drone {idx+1}: {name}" for idx, _, name in available_drones]
    default = [True] * min(3, len(available_drones))
    selected_options = st.multiselect(
        "Select Drones",
        options=options,
        default=options[:min(3, len(available_drones))]
    )
    
    # Map selected options back to drone IDs
    selected_indices = [options.index(opt) for opt in selected_options]
    selected_drones = [available_drones[idx][1] for idx in selected_indices] if selected_indices else []
    
    return selected_drones

def assign_mission_to_drones(
    selected_drones: List[str], 
    mission_type: str, 
    target_position: Tuple[float, float], 
    mission_params: Dict[str, Any]
) -> bool:
    """
    Assign the configured mission to selected drones
    
    # Function assigns subject mission
    # Method configures predicate drones
    # Operation executes object assignment
    
    Args:
        selected_drones: List of drone IDs to assign the mission to
        mission_type: Type of mission to assign
        target_position: Target position (lat, lon)
        mission_params: Mission-specific parameters
        
    Returns:
        True if mission was assigned successfully, False otherwise
    """
    if not selected_drones:
        st.error("No drones selected")
        return False
    
    base_position = None
    
    # Get base position from first drone
    if selected_drones:
        first_drone = st.session_state.drone_simulator.drones.get(selected_drones[0], {})
        if first_drone:
            base_position = (
                first_drone.get("position", {}).get("latitude", 0),
                first_drone.get("position", {}).get("longitude", 0)
            )
    
    if not base_position:
        st.error("Could not determine base position")
        return False
    
    # Assign missions to selected drones
    success = True
    for idx, drone_id in enumerate(selected_drones):
        # Map our mission types to simulator mission types
        sim_mission_type = "search"  # Default
        if mission_type == "Direct Attack":
            sim_mission_type = "direct"
        elif mission_type == "Surveillance":
            sim_mission_type = "orbit"
        elif mission_type == "Reconnaissance":
            sim_mission_type = "recon"
        
        mission_success = st.session_state.drone_simulator.assign_mission(
            drone_id=drone_id,
            mission_type=sim_mission_type,
            start_position=base_position,
            target_position=target_position,
            mission_params=mission_params
        )
        
        if mission_success:
            st.success(f"Mission assigned to {st.session_state.drone_simulator.drones[drone_id]['name']}")
        else:
            st.error(f"Failed to assign mission to {st.session_state.drone_simulator.drones[drone_id]['name']}")
            success = False
    
    # Start simulation if it's not already running
    if success and not st.session_state.simulator_running:
        st.session_state.simulator_running = True
        st.info("Simulation started automatically")
    
    return success

def render_mission_map(
    mission_type: str, 
    target_position: Tuple[float, float], 
    mission_params: Dict[str, Any]
) -> None:
    """
    Render the mission planning map with visualizations
    
    # Function renders subject map
    # Method visualizes predicate mission
    # UI displays object path
    
    Args:
        mission_type: Type of mission to visualize
        target_position: Target position (lat, lon)
        mission_params: Mission-specific parameters
    """
    target_lat, target_lon = target_position
    
    st.subheader("Mission Planning Map")
    
    # Find center point between base and target
    drones_data = []
    for drone_id in st.session_state.drone_ids:
        if drone_id in st.session_state.drone_simulator.drones:
            drone = st.session_state.drone_simulator.drones[drone_id]
            drones_data.append({
                "lat": drone["position"]["latitude"],
                "lon": drone["position"]["longitude"]
            })
    
    if not drones_data:
        st.warning("No drone data available to display map")
        return
    
    center_lat = sum(d["lat"] for d in drones_data) / len(drones_data)
    center_lon = sum(d["lon"] for d in drones_data) / len(drones_data)
    
    # Adjust center point to include target location
    center_lat = (center_lat + target_lat) / 2
    center_lon = (center_lon + target_lon) / 2
    
    # Create a Folium map
    tile, attr = get_tile_with_attribution("CartoDB positron")
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=7,
        tiles=tile,
        attr=attr
    )
    
    # Add drone markers
    for drone_id in st.session_state.drone_ids:
        if drone_id in st.session_state.drone_simulator.drones:
            drone = st.session_state.drone_simulator.drones[drone_id]
            
            # Choose icon color based on status
            color = "blue"
            if drone["status"]["state"] == "mission":
                color = "green"
            elif drone["status"]["state"] == "returning":
                color = "orange"
            elif drone["status"]["state"] == "emergency":
                color = "red"
            
            # Add drone marker
            folium.Marker(
                [drone["position"]["latitude"], drone["position"]["longitude"]],
                tooltip=f"{drone['name']} ({drone['type']})<br>Status: {drone['status']['state']}",
                icon=folium.Icon(color=color, icon="plane", prefix="fa")
            ).add_to(m)
            
            # Get and display drone path if on a mission
            if drone["status"]["state"] == "mission":
                path = st.session_state.drone_simulator.get_drone_path(drone_id)
                if path and len(path) > 1:
                    path_points = [(p["latitude"], p["longitude"]) for p in path]
                    folium.PolyLine(
                        path_points,
                        color=color,
                        weight=3,
                        opacity=0.7,
                        dash_array='5'
                    ).add_to(m)
    
    # Add target marker
    folium.Marker(
        [target_lat, target_lon],
        tooltip=f"Target: {mission_type}",
        icon=folium.Icon(color="red", icon="crosshairs", prefix="fa")
    ).add_to(m)
    
    # Draw mission-specific visualizations
    if mission_type == "Search Grid" and "grid_size" in mission_params and "grid_spacing" in mission_params:
        add_search_grid_to_map(m, target_lat, target_lon, mission_params["grid_size"], mission_params["grid_spacing"])
            
    elif mission_type == "Direct Attack" and "approach_vector" in mission_params and "attack_speed" in mission_params:
        add_direct_attack_to_map(m, target_lat, target_lon, mission_params["approach_vector"], mission_params["attack_speed"])
            
    elif mission_type == "Surveillance" and "orbit_radius" in mission_params and "orbit_time" in mission_params:
        add_orbit_to_map(m, target_lat, target_lon, mission_params["orbit_radius"], mission_params["orbit_time"])
    
    elif mission_type == "Reconnaissance" and "path_type" in mission_params and "path_length" in mission_params:
        add_reconnaissance_to_map(m, target_lat, target_lon, mission_params["path_type"], mission_params["path_length"])
    
    # Display the map
    st_folium(m, width=800, height=600)

def render_mission_planning() -> None:
    """
    Render the mission planning panel for drone operations
    
    # Function plans subject missions
    # Method configures predicate routes
    # Interface schedules object tasks
    """
    st.header("Mission Planning")
    
    if len(st.session_state.drone_ids) == 0:
        st.warning("Please create a squadron first in the Squadron Monitor tab")
        return
    
    # Create columns for the mission planning interface
    col1, col2 = st.columns([2, 1])
    
    with col2:
        # Render mission configuration panel
        mission_config = render_mission_config_panel()
        
        # Render drone selection panel
        selected_drones = render_drone_selection_panel()
        
        # Mission execution button
        if st.button("Execute Mission") and selected_drones:
            success = assign_mission_to_drones(
                selected_drones=selected_drones,
                mission_type=mission_config["mission_type"],
                target_position=mission_config["target_position"],
                mission_params=mission_config["mission_params"]
            )
            
            if success:
                # Force an update to refresh the map
                st.rerun()
    
    with col1:
        # Render mission map
        render_mission_map(
            mission_type=mission_config["mission_type"],
            target_position=mission_config["target_position"],
            mission_params=mission_config["mission_params"]
        )

# CTAS Module Line Count:
# USIM Header: 17 lines
# Comments: 47 lines
# Code: 293 lines
# Total: 357 lines