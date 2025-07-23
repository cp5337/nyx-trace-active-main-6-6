"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-DRONE-TELEMETRY-FEED-0001           │
// │ 📁 domain       : Drone, Telemetry, Monitoring              │
// │ 🧠 description  : Telemetry feed component with real-time    │
// │                  drone data visualization and history        │
// │ 🕸️ hash_type    : UUID → CUID-linked module                 │
// │ 🔄 parent_node  : NODE_DRONE_OPERATIONS                     │
// │ 🧩 dependencies : streamlit, pandas                         │
// │ 🔧 tool_usage   : Visualization, Monitoring                 │
// │ 📡 input_type   : Drone telemetry data                      │
// │ 🧪 test_status  : stable                                    │
// │ 🧠 cognitive_fn : monitoring, visualization                  │
// │ ⌛ TTL Policy   : 6.5 Persistent                            │
// └─────────────────────────────────────────────────────────────┘

Telemetry Feed Component
----------------------
This component provides a real-time visualization of drone telemetry data,
showing position, status, sensor readings, and communications metrics.
It supports historical data viewing and raw telemetry inspection.
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Any, Optional

from core.drone.simulation import DroneSimulator

def render_telemetry_feed():
    """
    Render the telemetry feed panel with live data streams
    
    # Function displays subject telemetry
    # Method visualizes predicate metrics
    # Interface presents object values
    """
    st.header("Telemetry Feed")
    
    # Get fresh telemetry data directly from the simulator
    telemetry_data = []
    telemetry_history = {}
    
    if len(st.session_state.drone_ids) > 0:
        # Create drone selector with telemetry data
        if len(st.session_state.drone_ids) > 0:
            # Create drone selector
            selected_drone_idx = st.selectbox(
                "Select Drone",
                range(len(st.session_state.drone_ids)),
                format_func=lambda i: f"Drone {i+1}: {st.session_state.drone_simulator.drones[st.session_state.drone_ids[i]]['name']}"
            )
            
            selected_drone_id = st.session_state.drone_ids[selected_drone_idx]
            
            # Get simulator drone data
            if selected_drone_id in st.session_state.drone_simulator.drones:
                drone = st.session_state.drone_simulator.drones[selected_drone_id]
                
                # Find matching telemetry data if available
                matching_telemetry = [t for t in telemetry_data if t.get('drone_id') == selected_drone_id]
                latest_telemetry = matching_telemetry[0] if matching_telemetry else None
                
                # Display basic drone info
                st.subheader(f"Drone: {drone['name']} ({drone['type']})")
                
                # Create columns for telemetry display
                col1, col2 = st.columns(2)
                
                with col1:
                    # Position telemetry
                    st.subheader("Position Telemetry")
                    
                    # If we have telemetry, use it, otherwise use simulator data directly
                    if latest_telemetry:
                        position_data = {
                            "Latitude": f"{latest_telemetry['lat']:.6f}°",
                            "Longitude": f"{latest_telemetry['lon']:.6f}°",
                            "Altitude": f"{latest_telemetry['alt']:.1f}m",
                            "Heading": f"{latest_telemetry['heading']:.1f}°",
                            "Speed": f"{latest_telemetry['speed_mps']:.1f}m/s",
                            "Last Update": latest_telemetry.get('timestamp', 'Unknown')
                        }
                    else:
                        position_data = {
                            "Latitude": f"{drone['position']['latitude']:.6f}°",
                            "Longitude": f"{drone['position']['longitude']:.6f}°",
                            "Altitude": f"{drone['position']['altitude']:.1f}m",
                            "Heading": f"{drone['position']['heading']:.1f}°",
                            "Speed": f"{drone['position']['speed']:.1f}m/s",
                            "Last Update": "Simulator data (no telemetry)"
                        }
                    st.json(position_data)
                    
                    # Status telemetry
                    st.subheader("Status Telemetry")
                    
                    # If we have telemetry, use it, otherwise use simulator data directly
                    if latest_telemetry and 'battery' in latest_telemetry and 'signal' in latest_telemetry:
                        status_data = {
                            "State": latest_telemetry.get('status', 'Unknown'),
                            "Battery": f"{latest_telemetry['battery']:.1f}%",
                            "Signal": f"{latest_telemetry['signal']:.1f}%"
                        }
                    else:
                        status_data = {
                            "State": drone['status']['state'],
                            "Battery": f"{drone['status']['battery']:.1f}%",
                            "Health": f"{drone['status']['health']:.1f}%",
                            "Signal": f"{drone['status']['signal_strength']:.1f}%",
                            "Mission Progress": f"{drone['status']['mission_progress']:.1f}%"
                        }
                    st.json(status_data)
                
                with col2:
                    # Sensor telemetry
                    st.subheader("Sensor Telemetry")
                    sensor_data = {}
                    for sensor in drone['sensors']:
                        if sensor in DroneSimulator.SENSOR_TYPES:
                            sensor_info = DroneSimulator.SENSOR_TYPES[sensor]
                            sensor_data[sensor_info['name']] = {
                                "Status": "Active",
                                "Data Rate": f"{sensor_info['data_rate']} Mbps"
                            }
                    st.json(sensor_data)
                    
                    # Communications telemetry
                    st.subheader("Communications Telemetry")
                    comms_data = {
                        "Primary": drone['comms']['primary'],
                        "Backup": drone['comms']['backup'],
                        "Encryption": drone['comms']['encryption'],
                        "Bandwidth": f"{drone['comms']['bandwidth']} Mbps",
                        "Range": f"{drone['comms']['range']} km",
                        "Latency": f"{drone['comms']['latency']} ms"
                    }
                    st.json(comms_data)
                
                # Show telemetry history if available
                if selected_drone_id in telemetry_history and telemetry_history[selected_drone_id]:
                    st.subheader("Telemetry History")
                    
                    # Display as a table
                    history_data = telemetry_history[selected_drone_id]
                    if history_data:
                        # Take the most recent 10 entries
                        recent_history = history_data[-10:]
                        
                        # Convert to DataFrame
                        df = pd.DataFrame(recent_history)
                        
                        # Select columns to display
                        cols_to_display = ['timestamp', 'lat', 'lon', 'alt', 'heading', 'speed_mps', 'battery', 'signal']
                        cols_to_display = [c for c in cols_to_display if c in df.columns]
                        
                        # Display the dataframe
                        if cols_to_display:
                            st.dataframe(df[cols_to_display], hide_index=True)
                
                # Full telemetry
                with st.expander("Raw Telemetry Data"):
                    if latest_telemetry:
                        st.json(latest_telemetry)
                    else:
                        st.json(drone['telemetry'])
            else:
                st.warning("Selected drone not found")
    else:
        st.info("Create a squadron to see telemetry data")

# CTAS Module Line Count:
# USIM Header: 17 lines
# Comments: 14 lines
# Code: 142 lines
# Total: 173 lines