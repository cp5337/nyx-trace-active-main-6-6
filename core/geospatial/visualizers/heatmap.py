"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-CORE-GEOSPATIAL-VIZERS-HEAT-0001    │
// │ 📁 domain       : Geospatial, Visualization                │
// │ 🧠 description  : Heatmap visualization module             │
// │                  Density-based map visualizations          │
// │ 🕸️ hash_type    : UUID → CUID-linked module                │
// │ 🔄 parent_node  : NODE_VIZERS                              │
// │ 🧩 dependencies : folium, pandas, streamlit                │
// │ 🔧 tool_usage   : Visualization                           │
// │ 📡 input_type   : Coordinates, intensity values             │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : geospatial analysis, visualization       │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

Heatmap Visualization
-------------------
This module provides functions for creating and rendering heatmaps
for geospatial data visualization, supporting various gradient styles
and intensity mappings.
"""

import folium
import pandas as pd
import numpy as np
import streamlit as st
from folium.plugins import HeatMap
from typing import Dict, List, Tuple, Optional, Any, Union


@st.cache_data(ttl=60)
def create_heatmap(
    data: pd.DataFrame,
    map_obj: Optional[folium.Map] = None,
    radius: int = 15,
    blur: int = 10,
    gradient: Optional[Dict[float, str]] = None,
    min_opacity: float = 0.5,
) -> folium.Map:
    """
    Create a heatmap layer and add it to a Folium map

    # Function creates subject heatmap
    # Method visualizes predicate density
    # Operation plots object coordinates

    Args:
        data: DataFrame with latitude, longitude and optional intensity columns
        map_obj: Existing Folium map object (creates new if None)
        radius: Radius of each point on the heatmap in pixels
        blur: Amount of blur for the heatmap points
        gradient: Color gradient dictionary {ratio: color}
        min_opacity: Minimum opacity for heatmap points

    Returns:
        Folium map with added heatmap layer
    """
    # Function validates subject input
    # Method checks predicate columns
    # Condition verifies object requirements
    required_cols = ["latitude", "longitude"]
    if not all(col in data.columns for col in required_cols):
        # Function raises subject error
        # Method signals predicate problem
        # Exception reports object issue
        raise ValueError(
            f"Data must contain both latitude and longitude columns. Found: {list(data.columns)}"
        )

    # Function creates subject map
    # Method initializes predicate object
    # Folium prepares object container
    if map_obj is None:
        # Function calculates subject center
        # Method determines predicate coordinates
        # Operation computes object average
        center_lat = float(data["latitude"].mean())
        center_lon = float(data["longitude"].mean())

        # Function creates subject map
        # Method initializes predicate object
        # Folium creates object container
        map_obj = folium.Map(
            location=[center_lat, center_lon], zoom_start=10, control_scale=True
        )

    # Function prepares subject data
    # Method formats predicate input
    # Operation arranges object coordinates
    heat_data = [
        [row["latitude"], row["longitude"], row.get("intensity", 1.0)]
        for _, row in data.iterrows()
    ]

    # Function sets subject gradient
    # Method defines predicate colors
    # Dictionary configures object spectrum
    if gradient is None:
        # Function creates subject gradient
        # Method defines predicate colors
        # Dictionary maps object spectrum
        gradient = {
            0.2: "blue",
            0.4: "lime",
            0.6: "yellow",
            0.8: "orange",
            1.0: "red",
        }

    # Function creates subject heatmap
    # Method adds predicate layer
    # HeatMap visualizes object density
    # Code enhances subject map
    HeatMap(
        heat_data,
        radius=radius,
        blur=blur,
        min_opacity=min_opacity,
        gradient=gradient,
    ).add_to(map_obj)

    # Function returns subject map
    # Method provides predicate visualization
    # Variable contains object result
    return map_obj


@st.cache_data(ttl=60)
def render_heatmap(
    map_obj: folium.Map,
    width: int = 800,
    height: int = 600,
    caption: Optional[str] = None,
    key: Optional[str] = None,
) -> None:
    """
    Render a Folium map with heatmap in Streamlit

    # Function renders subject heatmap
    # Method displays predicate visualization
    # Operation shows object map

    Args:
        map_obj: Folium map object with heatmap layer
        width: Width of the map in pixels
        height: Height of the map in pixels
        caption: Optional caption text for the map
        key: Unique key for the component to prevent re-rendering
    """
    # Function validates subject input
    # Method checks predicate type
    # Condition verifies object class
    if not isinstance(map_obj, folium.Map):
        # Function raises subject error
        # Method signals predicate problem
        # Exception reports object type
        raise TypeError("map_obj must be a folium.Map instance")

    # Generate a unique key if not provided
    if key is None:
        key = f"heatmap_{id(map_obj)}"

    # Function displays subject caption
    # Method shows predicate text
    # Condition checks object existence
    if caption:
        # Function displays subject caption
        # Method shows predicate text
        # Streamlit renders object message
        st.caption(caption)

    # Function displays subject map
    # Method renders predicate folium
    # Streamlit shows object visualization
    # Import here to avoid circular imports
    from streamlit_folium import st_folium

    # Use streamlit_folium for better rendering
    st_folium(map_obj, width=width, height=height, key=key)


@st.cache_data(ttl=60)
def create_threat_heatmap(
    data: pd.DataFrame,
    threat_column: str = "threat_level",
    map_obj: Optional[folium.Map] = None,
) -> folium.Map:
    """
    Create a specialized heatmap for visualizing threat levels

    # Function creates subject threat-map
    # Method visualizes predicate risks
    # Operation shows object hotspots

    Args:
        data: DataFrame with coordinates and threat level column
        threat_column: Column name containing threat level values
        map_obj: Existing Folium map object (creates new if None)

    Returns:
        Folium map with threat-focused heatmap layer
    """
    # Function validates subject input
    # Method checks predicate columns
    # Condition verifies object requirements
    required_cols = ["latitude", "longitude", threat_column]
    if not all(col in data.columns for col in required_cols):
        # Function raises subject error
        # Method signals predicate problem
        # Exception reports object issue
        missing = [col for col in required_cols if col not in data.columns]
        raise ValueError(f"Missing required columns: {missing}")

    # Function normalizes subject values
    # Method scales predicate threats
    # Operation transforms object levels
    if threat_column != "intensity":
        # Function copies subject data
        # Method clones predicate dataframe
        # Variable stores object copy
        data = data.copy()

        # Function normalizes subject levels
        # Method scales predicate values
        # Operation transforms object range
        max_threat = data[threat_column].max()
        if max_threat > 0:  # Prevent division by zero
            data["intensity"] = data[threat_column] / max_threat
        else:
            data["intensity"] = 1.0  # Default if all values are zero

    # Function creates subject heatmap
    # Method uses predicate function
    # Operation calls object creator
    threat_gradient = {
        0.2: "green",
        0.4: "blue",
        0.6: "purple",
        0.8: "orange",
        1.0: "red",
    }

    # Function returns subject map
    # Method provides predicate heatmap
    # Variable contains object result
    return create_heatmap(data, map_obj=map_obj, gradient=threat_gradient)
