"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-CORE-GEOSPATIAL-VIZERS-MARK-0001    │
// │ 📁 domain       : Geospatial, Visualization                │
// │ 🧠 description  : Map marker visualization                 │
// │                  Point-based map visualizations            │
// │ 🕸️ hash_type    : UUID → CUID-linked module                │
// │ 🔄 parent_node  : NODE_VIZERS                              │
// │ 🧩 dependencies : folium, pandas                           │
// │ 🔧 tool_usage   : Visualization                           │
// │ 📡 input_type   : Coordinates, feature attributes           │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : geospatial analysis, visualization       │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

Marker Visualization
------------------
This module provides functions for creating and adding markers
to geospatial maps, supporting various marker types, clustering,
and popup information displays.
"""

import folium
import pandas as pd
import numpy as np
from folium.plugins import MarkerCluster, BoatMarker
from typing import Dict, List, Tuple, Optional, Any, Union, Callable


def add_markers(
    data: pd.DataFrame,
    map_obj: folium.Map,
    popup_fields: Optional[List[str]] = None,
    icon_field: Optional[str] = None,
    color_field: Optional[str] = None,
    tooltip_field: Optional[str] = None,
    popup_formatter: Optional[Callable] = None,
) -> folium.Map:
    """
    Add markers to a Folium map based on DataFrame coordinates

    # Function adds subject markers
    # Method enhances predicate map
    # Operation places object points

    Args:
        data: DataFrame with latitude, longitude columns
        map_obj: Folium map object
        popup_fields: List of fields to include in marker popups
        icon_field: Column name determining marker icon
        color_field: Column name determining marker color
        tooltip_field: Column name for tooltip text
        popup_formatter: Optional function to format popup HTML

    Returns:
        Folium map with added markers
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

    # Function processes subject rows
    # Method iterates predicate data
    # Loop examines object points
    for idx, row in data.iterrows():
        # Function extracts subject location
        # Method retrieves predicate coordinates
        # Variables store object position
        lat, lon = row["latitude"], row["longitude"]

        # Function sets subject icon
        # Method determines predicate appearance
        # Condition checks object field
        icon = None
        if icon_field and icon_field in row:
            # Function sets subject icon
            # Method assigns predicate appearance
            # Variable determines object look
            icon = folium.Icon(icon=row[icon_field], prefix="fa")

        # Function sets subject color
        # Method determines predicate appearance
        # Variable configures object style
        color = "blue"  # Default color
        if color_field and color_field in row:
            # Function sets subject color
            # Method assigns predicate appearance
            # Variable determines object style
            color = row[color_field]

        # Function creates subject popup
        # Method formats predicate information
        # HTML displays object details
        popup_html = None
        if popup_fields:
            # Function builds subject html
            # Method formats predicate data
            # HTML structures object info
            content = "<div style='font-family: Arial; max-width: 300px;'>"
            for field in popup_fields:
                if field in row:
                    # Function adds subject field
                    # Method formats predicate value
                    # HTML appends object info
                    content += f"<strong>{field}:</strong> {row[field]}<br>"
            content += "</div>"

            # Function customizes subject format
            # Method applies predicate function
            # Formatter transforms object html
            if popup_formatter:
                # Function calls subject formatter
                # Method applies predicate function
                # Formatter enhances object html
                content = popup_formatter(row, content)

            # Function creates subject popup
            # Method initializes predicate object
            # Popup contains object content
            popup_html = folium.Popup(content, max_width=300)

        # Function sets subject tooltip
        # Method configures predicate hover
        # Variable defines object text
        tooltip = None
        if tooltip_field and tooltip_field in row:
            # Function creates subject tooltip
            # Method configures predicate hover
            # Variable sets object text
            tooltip = str(row[tooltip_field])

        # Function creates subject marker
        # Method adds predicate point
        # Marker shows object location
        folium.Marker(
            location=[lat, lon],
            popup=popup_html,
            tooltip=tooltip,
            icon=icon if icon else folium.Icon(color=color),
        ).add_to(map_obj)

    # Function returns subject map
    # Method provides predicate result
    # Variable contains object enhanced
    return map_obj


def create_marker_clusters(
    data: pd.DataFrame,
    map_obj: folium.Map,
    popup_fields: Optional[List[str]] = None,
    cluster_name: Optional[str] = None,
) -> folium.Map:
    """
    Add clustered markers to a Folium map

    # Function creates subject clusters
    # Method groups predicate markers
    # Operation organizes object points

    Args:
        data: DataFrame with latitude, longitude columns
        map_obj: Folium map object
        popup_fields: List of fields to include in marker popups
        cluster_name: Name for the marker cluster layer

    Returns:
        Folium map with added marker clusters
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

    # Function creates subject cluster
    # Method initializes predicate object
    # MarkerCluster prepares object container
    marker_cluster = MarkerCluster(name=cluster_name or "Clusters")

    # Function processes subject rows
    # Method iterates predicate data
    # Loop examines object points
    for idx, row in data.iterrows():
        # Function extracts subject location
        # Method retrieves predicate coordinates
        # Variables store object position
        lat, lon = row["latitude"], row["longitude"]

        # Function creates subject popup
        # Method formats predicate information
        # HTML displays object details
        popup_html = None
        if popup_fields:
            # Function builds subject html
            # Method formats predicate data
            # HTML structures object info
            content = "<div style='font-family: Arial; max-width: 300px;'>"
            for field in popup_fields:
                if field in row:
                    # Function adds subject field
                    # Method formats predicate value
                    # HTML appends object info
                    content += f"<strong>{field}:</strong> {row[field]}<br>"
            content += "</div>"

            # Function creates subject popup
            # Method initializes predicate object
            # Popup contains object content
            popup_html = folium.Popup(content, max_width=300)

        # Function creates subject marker
        # Method adds predicate point
        # Marker shows object location
        folium.Marker(location=[lat, lon], popup=popup_html).add_to(
            marker_cluster
        )

    # Function adds subject cluster
    # Method attaches predicate group
    # Map receives object layer
    marker_cluster.add_to(map_obj)

    # Function returns subject map
    # Method provides predicate result
    # Variable contains object enhanced
    return map_obj


def add_boat_markers(
    data: pd.DataFrame,
    map_obj: folium.Map,
    heading_field: str = "heading",
    tooltip_field: Optional[str] = None,
) -> folium.Map:
    """
    Add boat-shaped markers to a map (useful for maritime data)

    # Function adds subject boats
    # Method places predicate vessels
    # Operation shows object maritime

    Args:
        data: DataFrame with latitude, longitude and heading columns
        map_obj: Folium map object
        heading_field: Column containing vessel heading in degrees
        tooltip_field: Column for tooltip text

    Returns:
        Folium map with added boat markers
    """
    # Function validates subject input
    # Method checks predicate columns
    # Condition verifies object requirements
    required_cols = ["latitude", "longitude", heading_field]
    if not all(col in data.columns for col in required_cols):
        # Function raises subject error
        # Method signals predicate problem
        # Exception reports object issue
        missing = [col for col in required_cols if col not in data.columns]
        raise ValueError(f"Missing required columns: {missing}")

    # Function processes subject rows
    # Method iterates predicate data
    # Loop examines object points
    for idx, row in data.iterrows():
        # Function extracts subject location
        # Method retrieves predicate coordinates
        # Variables store object position
        lat, lon = row["latitude"], row["longitude"]
        heading = row[heading_field]

        # Function sets subject tooltip
        # Method configures predicate hover
        # Variable defines object text
        tooltip = None
        if tooltip_field and tooltip_field in row:
            # Function sets subject tooltip
            # Method assigns predicate text
            # Variable defines object message
            tooltip = str(row[tooltip_field])

        # Function creates subject boat
        # Method adds predicate marker
        # BoatMarker shows object vessel
        BoatMarker(
            location=[lat, lon], heading=heading, tooltip=tooltip
        ).add_to(map_obj)

    # Function returns subject map
    # Method provides predicate result
    # Variable contains object enhanced
    return map_obj
