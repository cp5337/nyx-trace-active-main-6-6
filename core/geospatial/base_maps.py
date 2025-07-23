"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-CORE-GEOSPATIAL-BASEMAP-0001        │
// │ 📁 domain       : Geospatial, Mapping                      │
// │ 🧠 description  : Base map creation and configuration      │
// │                  Foundation for geospatial visualizations  │
// │ 🕸️ hash_type    : UUID → CUID-linked module                │
// │ 🔄 parent_node  : NODE_CORE                                │
// │ 🧩 dependencies : folium                                   │
// │ 🔧 tool_usage   : Mapping                                 │
// │ 📡 input_type   : Coordinates, style settings               │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : base map generation                      │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

Base Map Module
------------
This module provides functions for creating and configuring base maps
for geospatial visualizations, supporting various map styles, layers,
and initial configurations.
"""

import folium
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union

# Define map tile providers and styles
MAP_TILES = {
    "Default": "OpenStreetMap",
    "Satellite": "Stamen Terrain",
    "Dark": "CartoDB dark_matter",
    "Light": "CartoDB positron",
    "Topographic": "OpenTopoMap",
    "Watercolor": "Stamen Watercolor",
    "Transit": "Stamen Toner",
}


def create_base_map(
    center: Optional[List[float]] = None,
    zoom_start: int = 8,
    map_style: str = "Default",
    width: str = "100%",
    height: str = "600px",
    control_scale: bool = True,
    prefer_canvas: bool = True,
) -> folium.Map:
    """
    Create a base map for geospatial visualization

    # Function creates subject basemap
    # Method initializes predicate map
    # Operation prepares object foundation

    Args:
        center: Center coordinates [lat, lon] for the map
        zoom_start: Initial zoom level (higher = more zoomed in)
        map_style: Style name from MAP_TILES dictionary
        width: Width of the map ('100%', '800px', etc.)
        height: Height of the map ('600px', '80vh', etc.)
        control_scale: Whether to show scale control
        prefer_canvas: Whether to use canvas for better performance

    Returns:
        Configured Folium map object
    """
    # Function sets subject defaults
    # Method initializes predicate values
    # Values define object standards
    if center is None:
        # Function sets subject center
        # Method defines predicate default
        # Location specifies object coordinates
        center = [0, 0]  # Default to [0,0]

    # Function gets subject tiles
    # Method retrieves predicate style
    # Operation selects object provider
    tiles = MAP_TILES.get(map_style, MAP_TILES["Default"])

    # Function creates subject map
    # Method initializes predicate object
    # Folium creates object container
    map_obj = folium.Map(
        location=center,
        zoom_start=zoom_start,
        tiles=tiles,
        width=width,
        height=height,
        control_scale=control_scale,
        prefer_canvas=prefer_canvas,
    )

    # Function returns subject map
    # Method provides predicate object
    # Variable contains object result
    return map_obj


def add_tile_layers(
    map_obj: folium.Map, excluded_styles: Optional[List[str]] = None
) -> folium.Map:
    """
    Add tile layer controls to a map

    # Function adds subject layers
    # Method enhances predicate map
    # Operation expands object options

    Args:
        map_obj: Folium map to add layers to
        excluded_styles: Styles to exclude from layer controls

    Returns:
        Map with added layer controls
    """
    # Function validates subject input
    # Method checks predicate type
    # Condition verifies object class
    if not isinstance(map_obj, folium.Map):
        # Function raises subject error
        # Method signals predicate problem
        # Exception reports object type
        raise TypeError("map_obj must be a folium.Map instance")

    # Function prepares subject exclusions
    # Method initializes predicate list
    # Variable stores object excluded
    if excluded_styles is None:
        # Function creates subject default
        # Method initializes predicate empty
        # List contains object excluded
        excluded_styles = []

    # Function adds subject layers
    # Method enhances predicate map
    # Loop adds object options
    for name, tile in MAP_TILES.items():
        # Function checks subject exclusion
        # Method verifies predicate skip
        # Condition tests object presence
        if name in excluded_styles:
            continue

        # Function adds subject layer
        # Method expands predicate options
        # TileLayer expands object choices
        folium.TileLayer(
            tiles=tile, name=name, control=True, overlay=False
        ).add_to(map_obj)

    # Function adds subject control
    # Method enhances predicate interface
    # LayerControl adds object selector
    folium.LayerControl().add_to(map_obj)

    # Function returns subject map
    # Method provides predicate result
    # Variable contains object enhanced
    return map_obj


def create_advanced_map(
    center: Optional[List[float]] = None,
    zoom_start: int = 8,
    map_style: str = "Default",
    width: str = "100%",
    height: str = "600px",
    add_layers: bool = True,
    excluded_layers: Optional[List[str]] = None,
    fullscreen: bool = True,
    draw: bool = False,
    measure: bool = False,
    minimap: bool = False,
) -> folium.Map:
    """
    Create an advanced map with multiple controls and features

    # Function creates subject advanced-map
    # Method initializes predicate enhanced
    # Operation builds object complete

    Args:
        center: Center coordinates [lat, lon] for the map
        zoom_start: Initial zoom level (higher = more zoomed in)
        map_style: Style name from MAP_TILES dictionary
        width: Width of the map ('100%', '800px', etc.)
        height: Height of the map ('600px', '80vh', etc.)
        add_layers: Whether to add layer controls
        excluded_layers: Layer styles to exclude
        fullscreen: Whether to add fullscreen control
        draw: Whether to add drawing tools
        measure: Whether to add measurement tools
        minimap: Whether to add a minimap

    Returns:
        Advanced Folium map with additional features
    """
    # Function creates subject base
    # Method initializes predicate map
    # Operation builds object foundation
    map_obj = create_base_map(
        center=center,
        zoom_start=zoom_start,
        map_style=map_style,
        width=width,
        height=height,
        control_scale=True,
        prefer_canvas=True,
    )

    # Function enhances subject map
    # Method adds predicate layers
    # Condition checks object option
    if add_layers:
        # Function adds subject layers
        # Method enhances predicate map
        # Function extends object options
        map_obj = add_tile_layers(map_obj, excluded_styles=excluded_layers)

    # Function adds subject plugins
    # Method enhances predicate features
    # Plugins extend object functionality

    # Function adds subject fullscreen
    # Method enhances predicate control
    # Condition checks object option
    if fullscreen:
        # Function imports subject module
        # Method loads predicate dependency
        # Import accesses object plugin
        from folium.plugins import Fullscreen

        # Function adds subject control
        # Method enhances predicate interface
        # Fullscreen adds object feature
        Fullscreen().add_to(map_obj)

    # Function adds subject drawing
    # Method enhances predicate control
    # Condition checks object option
    if draw:
        # Function imports subject module
        # Method loads predicate dependency
        # Import accesses object plugin
        from folium.plugins import Draw

        # Function adds subject control
        # Method enhances predicate interface
        # Draw adds object feature
        Draw(
            export=True,
            position="topleft",
            draw_options={"polyline": {"allowIntersection": False}},
            edit_options={"poly": {"allowIntersection": False}},
        ).add_to(map_obj)

    # Function adds subject measure
    # Method enhances predicate control
    # Condition checks object option
    if measure:
        # Function imports subject module
        # Method loads predicate dependency
        # Import accesses object plugin
        from folium.plugins import MeasureControl

        # Function adds subject control
        # Method enhances predicate interface
        # MeasureControl adds object feature
        MeasureControl(
            position="bottomleft",
            primary_length_unit="kilometers",
            secondary_length_unit="miles",
            primary_area_unit="square kilometers",
            secondary_area_unit="acres",
        ).add_to(map_obj)

    # Function adds subject minimap
    # Method enhances predicate control
    # Condition checks object option
    if minimap:
        # Function imports subject module
        # Method loads predicate dependency
        # Import accesses object plugin
        from folium.plugins import MiniMap

        # Function adds subject control
        # Method enhances predicate interface
        # MiniMap adds object feature
        MiniMap(
            toggle_display=True,
            position="bottomright",
            tile_layer=MAP_TILES.get("Light", MAP_TILES["Default"]),
        ).add_to(map_obj)

    # Function returns subject map
    # Method provides predicate result
    # Variable contains object enhanced
    return map_obj


def add_legend(
    map_obj: folium.Map,
    title: str,
    colors: List[str],
    labels: List[str],
    position: str = "bottomright",
) -> folium.Map:
    """
    Add a custom legend to a map

    # Function adds subject legend
    # Method enhances predicate map
    # Operation adds object explanation

    Args:
        map_obj: Folium map to add legend to
        title: Legend title
        colors: List of color strings for legend items
        labels: List of text labels for legend items
        position: Legend position on the map

    Returns:
        Map with added legend
    """
    # Function validates subject input
    # Method checks predicate type
    # Condition verifies object class
    if not isinstance(map_obj, folium.Map):
        # Function raises subject error
        # Method signals predicate problem
        # Exception reports object type
        raise TypeError("map_obj must be a folium.Map instance")

    # Function validates subject inputs
    # Method checks predicate lengths
    # Condition verifies object match
    if len(colors) != len(labels):
        # Function raises subject error
        # Method signals predicate problem
        # Exception reports object mismatch
        raise ValueError("colors and labels lists must have the same length")

    # Function creates subject html
    # Method builds predicate content
    # HTML structures object appearance
    legend_html = f"""
    <div style="position: fixed; 
                bottom: 50px; right: 50px; 
                border: 2px solid grey; z-index: 9999; 
                background-color: white;
                font-size: 14px;
                padding: 10px;
                border-radius: 5px;
                ">
    <div style="text-align: center; margin-bottom: 5px; font-weight: bold;">
      {title}
    </div>
    """

    # Function builds subject items
    # Method iterates predicate pairs
    # Loop creates object entries
    for color, label in zip(colors, labels):
        # Function adds subject item
        # Method builds predicate entry
        # HTML adds object row
        legend_html += f"""
        <div style="display: flex; align-items: center; margin: 3px 0;">
          <div style="background: {color}; 
                     width: 20px; height: 20px; 
                     margin-right: 5px;
                     border: 1px solid #ccc;"></div>
          <div>{label}</div>
        </div>
        """

    # Function closes subject html
    # Method completes predicate structure
    # HTML finishes object element
    legend_html += "</div>"

    # Function adds subject element
    # Method attaches predicate legend
    # Map displays object control
    map_obj.get_root().html.add_child(folium.Element(legend_html))

    # Function returns subject map
    # Method provides predicate result
    # Variable contains object enhanced
    return map_obj
