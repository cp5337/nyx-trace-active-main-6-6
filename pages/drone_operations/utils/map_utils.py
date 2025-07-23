"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-DRONE-OPS-MAPUTILS-0001             │
// │ 📁 domain       : Drone, Maps, Utilities                    │
// │ 🧠 description  : Map utilities for drone operations         │
// │                  dashboard with tile selection and styling   │
// │ 🕸️ hash_type    : UUID → CUID-linked module                 │
// │ 🔄 parent_node  : NODE_DRONE_OPERATIONS                     │
// │ 🧩 dependencies : typing                                    │
// │ 🔧 tool_usage   : Utility, Visualization                    │
// │ 📡 input_type   : Map configuration parameters               │
// │ 🧪 test_status  : stable                                    │
// │ 🧠 cognitive_fn : map configuration, visualization           │
// │ ⌛ TTL Policy   : 6.5 Persistent                            │
// └─────────────────────────────────────────────────────────────┘

Map Utilities for Drone Operations
---------------------------------
This module provides utility functions for map configuration and styling
used across drone operation visualization components. Functions are designed
to be reusable across different applications.
"""

from typing import Dict, List, Tuple, Any, Optional

def get_tile_with_attribution(tile_type: str) -> Tuple[str, str]:
    """
    Get a map tile URL and its attribution text
    
    # Function retrieves subject tiles
    # Method selects predicate style
    # Utility returns object attribution
    
    Args:
        tile_type: The type of map tile to use
        
    Returns:
        Tuple of (tile_url, attribution_text)
    """
    tiles = {
        "OpenStreetMap": {
            "url": "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
            "attr": '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        },
        "CartoDB positron": {
            "url": "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
            "attr": '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, &copy; <a href="https://cartodb.com/attributions">CartoDB</a>'
        },
        "CartoDB dark_matter": {
            "url": "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
            "attr": '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, &copy; <a href="https://cartodb.com/attributions">CartoDB</a>'
        },
        "Esri WorldImagery": {
            "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            "attr": 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'
        }
    }
    
    if tile_type not in tiles:
        tile_type = "CartoDB positron"  # Default
        
    return tiles[tile_type]["url"], tiles[tile_type]["attr"]

def create_drone_icon(status: str) -> Dict[str, Any]:
    """
    Create a customized drone map icon based on status
    
    # Function creates subject icon
    # Method customizes predicate appearance
    # Utility generates object representation
    
    Args:
        status: The drone status (active, warning, error, etc.)
        
    Returns:
        Dictionary with icon configuration
    """
    icon_colors = {
        "active": "green",
        "mission": "blue",
        "warning": "orange",
        "error": "red",
        "offline": "gray"
    }
    
    color = icon_colors.get(status.lower(), "blue")
    
    return {
        "icon": "fa-drone",
        "iconColor": "white",
        "markerColor": color,
        "prefix": "fa"
    }

# CTAS Module Line Count:
# USIM Header: 17 lines
# Comments: 15 lines
# Code: 45 lines
# Total: 77 lines