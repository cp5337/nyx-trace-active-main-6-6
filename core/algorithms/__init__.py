"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-ALGO-INIT-0001                      │
// │ 📁 domain       : Mathematics, Algorithms, Initialization   │
// │ 🧠 description  : Algorithms module initialization and      │
// │                  package exports                            │
// │ 🕸️ hash_type    : UUID → CUID-linked initialization         │
// │ 🔄 parent_node  : NODE_ALGORITHM                           │
// │ 🧩 dependencies : core.algorithms modules                   │
// │ 🔧 tool_usage   : Import, Module, Registry                 │
// │ 📡 input_type   : Module imports                           │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : module organization, namespace control    │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

CTAS Algorithms Module Initialization
-----------------------------------
This module serves as the initialization point for the algorithms package,
re-exporting all public interfaces from the individual algorithm modules.
It ensures proper module organization and namespace control.
"""

# Re-export from geospatial_algorithms.py for backward compatibility
from core.algorithms.geospatial_algorithms import (
    # Classes
    DistanceCalculator,
    HexagonalGrid,
    SpatialJoin,
    HotspotAnalysis,
    # Constants
    EARTH_RADIUS_KM,
    WGS84_A,
    WGS84_B,
    WGS84_F,
    DEG_TO_RAD,
    RAD_TO_DEG,
    # Utility functions
    degrees_to_radians,
    radians_to_degrees,
    normalize_longitude,
    geodetic_to_cartesian,
    cartesian_to_geodetic,
    get_bounding_box,
    points_to_numpy,
    numpy_to_points,
    simplify_polygon,
    is_valid_geometry,
    fix_invalid_geometry,
    interpolate_points,
    point_in_polygons,
)

# Module registry
__all__ = [
    # Classes
    "DistanceCalculator",
    "HexagonalGrid",
    "SpatialJoin",
    "HotspotAnalysis",
    # Constants
    "EARTH_RADIUS_KM",
    "WGS84_A",
    "WGS84_B",
    "WGS84_F",
    "DEG_TO_RAD",
    "RAD_TO_DEG",
    # Utility functions
    "degrees_to_radians",
    "radians_to_degrees",
    "normalize_longitude",
    "geodetic_to_cartesian",
    "cartesian_to_geodetic",
    "get_bounding_box",
    "points_to_numpy",
    "numpy_to_points",
    "simplify_polygon",
    "is_valid_geometry",
    "fix_invalid_geometry",
    "interpolate_points",
    "point_in_polygons",
]
