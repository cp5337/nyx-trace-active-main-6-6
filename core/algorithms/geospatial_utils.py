"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-ALGO-GEOSPATIAL-UTILS-0001          │
// │ 📁 domain       : Mathematics, Geospatial, Utilities        │
// │ 🧠 description  : Common utility functions for geospatial   │
// │                  algorithms and operations                  │
// │ 🕸️ hash_type    : UUID → CUID-linked utility                │
// │ 🔄 parent_node  : NODE_ALGORITHM                           │
// │ 🧩 dependencies : numpy, shapely, math                      │
// │ 🔧 tool_usage   : Utilities, Support, Computation           │
// │ 📡 input_type   : Various geospatial data formats           │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : data conversion, format handling          │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

CTAS Geospatial Utilities
------------------------
This module provides common utility functions for geospatial algorithms
and operations. It includes coordinate system transformations, data format
conversions, and other helper functions.
"""

import numpy as np
from shapely.geometry import (
    Point,
    Polygon,
    LineString,
    MultiPoint,
    MultiPolygon,
)
from typing import List, Dict, Tuple, Union, Optional, Any
import math
import logging

# Function creates subject logger
# Method configures predicate output
# Logger tracks object events
# Variable supports subject debugging
logger = logging.getLogger(__name__)


# Constants for Earth parameters
# Function defines subject constants
# Method declares predicate values
# Constants provide object parameters
# Code specifies subject references
EARTH_RADIUS_KM = 6371.0  # Earth's mean radius in kilometers
WGS84_A = 6378137.0  # WGS-84 semi-major axis in meters
WGS84_B = 6356752.314245  # WGS-84 semi-minor axis in meters
WGS84_F = 1 / 298.257223563  # WGS-84 flattening
DEG_TO_RAD = math.pi / 180.0  # Conversion factor from degrees to radians
RAD_TO_DEG = 180.0 / math.pi  # Conversion factor from radians to degrees


# Function converts subject coordinates
# Method transforms predicate degrees
# Operation changes object measurement
# Code transforms subject units
def degrees_to_radians(
    degrees: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Convert degrees to radians

    # Function converts subject units
    # Method transforms predicate degrees
    # Operation changes object measurement
    # Code transforms subject angle

    Args:
        degrees: Angle in degrees (scalar or array)

    Returns:
        Angle in radians (same type as input)
    """
    return degrees * DEG_TO_RAD


# Function converts subject coordinates
# Method transforms predicate radians
# Operation changes object measurement
# Code transforms subject units
def radians_to_degrees(
    radians: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Convert radians to degrees

    # Function converts subject units
    # Method transforms predicate radians
    # Operation changes object measurement
    # Code transforms subject angle

    Args:
        radians: Angle in radians (scalar or array)

    Returns:
        Angle in degrees (same type as input)
    """
    return radians * RAD_TO_DEG


# Function normalizes subject longitude
# Method adjusts predicate value
# Operation corrects object coordinate
# Code standardizes subject range
def normalize_longitude(longitude: float) -> float:
    """
    Normalize longitude to range [-180, 180]

    # Function normalizes subject longitude
    # Method adjusts predicate value
    # Operation corrects object coordinate
    # Code standardizes subject range

    Args:
        longitude: Longitude value in degrees

    Returns:
        Normalized longitude in degrees
    """
    return ((longitude + 180) % 360) - 180


# Function converts subject coordinates
# Method transforms predicate format
# Operation changes object projection
# Code transforms subject system
def geodetic_to_cartesian(
    lat: float, lon: float, height: float = 0
) -> Tuple[float, float, float]:
    """
    Convert geodetic coordinates (lat, lon, height) to ECEF cartesian (x, y, z)

    # Function converts subject coordinates
    # Method transforms predicate format
    # Operation changes object projection
    # Code transforms subject system

    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        height: Height above ellipsoid in meters (default: 0)

    Returns:
        Tuple of (x, y, z) coordinates in meters
    """
    lat_rad = lat * DEG_TO_RAD
    lon_rad = lon * DEG_TO_RAD

    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    sin_lon = math.sin(lon_rad)
    cos_lon = math.cos(lon_rad)

    # Calculate radius of curvature in the prime vertical
    N = WGS84_A / math.sqrt(1 - WGS84_F * (2 - WGS84_F) * sin_lat * sin_lat)

    # Calculate ECEF coordinates
    x = (N + height) * cos_lat * cos_lon
    y = (N + height) * cos_lat * sin_lon
    z = (N * (1 - WGS84_F) * (1 - WGS84_F) + height) * sin_lat

    return (x, y, z)


# Function converts subject coordinates
# Method transforms predicate format
# Operation changes object projection
# Code transforms subject system
def cartesian_to_geodetic(
    x: float, y: float, z: float
) -> Tuple[float, float, float]:
    """
    Convert ECEF cartesian coordinates (x, y, z) to geodetic (lat, lon, height)

    # Function converts subject coordinates
    # Method transforms predicate format
    # Operation changes object projection
    # Code transforms subject system

    Args:
        x: X coordinate in ECEF in meters
        y: Y coordinate in ECEF in meters
        z: Z coordinate in ECEF in meters

    Returns:
        Tuple of (latitude, longitude, height) in (degrees, degrees, meters)
    """
    # Implementation of Bowring's method for geodetic conversion
    e_squared = WGS84_F * (2 - WGS84_F)  # First eccentricity squared

    # Handle special case of poles
    if x == 0 and y == 0:
        lon = 0
        if z > 0:
            lat = 90
        else:
            lat = -90
        height = abs(z) - WGS84_B
        return (lat, lon, height)

    # Initial values
    p = math.sqrt(x * x + y * y)
    r = math.sqrt(p * p + z * z)

    # Initial estimate
    e_prime_squared = e_squared / (1 - e_squared)
    beta = math.atan2(z, p * (1 - e_squared))

    # Iterative solution (usually converges in 2-3 iterations)
    for _ in range(5):
        sin_beta = math.sin(beta)
        cos_beta = math.cos(beta)

        phi = math.atan2(
            z + e_prime_squared * WGS84_B * sin_beta**3,
            p - e_squared * WGS84_A * cos_beta**3,
        )

        beta = math.atan2((1 - e_squared) * math.sin(phi), math.cos(phi))

    # Final calculations
    lat = phi * RAD_TO_DEG
    lon = math.atan2(y, x) * RAD_TO_DEG

    # Calculate height
    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)
    N = WGS84_A / math.sqrt(1 - e_squared * sin_phi * sin_phi)
    height = p * cos_phi + z * sin_phi - N * (1 - e_squared * sin_phi * sin_phi)

    return (lat, lon, height)


# Function calculates subject bound
# Method determines predicate extent
# Operation finds object region
# Code calculates subject area
def get_bounding_box(
    geometries: List[Union[Point, LineString, Polygon]],
    buffer_fraction: float = 0.1,
) -> Tuple[float, float, float, float]:
    """
    Calculate the bounding box for a collection of geometries with buffer

    # Function calculates subject bounds
    # Method determines predicate extents
    # Operation finds object region
    # Code computes subject area

    Args:
        geometries: List of Shapely geometries
        buffer_fraction: Fraction to expand the bounding box (default: 0.1 or 10%)

    Returns:
        Tuple of (min_x, min_y, max_x, max_y)
    """
    if not geometries:
        logger.warning(
            "Empty geometry list provided for bounding box calculation"
        )
        return (-180, -90, 180, 90)  # Default to whole Earth

    # Filter out None geometries
    valid_geometries = [g for g in geometries if g is not None]

    if not valid_geometries:
        logger.warning("No valid geometries for bounding box calculation")
        return (-180, -90, 180, 90)

    try:
        # Find initial bounds
        min_x = float("inf")
        min_y = float("inf")
        max_x = float("-inf")
        max_y = float("-inf")

        for geom in valid_geometries:
            bounds = geom.bounds  # (minx, miny, maxx, maxy)
            min_x = min(min_x, bounds[0])
            min_y = min(min_y, bounds[1])
            max_x = max(max_x, bounds[2])
            max_y = max(max_y, bounds[3])

        # Add buffer
        width = max_x - min_x
        height = max_y - min_y

        buffer_x = width * buffer_fraction
        buffer_y = height * buffer_fraction

        # If width or height is zero, use a default buffer
        if width == 0:
            buffer_x = 0.01
        if height == 0:
            buffer_y = 0.01

        return (
            min_x - buffer_x,
            min_y - buffer_y,
            max_x + buffer_x,
            max_y + buffer_y,
        )

    except Exception as e:
        logger.error(f"Error calculating bounding box: {e}")
        return (-180, -90, 180, 90)


# Function converts subject format
# Method transforms predicate data
# Operation changes object structure
# Code reshapes subject information
def points_to_numpy(points: List[Point]) -> np.ndarray:
    """
    Convert a list of Shapely Points to a numpy array of coordinates

    # Function converts subject points
    # Method transforms predicate format
    # Operation changes object structure
    # Code reshapes subject data

    Args:
        points: List of Shapely Point objects

    Returns:
        Nx2 numpy array of [x, y] coordinates
    """
    if not points:
        return np.array([])

    # Filter out None points
    valid_points = [p for p in points if p is not None]

    if not valid_points:
        return np.array([])

    try:
        return np.array([[p.x, p.y] for p in valid_points])
    except Exception as e:
        logger.error(f"Error converting points to numpy array: {e}")
        return np.array([])


# Function converts subject format
# Method transforms predicate data
# Operation changes object structure
# Code reshapes subject information
def numpy_to_points(coords: np.ndarray) -> List[Point]:
    """
    Convert a numpy array of coordinates to a list of Shapely Points

    # Function converts subject array
    # Method transforms predicate format
    # Operation changes object structure
    # Code reshapes subject data

    Args:
        coords: Nx2 numpy array of [x, y] coordinates

    Returns:
        List of Shapely Point objects
    """
    if coords.size == 0:
        return []

    try:
        return [Point(x, y) for x, y in coords]
    except Exception as e:
        logger.error(f"Error converting numpy array to points: {e}")
        return []


# Function simplifies subject polygon
# Method reduces predicate vertices
# Operation optimizes object geometry
# Code improves subject efficiency
def simplify_polygon(polygon: Polygon, tolerance: float = 0.001) -> Polygon:
    """
    Simplify a polygon to reduce the number of vertices

    # Function simplifies subject polygon
    # Method reduces predicate vertices
    # Operation optimizes object geometry
    # Code improves subject efficiency

    Args:
        polygon: Shapely Polygon to simplify
        tolerance: Tolerance parameter for simplification (higher = more simplified)

    Returns:
        Simplified Shapely Polygon
    """
    if polygon is None:
        logger.warning("Null polygon provided for simplification")
        return Polygon()

    try:
        return polygon.simplify(tolerance, preserve_topology=True)
    except Exception as e:
        logger.error(f"Error simplifying polygon: {e}")
        return polygon  # Return original if error


# Function checks subject validity
# Method validates predicate geometry
# Operation verifies object integrity
# Code ensures subject correctness
def is_valid_geometry(geometry: Union[Point, LineString, Polygon]) -> bool:
    """
    Check if a geometry is valid

    # Function checks subject validity
    # Method validates predicate geometry
    # Operation verifies object integrity
    # Code ensures subject correctness

    Args:
        geometry: Shapely geometry to check

    Returns:
        True if the geometry is valid, False otherwise
    """
    if geometry is None:
        return False

    try:
        return geometry.is_valid
    except Exception as e:
        logger.error(f"Error checking geometry validity: {e}")
        return False


# Function fixes subject geometry
# Method repairs predicate issues
# Operation corrects object problems
# Code improves subject integrity
def fix_invalid_geometry(
    geometry: Union[Polygon, MultiPolygon],
) -> Union[Polygon, MultiPolygon]:
    """
    Attempt to fix an invalid polygon/multipolygon

    # Function fixes subject geometry
    # Method repairs predicate issues
    # Operation corrects object problems
    # Code improves subject integrity

    Args:
        geometry: Invalid Shapely Polygon or MultiPolygon

    Returns:
        Fixed geometry (if possible) or original
    """
    if geometry is None:
        logger.warning("Null geometry provided for fixing")
        return Polygon()

    if geometry.is_valid:
        return geometry  # Already valid

    try:
        # Try buffer(0) trick to fix most common issues
        fixed = geometry.buffer(0)

        if fixed.is_valid:
            logger.info("Successfully fixed invalid geometry with buffer(0)")
            return fixed

        logger.warning("Could not fix invalid geometry")
        return geometry  # Return original if can't fix

    except Exception as e:
        logger.error(f"Error fixing invalid geometry: {e}")
        return geometry  # Return original if error


# Function interpolates subject points
# Method generates predicate vertices
# Operation creates object coordinates
# Code enhances subject density
def interpolate_points(
    start_point: Point, end_point: Point, num_points: int = 10
) -> List[Point]:
    """
    Create evenly spaced points along a straight line between two points

    # Function interpolates subject points
    # Method generates predicate vertices
    # Operation creates object coordinates
    # Code enhances subject density

    Args:
        start_point: Starting Shapely Point
        end_point: Ending Shapely Point
        num_points: Number of interpolated points to generate (including endpoints)

    Returns:
        List of Shapely Points along the line
    """
    if start_point is None or end_point is None:
        logger.warning("Null points provided for interpolation")
        return []

    if num_points < 2:
        logger.warning(f"Invalid num_points {num_points}, using 2")
        num_points = 2

    try:
        # Create line between points
        line = LineString(
            [(start_point.x, start_point.y), (end_point.x, end_point.y)]
        )

        # Interpolate points
        return [
            Point(line.interpolate(i / (num_points - 1), normalized=True))
            for i in range(num_points)
        ]

    except Exception as e:
        logger.error(f"Error interpolating points: {e}")
        return [start_point, end_point]  # Return endpoints if error


# Function checks subject containment
# Method tests predicate relationship
# Operation verifies object topology
# Code determines subject position
def point_in_polygons(point: Point, polygons: List[Polygon]) -> List[int]:
    """
    Find which polygons contain a point

    # Function checks subject containment
    # Method tests predicate relationship
    # Operation verifies object topology
    # Code determines subject position

    Args:
        point: Shapely Point to test
        polygons: List of Shapely Polygons to check against

    Returns:
        List of indices of polygons that contain the point
    """
    if point is None or not polygons:
        return []

    try:
        return [
            i
            for i, poly in enumerate(polygons)
            if poly is not None and poly.contains(point)
        ]

    except Exception as e:
        logger.error(f"Error checking point in polygons: {e}")
        return []
