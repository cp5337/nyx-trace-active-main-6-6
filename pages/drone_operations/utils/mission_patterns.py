"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-DRONE-MISSION-PATTERNS-0001         │
// │ 📁 domain       : Drone, Mission, Planning                  │
// │ 🧠 description  : Mission pattern generator utilities        │
// │                  for various drone operations                │
// │ 🕸️ hash_type    : UUID → CUID-linked module                 │
// │ 🔄 parent_node  : NODE_DRONE_OPERATIONS                     │
// │ 🧩 dependencies : math, typing, folium                      │
// │ 🔧 tool_usage   : Utility, Visualization                    │
// │ 📡 input_type   : Mission parameters                        │
// │ 🧪 test_status  : stable                                    │
// │ 🧠 cognitive_fn : path generation, visualization             │
// │ ⌛ TTL Policy   : 6.5 Persistent                            │
// └─────────────────────────────────────────────────────────────┘

Mission Pattern Utilities
-----------------------
This module provides utilities for generating different mission patterns
for drone operations, including search grids, direct attack vectors,
surveillance orbits, and reconnaissance patterns.
"""

import math
import folium
from typing import List, Tuple, Dict, Any, Optional

# Constants for approximate conversions
KM_TO_DEG = 1/111  # Approximate conversion from km to degrees at the equator

def generate_search_grid_pattern(
    target_lat: float, 
    target_lon: float, 
    grid_size: float, 
    grid_spacing: float
) -> List[Tuple[float, float]]:
    """
    Generate points for an S-pattern search grid
    
    # Function generates subject grid
    # Method calculates predicate points
    # Utility produces object pattern
    
    Args:
        target_lat: Center latitude of the search area
        target_lon: Center longitude of the search area
        grid_size: Size of the search grid in km
        grid_spacing: Spacing between grid lines in km
        
    Returns:
        List of points (lat, lon) forming the search grid
    """
    # Convert to degrees
    grid_size_deg = grid_size * KM_TO_DEG
    grid_spacing_deg = grid_spacing * KM_TO_DEG
    
    half_size = grid_size_deg / 2
    grid_points = []
    
    # Create S-pattern grid
    y = target_lat - half_size
    step = 0
    while y <= target_lat + half_size:
        if step % 2 == 0:  # Even rows go left to right
            x_start = target_lon - half_size
            x_end = target_lon + half_size
            x_step = grid_spacing_deg
        else:  # Odd rows go right to left
            x_start = target_lon + half_size
            x_end = target_lon - half_size
            x_step = -grid_spacing_deg
            
        x = x_start
        while (x_step > 0 and x <= x_end) or (x_step < 0 and x >= x_end):
            grid_points.append((y, x))
            x += x_step
        
        y += grid_spacing_deg
        step += 1
    
    return grid_points

def add_search_grid_to_map(
    m: folium.Map, 
    target_lat: float, 
    target_lon: float, 
    grid_size: float, 
    grid_spacing: float
) -> None:
    """
    Add a search grid pattern to a Folium map
    
    # Function adds subject grid
    # Method visualizes predicate pattern
    # Utility updates object map
    
    Args:
        m: Folium map to add visualization to
        target_lat: Center latitude of the search area
        target_lon: Center longitude of the search area
        grid_size: Size of the search grid in km
        grid_spacing: Spacing between grid lines in km
    """
    # Generate grid points
    grid_points = generate_search_grid_pattern(target_lat, target_lon, grid_size, grid_spacing)
    
    # Convert to degrees for rectangle bounds
    grid_size_deg = grid_size * KM_TO_DEG
    half_size = grid_size_deg / 2
    
    # Draw the search grid pattern
    if grid_points:
        folium.PolyLine(
            grid_points,
            color="green",
            weight=2,
            opacity=0.7
        ).add_to(m)
        
        # Add rectangle showing search area
        folium.Rectangle(
            bounds=[
                [target_lat - half_size, target_lon - half_size],
                [target_lat + half_size, target_lon + half_size]
            ],
            color="green",
            weight=1,
            fill=True,
            fill_opacity=0.1
        ).add_to(m)

def generate_direct_attack_vector(
    target_lat: float, 
    target_lon: float, 
    approach_vector: float, 
    distance_km: float = 10.0
) -> List[Tuple[float, float]]:
    """
    Generate points for a direct attack vector
    
    # Function generates subject vector
    # Method calculates predicate approach
    # Utility produces object path
    
    Args:
        target_lat: Target latitude
        target_lon: Target longitude
        approach_vector: Approach angle in degrees
        distance_km: Distance from starting point to target in km
        
    Returns:
        List of points [(start_lat, start_lon), (target_lat, target_lon)]
    """
    # Convert approach vector to radians
    angle_rad = math.radians(approach_vector)
    
    # Calculate starting point (distance_km away from target in the specified direction)
    dist_deg = distance_km * KM_TO_DEG
    start_lat = target_lat - dist_deg * math.cos(angle_rad)
    start_lon = target_lon - dist_deg * math.sin(angle_rad)
    
    return [(start_lat, start_lon), (target_lat, target_lon)]

def add_direct_attack_to_map(
    m: folium.Map, 
    target_lat: float, 
    target_lon: float, 
    approach_vector: float, 
    attack_speed: float,
    distance_km: float = 10.0
) -> None:
    """
    Add a direct attack vector to a Folium map
    
    # Function adds subject vector
    # Method visualizes predicate approach
    # Utility updates object map
    
    Args:
        m: Folium map to add visualization to
        target_lat: Target latitude
        target_lon: Target longitude
        approach_vector: Approach angle in degrees
        attack_speed: Attack speed percentage
        distance_km: Distance from starting point to target in km
    """
    # Get attack vector points
    attack_points = generate_direct_attack_vector(target_lat, target_lon, approach_vector, distance_km)
    
    # Add basic line
    folium.PolyLine(
        attack_points,
        color="red",
        weight=3,
        opacity=0.8,
        dash_array='5, 10'
    ).add_to(m)
    
    # Add animated approach vector
    try:
        folium.plugins.AntPath(
            attack_points,
            color="red",
            weight=4,
            opacity=0.7,
            delay=2000 / (attack_speed / 50),  # Speed affects animation
            pulse_color="red"
        ).add_to(m)
    except Exception:
        # Fallback if AntPath plugin fails
        pass

def generate_orbit_pattern(
    target_lat: float, 
    target_lon: float, 
    radius_km: float,
    points: int = 36
) -> List[Tuple[float, float]]:
    """
    Generate points for a circular orbit pattern
    
    # Function generates subject orbit
    # Method calculates predicate circle
    # Utility produces object points
    
    Args:
        target_lat: Center latitude
        target_lon: Center longitude
        radius_km: Orbit radius in km
        points: Number of points to generate
        
    Returns:
        List of points forming a circular orbit
    """
    # Convert to degrees
    radius_deg = radius_km * KM_TO_DEG
    
    # Create points along the circle for the orbit path
    orbit_points = []
    angle_step = 360 / points
    for i in range(points + 1):  # +1 to close the loop
        angle = i * angle_step
        angle_rad = math.radians(angle)
        point_lat = target_lat + radius_deg * math.cos(angle_rad)
        point_lon = target_lon + radius_deg * math.sin(angle_rad)
        orbit_points.append((point_lat, point_lon))
    
    return orbit_points

def add_orbit_to_map(
    m: folium.Map, 
    target_lat: float, 
    target_lon: float, 
    radius_km: float,
    orbit_time: float
) -> None:
    """
    Add a surveillance orbit to a Folium map
    
    # Function adds subject orbit
    # Method visualizes predicate pattern
    # Utility updates object map
    
    Args:
        m: Folium map to add visualization to
        target_lat: Center latitude
        target_lon: Center longitude
        radius_km: Orbit radius in km
        orbit_time: Time to complete orbit in minutes
    """
    # Create a circle around the target
    folium.Circle(
        location=[target_lat, target_lon],
        radius=radius_km * 1000,  # Convert to meters
        color="blue",
        fill=True,
        fill_opacity=0.1
    ).add_to(m)
    
    # Get orbit points
    orbit_points = generate_orbit_pattern(target_lat, target_lon, radius_km)
    
    # Draw the orbit path
    folium.PolyLine(
        orbit_points,
        color="blue",
        weight=2,
        opacity=0.7,
        dash_array='5'
    ).add_to(m)

def generate_reconnaissance_path(
    target_lat: float, 
    target_lon: float, 
    path_type: str,
    path_length_km: float
) -> List[Tuple[float, float]]:
    """
    Generate points for a reconnaissance path
    
    # Function generates subject path
    # Method calculates predicate route
    # Utility produces object points
    
    Args:
        target_lat: Center latitude
        target_lon: Center longitude
        path_type: Type of path ("Linear", "S-Pattern", or "Circular")
        path_length_km: Path length in km
        
    Returns:
        List of points forming the reconnaissance path
    """
    # Convert to degrees
    path_length_deg = path_length_km * KM_TO_DEG
    
    if path_type == "Linear":
        # Linear path
        start_lat = target_lat - path_length_deg/2
        start_lon = target_lon - path_length_deg/2
        end_lat = target_lat + path_length_deg/2
        end_lon = target_lon + path_length_deg/2
        
        return [(start_lat, start_lon), (end_lat, end_lon)]
        
    elif path_type == "S-Pattern":
        # S-Pattern path
        segment_length = path_length_deg / 4
        return [
            (target_lat - path_length_deg/2, target_lon),
            (target_lat - path_length_deg/4, target_lon + segment_length),
            (target_lat, target_lon),
            (target_lat + path_length_deg/4, target_lon - segment_length),
            (target_lat + path_length_deg/2, target_lon)
        ]
        
    else:  # Circular
        # Use the orbit pattern function with a specific radius
        radius_deg = path_length_deg / (2 * math.pi)  # Convert path length to radius
        return generate_orbit_pattern(target_lat, target_lon, radius_deg * 111)  # Convert back to km

def add_reconnaissance_to_map(
    m: folium.Map, 
    target_lat: float, 
    target_lon: float, 
    path_type: str,
    path_length_km: float
) -> None:
    """
    Add a reconnaissance path to a Folium map
    
    # Function adds subject path
    # Method visualizes predicate route
    # Utility updates object map
    
    Args:
        m: Folium map to add visualization to
        target_lat: Center latitude
        target_lon: Center longitude
        path_type: Type of path ("Linear", "S-Pattern", or "Circular")
        path_length_km: Path length in km
    """
    # Get reconnaissance path points
    path_points = generate_reconnaissance_path(target_lat, target_lon, path_type, path_length_km)
    
    # Draw the path
    folium.PolyLine(
        path_points,
        color="purple",
        weight=3,
        opacity=0.7
    ).add_to(m)

# CTAS Module Line Count:
# USIM Header: 17 lines
# Comments: 74 lines
# Code: 294 lines
# Total: 385 lines