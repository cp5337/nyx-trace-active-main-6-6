"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-DRONE-FLIGHT-PROFILE-0001            │
// │ 📁 domain       : Drone, Simulation, Flight                  │
// │ 🧠 description  : Drone flight profiles defining movement    │
// │                  patterns and characteristics                │
// │ 🕸️ hash_type    : UUID → CUID-linked module                 │
// │ 🔄 parent_node  : NODE_DRONE_OPERATIONS                     │
// │ 🧩 dependencies : None                                      │
// │ 🔧 tool_usage   : Simulation                                │
// │ 📡 input_type   : Parameters                                │
// │ 🧪 test_status  : stable                                    │
// │ 🧠 cognitive_fn : modeling, patterns                         │
// │ ⌛ TTL Policy   : 6.5 Persistent                            │
// └─────────────────────────────────────────────────────────────┘

CTAS Drone Flight Profiles
-------------------------
This module defines drone flight profiles and characteristics
for use in simulation and visualization.
"""

import random
import math
from typing import Dict, List, Tuple, Optional


# Function defines subject profile
# Method describes predicate characteristics
# Definition models object behavior
class DroneProfile:
    """
    Drone profile containing flight characteristics

    # Class defines subject profile
    # Method describes predicate characteristics
    # Object models drone behavior
    """

    def __init__(self, model, speed_mps, turn_rate_deg, max_alt_m):
        self.model = model
        self.speed_mps = speed_mps
        self.turn_rate_deg = turn_rate_deg
        self.max_alt_m = max_alt_m


# Catalog of drone profiles
DRONE_PROFILES = {
    "nano_drone": DroneProfile("nano_drone", 3.0, 90, 50),
    "quadrotor_standard": DroneProfile("quadrotor_standard", 10.0, 45, 120),
    "tactical_uav": DroneProfile("tactical_uav", 28.0, 25, 500),
    "fixed_wing_long_range": DroneProfile(
        "fixed_wing_long_range", 65.0, 15, 1500
    ),
}


# Function provides subject profiles
# Method generates predicate patterns
# Class configures object movements
class DroneFlightProfiles:
    """
    Drone flight profiles providing movement patterns

    # Class provides subject profiles
    # Method generates predicate patterns
    # Object configures drone movements
    """

    def __init__(self):
        """Initialize flight profiles"""
        self.patterns = {
            "hover": self._hover_pattern,
            "circle": self._circle_pattern,
            "grid_search": self._grid_search_pattern,
            "linear": self._linear_pattern,
            "lawnmower": self._lawnmower_pattern,
            "spiral": self._spiral_pattern,
        }

    def get_pattern_generator(self, pattern_name: str, **kwargs):
        """
        Get a generator for a specific flight pattern

        Args:
            pattern_name: Name of the pattern to generate
            **kwargs: Parameters for the pattern generator

        Returns:
            Generator function for the requested pattern
        """
        if pattern_name not in self.patterns:
            pattern_name = "linear"  # Default

        return self.patterns[pattern_name](**kwargs)

    def _hover_pattern(
        self,
        center_point: Tuple[float, float],
        altitude: float = 100.0,
        duration: int = 300,
        jitter: float = 0.00005,
    ):
        """
        Generate hover pattern where drone stays in same area

        Args:
            center_point: (lat, lon) center position
            altitude: Altitude in meters
            duration: Duration of hover in seconds
            jitter: Small random movement amount

        Yields:
            (lat, lon, alt) position tuples
        """
        lat, lon = center_point
        for _ in range(duration):
            # Add small random movements to simulate hover instability
            jitter_lat = random.uniform(-jitter, jitter)
            jitter_lon = random.uniform(-jitter, jitter)
            jitter_alt = random.uniform(-1.0, 1.0)

            yield (lat + jitter_lat, lon + jitter_lon, altitude + jitter_alt)

    def _circle_pattern(
        self,
        center_point: Tuple[float, float],
        radius: float = 0.002,  # Approx 200m at equator
        altitude: float = 100.0,
        points: int = 120,
        clockwise: bool = True,
    ):
        """
        Generate circular flight pattern around point

        Args:
            center_point: (lat, lon) center position
            radius: Circle radius in degrees
            altitude: Altitude in meters
            points: Number of points in the circle
            clockwise: Direction of circle

        Yields:
            (lat, lon, alt) position tuples
        """
        lat, lon = center_point
        direction = 1 if clockwise else -1

        for i in range(points):
            angle = direction * 2 * math.pi * i / points
            dlat = radius * math.cos(angle)
            dlon = radius * math.sin(angle)

            yield (lat + dlat, lon + dlon, altitude)

    def _linear_pattern(
        self,
        start_point: Tuple[float, float],
        end_point: Tuple[float, float],
        altitude: float = 100.0,
        points: int = 50,
    ):
        """
        Generate linear flight pattern between two points

        Args:
            start_point: (lat, lon) starting position
            end_point: (lat, lon) ending position
            altitude: Altitude in meters
            points: Number of points in the path

        Yields:
            (lat, lon, alt) position tuples
        """
        start_lat, start_lon = start_point
        end_lat, end_lon = end_point

        for i in range(points):
            t = i / (points - 1)  # Parameter from 0 to 1
            lat = start_lat + t * (end_lat - start_lat)
            lon = start_lon + t * (end_lon - start_lon)

            yield (lat, lon, altitude)

    def _grid_search_pattern(
        self,
        start_point: Tuple[float, float],
        width: float = 0.01,  # ~1km at equator
        height: float = 0.01,
        rows: int = 5,
        altitude: float = 100.0,
    ):
        """
        Generate grid search pattern over rectangular area

        Args:
            start_point: (lat, lon) southwest corner position
            width: Width of grid in degrees longitude
            height: Height of grid in degrees latitude
            rows: Number of rows in the grid
            altitude: Altitude in meters

        Yields:
            (lat, lon, alt) position tuples
        """
        start_lat, start_lon = start_point

        # Distance between rows
        row_spacing = height / rows

        # Generate a lawnmower pattern
        for i in range(rows):
            row_lat = start_lat + i * row_spacing

            # Even rows go left to right
            if i % 2 == 0:
                for t in range(20):
                    t_norm = t / 19  # Parameter from 0 to 1
                    lon = start_lon + t_norm * width
                    yield (row_lat, lon, altitude)
            # Odd rows go right to left
            else:
                for t in range(20):
                    t_norm = t / 19  # Parameter from 0 to 1
                    lon = start_lon + width - t_norm * width
                    yield (row_lat, lon, altitude)

    def _lawnmower_pattern(
        self,
        start_point: Tuple[float, float],
        width: float = 0.01,
        height: float = 0.01,
        rows: int = 5,
        altitude: float = 100.0,
    ):
        """
        Generate lawnmower survey pattern over rectangular area

        Args:
            start_point: (lat, lon) southwest corner position
            width: Width of area in degrees longitude
            height: Height of area in degrees latitude
            rows: Number of passes
            altitude: Altitude in meters

        Yields:
            (lat, lon, alt) position tuples
        """
        # This is similar to grid_search but with smoother turns
        # Reuse the grid search implementation for now
        yield from self._grid_search_pattern(
            start_point=start_point,
            width=width,
            height=height,
            rows=rows,
            altitude=altitude,
        )

    def _spiral_pattern(
        self,
        center_point: Tuple[float, float],
        max_radius: float = 0.01,
        altitude: float = 100.0,
        turns: int = 3,
        points_per_turn: int = 30,
    ):
        """
        Generate spiral flight pattern from center outward

        Args:
            center_point: (lat, lon) center position
            max_radius: Maximum radius in degrees
            altitude: Altitude in meters
            turns: Number of turns in spiral
            points_per_turn: Number of points per turn

        Yields:
            (lat, lon, alt) position tuples
        """
        lat, lon = center_point
        total_points = turns * points_per_turn

        for i in range(total_points):
            angle = 2 * math.pi * i / points_per_turn
            radius_factor = i / total_points
            radius = max_radius * radius_factor

            dlat = radius * math.cos(angle)
            dlon = radius * math.sin(angle)

            yield (lat + dlat, lon + dlon, altitude)
