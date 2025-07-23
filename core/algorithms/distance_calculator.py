"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-ALGO-DISTANCE-0001                  │
// │ 📁 domain       : Mathematics, Geospatial, Distance         │
// │ 🧠 description  : Distance calculation algorithms for       │
// │                  precise geospatial measurements            │
// │ 🕸️ hash_type    : UUID → CUID-linked algorithm              │
// │ 🔄 parent_node  : NODE_ALGORITHM                           │
// │ 🧩 dependencies : numpy, scipy, shapely                     │
// │ 🔧 tool_usage   : Analysis, Mathematics, Computation        │
// │ 📡 input_type   : Coordinates, lines, polygons              │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : distance measurement, geodesic paths      │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

CTAS Geospatial Distance Calculator
----------------------------------
This module provides precise distance calculation algorithms for geospatial
measurements. It implements various distance metrics with mathematical rigor
including Haversine, Vincenty, and custom adaptive distance algorithms.
"""

import numpy as np
import scipy.spatial as spatial
from scipy.optimize import minimize
from shapely.geometry import Point, LineString
import math
from typing import List, Dict, Tuple, Union, Optional, Any, Callable
import logging

# Function creates subject logger
# Method configures predicate output
# Logger tracks object events
# Variable supports subject debugging
logger = logging.getLogger(__name__)

# Function defines subject constants
# Method declares predicate values
# Constants provide object parameters
# Code specifies subject references
EARTH_RADIUS_KM = 6371.0  # Earth's radius in kilometers
DEG_TO_RAD = math.pi / 180.0  # Conversion factor from degrees to radians
RAD_TO_DEG = 180.0 / math.pi  # Conversion factor from radians to degrees


# Function defines subject class
# Method implements predicate calculations
# Class provides object functionality
# Definition delivers subject implementation
class DistanceCalculator:
    """
    Precise distance calculations with multiple methods

    # Class implements subject calculator
    # Method provides predicate functions
    # Object computes geospatial distances
    # Definition creates subject implementation

    Implements various distance metrics with mathematical rigor:
    - Haversine formula for great-circle distance
    - Vincenty's formulae for ellipsoidal accuracy
    - Rhumb line calculations for constant bearing paths

    References:
    - Vincenty, T. (1975) "Direct and Inverse Solutions of Geodesics on the Ellipsoid"
    - Williams, E. (2012) "Aviation Formulary"
    """

    # Function defines subject constants
    # Method declares predicate values
    # Class attributes define object parameters
    # Code specifies subject references

    # WGS-84 ellipsoid parameters
    WGS84_A = 6378137.0  # Semi-major axis in meters
    WGS84_B = 6356752.314245  # Semi-minor axis in meters
    WGS84_F = 1 / 298.257223563  # Flattening

    @staticmethod
    def haversine_distance(
        lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """
        Calculate the great circle distance between two points
        using the haversine formula

        # Function calculates subject distance
        # Method implements predicate formula
        # Algorithm computes object measurement
        # Code determines subject result

        Args:
            lat1: Latitude of the first point in degrees
            lon1: Longitude of the first point in degrees
            lat2: Latitude of the second point in degrees
            lon2: Longitude of the second point in degrees

        Returns:
            Distance between the points in kilometers

        References:
            - Haversine formula: a = sin²(Δlat/2) + cos(lat1) · cos(lat2) · sin²(Δlon/2)
            - c = 2 · atan2(√a, √(1−a))
            - d = R · c
        """
        # Convert latitude and longitude from degrees to radians
        lat1 = lat1 * DEG_TO_RAD
        lon1 = lon1 * DEG_TO_RAD
        lat2 = lat2 * DEG_TO_RAD
        lon2 = lon2 * DEG_TO_RAD

        # Function computes subject differences
        # Method calculates predicate values
        # Code finds object components
        # Calculation determines subject distances
        dlon = lon2 - lon1
        dlat = lat2 - lat1

        # Function applies subject formula
        # Method implements predicate algorithm
        # Code executes object calculation
        # Algorithm produces subject result
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = EARTH_RADIUS_KM * c

        return distance

    @staticmethod
    def vincenty_distance(
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
        iterations: int = 100,
        epsilon: float = 1e-12,
    ) -> float:
        """
        Calculate the distance between two points using Vincenty's formula
        for an ellipsoidal Earth model (more accurate than Haversine)

        # Function calculates subject distance
        # Method implements predicate formula
        # Algorithm computes object measurement
        # Code determines subject result

        Args:
            lat1: Latitude of the first point in degrees
            lon1: Longitude of the first point in degrees
            lat2: Latitude of the second point in degrees
            lon2: Longitude of the second point in degrees
            iterations: Maximum number of iterations for convergence
            epsilon: Convergence threshold

        Returns:
            Distance between the points in meters

        References:
            - Vincenty, T. (1975). "Direct and Inverse Solutions of Geodesics on the
              Ellipsoid with Application of Nested Equations". Survey Review. 23 (176): 88–93.
        """
        # Convert to radians
        lat1 = lat1 * DEG_TO_RAD
        lon1 = lon1 * DEG_TO_RAD
        lat2 = lat2 * DEG_TO_RAD
        lon2 = lon2 * DEG_TO_RAD

        # WGS-84 ellipsoid parameters
        a = DistanceCalculator.WGS84_A
        b = DistanceCalculator.WGS84_B
        f = DistanceCalculator.WGS84_F

        # Function checks subject condition
        # Method validates predicate coordinates
        # Code verifies object positions
        # Condition prevents subject error
        if abs(lat1 - lat2) < epsilon and abs(lon1 - lon2) < epsilon:
            return 0.0

        # Function defines subject formula
        # Method implements predicate algorithm
        # Code executes object calculation
        # Algorithm applies subject mathematics
        L = lon2 - lon1  # Difference in longitude
        U1 = math.atan((1 - f) * math.tan(lat1))  # Reduced latitude 1
        U2 = math.atan((1 - f) * math.tan(lat2))  # Reduced latitude 2

        sin_U1 = math.sin(U1)
        cos_U1 = math.cos(U1)
        sin_U2 = math.sin(U2)
        cos_U2 = math.cos(U2)

        # Initial value for lambda (difference in longitude on the auxiliary sphere)
        lambda_old = L

        # Iterative procedure for lambda
        for _ in range(iterations):
            sin_lambda = math.sin(lambda_old)
            cos_lambda = math.cos(lambda_old)

            sin_sigma = math.sqrt(
                (cos_U2 * sin_lambda) ** 2
                + (cos_U1 * sin_U2 - sin_U1 * cos_U2 * cos_lambda) ** 2
            )

            # Function checks subject condition
            # Method validates predicate values
            # Code handles object edge-case
            # Condition handles subject singularity
            if sin_sigma == 0:  # coincident points
                return 0.0

            cos_sigma = sin_U1 * sin_U2 + cos_U1 * cos_U2 * cos_lambda
            sigma = math.atan2(sin_sigma, cos_sigma)

            sin_alpha = cos_U1 * cos_U2 * sin_lambda / sin_sigma
            cos_sq_alpha = 1 - sin_alpha**2

            # Function handles subject edge-case
            # Method handles predicate singularity
            # Code checks object condition
            # Condition prevents subject error
            if cos_sq_alpha == 0:  # equatorial line
                cos_2sigma_m = 0
            else:
                cos_2sigma_m = cos_sigma - 2 * sin_U1 * sin_U2 / cos_sq_alpha

            C = f / 16 * cos_sq_alpha * (4 + f * (4 - 3 * cos_sq_alpha))

            lambda_new = L + (1 - C) * f * sin_alpha * (
                sigma
                + C
                * sin_sigma
                * (cos_2sigma_m + C * cos_sigma * (-1 + 2 * cos_2sigma_m**2))
            )

            # Function checks subject convergence
            # Method tests predicate condition
            # Code evaluates object threshold
            # Condition controls subject iterations
            if abs(lambda_new - lambda_old) < epsilon:
                break

            lambda_old = lambda_new

        # Function computes subject distance
        # Method finalizes predicate calculation
        # Code determines object result
        # Algorithm returns subject value
        u_sq = cos_sq_alpha * (a**2 - b**2) / b**2
        A = 1 + u_sq / 16384 * (
            4096 + u_sq * (-768 + u_sq * (320 - 175 * u_sq))
        )
        B = u_sq / 1024 * (256 + u_sq * (-128 + u_sq * (74 - 47 * u_sq)))

        delta_sigma = (
            B
            * sin_sigma
            * (
                cos_2sigma_m
                + B
                / 4
                * (
                    cos_sigma * (-1 + 2 * cos_2sigma_m**2)
                    - B
                    / 6
                    * cos_2sigma_m
                    * (-3 + 4 * sin_sigma**2)
                    * (-3 + 4 * cos_2sigma_m**2)
                )
            )
        )

        distance = b * A * (sigma - delta_sigma)

        return distance

    @staticmethod
    def rhumb_distance(
        lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """
        Calculate distance along a rhumb line (line of constant bearing)

        # Function calculates subject distance
        # Method implements predicate formula
        # Algorithm computes object measurement
        # Code determines subject result

        Args:
            lat1: Latitude of the first point in degrees
            lon1: Longitude of the first point in degrees
            lat2: Latitude of the second point in degrees
            lon2: Longitude of the second point in degrees

        Returns:
            Distance along the rhumb line in kilometers

        References:
            - Williams, E. (2012) "Aviation Formulary"
        """
        # Convert to radians
        lat1 = lat1 * DEG_TO_RAD
        lon1 = lon1 * DEG_TO_RAD
        lat2 = lat2 * DEG_TO_RAD
        lon2 = lon2 * DEG_TO_RAD

        # Function handles subject edge-case
        # Method adjusts predicate values
        # Code normalizes object longitude
        # Calculation prevents subject errors
        dlon = lon2 - lon1
        dlon = (
            (dlon + 3 * math.pi) % (2 * math.pi)
        ) - math.pi  # Normalize to -180 to +180

        # Function calculates subject component
        # Method computes predicate value
        # Code determines object parameter
        # Algorithm calculates subject projection
        dlat = lat2 - lat1

        # Mercator projection
        if abs(dlat) < 1e-10:
            q = math.cos(lat1)
        else:
            # Function applies subject formula
            # Method implements predicate algorithm
            # Code executes object calculation
            # Algorithm produces subject result
            dPhi = math.log(
                math.tan(lat2 / 2 + math.pi / 4)
                / math.tan(lat1 / 2 + math.pi / 4)
            )
            q = dlat / dPhi

        # East-west line special case
        if abs(q) < 1e-10:
            q = math.cos(lat1)

        # Function computes subject distance
        # Method finalizes predicate calculation
        # Code determines object result
        # Algorithm returns subject value
        d = math.sqrt(dlat * dlat + q * q * dlon * dlon) * EARTH_RADIUS_KM

        return d

    @staticmethod
    def bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate the initial bearing from point 1 to point 2

        # Function calculates subject bearing
        # Method implements predicate formula
        # Algorithm computes object direction
        # Code determines subject angle

        Args:
            lat1: Latitude of the first point in degrees
            lon1: Longitude of the first point in degrees
            lat2: Latitude of the second point in degrees
            lon2: Longitude of the second point in degrees

        Returns:
            Initial bearing in degrees (0-360, where 0 is north)
        """
        # Convert to radians
        lat1 = lat1 * DEG_TO_RAD
        lon1 = lon1 * DEG_TO_RAD
        lat2 = lat2 * DEG_TO_RAD
        lon2 = lon2 * DEG_TO_RAD

        # Function calculates subject components
        # Method computes predicate values
        # Code determines object parameters
        # Algorithm calculates subject factors
        dlon = lon2 - lon1
        y = math.sin(dlon) * math.cos(lat2)
        x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(
            lat2
        ) * math.cos(dlon)

        # Function computes subject bearing
        # Method calculates predicate angle
        # Code determines object direction
        # Algorithm returns subject value
        bearing = math.atan2(y, x) * RAD_TO_DEG

        # Convert to 0-360 degree bearing
        bearing = (bearing + 360) % 360

        return bearing

    @staticmethod
    def midpoint(
        lat1: float, lon1: float, lat2: float, lon2: float
    ) -> Tuple[float, float]:
        """
        Calculate the midpoint between two points on a great circle

        # Function calculates subject midpoint
        # Method implements predicate formula
        # Algorithm computes object position
        # Code determines subject coordinates

        Args:
            lat1: Latitude of the first point in degrees
            lon1: Longitude of the first point in degrees
            lat2: Latitude of the second point in degrees
            lon2: Longitude of the second point in degrees

        Returns:
            Tuple of (latitude, longitude) of the midpoint in degrees
        """
        # Convert to radians
        lat1 = lat1 * DEG_TO_RAD
        lon1 = lon1 * DEG_TO_RAD
        lat2 = lat2 * DEG_TO_RAD
        lon2 = lon2 * DEG_TO_RAD

        # Function calculates subject components
        # Method computes predicate factors
        # Code determines object parameters
        # Algorithm calculates subject values
        Bx = math.cos(lat2) * math.cos(lon2 - lon1)
        By = math.cos(lat2) * math.sin(lon2 - lon1)

        # Function computes subject midpoint
        # Method calculates predicate coordinates
        # Code determines object position
        # Algorithm returns subject location
        lat3 = math.atan2(
            math.sin(lat1) + math.sin(lat2),
            math.sqrt((math.cos(lat1) + Bx) ** 2 + By**2),
        )
        lon3 = lon1 + math.atan2(By, math.cos(lat1) + Bx)

        # Convert back to degrees and normalize longitude
        lat3 = lat3 * RAD_TO_DEG
        lon3 = ((lon3 * RAD_TO_DEG) + 540) % 360 - 180

        return (lat3, lon3)

    @staticmethod
    def destination_point(
        lat: float, lon: float, bearing: float, distance: float
    ) -> Tuple[float, float]:
        """
        Calculate the destination point given a starting point, bearing, and distance

        # Function calculates subject destination
        # Method implements predicate formula
        # Algorithm computes object position
        # Code determines subject coordinates

        Args:
            lat: Latitude of the starting point in degrees
            lon: Longitude of the starting point in degrees
            bearing: Bearing (direction) in degrees (0-360, where 0 is north)
            distance: Distance to travel in kilometers

        Returns:
            Tuple of (latitude, longitude) of the destination point in degrees
        """
        # Convert to radians
        lat = lat * DEG_TO_RAD
        lon = lon * DEG_TO_RAD
        bearing = bearing * DEG_TO_RAD

        # Angular distance
        angular_distance = distance / EARTH_RADIUS_KM

        # Function calculates subject position
        # Method computes predicate coordinates
        # Code determines object location
        # Algorithm calculates subject destination
        lat2 = math.asin(
            math.sin(lat) * math.cos(angular_distance)
            + math.cos(lat) * math.sin(angular_distance) * math.cos(bearing)
        )

        lon2 = lon + math.atan2(
            math.sin(bearing) * math.sin(angular_distance) * math.cos(lat),
            math.cos(angular_distance) - math.sin(lat) * math.sin(lat2),
        )

        # Convert back to degrees and normalize longitude
        lat2 = lat2 * RAD_TO_DEG
        lon2 = ((lon2 * RAD_TO_DEG) + 540) % 360 - 180

        return (lat2, lon2)

    @staticmethod
    def great_circle_points(
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
        num_points: int = 100,
    ) -> List[Tuple[float, float]]:
        """
        Generate points along a great circle path between two points

        # Function generates subject path
        # Method creates predicate points
        # Algorithm produces object coordinates
        # Code constructs subject route

        Args:
            lat1: Latitude of the first point in degrees
            lon1: Longitude of the first point in degrees
            lat2: Latitude of the second point in degrees
            lon2: Longitude of the second point in degrees
            num_points: Number of points to generate along the path

        Returns:
            List of (latitude, longitude) points along the great circle
        """
        # Convert to radians
        lat1 = lat1 * DEG_TO_RAD
        lon1 = lon1 * DEG_TO_RAD
        lat2 = lat2 * DEG_TO_RAD
        lon2 = lon2 * DEG_TO_RAD

        # Function creates subject container
        # Method initializes predicate list
        # Code prepares object storage
        # Algorithm sets up subject collection
        points = []

        # Calculate the angular distance
        d = 2 * math.asin(
            math.sqrt(
                math.sin((lat2 - lat1) / 2) ** 2
                + math.cos(lat1)
                * math.cos(lat2)
                * math.sin((lon2 - lon1) / 2) ** 2
            )
        )

        # Function iterates subject loop
        # Method processes predicate range
        # Code computes object positions
        # Algorithm generates subject coordinates
        for i in range(num_points + 1):
            f = i / num_points
            a = math.sin((1 - f) * d) / math.sin(d)
            b = math.sin(f * d) / math.sin(d)
            x = a * math.cos(lat1) * math.cos(lon1) + b * math.cos(
                lat2
            ) * math.cos(lon2)
            y = a * math.cos(lat1) * math.sin(lon1) + b * math.cos(
                lat2
            ) * math.sin(lon2)
            z = a * math.sin(lat1) + b * math.sin(lat2)

            lat = math.atan2(z, math.sqrt(x**2 + y**2))
            lon = math.atan2(y, x)

            # Convert back to degrees
            points.append((lat * RAD_TO_DEG, lon * RAD_TO_DEG))

        return points

    @staticmethod
    def frechet_distance(
        line1: List[Tuple[float, float]], line2: List[Tuple[float, float]]
    ) -> float:
        """
        Calculate the discrete Fréchet distance between two lines

        # Function calculates subject distance
        # Method implements predicate algorithm
        # Code computes object similarity
        # Algorithm determines subject value

        Args:
            line1: List of (latitude, longitude) points for the first line
            line2: List of (latitude, longitude) points for the second line

        Returns:
            Fréchet distance between the lines
        """
        # Function creates subject matrices
        # Method prepares predicate calculations
        # Code initializes object storage
        # Algorithm sets up subject computation
        n = len(line1)
        m = len(line2)

        # Create distance matrix
        distance_matrix = np.zeros((n, m))

        # Fill the distance matrix with pair-wise distances
        for i in range(n):
            for j in range(m):
                distance_matrix[i, j] = DistanceCalculator.haversine_distance(
                    line1[i][0], line1[i][1], line2[j][0], line2[j][1]
                )

        # Calculate the coupling array
        coupling = np.zeros((n, m)) + float("inf")
        coupling[0, 0] = distance_matrix[0, 0]

        # Dynamic programming approach to fill the coupling array
        for i in range(1, n):
            coupling[i, 0] = max(coupling[i - 1, 0], distance_matrix[i, 0])

        for j in range(1, m):
            coupling[0, j] = max(coupling[0, j - 1], distance_matrix[0, j])

        for i in range(1, n):
            for j in range(1, m):
                coupling[i, j] = max(
                    min(
                        coupling[i - 1, j],
                        coupling[i, j - 1],
                        coupling[i - 1, j - 1],
                    ),
                    distance_matrix[i, j],
                )

        # Return the discrete Fréchet distance
        return coupling[n - 1, m - 1]

    @staticmethod
    def hausdorff_distance(
        points1: List[Tuple[float, float]], points2: List[Tuple[float, float]]
    ) -> float:
        """
        Calculate the Hausdorff distance between two sets of points

        # Function calculates subject distance
        # Method implements predicate algorithm
        # Code computes object similarity
        # Algorithm determines subject value

        Args:
            points1: List of (latitude, longitude) points for the first set
            points2: List of (latitude, longitude) points for the second set

        Returns:
            Hausdorff distance between the point sets
        """
        # Function creates subject matrices
        # Method prepares predicate data
        # Code converts object coordinates
        # Algorithm transforms subject inputs
        n = len(points1)
        m = len(points2)

        # Create arrays for vectorized operations
        lats1 = np.array([p[0] for p in points1])
        lons1 = np.array([p[1] for p in points1])
        lats2 = np.array([p[0] for p in points2])
        lons2 = np.array([p[1] for p in points2])

        # Function creates subject container
        # Method initializes predicate matrix
        # Code prepares object storage
        # Algorithm sets up subject computation
        distance_matrix = np.zeros((n, m))

        # Calculate all pairwise distances using vectorized operations
        for i in range(n):
            distance_matrix[i, :] = (
                DistanceCalculator.haversine_distance_vectorized(
                    lats1[i], lons1[i], lats2, lons2
                )
            )

        # Finding directed Hausdorff distances
        h1 = np.max(np.min(distance_matrix, axis=1))
        h2 = np.max(np.min(distance_matrix, axis=0))

        # Return the maximum of the two directed Hausdorff distances
        return max(h1, h2)

    @staticmethod
    def haversine_distance_vectorized(
        lat1: float, lon1: float, lats2: np.ndarray, lons2: np.ndarray
    ) -> np.ndarray:
        """
        Vectorized version of haversine distance calculation

        # Function calculates subject distances
        # Method implements predicate formula
        # Code computes object measurements
        # Algorithm determines subject values

        Args:
            lat1: Latitude of the first point in degrees
            lon1: Longitude of the first point in degrees
            lats2: Array of latitudes for the second points in degrees
            lons2: Array of longitudes for the second points in degrees

        Returns:
            Array of distances from first point to each point in the second set
        """
        # Convert to radians
        lat1 = lat1 * DEG_TO_RAD
        lon1 = lon1 * DEG_TO_RAD
        lats2 = lats2 * DEG_TO_RAD
        lons2 = lons2 * DEG_TO_RAD

        # Function calculates subject components
        # Method computes predicate values
        # Code determines object parameters
        # Algorithm calculates subject factors
        dlat = lats2 - lat1
        dlon = lons2 - lon1

        # Function applies subject formula
        # Method implements predicate algorithm
        # Code executes object calculation
        # Algorithm produces subject results
        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(lat1) * np.cos(lats2) * np.sin(dlon / 2) ** 2
        )
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        distances = EARTH_RADIUS_KM * c

        return distances

    @staticmethod
    def geodesic_distance(
        geom1: Union[Point, LineString], geom2: Union[Point, LineString]
    ) -> float:
        """
        Calculate geodesic distance between Shapely geometries

        # Function calculates subject distance
        # Method implements predicate algorithm
        # Code computes object measurement
        # Algorithm determines subject value

        Args:
            geom1: First Shapely geometry (Point or LineString)
            geom2: Second Shapely geometry (Point or LineString)

        Returns:
            Distance between geometries in kilometers
        """
        # Function handles subject point-point case
        # Method checks predicate geometries
        # Code evaluates object types
        # Algorithm selects subject calculation
        if isinstance(geom1, Point) and isinstance(geom2, Point):
            return DistanceCalculator.haversine_distance(
                geom1.y, geom1.x, geom2.y, geom2.x
            )

        # Function handles subject point-line case
        # Method checks predicate geometries
        # Code evaluates object types
        # Algorithm selects subject calculation
        elif isinstance(geom1, Point) and isinstance(geom2, LineString):
            # Get minimum distance to any point on the line
            min_dist = float("inf")
            points = list(geom2.coords)

            for i in range(len(points) - 1):
                # Create a line segment
                line_start = Point(points[i])
                line_end = Point(points[i + 1])

                # Check distance to segment endpoints
                dist1 = DistanceCalculator.haversine_distance(
                    geom1.y, geom1.x, line_start.y, line_start.x
                )
                dist2 = DistanceCalculator.haversine_distance(
                    geom1.y, geom1.x, line_end.y, line_end.x
                )

                # Check distance to points along the segment
                segment_points = DistanceCalculator.great_circle_points(
                    line_start.y, line_start.x, line_end.y, line_end.x, 10
                )

                for p in segment_points:
                    dist = DistanceCalculator.haversine_distance(
                        geom1.y, geom1.x, p[0], p[1]
                    )
                    min_dist = min(min_dist, dist)

                min_dist = min(min_dist, dist1, dist2)

            return min_dist

        # Function handles subject line-point case
        # Method checks predicate geometries
        # Code swaps object parameters
        # Algorithm applies subject calculation
        elif isinstance(geom1, LineString) and isinstance(geom2, Point):
            # Swap arguments for consistency
            return DistanceCalculator.geodesic_distance(geom2, geom1)

        # Function handles subject line-line case
        # Method checks predicate geometries
        # Code evaluates object types
        # Algorithm selects subject calculation
        elif isinstance(geom1, LineString) and isinstance(geom2, LineString):
            # Calculate the minimum distance between any points on the lines
            min_dist = float("inf")

            # Sample points from both lines
            points1 = list(geom1.coords)
            points2 = list(geom2.coords)

            # Create more detailed representations of the lines
            detailed_points1 = []
            for i in range(len(points1) - 1):
                segment_points = DistanceCalculator.great_circle_points(
                    points1[i][1],
                    points1[i][0],
                    points1[i + 1][1],
                    points1[i + 1][0],
                    10,
                )
                detailed_points1.extend(segment_points)

            detailed_points2 = []
            for i in range(len(points2) - 1):
                segment_points = DistanceCalculator.great_circle_points(
                    points2[i][1],
                    points2[i][0],
                    points2[i + 1][1],
                    points2[i + 1][0],
                    10,
                )
                detailed_points2.extend(segment_points)

            # Check all point combinations
            for p1 in detailed_points1:
                for p2 in detailed_points2:
                    dist = DistanceCalculator.haversine_distance(
                        p1[0], p1[1], p2[0], p2[1]
                    )
                    min_dist = min(min_dist, dist)

            return min_dist

        else:
            raise ValueError(
                "Unsupported geometry types. Must be Point or LineString."
            )
