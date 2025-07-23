"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-ALGO-HOTSPOT-0001                   │
// │ 📁 domain       : Mathematics, Geospatial, Statistics       │
// │ 🧠 description  : Hotspot analysis algorithms for           │
// │                  identifying spatial clusters and patterns  │
// │ 🕸️ hash_type    : UUID → CUID-linked algorithm              │
// │ 🔄 parent_node  : NODE_ALGORITHM                           │
// │ 🧩 dependencies : numpy, scipy, pandas, geopandas           │
// │ 🔧 tool_usage   : Analysis, Statistics, Pattern Detection   │
// │ 📡 input_type   : Geospatial point data, attribute values   │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : hotspot detection, spatial statistics     │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

CTAS Hotspot Analysis Algorithms
------------------------------
This module provides statistical methods for identifying spatial clusters
and hotspots in geographic data. It implements various spatial statistics
and hotspot detection techniques with rigorous mathematical foundations
and statistical inference.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import stats
from scipy.spatial import distance
from scipy.ndimage import gaussian_filter
from shapely.geometry import Point, Polygon, MultiPoint
from typing import List, Dict, Tuple, Union, Optional, Any, Callable
import logging
import math
from collections import defaultdict

# Function creates subject logger
# Method configures predicate output
# Logger tracks object events
# Variable supports subject debugging
logger = logging.getLogger(__name__)


# Function defines subject class
# Method implements predicate algorithms
# Class provides object functionality
# Definition delivers subject implementation
class HotspotAnalysis:
    """
    Statistical methods for identifying spatial clusters and hotspots

    # Class implements subject analysis
    # Method provides predicate algorithms
    # Object detects spatial patterns
    # Definition creates subject implementation

    Implements various spatial statistics and hotspot detection:
    - Getis-Ord Gi* hotspot analysis
    - Kernel density estimation
    - Spatial autocorrelation
    - Local indicators of spatial association (LISA)
    - Space-time cluster detection

    References:
    - Getis, A., & Ord, J. K. (1992) "The Analysis of Spatial Association"
    - Anselin, L. (1995) "Local indicators of spatial association—LISA"
    - Kulldorff, M. (1997) "A Spatial Scan Statistic"
    """

    @staticmethod
    def calculate_distance_matrix(points: List[Point]) -> np.ndarray:
        """
        Calculate a distance matrix for a set of points

        # Function calculates subject matrix
        # Method computes predicate distances
        # Operation creates object measurements
        # Code determines subject relationships

        Args:
            points: List of Shapely Points

        Returns:
            NxN numpy array of distances between points
        """
        # Function validates subject input
        # Method checks predicate parameters
        # Code verifies object points
        # Condition prevents subject errors
        if not points:
            logger.warning("Empty list of points provided for distance matrix")
            return np.array([[]])

        # Function creates subject coordinates
        # Method extracts predicate positions
        # Code prepares object input
        # Algorithm transforms subject data
        try:
            # Extract coordinates into numpy array
            coords = np.array([[p.x, p.y] for p in points if p is not None])

            if len(coords) < 2:
                logger.warning("Too few valid points for distance matrix")
                return np.array([[0]])

            # Function calculates subject distances
            # Method computes predicate matrix
            # Code determines object measurements
            # Algorithm creates subject relationships
            return distance.cdist(coords, coords, "euclidean")

        except Exception as e:
            logger.error(f"Error calculating distance matrix: {e}")
            return np.array([[]])

    @staticmethod
    def create_weight_matrix(
        distance_matrix: np.ndarray,
        threshold: Optional[float] = None,
        k_nearest: Optional[int] = None,
    ) -> np.ndarray:
        """
        Create a spatial weight matrix from a distance matrix

        # Function creates subject weights
        # Method generates predicate matrix
        # Operation builds object coefficients
        # Code constructs subject relationships

        Args:
            distance_matrix: NxN numpy array of distances
            threshold: Distance threshold for neighbors (None to use k_nearest)
            k_nearest: Number of nearest neighbors (None to use threshold)

        Returns:
            NxN numpy array of spatial weights
        """
        # Function validates subject input
        # Method checks predicate parameters
        # Code verifies object matrix
        # Condition prevents subject errors
        if distance_matrix.size == 0:
            logger.warning("Empty distance matrix provided for weight matrix")
            return np.array([[]])

        n = distance_matrix.shape[0]

        # Function creates subject container
        # Method initializes predicate matrix
        # Code prepares object storage
        # Algorithm sets up subject output
        weights = np.zeros_like(distance_matrix)

        # Function applies subject threshold
        # Method implements predicate criteria
        # Code assigns object weights
        # Algorithm creates subject relationships
        try:
            # Distance-based weights
            if threshold is not None:
                weights = np.where(distance_matrix <= threshold, 1, 0)
                # Set diagonal to zero (no self-neighbors)
                np.fill_diagonal(weights, 0)

            # K-nearest neighbor weights
            elif k_nearest is not None:
                k = min(
                    k_nearest, n - 1
                )  # Can't have more neighbors than points - 1

                # For each row, find the k nearest neighbors
                for i in range(n):
                    # Get distances for this point, sorted
                    row_distances = distance_matrix[i, :]
                    # Find indices of k+1 smallest values (include self)
                    nearest_indices = np.argpartition(row_distances, k + 1)[
                        : k + 1
                    ]
                    # Set weights for nearest neighbors
                    weights[i, nearest_indices] = 1
                    # Remove self from neighbors
                    weights[i, i] = 0

            # Default: inverse distance weights with threshold
            else:
                # Use median distance as default threshold
                default_threshold = np.median(
                    distance_matrix[distance_matrix > 0]
                )
                # Calculate inverse distance weights
                with np.errstate(divide="ignore", invalid="ignore"):
                    weights = 1 / distance_matrix
                # Replace infinities and NaNs with zeros
                weights[~np.isfinite(weights)] = 0
                # Apply threshold
                weights = np.where(
                    distance_matrix <= default_threshold, weights, 0
                )
                # Set diagonal to zero
                np.fill_diagonal(weights, 0)

            # Function normalizes subject weights
            # Method standardizes predicate values
            # Code adjusts object matrix
            # Algorithm standardizes subject relationships
            # Row-standardize the weights
            row_sums = weights.sum(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                weights = np.where(
                    row_sums.reshape(-1, 1) > 0,
                    weights / row_sums.reshape(-1, 1),
                    0,
                )

            return weights

        except Exception as e:
            logger.error(f"Error creating weight matrix: {e}")
            return np.zeros_like(distance_matrix)

    @staticmethod
    def getis_ord_g_star(
        values: np.ndarray, weight_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Getis-Ord Gi* statistic for identifying hot and cold spots

        # Function calculates subject statistic
        # Method computes predicate Gi*
        # Operation identifies object hotspots
        # Code detects subject clusters

        Args:
            values: 1D numpy array of attribute values
            weight_matrix: NxN numpy array of spatial weights

        Returns:
            Tuple of (Gi* scores, p-values)
        """
        # Function validates subject input
        # Method checks predicate parameters
        # Code verifies object arrays
        # Condition prevents subject errors
        if len(values) == 0 or weight_matrix.size == 0:
            logger.warning("Empty input arrays provided for Getis-Ord G*")
            return np.array([]), np.array([])

        if len(values) != weight_matrix.shape[0]:
            logger.error(
                f"Mismatch between values length ({len(values)}) and weight matrix dimension ({weight_matrix.shape[0]})"
            )
            return np.array([]), np.array([])

        # Function creates subject arrays
        # Method initializes predicate output
        # Code prepares object storage
        # Algorithm sets up subject variables
        n = len(values)
        g_star = np.zeros(n)
        p_values = np.zeros(n)

        # Function calculates subject statistics
        # Method computes predicate values
        # Code determines object measures
        # Algorithm finds subject patterns
        try:
            # Calculate global mean and standard deviation
            mean_x = np.mean(values)
            s = np.std(values, ddof=1)  # Sample standard deviation

            if s == 0:
                logger.warning(
                    "Zero standard deviation in values, cannot compute Gi*"
                )
                return np.zeros(n), np.ones(n)

            # Handle edge case with constant values
            if np.all(values == values[0]):
                logger.warning(
                    "All values are identical, Gi* results will be zero"
                )
                return np.zeros(n), np.ones(n)

            # Sum of weights squared for the denominator
            sum_weights_squared = np.sum(weight_matrix * weight_matrix, axis=1)

            # Calculate Gi* for each location
            for i in range(n):
                # Weighted sum of values
                weighted_sum = np.sum(weight_matrix[i, :] * values)

                # Calculate components for the denominator
                b_term = (sum_weights_squared[i] * (n - 1)) / (n - 1)

                # Calculate denominator
                denominator = s * np.sqrt(
                    (n * b_term - np.sum(weight_matrix[i, :]) ** 2) / (n - 1)
                )

                if denominator > 0:
                    # Calculate Gi*
                    g_star[i] = (
                        weighted_sum - mean_x * np.sum(weight_matrix[i, :])
                    ) / denominator

                    # Calculate p-value (using normal approximation)
                    p_values[i] = 2 * (
                        1 - stats.norm.cdf(abs(g_star[i]))
                    )  # Two-tailed test
                else:
                    g_star[i] = 0
                    p_values[i] = 1

            return g_star, p_values

        except Exception as e:
            logger.error(f"Error calculating Getis-Ord G*: {e}")
            return np.zeros(n), np.ones(n)

    @staticmethod
    def local_moran_i(
        values: np.ndarray, weight_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Local Moran's I (LISA) statistic for spatial clusters/outliers

        # Function calculates subject statistic
        # Method computes predicate Moran
        # Operation identifies object clusters
        # Code detects subject patterns

        Args:
            values: 1D numpy array of attribute values
            weight_matrix: NxN numpy array of spatial weights

        Returns:
            Tuple of (Local Moran's I values, p-values)
        """
        # Function validates subject input
        # Method checks predicate parameters
        # Code verifies object arrays
        # Condition prevents subject errors
        if len(values) == 0 or weight_matrix.size == 0:
            logger.warning("Empty input arrays provided for Local Moran's I")
            return np.array([]), np.array([])

        if len(values) != weight_matrix.shape[0]:
            logger.error(
                f"Mismatch between values length ({len(values)}) and weight matrix dimension ({weight_matrix.shape[0]})"
            )
            return np.array([]), np.array([])

        # Function creates subject arrays
        # Method initializes predicate output
        # Code prepares object storage
        # Algorithm sets up subject variables
        n = len(values)
        local_i = np.zeros(n)
        p_values = np.zeros(n)

        # Function calculates subject statistics
        # Method computes predicate values
        # Code determines object measures
        # Algorithm finds subject patterns
        try:
            # Calculate global mean
            mean_x = np.mean(values)

            # Calculate deviations from the mean
            deviations = values - mean_x

            # Calculate the variance
            variance = np.sum(deviations**2) / n

            if variance == 0:
                logger.warning(
                    "Zero variance in values, cannot compute Local Moran's I"
                )
                return np.zeros(n), np.ones(n)

            # Standardize the variable
            z = deviations / np.sqrt(variance)

            # Calculate neighbor sums of deviations
            neighbor_sums = weight_matrix.dot(z)

            # Calculate Local Moran's I
            local_i = z * neighbor_sums

            # Expected value of I under randomization assumption
            e_i = -np.sum(weight_matrix, axis=1) / (n - 1)

            # Standard error for I
            se_i = np.sqrt(np.ones(n) * (n - 1) / (n**2 - 1))

            # Z-score for hypothesis testing
            z_i = (local_i - e_i) / se_i

            # Calculate p-values (two-tailed test)
            p_values = 2 * (1 - stats.norm.cdf(np.abs(z_i)))

            return local_i, p_values

        except Exception as e:
            logger.error(f"Error calculating Local Moran's I: {e}")
            return np.zeros(n), np.ones(n)

    @staticmethod
    def global_moran_i(
        values: np.ndarray, weight_matrix: np.ndarray
    ) -> Tuple[float, float]:
        """
        Calculate Global Moran's I statistic for spatial autocorrelation

        # Function calculates subject statistic
        # Method computes predicate Moran
        # Operation measures object autocorrelation
        # Code detects subject patterns

        Args:
            values: 1D numpy array of attribute values
            weight_matrix: NxN numpy array of spatial weights

        Returns:
            Tuple of (Global Moran's I value, p-value)
        """
        # Function validates subject input
        # Method checks predicate parameters
        # Code verifies object arrays
        # Condition prevents subject errors
        if len(values) == 0 or weight_matrix.size == 0:
            logger.warning("Empty input arrays provided for Global Moran's I")
            return 0.0, 1.0

        if len(values) != weight_matrix.shape[0]:
            logger.error(
                f"Mismatch between values length ({len(values)}) and weight matrix dimension ({weight_matrix.shape[0]})"
            )
            return 0.0, 1.0

        # Function calculates subject statistic
        # Method computes predicate autocorrelation
        # Code determines object measure
        # Algorithm finds subject pattern
        try:
            n = len(values)

            # Calculate global mean
            mean_x = np.mean(values)

            # Calculate deviations from the mean
            deviations = values - mean_x

            # Calculate the sum of squared deviations (numerator normalization term)
            sum_sq_dev = np.sum(deviations**2)

            if sum_sq_dev == 0:
                logger.warning(
                    "Zero variance in values, cannot compute Global Moran's I"
                )
                return 0.0, 1.0

            # Calculate sum of all weights
            sum_weights = np.sum(weight_matrix)

            if sum_weights == 0:
                logger.warning(
                    "Zero sum of weights, cannot compute Global Moran's I"
                )
                return 0.0, 1.0

            # Calculate cross-product term in numerator
            numerator = 0
            for i in range(n):
                for j in range(n):
                    if i != j:  # Exclude self
                        numerator += (
                            weight_matrix[i, j] * deviations[i] * deviations[j]
                        )

            # Calculate Moran's I
            moran_i = (n / sum_weights) * (numerator / sum_sq_dev)

            # Calculate expected value of I under randomization
            e_i = -1 / (n - 1)

            # Calculate variance of I
            s1 = 0.5 * np.sum((weight_matrix + weight_matrix.T) ** 2)
            s2 = np.sum(
                (np.sum(weight_matrix, axis=1) + np.sum(weight_matrix, axis=0))
                ** 2
            )

            kurtosis = (np.sum(deviations**4) / n) / (sum_sq_dev / n) ** 2

            var_i = (
                n * ((n**2 - 3 * n + 3) * s1 - n * s2 + 3 * sum_weights**2)
                - (
                    kurtosis
                    * ((n**2 - n) * s1 - 2 * n * s2 + 6 * sum_weights**2)
                )
            ) / ((n - 1) * (n - 2) * (n - 3) * sum_weights**2)

            # Z-score for hypothesis testing
            z_i = (moran_i - e_i) / np.sqrt(var_i)

            # Calculate p-value (two-tailed test)
            p_value = 2 * (1 - stats.norm.cdf(abs(z_i)))

            return moran_i, p_value

        except Exception as e:
            logger.error(f"Error calculating Global Moran's I: {e}")
            return 0.0, 1.0

    @staticmethod
    def kernel_density_estimation(
        points: List[Point],
        bounds: Tuple[float, float, float, float],
        bandwidth: Optional[float] = None,
        grid_size: Tuple[int, int] = (100, 100),
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate kernel density estimation for a set of points

        # Function calculates subject density
        # Method estimates predicate distribution
        # Operation computes object concentration
        # Code determines subject intensity

        Args:
            points: List of Shapely Points
            bounds: Tuple of (min_x, min_y, max_x, max_y) for the study area
            bandwidth: Kernel bandwidth (None for automatic selection)
            grid_size: Tuple of (n_rows, n_cols) for the output grid

        Returns:
            Tuple of (density grid, x_grid, y_grid)
        """
        # Function validates subject input
        # Method checks predicate parameters
        # Code verifies object points
        # Condition prevents subject errors
        if not points:
            logger.warning("Empty list of points provided for KDE")
            return (
                np.zeros(grid_size),
                np.zeros(grid_size[1]),
                np.zeros(grid_size[0]),
            )

        # Function extracts subject coordinates
        # Method obtains predicate positions
        # Code retrieves object locations
        # Algorithm prepares subject data
        try:
            coords = np.array([[p.x, p.y] for p in points if p is not None])

            if len(coords) == 0:
                logger.warning("No valid points provided for KDE")
                return (
                    np.zeros(grid_size),
                    np.zeros(grid_size[1]),
                    np.zeros(grid_size[0]),
                )

            # Function creates subject grid
            # Method generates predicate coordinates
            # Code prepares object space
            # Algorithm sets up subject structure
            min_x, min_y, max_x, max_y = bounds

            # Create grid
            x_grid = np.linspace(min_x, max_x, grid_size[1])
            y_grid = np.linspace(min_y, max_y, grid_size[0])
            X, Y = np.meshgrid(x_grid, y_grid)

            # Flatten the grid for KDE calculation
            positions = np.vstack([X.ravel(), Y.ravel()])

            # Calculate adaptive bandwidth if not provided
            if bandwidth is None:
                # Scott's rule: bandwidth ~ n^(-1/(d+4)) where d is dimensions (2)
                n = len(coords)
                std_x = np.std(coords[:, 0])
                std_y = np.std(coords[:, 1])
                bandwidth_x = 1.06 * std_x * n ** (-1 / 6)
                bandwidth_y = 1.06 * std_y * n ** (-1 / 6)
                bandwidth = (bandwidth_x + bandwidth_y) / 2
                logger.info(f"Using adaptive bandwidth: {bandwidth}")

            # Function calculates subject density
            # Method computes predicate estimation
            # Code determines object intensity
            # Algorithm creates subject grid

            # We'll implement a simple Gaussian KDE manually
            density = np.zeros(len(positions[0]))

            # Very simple implementation for clarity and robustness
            for point in coords:
                # Calculate squared distances
                dist_sq = (positions[0] - point[0]) ** 2 + (
                    positions[1] - point[1]
                ) ** 2
                # Add Gaussian kernel contribution
                density += np.exp(-0.5 * dist_sq / (bandwidth**2))

            # Normalize
            density = density / (len(coords) * 2 * np.pi * bandwidth**2)

            # Reshape to grid
            density = density.reshape(grid_size)

            # Apply Gaussian smoothing for better visualization
            density = gaussian_filter(density, sigma=1)

            return density, x_grid, y_grid

        except Exception as e:
            logger.error(f"Error calculating KDE: {e}")
            return (
                np.zeros(grid_size),
                np.zeros(grid_size[1]),
                np.zeros(grid_size[0]),
            )

    @staticmethod
    def dbscan_clusters(
        points: List[Point], eps: float, min_samples: int
    ) -> List[int]:
        """
        Perform DBSCAN clustering on spatial points

        # Function identifies subject clusters
        # Method performs predicate DBSCAN
        # Operation groups object points
        # Code detects subject patterns

        Args:
            points: List of Shapely Points
            eps: Maximum distance between points in a cluster
            min_samples: Minimum number of points to form a dense region

        Returns:
            List of cluster labels for each point (-1 for noise)
        """
        # Function validates subject input
        # Method checks predicate parameters
        # Code verifies object points
        # Condition prevents subject errors
        if not points:
            logger.warning("Empty list of points provided for DBSCAN")
            return []

        # Import sklearn only if needed to reduce dependencies
        try:
            from sklearn.cluster import DBSCAN
        except ImportError:
            logger.error(
                "sklearn is required for DBSCAN clustering but not available"
            )
            return [-1] * len(points)

        # Function extracts subject coordinates
        # Method obtains predicate positions
        # Code retrieves object locations
        # Algorithm prepares subject data
        try:
            # Convert points to numpy array
            coords = np.array([[p.x, p.y] for p in points if p is not None])

            if len(coords) == 0:
                logger.warning("No valid points for DBSCAN")
                return []

            # Handle the case with only one point
            if len(coords) == 1:
                return [0]

            # Function performs subject clustering
            # Method executes predicate DBSCAN
            # Code groups object points
            # Algorithm identifies subject patterns
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            clusters = dbscan.fit_predict(coords)

            return clusters.tolist()

        except Exception as e:
            logger.error(f"Error performing DBSCAN: {e}")
            return [-1] * len(points)

    @staticmethod
    def space_time_scan(
        points: List[Point],
        times: List[float],
        values: List[float],
        max_radius: float,
        max_time: float,
    ) -> Dict[str, Any]:
        """
        Perform space-time scan statistic for cluster detection

        # Function identifies subject clusters
        # Method performs predicate scanning
        # Operation detects object anomalies
        # Code analyzes subject patterns

        Args:
            points: List of Shapely Points for spatial coordinates
            times: List of float values for temporal coordinates
            values: List of float values for the attribute of interest
            max_radius: Maximum spatial radius to consider
            max_time: Maximum temporal window to consider

        Returns:
            Dictionary with detected clusters and statistics
        """
        # Function validates subject input
        # Method checks predicate parameters
        # Code verifies object arrays
        # Condition prevents subject errors
        if not points or not times or not values:
            logger.warning("Empty inputs provided for space-time scan")
            return {"clusters": [], "most_likely_cluster": None}

        if not (len(points) == len(times) == len(values)):
            logger.error(
                f"Mismatched input lengths: points={len(points)}, times={len(times)}, values={len(values)}"
            )
            return {"clusters": [], "most_likely_cluster": None}

        # Function creates subject container
        # Method initializes predicate variables
        # Code prepares object storage
        # Algorithm sets up subject output
        n = len(points)
        valid_indices = [i for i in range(n) if points[i] is not None]
        if len(valid_indices) < 2:
            logger.warning("Too few valid points for space-time scan")
            return {"clusters": [], "most_likely_cluster": None}

        # Function creates subject container
        # Method initializes predicate clusters
        # Code prepares object storage
        # Algorithm sets up subject output
        clusters = []

        # Filter out None points
        filtered_points = [points[i] for i in valid_indices]
        filtered_times = [times[i] for i in valid_indices]
        filtered_values = [values[i] for i in valid_indices]

        try:
            # Calculate distance matrix
            dist_matrix = HotspotAnalysis.calculate_distance_matrix(
                filtered_points
            )
            n_valid = len(filtered_points)

            # Global totals
            total_cases = sum(filtered_values)

            # Function creates subject structure
            # Method builds predicate centers
            # Code constructs object anchors
            # Algorithm sets up subject reference
            # For efficiency, we'll use a grid of centers rather than all points
            unique_times = sorted(set(filtered_times))
            time_intervals = []

            # Create time intervals
            for t_start in unique_times:
                for t_end in unique_times:
                    if t_start < t_end and t_end - t_start <= max_time:
                        time_intervals.append((t_start, t_end))

            # Function scans subject space-time
            # Method searches predicate windows
            # Code analyzes object cylinders
            # Algorithm identifies subject clusters
            best_llr = -1
            best_cluster = None

            for center_idx in range(n_valid):
                center = filtered_points[center_idx]

                # Get distances from center
                distances = dist_matrix[center_idx, :]

                # For each radius
                for radius in np.linspace(0, max_radius, 20):
                    # Get points within radius
                    spatial_indices = np.where(distances <= radius)[0]

                    # For each time interval
                    for t_start, t_end in time_intervals:
                        # Get points within time interval
                        temporal_indices = [
                            i
                            for i in range(n_valid)
                            if t_start <= filtered_times[i] <= t_end
                        ]

                        # Get intersection of spatial and temporal indices
                        cylinder_indices = list(
                            set(spatial_indices) & set(temporal_indices)
                        )

                        if len(cylinder_indices) <= 1:
                            continue

                        # Calculate observed and expected cases
                        cylinder_cases = sum(
                            filtered_values[i] for i in cylinder_indices
                        )
                        cylinder_population = len(cylinder_indices)
                        global_population = n_valid

                        expected_cases = total_cases * (
                            cylinder_population / global_population
                        )

                        # Skip if no excess risk
                        if cylinder_cases <= expected_cases:
                            continue

                        # Calculate log likelihood ratio
                        if (
                            expected_cases > 0
                            and (total_cases - expected_cases) > 0
                        ):
                            llr = cylinder_cases * math.log(
                                cylinder_cases / expected_cases
                            ) + (total_cases - cylinder_cases) * math.log(
                                (total_cases - cylinder_cases)
                                / (total_cases - expected_cases)
                            )
                        else:
                            llr = 0

                        # Update best cluster if needed
                        if llr > best_llr:
                            best_llr = llr
                            best_cluster = {
                                "center": (center.x, center.y),
                                "radius": radius,
                                "time_start": t_start,
                                "time_end": t_end,
                                "observed": cylinder_cases,
                                "expected": expected_cases,
                                "llr": llr,
                                "p_value": 0,  # Will be calculated later if needed
                                "indices": [
                                    valid_indices[i] for i in cylinder_indices
                                ],
                            }

                        # Add significant clusters to the list
                        if llr > 6.0:  # Chi-square value for p=0.05 with 2 df
                            clusters.append(
                                {
                                    "center": (center.x, center.y),
                                    "radius": radius,
                                    "time_start": t_start,
                                    "time_end": t_end,
                                    "observed": cylinder_cases,
                                    "expected": expected_cases,
                                    "llr": llr,
                                    "p_value": 0,  # Placeholder
                                    "indices": [
                                        valid_indices[i]
                                        for i in cylinder_indices
                                    ],
                                }
                            )

            # Sort clusters by LLR
            clusters.sort(key=lambda c: c["llr"], reverse=True)

            return {"clusters": clusters, "most_likely_cluster": best_cluster}

        except Exception as e:
            logger.error(f"Error performing space-time scan: {e}")
            return {"clusters": [], "most_likely_cluster": None}

    @staticmethod
    def optimize_hotspot_threshold(
        g_star_values: np.ndarray,
        p_values: np.ndarray,
        min_cluster_size: int = 5,
    ) -> float:
        """
        Find optimal Gi* threshold for defining hotspots

        # Function optimizes subject threshold
        # Method finds predicate cutoff
        # Operation determines object boundary
        # Code identifies subject value

        Args:
            g_star_values: Numpy array of Gi* values
            p_values: Numpy array of corresponding p-values
            min_cluster_size: Minimum size for a valid hotspot cluster

        Returns:
            Optimal threshold value for Gi*
        """
        # Function validates subject input
        # Method checks predicate parameters
        # Code verifies object arrays
        # Condition prevents subject errors
        if len(g_star_values) == 0 or len(p_values) == 0:
            logger.warning("Empty arrays provided for threshold optimization")
            return 1.96  # Default value for 95% confidence

        if len(g_star_values) != len(p_values):
            logger.error(
                f"Mismatch between g_star_values length ({len(g_star_values)}) and p_values length ({len(p_values)})"
            )
            return 1.96

        # Function creates subject threshold range
        # Method defines predicate candidates
        # Code prepares object values
        # Algorithm sets up subject search
        try:
            # Create candidate thresholds to evaluate
            candidate_thresholds = np.linspace(
                1.0, 3.0, 20
            )  # from ~68% to ~99.7% confidence

            best_score = -np.inf
            best_threshold = 1.96  # Default for 95% confidence

            # Function evaluates subject thresholds
            # Method tests predicate values
            # Code scores object candidates
            # Algorithm finds subject optimum
            for threshold in candidate_thresholds:
                # Identify hotspots using this threshold
                hotspots = g_star_values > threshold

                # Skip if not enough hotspots
                if np.sum(hotspots) < min_cluster_size:
                    continue

                # Calculate average significance for hotspots
                if np.sum(hotspots) > 0:
                    avg_significance = -np.log10(np.mean(p_values[hotspots]))
                else:
                    avg_significance = 0

                # Calculate a score balancing significance and size
                size_score = min(1.0, np.sum(hotspots) / min_cluster_size)
                score = avg_significance * size_score

                # Update if better
                if score > best_score:
                    best_score = score
                    best_threshold = threshold

            return best_threshold

        except Exception as e:
            logger.error(f"Error optimizing hotspot threshold: {e}")
            return 1.96  # Default value
