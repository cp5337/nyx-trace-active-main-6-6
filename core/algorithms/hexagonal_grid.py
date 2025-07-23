"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-ALGO-HEXGRID-0001                   │
// │ 📁 domain       : Mathematics, Geospatial, Tessellation     │
// │ 🧠 description  : Hexagonal grid systems for geospatial     │
// │                  indexing and spatial binning               │
// │ 🕸️ hash_type    : UUID → CUID-linked algorithm              │
// │ 🔄 parent_node  : NODE_ALGORITHM                           │
// │ 🧩 dependencies : numpy, h3, shapely                        │
// │ 🔧 tool_usage   : Analysis, Mathematics, Indexing           │
// │ 📡 input_type   : Coordinates, regions, areas               │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : spatial binning, hierarchical indexing    │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

CTAS Hexagonal Grid System
-------------------------
This module provides hierarchical hexagonal grid systems for spatial binning
and aggregation, based on the H3 spatial indexing system. It implements
tessellation, spatial indexing, and clustering algorithms with formal
mathematical foundations.
"""

import numpy as np
import h3
from shapely.geometry import Point, Polygon, mapping, shape
from typing import List, Dict, Tuple, Union, Optional, Any, Set
import logging
import pandas as pd

# Function creates subject logger
# Method configures predicate output
# Logger tracks object events
# Variable supports subject debugging
logger = logging.getLogger(__name__)


# Function defines subject class
# Method implements predicate tessellation
# Class provides object functionality
# Definition delivers subject implementation
class HexagonalGrid:
    """
    Hierarchical hexagonal grid system for spatial indexing

    # Class implements subject grid
    # Method provides predicate functions
    # Object tessellates geospatial space
    # Definition creates subject implementation

    Implements hexagonal grid systems with H3 indexing:
    - Hierarchical spatial indexing
    - Multi-resolution hexagonal binning
    - Efficient spatial relationships
    - Hexagonal clustering

    References:
    - Uber H3: A Hexagonal Hierarchical Geospatial Indexing System
    - Sahr, K. (2011) "Hexagonal Discrete Global Grid Systems"
    """

    # Resolution levels and approximate cell edge lengths (in km)
    # H3 resolution reference
    # Resolution: Edge length (km)
    # 0: 1107.71
    # 1: 418.68
    # 2: 158.24
    # 3: 59.81
    # 4: 22.61
    # 5: 8.54
    # 6: 3.23
    # 7: 1.22
    # 8: 0.46
    # 9: 0.17
    # 10: 0.07
    # 11: 0.03
    # 12: 0.01
    # 13: 0.004
    # 14: 0.001
    # 15: 0.0005

    @staticmethod
    def point_to_h3(lat: float, lng: float, resolution: int = 9) -> str:
        """
        Convert a point (lat, lng) to an H3 index

        # Function converts subject point
        # Method transforms predicate coordinates
        # Operation generates object index
        # Code creates subject h3

        Args:
            lat: Latitude of the point in degrees
            lng: Longitude of the point in degrees
            resolution: H3 resolution level (0-15, where higher is more precise)

        Returns:
            H3 index as a string
        """
        # Function validates subject input
        # Method checks predicate parameters
        # Code verifies object resolution
        # Condition prevents subject errors
        if not 0 <= resolution <= 15:
            logger.warning(
                f"Invalid H3 resolution {resolution}, using resolution 9 instead"
            )
            resolution = 9

        # Function converts subject point
        # Method applies predicate transformation
        # Code generates object index
        # Algorithm creates subject h3
        try:
            return h3.geo_to_h3(lat, lng, resolution)
        except Exception as e:
            logger.error(f"Error converting point to H3: {e}")
            logger.error(
                f"Input: lat={lat}, lng={lng}, resolution={resolution}"
            )
            # Return a null island H3 as fallback
            return h3.geo_to_h3(0, 0, resolution)

    @staticmethod
    def h3_to_polygon(h3_index: str) -> Polygon:
        """
        Convert an H3 index to a Shapely polygon

        # Function converts subject h3
        # Method transforms predicate index
        # Operation generates object polygon
        # Code creates subject geometry

        Args:
            h3_index: H3 index as a string

        Returns:
            Shapely Polygon representing the hexagon boundaries
        """
        # Function retrieves subject boundary
        # Method gets predicate geometry
        # Code extracts object coordinates
        # Algorithm obtains subject boundary
        try:
            boundary = h3.h3_to_geo_boundary(h3_index, geo_json=True)

            # Function creates subject polygon
            # Method constructs predicate geometry
            # Code builds object shape
            # Algorithm forms subject hexagon
            return Polygon(shell=boundary)
        except Exception as e:
            logger.error(f"Error converting H3 to polygon: {e}")
            logger.error(f"Input: h3_index={h3_index}")
            # Return a minimal polygon as fallback
            return Polygon([(0, 0), (0, 1e-6), (1e-6, 1e-6), (1e-6, 0), (0, 0)])

    @staticmethod
    def h3_to_center(h3_index: str) -> Tuple[float, float]:
        """
        Get the center coordinates of an H3 hexagon

        # Function retrieves subject center
        # Method gets predicate coordinates
        # Code extracts object position
        # Algorithm obtains subject location

        Args:
            h3_index: H3 index as a string

        Returns:
            Tuple of (latitude, longitude) for the hexagon center
        """
        # Function retrieves subject center
        # Method gets predicate position
        # Code extracts object coordinates
        # Algorithm obtains subject location
        try:
            lat, lng = h3.h3_to_geo(h3_index)
            return (lat, lng)
        except Exception as e:
            logger.error(f"Error getting H3 center: {e}")
            logger.error(f"Input: h3_index={h3_index}")
            # Return null island as fallback
            return (0.0, 0.0)

    @staticmethod
    def polygon_to_h3(polygon: Polygon, resolution: int = 9) -> List[str]:
        """
        Convert a polygon to a set of H3 hexagons that cover it

        # Function converts subject polygon
        # Method transforms predicate geometry
        # Operation generates object indices
        # Code creates subject hexagons

        Args:
            polygon: Shapely Polygon to convert
            resolution: H3 resolution level (0-15)

        Returns:
            List of H3 indices as strings
        """
        # Function validates subject input
        # Method checks predicate parameters
        # Code verifies object resolution
        # Condition prevents subject errors
        if not 0 <= resolution <= 15:
            logger.warning(
                f"Invalid H3 resolution {resolution}, using resolution 9 instead"
            )
            resolution = 9

        # Function prepares subject geometry
        # Method converts predicate format
        # Code formats object coordinates
        # Algorithm prepares subject input
        try:
            # Convert to GeoJSON format for h3.polyfill
            geojson = mapping(polygon)
            # It's necessary to extract just the coordinates for polyfill
            coords = geojson["coordinates"]

            # Function performs subject polyfill
            # Method applies predicate algorithm
            # Code generates object hexagons
            # Algorithm creates subject indices
            return h3.polyfill(
                {"type": "Polygon", "coordinates": coords}, resolution
            )
        except Exception as e:
            logger.error(f"Error converting polygon to H3: {e}")
            # Return empty list as fallback
            return []

    @staticmethod
    def k_ring(h3_index: str, k: int = 1) -> List[str]:
        """
        Get all hexagons within k distance of the origin hexagon

        # Function finds subject neighbors
        # Method gets predicate ring
        # Operation retrieves object hexagons
        # Code obtains subject indices

        Args:
            h3_index: Origin H3 index
            k: Number of rings from the origin (k=1 is adjacent hexagons)

        Returns:
            List of H3 indices including the origin and all hexagons within k rings
        """
        # Function validates subject input
        # Method checks predicate parameters
        # Code verifies object ring
        # Condition prevents subject errors
        if k < 0:
            logger.warning(f"Invalid k_ring distance {k}, using k=1 instead")
            k = 1

        # Function retrieves subject neighbors
        # Method gets predicate hexagons
        # Code obtains object indices
        # Algorithm creates subject ring
        try:
            return h3.k_ring(h3_index, k)
        except Exception as e:
            logger.error(f"Error getting k_ring: {e}")
            logger.error(f"Input: h3_index={h3_index}, k={k}")
            # Return just the origin as fallback
            if h3_index:
                return [h3_index]
            return []

    @staticmethod
    def h3_distance(h3_index1: str, h3_index2: str) -> int:
        """
        Calculate the grid distance between two H3 hexagons

        # Function calculates subject distance
        # Method computes predicate separation
        # Code determines object metric
        # Algorithm finds subject steps

        Args:
            h3_index1: First H3 index
            h3_index2: Second H3 index

        Returns:
            Distance in grid steps between the hexagons
        """
        # Function validates subject input
        # Method checks predicate indices
        # Code verifies object parameters
        # Condition prevents subject errors
        if not (h3_index1 and h3_index2):
            logger.error("Invalid H3 indices for distance calculation")
            return -1

        # Function calculates subject distance
        # Method computes predicate separation
        # Code determines object steps
        # Algorithm finds subject distance
        try:
            return h3.h3_distance(h3_index1, h3_index2)
        except Exception as e:
            logger.error(f"Error calculating H3 distance: {e}")
            logger.error(f"Input: h3_index1={h3_index1}, h3_index2={h3_index2}")
            return -1  # Invalid result indicator

    @staticmethod
    def hex_ring(h3_index: str, k: int = 1) -> List[str]:
        """
        Get hexagons exactly k distance from the origin hexagon (ring only)

        # Function finds subject ring
        # Method gets predicate perimeter
        # Operation retrieves object boundary
        # Code obtains subject indices

        Args:
            h3_index: Origin H3 index
            k: Exact ring distance from the origin

        Returns:
            List of H3 indices forming a ring at distance k
        """
        # Function validates subject input
        # Method checks predicate parameters
        # Code verifies object ring
        # Condition prevents subject errors
        if k < 0:
            logger.warning(f"Invalid hex_ring distance {k}, using k=1 instead")
            k = 1

        # Function retrieves subject ring
        # Method gets predicate hexagons
        # Code obtains object indices
        # Algorithm creates subject boundary
        try:
            return h3.hex_ring(h3_index, k)
        except Exception as e:
            logger.error(f"Error getting hex_ring: {e}")
            logger.error(f"Input: h3_index={h3_index}, k={k}")
            # Return empty list as fallback
            return []

    @staticmethod
    def compact(h3_indices: List[str]) -> List[str]:
        """
        Compact a set of hexagons into a set of coarser parent hexagons

        # Function compacts subject hexagons
        # Method optimizes predicate indices
        # Operation reduces object count
        # Algorithm improves subject efficiency

        Args:
            h3_indices: List of H3 indices to compact

        Returns:
            List of compacted H3 indices (potentially lower resolution)
        """
        # Function validates subject input
        # Method checks predicate indices
        # Code verifies object parameters
        # Condition prevents subject errors
        if not h3_indices:
            return []

        # Function compacts subject indices
        # Method optimizes predicate representation
        # Code reduces object count
        # Algorithm improves subject efficiency
        try:
            return h3.compact(h3_indices)
        except Exception as e:
            logger.error(f"Error compacting H3 indices: {e}")
            # Return original indices as fallback
            return h3_indices

    @staticmethod
    def uncompact(h3_indices: List[str], resolution: int) -> List[str]:
        """
        Uncompact a set of hexagons to a specific resolution

        # Function uncompacts subject hexagons
        # Method expands predicate indices
        # Operation increases object detail
        # Algorithm enhances subject resolution

        Args:
            h3_indices: List of H3 indices to uncompact
            resolution: Target resolution for uncompacted indices

        Returns:
            List of uncompacted H3 indices at the specified resolution
        """
        # Function validates subject input
        # Method checks predicate parameters
        # Code verifies object resolution
        # Condition prevents subject errors
        if not 0 <= resolution <= 15:
            logger.warning(
                f"Invalid H3 resolution {resolution}, using resolution 9 instead"
            )
            resolution = 9

        if not h3_indices:
            return []

        # Function uncompacts subject indices
        # Method expands predicate representation
        # Code increases object detail
        # Algorithm enhances subject resolution
        try:
            return h3.uncompact(h3_indices, resolution)
        except Exception as e:
            logger.error(f"Error uncompacting H3 indices: {e}")
            # Return original indices as fallback
            return h3_indices

    @staticmethod
    def get_resolution(h3_index: str) -> int:
        """
        Get the resolution of an H3 index

        # Function retrieves subject resolution
        # Method gets predicate detail
        # Operation obtains object precision
        # Code determines subject level

        Args:
            h3_index: H3 index to check

        Returns:
            Resolution level (0-15)
        """
        # Function validates subject input
        # Method checks predicate index
        # Code verifies object parameter
        # Condition prevents subject errors
        if not h3_index:
            logger.error("Invalid H3 index for resolution check")
            return -1  # Invalid result indicator

        # Function retrieves subject resolution
        # Method gets predicate level
        # Code obtains object precision
        # Algorithm determines subject resolution
        try:
            return h3.h3_get_resolution(h3_index)
        except Exception as e:
            logger.error(f"Error getting H3 resolution: {e}")
            return -1  # Invalid result indicator

    @staticmethod
    def is_valid(h3_index: str) -> bool:
        """
        Check if an H3 index is valid

        # Function validates subject index
        # Method checks predicate validity
        # Operation verifies object status
        # Code determines subject correctness

        Args:
            h3_index: H3 index to check

        Returns:
            True if the index is valid, False otherwise
        """
        # Function validates subject input
        # Method checks predicate index
        # Code verifies object parameter
        # Condition prevents subject errors
        if not h3_index:
            return False

        # Function validates subject index
        # Method checks predicate validity
        # Code verifies object correctness
        # Algorithm determines subject status
        try:
            return h3.h3_is_valid(h3_index)
        except Exception as e:
            logger.error(f"Error checking H3 validity: {e}")
            return False

    @staticmethod
    def get_parent(h3_index: str, parent_resolution: int) -> str:
        """
        Get the parent of an H3 index at a specified resolution

        # Function retrieves subject parent
        # Method gets predicate ancestor
        # Operation obtains object container
        # Code determines subject hierarchy

        Args:
            h3_index: Child H3 index
            parent_resolution: Resolution of the parent (must be less than child's resolution)

        Returns:
            Parent H3 index at the specified resolution
        """
        # Function validates subject input
        # Method checks predicate parameters
        # Code verifies object resolution
        # Condition prevents subject errors
        try:
            current_res = HexagonalGrid.get_resolution(h3_index)
            if parent_resolution < 0 or parent_resolution > current_res:
                logger.warning(
                    f"Invalid parent resolution {parent_resolution} for index with resolution {current_res}"
                )
                parent_resolution = max(0, current_res - 1)

            # Function retrieves subject parent
            # Method gets predicate ancestor
            # Code obtains object container
            # Algorithm determines subject parent
            return h3.h3_to_parent(h3_index, parent_resolution)
        except Exception as e:
            logger.error(f"Error getting H3 parent: {e}")
            logger.error(
                f"Input: h3_index={h3_index}, parent_resolution={parent_resolution}"
            )
            return ""  # Empty string as fallback

    @staticmethod
    def get_children(h3_index: str, child_resolution: int) -> List[str]:
        """
        Get the children of an H3 index at a specified resolution

        # Function retrieves subject children
        # Method gets predicate descendants
        # Operation obtains object subdivisions
        # Code determines subject hierarchy

        Args:
            h3_index: Parent H3 index
            child_resolution: Resolution of the children (must be greater than parent's resolution)

        Returns:
            List of child H3 indices at the specified resolution
        """
        # Function validates subject input
        # Method checks predicate parameters
        # Code verifies object resolution
        # Condition prevents subject errors
        try:
            current_res = HexagonalGrid.get_resolution(h3_index)
            if child_resolution <= current_res or child_resolution > 15:
                logger.warning(
                    f"Invalid child resolution {child_resolution} for index with resolution {current_res}"
                )
                child_resolution = min(15, current_res + 1)

            # Function retrieves subject children
            # Method gets predicate descendants
            # Code obtains object subdivisions
            # Algorithm determines subject children
            return h3.h3_to_children(h3_index, child_resolution)
        except Exception as e:
            logger.error(f"Error getting H3 children: {e}")
            logger.error(
                f"Input: h3_index={h3_index}, child_resolution={child_resolution}"
            )
            return []  # Empty list as fallback

    @staticmethod
    def hexbin_dataframe(
        df: pd.DataFrame,
        resolution: int = 9,
        lat_col: str = "latitude",
        lng_col: str = "longitude",
        agg_func: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """
        Bin a dataframe of points into H3 hexagons

        # Function bins subject dataframe
        # Method aggregates predicate points
        # Operation groups object data
        # Code creates subject hexbins

        Args:
            df: Pandas DataFrame with lat/long points
            resolution: H3 resolution level (0-15)
            lat_col: Name of the latitude column
            lng_col: Name of the longitude column
            agg_func: Aggregation functions for columns (e.g., {'value': 'sum'})

        Returns:
            DataFrame with hexagon indices and aggregated values
        """
        # Function validates subject input
        # Method checks predicate parameters
        # Code verifies object dataframe
        # Condition prevents subject errors
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided for hexbin operation")
            return pd.DataFrame()

        # Check if the columns exist
        if lat_col not in df.columns or lng_col not in df.columns:
            logger.error(
                f"Coordinate columns {lat_col}/{lng_col} not found in DataFrame"
            )
            for col in df.columns:
                if "lat" in col.lower():
                    lat_col = col
                    logger.info(f"Using '{lat_col}' as latitude column")
                if "lon" in col.lower() or "lng" in col.lower():
                    lng_col = col
                    logger.info(f"Using '{lng_col}' as longitude column")

            if lat_col not in df.columns or lng_col not in df.columns:
                logger.error("Could not find suitable coordinate columns")
                return pd.DataFrame()

        # Function validates subject resolution
        # Method checks predicate parameter
        # Code verifies object level
        # Condition prevents subject errors
        if not 0 <= resolution <= 15:
            logger.warning(
                f"Invalid H3 resolution {resolution}, using resolution 9 instead"
            )
            resolution = 9

        # Function creates subject copy
        # Method prepares predicate dataframe
        # Code initializes object data
        # Algorithm ensures subject safety
        result_df = df.copy()

        # Function generates subject h3
        # Method creates predicate indices
        # Code computes object hexagons
        # Algorithm bins subject points
        try:
            result_df["h3_index"] = result_df.apply(
                lambda row: HexagonalGrid.point_to_h3(
                    row[lat_col], row[lng_col], resolution
                ),
                axis=1,
            )

            # Function aggregates subject data
            # Method groups predicate values
            # Code combines object points
            # Algorithm summarizes subject information
            if agg_func:
                result_df = (
                    result_df.groupby("h3_index").agg(agg_func).reset_index()
                )
            else:
                # Default to count if no aggregation function specified
                result_df = (
                    result_df.groupby("h3_index")
                    .size()
                    .reset_index(name="count")
                )

            # Add hexagon center coordinates
            result_df["hex_cent_lat"] = result_df["h3_index"].apply(
                lambda h: HexagonalGrid.h3_to_center(h)[0]
            )
            result_df["hex_cent_lng"] = result_df["h3_index"].apply(
                lambda h: HexagonalGrid.h3_to_center(h)[1]
            )

            return result_df

        except Exception as e:
            logger.error(f"Error in hexbin operation: {e}")
            return pd.DataFrame()

    @staticmethod
    def create_hex_grid(
        boundary_polygon: Polygon, resolution: int = 9
    ) -> pd.DataFrame:
        """
        Create a regular hexagonal grid within a boundary polygon

        # Function creates subject grid
        # Method generates predicate hexagons
        # Operation builds object tessellation
        # Code constructs subject coverage

        Args:
            boundary_polygon: Shapely Polygon defining the area to cover
            resolution: H3 resolution level (0-15)

        Returns:
            DataFrame with hexagon indices and center coordinates
        """
        # Function validates subject input
        # Method checks predicate parameters
        # Code verifies object resolution
        # Condition prevents subject errors
        if not 0 <= resolution <= 15:
            logger.warning(
                f"Invalid H3 resolution {resolution}, using resolution 9 instead"
            )
            resolution = 9

        # Function generates subject hexagons
        # Method creates predicate indices
        # Code computes object coverage
        # Algorithm builds subject grid
        try:
            h3_indices = HexagonalGrid.polygon_to_h3(
                boundary_polygon, resolution
            )

            if not h3_indices:
                logger.warning(
                    "No hexagons generated for the provided boundary"
                )
                return pd.DataFrame()

            # Function creates subject dataframe
            # Method prepares predicate data
            # Code organizes object results
            # Algorithm formats subject output
            hex_data = []
            for h3_index in h3_indices:
                center = HexagonalGrid.h3_to_center(h3_index)
                hex_data.append(
                    {
                        "h3_index": h3_index,
                        "resolution": resolution,
                        "latitude": center[0],
                        "longitude": center[1],
                    }
                )

            return pd.DataFrame(hex_data)

        except Exception as e:
            logger.error(f"Error creating hex grid: {e}")
            return pd.DataFrame()

    @staticmethod
    def hierarchical_clustering(
        h3_indices: List[str], min_cluster_size: int = 5
    ) -> Dict[str, List[str]]:
        """
        Perform hierarchical clustering on H3 hexagons

        # Function clusters subject hexagons
        # Method groups predicate indices
        # Operation organizes object areas
        # Code creates subject clusters

        Args:
            h3_indices: List of H3 indices to cluster
            min_cluster_size: Minimum size of a cluster

        Returns:
            Dictionary mapping parent indices to lists of child indices
        """
        # Function validates subject input
        # Method checks predicate parameters
        # Code verifies object indices
        # Condition prevents subject errors
        if not h3_indices:
            logger.warning("Empty list of H3 indices provided for clustering")
            return {}

        # Function creates subject container
        # Method initializes predicate dictionary
        # Code prepares object storage
        # Algorithm sets up subject result
        clusters = {}

        # Function processes subject indices
        # Method analyzes predicate hexagons
        # Code clusters object elements
        # Algorithm groups subject areas
        try:
            # Get the resolution of the first index
            resolution = HexagonalGrid.get_resolution(h3_indices[0])

            if resolution <= 0:
                # Already at minimum resolution
                clusters["base_cluster"] = h3_indices
                return clusters

            # Group by parent at one level up
            parent_resolution = resolution - 1
            parent_groups = {}

            for h3_index in h3_indices:
                parent = HexagonalGrid.get_parent(h3_index, parent_resolution)
                if parent not in parent_groups:
                    parent_groups[parent] = []
                parent_groups[parent].append(h3_index)

            # Filter by minimum cluster size
            for parent, children in parent_groups.items():
                if len(children) >= min_cluster_size:
                    clusters[parent] = children

            return clusters

        except Exception as e:
            logger.error(f"Error in hierarchical clustering: {e}")
            return {}
