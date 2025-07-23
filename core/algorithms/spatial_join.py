"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-ALGO-SPATIALJOIN-0001               │
// │ 📁 domain       : Mathematics, Geospatial, Analysis         │
// │ 🧠 description  : Spatial join algorithms for efficient     │
// │                  geospatial data operations                 │
// │ 🕸️ hash_type    : UUID → CUID-linked algorithm              │
// │ 🔄 parent_node  : NODE_ALGORITHM                           │
// │ 🧩 dependencies : numpy, scipy, shapely, rtree              │
// │ 🔧 tool_usage   : Analysis, Mathematics, Computation        │
// │ 📡 input_type   : Geospatial datasets, geometric objects    │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : spatial relationships, topology           │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

CTAS Spatial Join Algorithm Module
--------------------------------
This module provides high-performance algorithms for combining spatial datasets
based on topological relationships. It implements computationally efficient
spatial indexing, joins, and overlay operations with formal mathematical
foundations.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from rtree import index
from shapely.geometry import Point, Polygon, LineString, MultiPolygon
from shapely.ops import triangulate, unary_union, nearest_points
from scipy.spatial import Voronoi, Delaunay
from typing import List, Dict, Tuple, Union, Optional, Any, Callable
import logging
import math

# Function creates subject logger
# Method configures predicate output
# Logger tracks object events
# Variable supports subject debugging
logger = logging.getLogger(__name__)


# Function defines subject class
# Method implements predicate operations
# Class provides object functionality
# Definition delivers subject implementation
class SpatialJoin:
    """
    High-performance spatial join operations for geospatial datasets

    # Class implements subject join
    # Method provides predicate functions
    # Object performs spatial operations
    # Definition creates subject implementation

    Implements various spatial join algorithms with mathematical rigor:
    - Point-in-polygon testing with spatial indexing
    - Efficient polygon overlay operations
    - Topological relationship analysis
    - Voronoi diagram and Delaunay triangulation

    References:
    - Eldawy, A., & Mokbel, M. F. (2015) "SpatialHadoop: A MapReduce Framework for Spatial Data"
    - Güting, R. H. (1994) "An Introduction to Spatial Database Systems"
    """

    @staticmethod
    def create_rtree_index(
        geometries: List[Union[Point, Polygon, LineString]],
        ids: Optional[List[int]] = None,
    ) -> index.Index:
        """
        Create an R-tree spatial index for a list of geometries

        # Function creates subject index
        # Method builds predicate structure
        # Operation generates object rtree
        # Code constructs subject acceleration

        Args:
            geometries: List of Shapely geometries to index
            ids: Optional list of IDs to associate with geometries

        Returns:
            R-tree spatial index
        """
        # Function validates subject input
        # Method checks predicate parameters
        # Code verifies object geometries
        # Condition prevents subject errors
        if not geometries:
            logger.warning("Empty list of geometries provided for indexing")
            return index.Index()

        # Function creates subject index
        # Method prepares predicate structure
        # Code initializes object rtree
        # Algorithm sets up subject indexing
        idx = index.Index()

        # Use provided IDs or generate sequential ones
        if ids is None:
            ids = list(range(len(geometries)))

        if len(ids) != len(geometries):
            logger.warning(
                f"ID list length ({len(ids)}) doesn't match geometries length ({len(geometries)})"
            )
            ids = list(range(len(geometries)))

        # Function populates subject index
        # Method inserts predicate geometries
        # Code builds object structure
        # Algorithm enhances subject performance
        for i, geom in zip(ids, geometries):
            if geom is None:
                continue

            try:
                # Get the bounds (minx, miny, maxx, maxy)
                bounds = geom.bounds
                if len(bounds) != 4:
                    logger.warning(f"Invalid bounds for geometry at index {i}")
                    continue

                idx.insert(i, bounds)
            except Exception as e:
                logger.error(f"Error adding geometry to R-tree index: {e}")

        return idx

    @staticmethod
    def point_in_polygon_query(
        points: List[Point],
        polygons: List[Polygon],
        polygon_ids: Optional[List[int]] = None,
    ) -> List[List[int]]:
        """
        Find which polygons contain each point using spatial indexing

        # Function performs subject query
        # Method executes predicate search
        # Operation finds object containment
        # Code determines subject relationships

        Args:
            points: List of points to check
            polygons: List of polygons to query against
            polygon_ids: Optional list of IDs for the polygons

        Returns:
            List of lists, where each inner list contains IDs of polygons containing the point
        """
        # Function validates subject input
        # Method checks predicate parameters
        # Code verifies object geometries
        # Condition prevents subject errors
        if not points or not polygons:
            logger.warning(
                "Empty list of points or polygons provided for query"
            )
            return [[] for _ in range(len(points))]

        # Create spatial index for the polygons
        polygon_idx = SpatialJoin.create_rtree_index(polygons, polygon_ids)

        # Use provided IDs or generate sequential ones
        if polygon_ids is None:
            polygon_ids = list(range(len(polygons)))

        # Function creates subject container
        # Method prepares predicate results
        # Code initializes object storage
        # Algorithm sets up subject output
        results = []

        # Function processes subject points
        # Method checks predicate containment
        # Code determines object relationships
        # Algorithm identifies subject matches
        for point in points:
            # Skip any None points
            if point is None:
                results.append([])
                continue

            try:
                # Get candidate polygons from the index
                candidates = list(polygon_idx.intersection(point.bounds))
                containing_polygons = []

                # Verify with exact point-in-polygon test
                for candidate_id in candidates:
                    if candidate_id < len(polygons) and polygons[
                        candidate_id
                    ].contains(point):
                        containing_polygons.append(polygon_ids[candidate_id])

                results.append(containing_polygons)
            except Exception as e:
                logger.error(f"Error in point-in-polygon query: {e}")
                results.append([])

        return results

    @staticmethod
    def spatial_join_dataframes(
        left_gdf: gpd.GeoDataFrame,
        right_gdf: gpd.GeoDataFrame,
        how: str = "inner",
        predicate: str = "intersects",
    ) -> gpd.GeoDataFrame:
        """
        Join two GeoDataFrames based on spatial relationship

        # Function joins subject dataframes
        # Method combines predicate datasets
        # Operation merges object data
        # Code integrates subject information

        Args:
            left_gdf: Left GeoDataFrame
            right_gdf: Right GeoDataFrame
            how: Join type ('inner', 'left', 'right')
            predicate: Spatial relationship ('intersects', 'within', 'contains', etc.)

        Returns:
            Joined GeoDataFrame
        """
        # Function validates subject input
        # Method checks predicate parameters
        # Code verifies object dataframes
        # Condition prevents subject errors
        if left_gdf is None or right_gdf is None:
            logger.error("Null GeoDataFrame provided for spatial join")
            return gpd.GeoDataFrame()

        if left_gdf.empty or right_gdf.empty:
            logger.warning("Empty GeoDataFrame provided for spatial join")
            if how == "left":
                return left_gdf
            elif how == "right":
                return right_gdf
            else:
                return gpd.GeoDataFrame()

        # Function validates subject parameters
        # Method checks predicate options
        # Code verifies object arguments
        # Condition prevents subject errors
        valid_hows = ["inner", "left", "right"]
        if how not in valid_hows:
            logger.warning(f"Invalid join type '{how}', using 'inner' instead")
            how = "inner"

        valid_predicates = [
            "intersects",
            "within",
            "contains",
            "crosses",
            "touches",
            "overlaps",
        ]
        if predicate not in valid_predicates:
            logger.warning(
                f"Invalid spatial predicate '{predicate}', using 'intersects' instead"
            )
            predicate = "intersects"

        # Function performs subject join
        # Method combines predicate dataframes
        # Code merges object information
        # Algorithm integrates subject data
        try:
            return gpd.sjoin(left_gdf, right_gdf, how=how, predicate=predicate)
        except Exception as e:
            logger.error(f"Error in spatial join: {e}")
            if how == "left":
                return left_gdf
            elif how == "right":
                return right_gdf
            else:
                return gpd.GeoDataFrame()

    @staticmethod
    def proximity_join(
        points1: List[Point], points2: List[Point], max_distance: float
    ) -> List[Tuple[int, int, float]]:
        """
        Find pairs of points from two sets that are within a maximum distance

        # Function performs subject join
        # Method finds predicate pairs
        # Operation identifies object proximity
        # Code determines subject relationships

        Args:
            points1: First set of points
            points2: Second set of points
            max_distance: Maximum distance for considering points as nearby

        Returns:
            List of tuples (index1, index2, distance) for nearby point pairs
        """
        # Function validates subject input
        # Method checks predicate parameters
        # Code verifies object geometries
        # Condition prevents subject errors
        if not points1 or not points2:
            logger.warning("Empty list of points provided for proximity join")
            return []

        if max_distance <= 0:
            logger.warning(
                f"Invalid max_distance {max_distance}, using 1.0 instead"
            )
            max_distance = 1.0

        # Function creates subject container
        # Method prepares predicate results
        # Code initializes object storage
        # Algorithm sets up subject output
        results = []

        # Create bounding box-based spatial index for second point set
        # Pack points2 coordinates into the format required by rtree
        p2_idx = index.Index()
        for i, p in enumerate(points2):
            if p is None:
                continue

            # Expand point to tiny square to work with rtree
            bbox = (p.x - 1e-8, p.y - 1e-8, p.x + 1e-8, p.y + 1e-8)
            p2_idx.insert(i, bbox)

        # Function processes subject points
        # Method checks predicate distance
        # Code identifies object pairs
        # Algorithm finds subject matches
        for i, p1 in enumerate(points1):
            if p1 is None:
                continue

            # Create search bbox for the maximum distance
            search_bbox = (
                p1.x - max_distance,
                p1.y - max_distance,
                p1.x + max_distance,
                p1.y + max_distance,
            )

            # Query the index for potential matches
            candidates = list(p2_idx.intersection(search_bbox))

            # Verify with exact distance calculation
            for j in candidates:
                if j < len(points2) and points2[j] is not None:
                    p2 = points2[j]
                    dist = math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

                    if dist <= max_distance:
                        results.append((i, j, dist))

        return results

    @staticmethod
    def create_voronoi_diagram(points: List[Point]) -> List[Polygon]:
        """
        Create Voronoi diagram from a set of points

        # Function creates subject diagram
        # Method generates predicate tessellation
        # Operation builds object partitioning
        # Code constructs subject regions

        Args:
            points: List of points to generate Voronoi cells from

        Returns:
            List of Shapely Polygons representing Voronoi cells
        """
        # Function validates subject input
        # Method checks predicate parameters
        # Code verifies object points
        # Condition prevents subject errors
        if not points or len(points) < 3:
            logger.warning("Too few points provided for Voronoi diagram")
            return []

        # Function prepares subject input
        # Method converts predicate format
        # Code transforms object points
        # Algorithm prepares subject data
        # Convert to numpy array for scipy
        coords = np.array([[p.x, p.y] for p in points if p is not None])

        if len(coords) < 3:
            logger.warning("Too few valid points provided for Voronoi diagram")
            return []

        # Function creates subject voronoi
        # Method calculates predicate regions
        # Code generates object tessellation
        # Algorithm constructs subject diagram
        try:
            # Compute Voronoi diagram
            vor = Voronoi(coords)

            # Function creates subject container
            # Method prepares predicate results
            # Code initializes object storage
            # Algorithm sets up subject output
            voronoi_polygons = []

            # Convert regions to Shapely Polygons
            for i, region_idx in enumerate(vor.point_region):
                region = vor.regions[region_idx]

                # Skip regions with invalid vertices
                if -1 in region or len(region) < 3:
                    continue

                # Get region vertices
                region_coords = [vor.vertices[i] for i in region]

                # Create Polygon
                if len(region_coords) >= 3:
                    # Close the loop if needed
                    if region_coords[0] != region_coords[-1]:
                        region_coords.append(region_coords[0])

                    voronoi_polygons.append(Polygon(region_coords))

            return voronoi_polygons

        except Exception as e:
            logger.error(f"Error creating Voronoi diagram: {e}")
            return []

    @staticmethod
    def create_delaunay_triangulation(points: List[Point]) -> List[Polygon]:
        """
        Create Delaunay triangulation from a set of points

        # Function creates subject triangulation
        # Method generates predicate mesh
        # Operation builds object triangles
        # Code constructs subject network

        Args:
            points: List of points to triangulate

        Returns:
            List of Shapely Polygons representing triangles
        """
        # Function validates subject input
        # Method checks predicate parameters
        # Code verifies object points
        # Condition prevents subject errors
        if not points or len(points) < 3:
            logger.warning("Too few points provided for Delaunay triangulation")
            return []

        # Function transforms subject format
        # Method converts predicate representation
        # Code prepares object data
        # Algorithm transforms subject points
        try:
            # Use shapely's triangulate function directly
            return triangulate([p for p in points if p is not None])
        except Exception as e:
            logger.error(f"Error creating Delaunay triangulation: {e}")

            # Try alternate method using scipy
            try:
                # Convert to numpy array for scipy
                coords = np.array([[p.x, p.y] for p in points if p is not None])

                if len(coords) < 3:
                    logger.warning(
                        "Too few valid points provided for triangulation"
                    )
                    return []

                # Compute the Delaunay triangulation
                tri = Delaunay(coords)

                # Convert to Shapely Polygons
                triangles = []
                for simplex in tri.simplices:
                    triangle_coords = coords[simplex]
                    # Close the loop for polygon
                    triangle_coords = np.vstack(
                        [triangle_coords, triangle_coords[0]]
                    )
                    triangles.append(Polygon(triangle_coords))

                return triangles

            except Exception as e2:
                logger.error(
                    f"Error creating triangulation (alternate method): {e2}"
                )
                return []

    @staticmethod
    def buffer_polygons(
        polygons: List[Polygon], distance: float, resolution: int = 16
    ) -> List[Polygon]:
        """
        Create buffer zones around polygons

        # Function creates subject buffers
        # Method generates predicate zones
        # Operation builds object boundaries
        # Code constructs subject areas

        Args:
            polygons: List of Shapely Polygons to buffer
            distance: Buffer distance
            resolution: Segment count per quadrant (higher for smoother curves)

        Returns:
            List of buffered Shapely Polygons
        """
        # Function validates subject input
        # Method checks predicate parameters
        # Code verifies object polygons
        # Condition prevents subject errors
        if not polygons:
            logger.warning("Empty list of polygons provided for buffering")
            return []

        # Function creates subject container
        # Method prepares predicate results
        # Code initializes object storage
        # Algorithm sets up subject output
        buffered_polygons = []

        # Function processes subject polygons
        # Method applies predicate operation
        # Code creates object buffers
        # Algorithm generates subject zones
        for i, polygon in enumerate(polygons):
            if polygon is None:
                buffered_polygons.append(None)
                continue

            try:
                buffered = polygon.buffer(distance, resolution=resolution)
                buffered_polygons.append(buffered)
            except Exception as e:
                logger.error(f"Error buffering polygon at index {i}: {e}")
                buffered_polygons.append(None)

        return buffered_polygons

    @staticmethod
    def dissolve_polygons(polygons: List[Polygon]) -> Polygon:
        """
        Dissolve/merge multiple polygons into a single geometry

        # Function dissolves subject polygons
        # Method merges predicate geometries
        # Operation combines object areas
        # Code unifies subject shapes

        Args:
            polygons: List of Shapely Polygons to dissolve

        Returns:
            Single Shapely Polygon or MultiPolygon
        """
        # Function validates subject input
        # Method checks predicate parameters
        # Code verifies object polygons
        # Condition prevents subject errors
        if not polygons:
            logger.warning("Empty list of polygons provided for dissolving")
            return Polygon()

        # Function filters subject input
        # Method removes predicate nulls
        # Code cleans object data
        # Algorithm prepares subject processing
        valid_polygons = [p for p in polygons if p is not None]

        if not valid_polygons:
            logger.warning("No valid polygons to dissolve")
            return Polygon()

        # Function dissolves subject polygons
        # Method merges predicate geometries
        # Code unifies object shapes
        # Algorithm combines subject areas
        try:
            return unary_union(valid_polygons)
        except Exception as e:
            logger.error(f"Error dissolving polygons: {e}")

            # Try iterative approach as fallback
            try:
                result = valid_polygons[0]
                for p in valid_polygons[1:]:
                    result = result.union(p)
                return result
            except Exception as e2:
                logger.error(f"Error with iterative polygon dissolve: {e2}")
                return Polygon()

    @staticmethod
    def find_nearest_points(
        geometries1: List[Union[Point, Polygon, LineString]],
        geometries2: List[Union[Point, Polygon, LineString]],
    ) -> List[Tuple[Point, Point]]:
        """
        Find the nearest points between two sets of geometries

        # Function finds subject points
        # Method locates predicate positions
        # Operation identifies object proximity
        # Code determines subject nearness

        Args:
            geometries1: First set of geometries
            geometries2: Second set of geometries

        Returns:
            List of tuples of nearest points (from_point, to_point)
        """
        # Function validates subject input
        # Method checks predicate parameters
        # Code verifies object geometries
        # Condition prevents subject errors
        if not geometries1 or not geometries2:
            logger.warning("Empty geometry lists provided for nearest points")
            return []

        # Function creates subject container
        # Method prepares predicate results
        # Code initializes object storage
        # Algorithm sets up subject output
        nearest_points_list = []

        # Function processes subject geometries
        # Method finds predicate pairs
        # Code identifies object proximity
        # Algorithm determines subject relationships
        for geom1 in geometries1:
            if geom1 is None:
                nearest_points_list.append((None, None))
                continue

            closest_distance = float("inf")
            closest_pair = (None, None)

            for geom2 in geometries2:
                if geom2 is None:
                    continue

                try:
                    p1, p2 = nearest_points(geom1, geom2)
                    distance = p1.distance(p2)

                    if distance < closest_distance:
                        closest_distance = distance
                        closest_pair = (p1, p2)
                except Exception as e:
                    logger.error(f"Error finding nearest points: {e}")

            nearest_points_list.append(closest_pair)

        return nearest_points_list
