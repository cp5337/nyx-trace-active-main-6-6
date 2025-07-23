"""
Spatial Matroid Module
--------------------
Implements specialized matroid classes for geospatial intelligence analysis.
These matroids handle spatial data, geographic relationships, and geotemporal patterns.
"""

from typing import (
    Dict,
    Any,
    List,
    Set,
    Tuple,
    Optional,
    TypeVar,
    Generic,
    FrozenSet,
    Callable,
    Union,
)
import math
import uuid
import logging
from datetime import datetime, timedelta

from core.matroid.base import Matroid, RankMatroid, IndependenceMatroid

logger = logging.getLogger(__name__)

# Type definitions
Location = Tuple[float, float]  # (latitude, longitude)
GeoElement = Dict[
    str, Any
]  # Dictionary with at least 'latitude' and 'longitude' keys


class SpatialMatroid(Matroid[str]):
    """
    Matroid for spatial independence in geographical contexts.

    Elements are locations or geographic features. Sets are independent if they
    satisfy spatial constraints like minimum distance, coverage requirements, etc.
    """

    def __init__(
        self,
        name: str = None,
        min_distance_km: float = 5.0,
        max_elements_per_region: Optional[int] = None,
    ):
        """
        Initialize a spatial matroid

        Args:
            name: Optional name for the matroid
            min_distance_km: Minimum distance between elements in kilometers
            max_elements_per_region: Maximum number of elements in a region
        """
        super().__init__(name)
        self.min_distance_km = min_distance_km
        self.max_elements_per_region = max_elements_per_region
        self._locations: Dict[str, Tuple[float, float]] = (
            {}
        )  # element_id -> (lat, lon)
        self._regions: Dict[str, Set[str]] = (
            {}
        )  # region_id -> set of element_ids

    def add_location(
        self,
        element_id: str,
        latitude: float,
        longitude: float,
        region_id: Optional[str] = None,
    ) -> None:
        """
        Add a location to the matroid

        Args:
            element_id: Element identifier
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            region_id: Optional region identifier
        """
        self._locations[element_id] = (latitude, longitude)
        self.add_to_ground_set(element_id)

        # Add to region if specified
        if region_id:
            if region_id not in self._regions:
                self._regions[region_id] = set()
            self._regions[region_id].add(element_id)

    def _check_independence(self, subset: Set[str]) -> bool:
        """
        Check if a subset of locations is independent

        Locations are independent if:
        1. No two locations are too close to each other
        2. No region has too many elements

        Args:
            subset: Subset of locations to check

        Returns:
            True if the subset is independent, False otherwise
        """
        # Check for minimum distance
        locations = [self._locations.get(element_id) for element_id in subset]
        locations = [loc for loc in locations if loc is not None]

        for i in range(len(locations)):
            for j in range(i + 1, len(locations)):
                if (
                    self._haversine_distance(locations[i], locations[j])
                    < self.min_distance_km
                ):
                    return False

        # Check for maximum elements per region
        if self.max_elements_per_region is not None:
            region_counts = {}
            for region_id, elements in self._regions.items():
                region_counts[region_id] = len(elements.intersection(subset))
                if region_counts[region_id] > self.max_elements_per_region:
                    return False

        return True

    def _haversine_distance(
        self, location1: Location, location2: Location
    ) -> float:
        """
        Calculate the haversine distance between two locations in kilometers

        Args:
            location1: First location (latitude, longitude)
            location2: Second location (latitude, longitude)

        Returns:
            Distance in kilometers
        """
        # Convert latitude and longitude to radians
        lat1, lon1 = math.radians(location1[0]), math.radians(location1[1])
        lat2, lon2 = math.radians(location2[0]), math.radians(location2[1])

        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.asin(math.sqrt(a))
        r = 6371  # Radius of Earth in kilometers

        return c * r

    def get_nearest_neighbors(
        self, element_id: str, k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Get the k nearest neighbors of a location

        Args:
            element_id: Element identifier
            k: Number of neighbors to return

        Returns:
            List of (element_id, distance) tuples
        """
        if element_id not in self._locations:
            return []

        location = self._locations[element_id]

        distances = []
        for other_id, other_location in self._locations.items():
            if other_id != element_id:
                distance = self._haversine_distance(location, other_location)
                distances.append((other_id, distance))

        # Sort by distance and return the k nearest
        return sorted(distances, key=lambda x: x[1])[:k]

    def find_optimal_coverage(
        self,
        required_elements: Optional[Set[str]] = None,
        max_elements: Optional[int] = None,
    ) -> Set[str]:
        """
        Find an optimal subset of locations that provides good geographic coverage

        Args:
            required_elements: Elements that must be included
            max_elements: Maximum number of elements to include

        Returns:
            Set of element identifiers
        """
        if required_elements is None:
            required_elements = set()

        if max_elements is None:
            max_elements = len(self._ground_set)

        # Start with required elements
        result = set(required_elements)

        # Greedily add elements to maximize coverage
        candidates = self._ground_set - result
        while len(result) < max_elements and candidates:
            best_element = None
            best_score = -1

            for element_id in candidates:
                # Calculate the score for this element
                # (Sum of distances to existing elements)
                score = 0
                location = self._locations.get(element_id)
                if location:
                    for existing in result:
                        existing_location = self._locations.get(existing)
                        if existing_location:
                            score += self._haversine_distance(
                                location, existing_location
                            )

                if score > best_score:
                    best_score = score
                    best_element = element_id

            if best_element:
                result.add(best_element)
                candidates.remove(best_element)
            else:
                break

        return result

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert spatial matroid to dictionary representation

        Returns:
            Dictionary representation
        """
        base_dict = super().to_dict()

        # Add spatial-specific information
        base_dict.update(
            {
                "min_distance_km": self.min_distance_km,
                "max_elements_per_region": self.max_elements_per_region,
                "locations": {
                    element_id: {"latitude": lat, "longitude": lon}
                    for element_id, (lat, lon) in self._locations.items()
                },
                "regions": {
                    region_id: list(elements)
                    for region_id, elements in self._regions.items()
                },
            }
        )

        return base_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpatialMatroid":
        """
        Create spatial matroid from dictionary representation

        Args:
            data: Dictionary representation

        Returns:
            SpatialMatroid instance
        """
        matroid = cls(
            name=data.get("name"),
            min_distance_km=data.get("min_distance_km", 5.0),
            max_elements_per_region=data.get("max_elements_per_region"),
        )

        # Add locations
        for element_id, location in data.get("locations", {}).items():
            matroid.add_location(
                element_id, location["latitude"], location["longitude"]
            )

        # Add elements to regions
        for region_id, elements in data.get("regions", {}).items():
            for element_id in elements:
                if element_id in matroid._locations:
                    if region_id not in matroid._regions:
                        matroid._regions[region_id] = set()
                    matroid._regions[region_id].add(element_id)

        return matroid


class SpatioTemporalMatroid(SpatialMatroid):
    """
    Matroid for spatiotemporal independence in geographical contexts.

    Elements are spatiotemporal events. Sets are independent if they
    satisfy both spatial and temporal constraints.
    """

    def __init__(
        self,
        name: str = None,
        min_distance_km: float = 5.0,
        min_time_interval: timedelta = timedelta(hours=1),
        max_elements_per_region: Optional[int] = None,
    ):
        """
        Initialize a spatiotemporal matroid

        Args:
            name: Optional name for the matroid
            min_distance_km: Minimum distance between elements in kilometers
            min_time_interval: Minimum time interval between events
            max_elements_per_region: Maximum number of elements in a region
        """
        super().__init__(name, min_distance_km, max_elements_per_region)
        self.min_time_interval = min_time_interval
        self._timestamps: Dict[str, datetime] = {}  # element_id -> timestamp

    def add_event(
        self,
        element_id: str,
        latitude: float,
        longitude: float,
        timestamp: datetime,
        region_id: Optional[str] = None,
    ) -> None:
        """
        Add a spatiotemporal event to the matroid

        Args:
            element_id: Element identifier
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            timestamp: Event timestamp
            region_id: Optional region identifier
        """
        super().add_location(element_id, latitude, longitude, region_id)
        self._timestamps[element_id] = timestamp

    def _check_independence(self, subset: Set[str]) -> bool:
        """
        Check if a subset of events is independent

        Events are independent if:
        1. They satisfy spatial independence (from SpatialMatroid)
        2. No two events are too close in time

        Args:
            subset: Subset of events to check

        Returns:
            True if the subset is independent, False otherwise
        """
        # Check spatial independence
        if not super()._check_independence(subset):
            return False

        # Check temporal independence
        events = []
        for element_id in subset:
            if element_id in self._locations and element_id in self._timestamps:
                events.append(
                    (
                        element_id,
                        self._locations[element_id],
                        self._timestamps[element_id],
                    )
                )

        # Sort by timestamp
        events.sort(key=lambda x: x[2])

        # Check for minimum time interval between nearby events
        for i in range(len(events)):
            for j in range(i + 1, len(events)):
                # Check if events are close in space
                if (
                    self._haversine_distance(events[i][1], events[j][1])
                    < self.min_distance_km * 3
                ):
                    # Check if events are close in time
                    time_diff = abs(
                        (events[j][2] - events[i][2]).total_seconds()
                    )
                    if time_diff < self.min_time_interval.total_seconds():
                        return False

        return True

    def find_temporal_clusters(
        self,
        time_window: timedelta = timedelta(hours=24),
        spatial_radius_km: float = 10.0,
    ) -> List[Set[str]]:
        """
        Find clusters of events that are close in both space and time

        Args:
            time_window: Time window for clustering
            spatial_radius_km: Spatial radius for clustering in kilometers

        Returns:
            List of clusters (sets of element identifiers)
        """
        events = []
        for element_id, location in self._locations.items():
            if element_id in self._timestamps:
                events.append(
                    (element_id, location, self._timestamps[element_id])
                )

        # Sort by timestamp
        events.sort(key=lambda x: x[2])

        clusters = []
        processed = set()

        for i, (element_id, location, timestamp) in enumerate(events):
            if element_id in processed:
                continue

            # Start a new cluster
            cluster = {element_id}
            processed.add(element_id)

            # Find related events
            for j in range(i + 1, len(events)):
                other_id, other_location, other_timestamp = events[j]

                if other_id in processed:
                    continue

                # Check if the event is within the time window
                time_diff = abs((other_timestamp - timestamp).total_seconds())
                if time_diff <= time_window.total_seconds():
                    # Check if the event is within the spatial radius
                    space_diff = self._haversine_distance(
                        location, other_location
                    )
                    if space_diff <= spatial_radius_km:
                        cluster.add(other_id)
                        processed.add(other_id)

            if len(cluster) > 1:
                clusters.append(cluster)

        return clusters

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert spatiotemporal matroid to dictionary representation

        Returns:
            Dictionary representation
        """
        base_dict = super().to_dict()

        # Add temporal-specific information
        base_dict.update(
            {
                "min_time_interval_seconds": self.min_time_interval.total_seconds(),
                "timestamps": {
                    element_id: timestamp.isoformat()
                    for element_id, timestamp in self._timestamps.items()
                },
            }
        )

        return base_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpatioTemporalMatroid":
        """
        Create spatiotemporal matroid from dictionary representation

        Args:
            data: Dictionary representation

        Returns:
            SpatioTemporalMatroid instance
        """
        min_time_interval = timedelta(
            seconds=data.get("min_time_interval_seconds", 3600)
        )

        matroid = cls(
            name=data.get("name"),
            min_distance_km=data.get("min_distance_km", 5.0),
            min_time_interval=min_time_interval,
            max_elements_per_region=data.get("max_elements_per_region"),
        )

        # Add locations
        for element_id, location in data.get("locations", {}).items():
            matroid.add_location(
                element_id, location["latitude"], location["longitude"]
            )

        # Add timestamps
        for element_id, timestamp_str in data.get("timestamps", {}).items():
            if element_id in matroid._locations:
                matroid._timestamps[element_id] = datetime.fromisoformat(
                    timestamp_str
                )

        # Add elements to regions
        for region_id, elements in data.get("regions", {}).items():
            for element_id in elements:
                if element_id in matroid._locations:
                    if region_id not in matroid._regions:
                        matroid._regions[region_id] = set()
                    matroid._regions[region_id].add(element_id)

        return matroid


class GraphMatroid(Matroid[str]):
    """
    Matroid based on a graph structure.

    Elements are edges in a graph. Sets are independent if they form a forest
    (contain no cycles).
    """

    def __init__(self, name: str = None):
        """
        Initialize a graph matroid

        Args:
            name: Optional name for the matroid
        """
        super().__init__(name)
        self._vertices: Set[str] = set()
        self._edges: Dict[str, Tuple[str, str]] = (
            {}
        )  # edge_id -> (vertex1, vertex2)

    def add_vertex(self, vertex_id: str) -> None:
        """
        Add a vertex to the graph

        Args:
            vertex_id: Vertex identifier
        """
        self._vertices.add(vertex_id)

    def add_edge(self, edge_id: str, vertex1: str, vertex2: str) -> None:
        """
        Add an edge to the graph

        Args:
            edge_id: Edge identifier
            vertex1: First vertex
            vertex2: Second vertex
        """
        # Add vertices if they don't exist
        self.add_vertex(vertex1)
        self.add_vertex(vertex2)

        # Add edge
        self._edges[edge_id] = (vertex1, vertex2)
        self.add_to_ground_set(edge_id)

    def _check_independence(self, subset: Set[str]) -> bool:
        """
        Check if a subset of edges is independent (forms a forest)

        Args:
            subset: Subset of edges to check

        Returns:
            True if the subset is independent, False otherwise
        """
        # Get the edge tuples
        edges = [self._edges.get(edge_id) for edge_id in subset]
        edges = [edge for edge in edges if edge is not None]

        # Build an adjacency list
        adjacency: Dict[str, List[str]] = {}
        for v1, v2 in edges:
            if v1 not in adjacency:
                adjacency[v1] = []
            if v2 not in adjacency:
                adjacency[v2] = []
            adjacency[v1].append(v2)
            adjacency[v2].append(v1)

        # Check for cycles using DFS
        visited: Set[str] = set()

        def has_cycle(vertex: str, parent: Optional[str] = None) -> bool:
            visited.add(vertex)

            for neighbor in adjacency.get(vertex, []):
                if neighbor not in visited:
                    if has_cycle(neighbor, vertex):
                        return True
                elif neighbor != parent:
                    return True

            return False

        # Check each connected component
        for vertex in adjacency:
            if vertex not in visited:
                if has_cycle(vertex):
                    return False

        return True

    def find_minimum_spanning_tree(self, weights: Dict[str, float]) -> Set[str]:
        """
        Find a minimum spanning tree using Kruskal's algorithm

        Args:
            weights: Edge weights

        Returns:
            Set of edge identifiers forming a minimum spanning tree
        """
        # Sort edges by weight
        edges = [
            (edge_id, self._edges[edge_id], weights.get(edge_id, 0))
            for edge_id in self._edges
        ]
        edges.sort(key=lambda x: x[2])

        # Initialize union-find data structure
        parent: Dict[str, str] = {vertex: vertex for vertex in self._vertices}

        def find(vertex: str) -> str:
            if parent[vertex] != vertex:
                parent[vertex] = find(parent[vertex])
            return parent[vertex]

        def union(v1: str, v2: str) -> None:
            parent[find(v1)] = find(v2)

        # Kruskal's algorithm
        mst = set()
        for edge_id, (v1, v2), weight in edges:
            if find(v1) != find(v2):
                union(v1, v2)
                mst.add(edge_id)

        return mst

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert graph matroid to dictionary representation

        Returns:
            Dictionary representation
        """
        base_dict = super().to_dict()

        # Add graph-specific information
        base_dict.update(
            {
                "vertices": list(self._vertices),
                "edges": {
                    edge_id: {"vertex1": v1, "vertex2": v2}
                    for edge_id, (v1, v2) in self._edges.items()
                },
            }
        )

        return base_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphMatroid":
        """
        Create graph matroid from dictionary representation

        Args:
            data: Dictionary representation

        Returns:
            GraphMatroid instance
        """
        matroid = cls(name=data.get("name"))

        # Add vertices
        for vertex in data.get("vertices", []):
            matroid.add_vertex(vertex)

        # Add edges
        for edge_id, edge in data.get("edges", {}).items():
            matroid.add_edge(edge_id, edge["vertex1"], edge["vertex2"])

        return matroid
