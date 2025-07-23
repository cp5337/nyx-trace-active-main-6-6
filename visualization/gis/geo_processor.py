"""
Geospatial Data Processor
-----------------------
This module provides core geospatial data processing functionality for the NyxTrace platform.
"""

import os
import json
import logging
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, Polygon, LineString, shape
from shapely.ops import transform, unary_union
import pyproj
from functools import partial
from rtree import index
from typing import Dict, List, Any, Union, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('geo_processor')


class GeoProcessor:
    """
    Core class for geospatial data processing
    
    This class provides methods for:
    - Coordinate transformations
    - Spatial operations
    - Data standardization
    - Spatial indexing
    """
    
    def __init__(self, default_crs: str = 'EPSG:4326'):
        """
        Initialize the geospatial processor
        
        Args:
            default_crs: Default coordinate reference system (default: WGS84)
        """
        self.default_crs = default_crs
        self.spatial_index = None
        logger.info(f"Geospatial processor initialized with default CRS: {default_crs}")
    
    def create_point_geodataframe(self, 
                                 df: pd.DataFrame, 
                                 lat_col: str = 'lat', 
                                 lon_col: str = 'lon', 
                                 crs: str = 'EPSG:4326') -> gpd.GeoDataFrame:
        """
        Create a GeoDataFrame from a DataFrame with lat/lon columns
        
        Args:
            df: DataFrame with latitude and longitude columns
            lat_col: Name of latitude column
            lon_col: Name of longitude column
            crs: Coordinate reference system
            
        Returns:
            GeoDataFrame with Point geometry
        """
        if df.empty:
            return gpd.GeoDataFrame(columns=df.columns, geometry=[], crs=crs)
        
        # Create geometry column
        geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=crs)
        
        logger.info(f"Created point GeoDataFrame with {len(gdf)} points")
        
        return gdf
    
    def load_geojson(self, file_path: str) -> gpd.GeoDataFrame:
        """
        Load a GeoJSON file into a GeoDataFrame
        
        Args:
            file_path: Path to GeoJSON file
            
        Returns:
            GeoDataFrame with loaded features
        """
        try:
            gdf = gpd.read_file(file_path)
            logger.info(f"Loaded GeoJSON from {file_path} with {len(gdf)} features")
            return gdf
        except Exception as e:
            logger.error(f"Error loading GeoJSON from {file_path}: {str(e)}")
            return gpd.GeoDataFrame()
    
    def save_geojson(self, gdf: gpd.GeoDataFrame, file_path: str) -> bool:
        """
        Save a GeoDataFrame to a GeoJSON file
        
        Args:
            gdf: GeoDataFrame to save
            file_path: Path to save GeoJSON file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save to GeoJSON
            gdf.to_file(file_path, driver="GeoJSON")
            
            logger.info(f"Saved GeoDataFrame to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving GeoDataFrame to {file_path}: {str(e)}")
            return False
    
    def reproject_geodataframe(self, 
                              gdf: gpd.GeoDataFrame, 
                              target_crs: str = 'EPSG:4326') -> gpd.GeoDataFrame:
        """
        Reproject a GeoDataFrame to a different coordinate reference system
        
        Args:
            gdf: GeoDataFrame to reproject
            target_crs: Target coordinate reference system
            
        Returns:
            Reprojected GeoDataFrame
        """
        if gdf.empty:
            return gdf
        
        # Check if reprojection is needed
        if gdf.crs == target_crs:
            return gdf
        
        # Reproject
        reprojected = gdf.copy()
        reprojected = reprojected.to_crs(target_crs)
        
        logger.info(f"Reprojected GeoDataFrame from {gdf.crs} to {target_crs}")
        
        return reprojected
    
    def calculate_distance(self, 
                          point1: Tuple[float, float], 
                          point2: Tuple[float, float], 
                          crs: str = 'EPSG:4326', 
                          unit: str = 'km') -> float:
        """
        Calculate the distance between two points
        
        Args:
            point1: First point (lon, lat)
            point2: Second point (lon, lat)
            crs: Coordinate reference system
            unit: Unit of measurement ('m', 'km', 'mi')
            
        Returns:
            Distance in specified units
        """
        # Create Point objects
        p1 = Point(point1)
        p2 = Point(point2)
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(geometry=[p1, p2], crs=crs)
        
        # Reproject to a projected CRS for accurate measurements
        # UTM zones are ideal for distance calculations
        # This is a approximation - ideally we would select the appropriate UTM zone
        utm_crs = 'EPSG:3857'  # Web Mercator
        gdf_projected = gdf.to_crs(utm_crs)
        
        # Calculate distance in meters
        distance_m = gdf_projected.iloc[0].geometry.distance(gdf_projected.iloc[1].geometry)
        
        # Convert to requested unit
        if unit == 'km':
            return distance_m / 1000
        elif unit == 'mi':
            return distance_m / 1609.34
        else:
            return distance_m
    
    def buffer_points(self, 
                     gdf: gpd.GeoDataFrame, 
                     distance: float, 
                     distance_unit: str = 'km',
                     dissolve: bool = False) -> gpd.GeoDataFrame:
        """
        Create buffer around points
        
        Args:
            gdf: GeoDataFrame with Point geometries
            distance: Buffer distance
            distance_unit: Unit of buffer distance ('m', 'km', 'mi')
            dissolve: Whether to dissolve buffers into a single polygon
            
        Returns:
            GeoDataFrame with buffered geometries
        """
        if gdf.empty:
            return gdf
        
        # Convert distance to meters
        if distance_unit == 'km':
            distance_m = distance * 1000
        elif distance_unit == 'mi':
            distance_m = distance * 1609.34
        else:
            distance_m = distance
        
        # Reproject to a projected CRS for accurate buffer
        utm_crs = 'EPSG:3857'  # Web Mercator
        gdf_projected = gdf.to_crs(utm_crs)
        
        # Create buffer
        buffered = gdf_projected.copy()
        buffered['geometry'] = gdf_projected.buffer(distance_m)
        
        # Dissolve if requested
        if dissolve and len(buffered) > 1:
            buffered = buffered.dissolve()
        
        # Reproject back to original CRS
        buffered = buffered.to_crs(gdf.crs)
        
        logger.info(f"Created buffer with distance {distance} {distance_unit}")
        
        return buffered
    
    def create_hexagons(self, 
                      bounds: Tuple[float, float, float, float], 
                      resolution: int = 9,
                      crs: str = 'EPSG:4326') -> gpd.GeoDataFrame:
        """
        Create a hexagonal grid within the given bounds
        
        Args:
            bounds: Bounds (minx, miny, maxx, maxy)
            resolution: H3 resolution (0-15, with higher numbers being more granular)
            crs: Coordinate reference system
            
        Returns:
            GeoDataFrame with hexagonal grid
        """
        try:
            import h3
            
            # Create a polygon from the bounds
            bbox = Polygon([
                (bounds[0], bounds[1]),
                (bounds[2], bounds[1]),
                (bounds[2], bounds[3]),
                (bounds[0], bounds[3]),
                (bounds[0], bounds[1])
            ])
            
            # Get the center point of the bounds
            center_lat = (bounds[1] + bounds[3]) / 2
            center_lng = (bounds[0] + bounds[2]) / 2
            
            # Get the H3 index for the center point
            center_h3 = h3.geo_to_h3(center_lat, center_lng, resolution)
            
            # Get the hexagon
            center_hex = h3.h3_to_geo_boundary(center_h3, geo_json=True)
            
            # Create a buffer around the bounds to ensure we cover the entire area
            # The buffer size depends on the resolution
            buffer_size = 2 ** (15 - resolution) * 0.01  # Approximate buffer size
            
            # Get all hexagons within the bounding box
            hex_indexes = h3.polyfill(
                {"type": "Polygon", "coordinates": [bbox.exterior.coords]},
                resolution,
                geo_json_conformant=True
            )
            
            # Convert H3 indexes to polygons
            hexagons = []
            for h3_index in hex_indexes:
                hex_boundary = h3.h3_to_geo_boundary(h3_index, geo_json=True)
                hex_polygon = Polygon(hex_boundary)
                hexagons.append({
                    'geometry': hex_polygon,
                    'h3_index': h3_index
                })
            
            # Create GeoDataFrame
            hex_gdf = gpd.GeoDataFrame(hexagons, crs=crs)
            
            logger.info(f"Created hexagonal grid with {len(hex_gdf)} hexagons at resolution {resolution}")
            
            return hex_gdf
            
        except ImportError:
            logger.error("H3 library not installed. Install with 'pip install h3'")
            return gpd.GeoDataFrame()
    
    def spatial_join(self, 
                    left_gdf: gpd.GeoDataFrame, 
                    right_gdf: gpd.GeoDataFrame, 
                    how: str = 'inner', 
                    predicate: str = 'intersects') -> gpd.GeoDataFrame:
        """
        Perform a spatial join between two GeoDataFrames
        
        Args:
            left_gdf: Left GeoDataFrame
            right_gdf: Right GeoDataFrame
            how: Join type ('inner', 'left', 'right')
            predicate: Spatial predicate ('intersects', 'contains', 'within', etc.)
            
        Returns:
            Joined GeoDataFrame
        """
        if left_gdf.empty or right_gdf.empty:
            return gpd.GeoDataFrame()
        
        # Make sure both GeoDataFrames have the same CRS
        if left_gdf.crs != right_gdf.crs:
            right_gdf = right_gdf.to_crs(left_gdf.crs)
        
        # Perform spatial join
        joined = gpd.sjoin(left_gdf, right_gdf, how=how, predicate=predicate)
        
        logger.info(f"Performed spatial join with {len(joined)} resulting features")
        
        return joined
    
    def build_spatial_index(self, gdf: gpd.GeoDataFrame) -> None:
        """
        Build a spatial index for the GeoDataFrame
        
        Args:
            gdf: GeoDataFrame to index
        """
        if gdf.empty:
            self.spatial_index = None
            return
        
        # Create R-tree index
        idx = index.Index()
        
        # Populate index
        for i, geom in enumerate(gdf.geometry):
            idx.insert(i, geom.bounds)
        
        self.spatial_index = {
            'index': idx,
            'gdf': gdf
        }
        
        logger.info(f"Built spatial index with {len(gdf)} features")
    
    def query_spatial_index(self, 
                          geom: Any, 
                          return_geometries: bool = False) -> Union[List[int], gpd.GeoDataFrame]:
        """
        Query the spatial index
        
        Args:
            geom: Geometry to query with (Point, Polygon, etc.)
            return_geometries: Whether to return geometries or indices
            
        Returns:
            List of indices or GeoDataFrame with matching features
        """
        if self.spatial_index is None:
            logger.warning("Spatial index not built. Call build_spatial_index() first")
            return [] if not return_geometries else gpd.GeoDataFrame()
        
        # Get bounds
        bounds = geom.bounds if hasattr(geom, 'bounds') else geom
        
        # Query index
        matching_indices = list(self.spatial_index['index'].intersection(bounds))
        
        # Return indices or geometries
        if return_geometries:
            return self.spatial_index['gdf'].iloc[matching_indices]
        else:
            return matching_indices
    
    def extract_coordinates(self, 
                           address: str, 
                           provider: str = 'nominatim',
                           timeout: int = 5) -> Optional[Tuple[float, float]]:
        """
        Extract coordinates from an address using geocoding
        
        Args:
            address: Address to geocode
            provider: Geocoding provider ('nominatim', 'google', etc.)
            timeout: Timeout in seconds
            
        Returns:
            Tuple of (longitude, latitude) or None if geocoding failed
        """
        try:
            from geopy.geocoders import Nominatim, GoogleV3
            from geopy.extra.rate_limiter import RateLimiter
            
            # Set up geocoder
            if provider == 'nominatim':
                geocoder = Nominatim(user_agent="nyxtrace-geospatial-platform")
            elif provider == 'google':
                api_key = os.environ.get('GOOGLE_MAPS_API_KEY')
                if not api_key:
                    logger.warning("No Google Maps API key found. Using Nominatim instead")
                    geocoder = Nominatim(user_agent="nyxtrace-geospatial-platform")
                else:
                    geocoder = GoogleV3(api_key=api_key)
            else:
                logger.warning(f"Unknown geocoding provider: {provider}. Using Nominatim")
                geocoder = Nominatim(user_agent="nyxtrace-geospatial-platform")
            
            # Add rate limiting
            geocode = RateLimiter(geocoder.geocode, min_delay_seconds=1)
            
            # Geocode
            location = geocode(address, timeout=timeout)
            
            if location is None:
                logger.warning(f"Could not geocode address: {address}")
                return None
            
            return (location.longitude, location.latitude)
            
        except ImportError:
            logger.error("Geopy library not installed. Install with 'pip install geopy'")
            return None
        except Exception as e:
            logger.error(f"Error geocoding address {address}: {str(e)}")
            return None
    
    def calculate_viewshed(self, 
                          observer_point: Point, 
                          terrain_raster: str, 
                          max_distance: float = 5000,
                          observer_height: float = 1.8) -> Optional[gpd.GeoDataFrame]:
        """
        Calculate the viewshed from an observer point
        
        This is a placeholder - implementing a full viewshed analysis
        would require additional libraries and elevation data.
        
        Args:
            observer_point: Point representing the observer location
            terrain_raster: Path to terrain raster file
            max_distance: Maximum visible distance in meters
            observer_height: Height of the observer in meters
            
        Returns:
            GeoDataFrame with viewshed polygon or None if calculation failed
        """
        # This would require a more complex integration with GDAL/rasterio
        # and elevation data. It's a placeholder for future implementation.
        
        logger.warning("Viewshed analysis not fully implemented yet")
        return None
    
    def calculate_isochrones(self, 
                           center_point: Point, 
                           time_limits: List[int] = [5, 10, 15],
                           mode: str = 'drive',
                           crs: str = 'EPSG:4326') -> Optional[gpd.GeoDataFrame]:
        """
        Calculate isochrones (travel time polygons) from a center point
        
        This uses OpenStreetMap data and the OSMnx library to calculate
        areas reachable within specified time limits.
        
        Args:
            center_point: Center point for isochrone calculation
            time_limits: List of time limits in minutes
            mode: Travel mode ('drive', 'walk', 'bike')
            crs: Coordinate reference system
            
        Returns:
            GeoDataFrame with isochrone polygons or None if calculation failed
        """
        try:
            import osmnx as ox
            import networkx as nx
            
            # Configure OSMnx
            ox.config(use_cache=True, log_console=False)
            
            # Convert mode to network type
            if mode == 'drive':
                network_type = 'drive'
            elif mode == 'walk':
                network_type = 'walk'
            elif mode == 'bike':
                network_type = 'bike'
            else:
                logger.warning(f"Unknown travel mode: {mode}. Using 'drive'")
                network_type = 'drive'
            
            # Get coordinates
            lat, lng = center_point.y, center_point.x
            
            # Calculate max distance to cover all time limits
            # Assuming average speeds: drive=50 km/h, walk=5 km/h, bike=15 km/h
            speeds = {'drive': 50, 'walk': 5, 'bike': 15}
            speed = speeds.get(mode, 50)  # km/h
            
            # Max distance in meters (convert minutes to hours, then to meters)
            max_time = max(time_limits) / 60  # hours
            max_distance = max_time * speed * 1000 * 1.5  # meters, with 50% buffer
            
            # Download street network around center point
            graph = ox.graph_from_point((lat, lng), dist=max_distance, network_type=network_type)
            
            # Get nearest node to center point
            center_node = ox.distance.nearest_nodes(graph, lng, lat)
            
            # Calculate travel time for each edge
            # Assuming speed limits according to road type
            speeds = {'motorway': 100, 'trunk': 80, 'primary': 60, 'secondary': 50, 
                     'tertiary': 40, 'residential': 30, 'service': 20, 'unclassified': 30}
            
            # Set default speed based on mode
            default_speed = speed
            
            # Add travel time in seconds to each edge
            for u, v, k, data in graph.edges(keys=True, data=True):
                if 'highway' in data:
                    highway = data['highway']
                    if isinstance(highway, list):
                        highway = highway[0]
                    speed_kph = speeds.get(highway, default_speed)
                else:
                    speed_kph = default_speed
                
                # Length is in meters, speed in km/h, time in seconds
                data['travel_time'] = data['length'] / (speed_kph * 1000 / 3600)
            
            # Calculate shortest paths
            # This will get all nodes reachable within time limits
            isochrone_polys = []
            
            for time_limit in sorted(time_limits):
                # Convert minutes to seconds
                time_limit_s = time_limit * 60
                
                # Calculate subgraph of nodes within time limit
                reachable_nodes = nx.ego_graph(graph, center_node, radius=time_limit_s, distance='travel_time')
                
                # Create a polygon from the convex hull of reachable nodes
                node_points = [Point(data['x'], data['y']) for _, data in reachable_nodes.nodes(data=True)]
                
                if len(node_points) < 3:
                    logger.warning(f"Not enough nodes for isochrone at {time_limit} minutes")
                    continue
                
                # Create convex hull
                node_gdf = gpd.GeoDataFrame(geometry=node_points, crs=crs)
                convex_hull = node_gdf.unary_union.convex_hull
                
                # Add to results
                isochrone_polys.append({
                    'geometry': convex_hull,
                    'time_limit': time_limit
                })
            
            # Create GeoDataFrame
            isochrones = gpd.GeoDataFrame(isochrone_polys, crs=crs)
            
            logger.info(f"Calculated isochrones for {len(time_limits)} time limits")
            
            return isochrones
            
        except ImportError:
            logger.error("Required libraries not installed. Install with 'pip install osmnx networkx'")
            return None
        except Exception as e:
            logger.error(f"Error calculating isochrones: {str(e)}")
            return None
    
    def detect_hotspots(self, 
                      point_gdf: gpd.GeoDataFrame, 
                      intensity_col: Optional[str] = None,
                      method: str = 'kde',
                      n_clusters: int = 5) -> gpd.GeoDataFrame:
        """
        Detect hotspots in point data
        
        Args:
            point_gdf: GeoDataFrame with Point geometries
            intensity_col: Optional column with intensity values
            method: Method to use ('kde', 'dbscan', 'kmeans')
            n_clusters: Number of clusters for KMeans
            
        Returns:
            GeoDataFrame with hotspot polygons
        """
        if point_gdf.empty:
            return gpd.GeoDataFrame()
        
        # Extract coordinates
        coords = np.array([(p.x, p.y) for p in point_gdf.geometry])
        
        if method == 'kde':
            try:
                from sklearn.neighbors import KernelDensity
                from scipy.ndimage import gaussian_filter
                from shapely.geometry import Polygon, MultiPolygon
                
                # Compute KDE
                kde = KernelDensity(bandwidth=0.01, metric='haversine')
                kde.fit(coords)
                
                # Create a grid of points
                x_min, y_min, x_max, y_max = point_gdf.total_bounds
                x_grid, y_grid = np.meshgrid(
                    np.linspace(x_min, x_max, 100),
                    np.linspace(y_min, y_max, 100)
                )
                grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T
                
                # Compute density at each grid point
                density = kde.score_samples(grid_points)
                density = density.reshape(x_grid.shape)
                
                # Apply Gaussian filter to smooth
                density = gaussian_filter(density, sigma=1)
                
                # Identify hotspots (points above a threshold)
                threshold = np.percentile(density, 75)  # Top 25%
                
                # Create contour polygons using matplotlib
                import matplotlib.pyplot as plt
                from matplotlib import path
                
                fig, ax = plt.subplots()
                contour = ax.contour(x_grid, y_grid, density, levels=[threshold])
                plt.close(fig)
                
                # Convert contours to polygons
                polygons = []
                for i, collection in enumerate(contour.collections):
                    for path in collection.get_paths():
                        path = path.to_polygons()[0]
                        if len(path) < 3:
                            continue
                        polygon = Polygon(path)
                        if polygon.is_valid:
                            polygons.append({
                                'geometry': polygon,
                                'density': threshold,
                                'rank': i + 1
                            })
                
                # Create GeoDataFrame
                hotspots = gpd.GeoDataFrame(polygons, crs=point_gdf.crs)
                
                logger.info(f"Detected {len(hotspots)} hotspot areas using KDE")
                
                return hotspots
                
            except ImportError:
                logger.error("Required libraries not installed. Install with 'pip install scikit-learn scipy'")
                return gpd.GeoDataFrame()
                
        elif method == 'dbscan':
            try:
                from sklearn.cluster import DBSCAN
                from shapely.geometry import MultiPoint
                
                # Compute DBSCAN clustering
                # eps is in degrees, so this is approximate
                eps = 0.01  # ~1km at the equator
                min_samples = 5
                
                db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
                
                # Get cluster labels
                labels = db.labels_
                
                # Number of clusters (excluding noise points with label -1)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                
                # Create a GeoDataFrame with cluster information
                point_gdf['cluster'] = labels
                
                # Create convex hulls for each cluster
                hulls = []
                
                for cluster_id in range(n_clusters):
                    # Get points in cluster
                    cluster_points = point_gdf[point_gdf['cluster'] == cluster_id]
                    
                    if len(cluster_points) < 3:
                        continue
                    
                    # Create convex hull
                    hull = cluster_points.unary_union.convex_hull
                    
                    hulls.append({
                        'geometry': hull,
                        'cluster_id': cluster_id,
                        'point_count': len(cluster_points)
                    })
                
                # Create GeoDataFrame
                hotspots = gpd.GeoDataFrame(hulls, crs=point_gdf.crs)
                
                logger.info(f"Detected {len(hotspots)} hotspot areas using DBSCAN")
                
                return hotspots
                
            except ImportError:
                logger.error("Scikit-learn not installed. Install with 'pip install scikit-learn'")
                return gpd.GeoDataFrame()
                
        elif method == 'kmeans':
            try:
                from sklearn.cluster import KMeans
                
                # Compute KMeans clustering
                kmeans = KMeans(n_clusters=n_clusters).fit(coords)
                
                # Get cluster labels
                labels = kmeans.labels_
                
                # Add to original GeoDataFrame
                point_gdf['cluster'] = labels
                
                # Create convex hulls for each cluster
                hulls = []
                
                for cluster_id in range(n_clusters):
                    # Get points in cluster
                    cluster_points = point_gdf[point_gdf['cluster'] == cluster_id]
                    
                    if len(cluster_points) < 3:
                        continue
                    
                    # Create convex hull
                    hull = cluster_points.unary_union.convex_hull
                    
                    hulls.append({
                        'geometry': hull,
                        'cluster_id': cluster_id,
                        'point_count': len(cluster_points)
                    })
                
                # Create GeoDataFrame
                hotspots = gpd.GeoDataFrame(hulls, crs=point_gdf.crs)
                
                logger.info(f"Detected {len(hotspots)} hotspot areas using KMeans")
                
                return hotspots
                
            except ImportError:
                logger.error("Scikit-learn not installed. Install with 'pip install scikit-learn'")
                return gpd.GeoDataFrame()
        
        else:
            logger.warning(f"Unknown hotspot detection method: {method}")
            return gpd.GeoDataFrame()