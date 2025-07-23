"""
GeoJSON Utilities Module
----------------------
This module provides utilities for working with GeoJSON data
in the NyxTrace platform.
"""

import os
import json
import logging
import pandas as pd
import geopandas as gpd
import numpy as np
from typing import Dict, List, Any, Union, Optional, Tuple
from shapely.geometry import shape, mapping, Point, LineString, Polygon, MultiPolygon
import tempfile
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('geojson_utils')


class GeoJSONHandler:
    """
    Utilities for working with GeoJSON data
    
    This class provides methods for:
    - Loading and saving GeoJSON files
    - Manipulating GeoJSON data
    - Converting between GeoJSON and other formats
    - Working with GeoJSON in memory
    """
    
    def __init__(self, default_crs: str = "EPSG:4326"):
        """
        Initialize GeoJSON handler
        
        Args:
            default_crs: Default coordinate reference system
        """
        self.default_crs = default_crs
        # Create a temporary directory for temporary files
        self.temp_dir = tempfile.mkdtemp()
        logger.info(f"GeoJSON handler initialized with temporary directory: {self.temp_dir}")
    
    def load_geojson(self, file_path: str) -> Union[Dict[str, Any], None]:
        """
        Load GeoJSON from file
        
        Args:
            file_path: Path to GeoJSON file
            
        Returns:
            GeoJSON data as dictionary or None if error
        """
        try:
            with open(file_path, 'r') as f:
                geojson_data = json.load(f)
            
            logger.info(f"Loaded GeoJSON from {file_path}")
            return geojson_data
        except Exception as e:
            logger.error(f"Error loading GeoJSON from {file_path}: {str(e)}")
            return None
    
    def save_geojson(self, geojson_data: Dict[str, Any], file_path: str) -> bool:
        """
        Save GeoJSON to file
        
        Args:
            geojson_data: GeoJSON data
            file_path: Output file path
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            with open(file_path, 'w') as f:
                json.dump(geojson_data, f, indent=2)
            
            logger.info(f"Saved GeoJSON to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving GeoJSON to {file_path}: {str(e)}")
            return False
    
    def geojson_to_geodataframe(self, geojson_data: Dict[str, Any]) -> gpd.GeoDataFrame:
        """
        Convert GeoJSON to GeoDataFrame
        
        Args:
            geojson_data: GeoJSON data
            
        Returns:
            GeoDataFrame
        """
        try:
            gdf = gpd.GeoDataFrame.from_features(geojson_data["features"], crs=self.default_crs)
            logger.info(f"Converted GeoJSON to GeoDataFrame with {len(gdf)} features")
            return gdf
        except Exception as e:
            logger.error(f"Error converting GeoJSON to GeoDataFrame: {str(e)}")
            return gpd.GeoDataFrame()
    
    def geodataframe_to_geojson(self, gdf: gpd.GeoDataFrame) -> Dict[str, Any]:
        """
        Convert GeoDataFrame to GeoJSON
        
        Args:
            gdf: GeoDataFrame
            
        Returns:
            GeoJSON data as dictionary
        """
        try:
            # Ensure the GeoDataFrame is in the correct CRS
            if gdf.crs != self.default_crs:
                gdf = gdf.to_crs(self.default_crs)
            
            # Convert to GeoJSON
            geojson_data = json.loads(gdf.to_json())
            
            logger.info(f"Converted GeoDataFrame to GeoJSON with {len(gdf)} features")
            return geojson_data
        except Exception as e:
            logger.error(f"Error converting GeoDataFrame to GeoJSON: {str(e)}")
            return {"type": "FeatureCollection", "features": []}
    
    def filter_geojson(self, 
                     geojson_data: Dict[str, Any], 
                     property_filters: Optional[Dict[str, Any]] = None,
                     spatial_filter: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Filter GeoJSON features based on properties and/or spatial extent
        
        Args:
            geojson_data: GeoJSON data
            property_filters: Dictionary of property filters {property_name: value}
            spatial_filter: Spatial filter as GeoJSON geometry
            
        Returns:
            Filtered GeoJSON data
        """
        try:
            if not property_filters and not spatial_filter:
                return geojson_data
            
            # Start with all features
            features = geojson_data.get("features", [])
            filtered_features = []
            
            # Process each feature
            for feature in features:
                # Check property filters
                include_feature = True
                
                if property_filters:
                    properties = feature.get("properties", {})
                    for prop_name, prop_value in property_filters.items():
                        # Handle lists of values (OR condition)
                        if isinstance(prop_value, list):
                            if prop_name not in properties or properties[prop_name] not in prop_value:
                                include_feature = False
                                break
                        # Handle wildcards
                        elif isinstance(prop_value, str) and '*' in prop_value:
                            import re
                            pattern = '^' + prop_value.replace('*', '.*') + '$'
                            if prop_name not in properties or not re.match(pattern, str(properties[prop_name])):
                                include_feature = False
                                break
                        # Regular exact match
                        elif prop_name not in properties or properties[prop_name] != prop_value:
                            include_feature = False
                            break
                
                # Check spatial filter
                if include_feature and spatial_filter:
                    from shapely.geometry import shape
                    
                    # Convert features to shapely geometries
                    feature_geom = shape(feature.get("geometry", {}))
                    filter_geom = shape(spatial_filter)
                    
                    # Check intersection
                    if not feature_geom.intersects(filter_geom):
                        include_feature = False
                
                if include_feature:
                    filtered_features.append(feature)
            
            # Create new GeoJSON with filtered features
            filtered_geojson = {
                "type": "FeatureCollection",
                "features": filtered_features
            }
            
            logger.info(f"Filtered GeoJSON from {len(features)} to {len(filtered_features)} features")
            return filtered_geojson
        except Exception as e:
            logger.error(f"Error filtering GeoJSON: {str(e)}")
            return geojson_data
    
    def merge_geojson(self, geojson_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge multiple GeoJSON objects
        
        Args:
            geojson_list: List of GeoJSON data objects
            
        Returns:
            Merged GeoJSON data
        """
        try:
            all_features = []
            
            # Collect all features
            for geojson in geojson_list:
                features = geojson.get("features", [])
                all_features.extend(features)
            
            # Create merged GeoJSON
            merged_geojson = {
                "type": "FeatureCollection",
                "features": all_features
            }
            
            logger.info(f"Merged {len(geojson_list)} GeoJSON objects with total {len(all_features)} features")
            return merged_geojson
        except Exception as e:
            logger.error(f"Error merging GeoJSON objects: {str(e)}")
            return {"type": "FeatureCollection", "features": []}
    
    def simplify_geojson(self, 
                       geojson_data: Dict[str, Any], 
                       tolerance: float = 0.001) -> Dict[str, Any]:
        """
        Simplify GeoJSON geometries
        
        Args:
            geojson_data: GeoJSON data
            tolerance: Simplification tolerance
            
        Returns:
            Simplified GeoJSON data
        """
        try:
            # Convert to GeoDataFrame for simplification
            gdf = self.geojson_to_geodataframe(geojson_data)
            
            if gdf.empty:
                return geojson_data
            
            # Simplify geometries
            gdf['geometry'] = gdf['geometry'].simplify(tolerance)
            
            # Convert back to GeoJSON
            simplified_geojson = self.geodataframe_to_geojson(gdf)
            
            logger.info(f"Simplified GeoJSON geometries with tolerance {tolerance}")
            return simplified_geojson
        except Exception as e:
            logger.error(f"Error simplifying GeoJSON: {str(e)}")
            return geojson_data
    
    def add_property(self, 
                   geojson_data: Dict[str, Any], 
                   property_name: str, 
                   value: Any,
                   filter_func: Optional[callable] = None) -> Dict[str, Any]:
        """
        Add property to GeoJSON features
        
        Args:
            geojson_data: GeoJSON data
            property_name: Name of property to add
            value: Property value (or function to compute value)
            filter_func: Optional function to filter features
            
        Returns:
            GeoJSON with added property
        """
        try:
            features = geojson_data.get("features", [])
            
            for i, feature in enumerate(features):
                # Apply filter if provided
                if filter_func is None or filter_func(feature):
                    # Ensure properties exists
                    if "properties" not in feature:
                        feature["properties"] = {}
                    
                    # Compute value if it's a function
                    if callable(value):
                        property_value = value(feature)
                    else:
                        property_value = value
                    
                    # Add property
                    feature["properties"][property_name] = property_value
            
            logger.info(f"Added property '{property_name}' to GeoJSON features")
            return geojson_data
        except Exception as e:
            logger.error(f"Error adding property to GeoJSON: {str(e)}")
            return geojson_data
    
    def create_buffer(self, 
                    geojson_data: Dict[str, Any], 
                    distance: float,
                    dissolve: bool = False) -> Dict[str, Any]:
        """
        Create buffer around GeoJSON features
        
        Args:
            geojson_data: GeoJSON data
            distance: Buffer distance in degrees (for WGS84) or units (for projected CRS)
            dissolve: Whether to dissolve buffers into a single geometry
            
        Returns:
            Buffered GeoJSON data
        """
        try:
            # Convert to GeoDataFrame for buffer operation
            gdf = self.geojson_to_geodataframe(geojson_data)
            
            if gdf.empty:
                return geojson_data
            
            # For more accurate buffers, reproject to an appropriate projected CRS
            # This is a simple approximation using Web Mercator (not ideal for all locations)
            gdf_projected = gdf.to_crs("EPSG:3857")
            
            # Buffer in meters (Web Mercator units are meters)
            buffered = gdf_projected.buffer(distance)
            
            # Dissolve if requested
            if dissolve:
                from shapely.ops import unary_union
                buffered = gpd.GeoDataFrame(geometry=[unary_union(buffered)], crs="EPSG:3857")
            else:
                buffered = gpd.GeoDataFrame(geometry=buffered, crs="EPSG:3857")
            
            # Reproject back to WGS84
            buffered = buffered.to_crs(self.default_crs)
            
            # Convert to GeoJSON
            buffer_geojson = self.geodataframe_to_geojson(buffered)
            
            logger.info(f"Created buffer with distance {distance} around GeoJSON features")
            return buffer_geojson
        except Exception as e:
            logger.error(f"Error creating buffer around GeoJSON: {str(e)}")
            return geojson_data
    
    def create_grid(self, 
                  bounds: Tuple[float, float, float, float], 
                  cell_size: float, 
                  cell_type: str = 'square') -> Dict[str, Any]:
        """
        Create a grid of cells within bounds
        
        Args:
            bounds: Bounds (minx, miny, maxx, maxy)
            cell_size: Cell size in degrees (for WGS84)
            cell_type: Cell type ('square' or 'hexagon')
            
        Returns:
            GeoJSON grid
        """
        try:
            if cell_type == 'square':
                # Create square grid
                minx, miny, maxx, maxy = bounds
                
                # Calculate number of cells
                nx = int((maxx - minx) / cell_size)
                ny = int((maxy - miny) / cell_size)
                
                # Create cells
                cells = []
                
                for i in range(nx):
                    for j in range(ny):
                        # Calculate cell bounds
                        cell_minx = minx + i * cell_size
                        cell_miny = miny + j * cell_size
                        cell_maxx = minx + (i + 1) * cell_size
                        cell_maxy = miny + (j + 1) * cell_size
                        
                        # Create cell polygon
                        cell = Polygon([
                            (cell_minx, cell_miny),
                            (cell_maxx, cell_miny),
                            (cell_maxx, cell_maxy),
                            (cell_minx, cell_maxy),
                            (cell_minx, cell_miny)
                        ])
                        
                        # Add to cells
                        cells.append({
                            "type": "Feature",
                            "properties": {
                                "cell_id": f"{i}_{j}",
                                "cell_x": i,
                                "cell_y": j
                            },
                            "geometry": mapping(cell)
                        })
                
                # Create GeoJSON
                grid_geojson = {
                    "type": "FeatureCollection",
                    "features": cells
                }
                
                logger.info(f"Created square grid with {len(cells)} cells")
                return grid_geojson
                
            elif cell_type == 'hexagon':
                # For hexagonal grid, use H3 if available
                try:
                    import h3
                    
                    # Convert bounds to center point
                    center_lat = (bounds[1] + bounds[3]) / 2
                    center_lng = (bounds[0] + bounds[2]) / 2
                    
                    # Determine appropriate H3 resolution
                    # Resolution 9 is approximately 0.1 km²
                    resolution = 9
                    
                    # Get hexagon at center
                    center_hex = h3.geo_to_h3(center_lat, center_lng, resolution)
                    
                    # Get hexagon size
                    hex_size = h3.edge_length(resolution, unit='km')
                    
                    # Calculate number of rings needed to cover bounds
                    diagonal = ((bounds[2] - bounds[0])**2 + (bounds[3] - bounds[1])**2)**0.5
                    rings = int(diagonal / (hex_size * 0.009))  # Convert km to degrees (rough approximation)
                    
                    # Get hexagons around center
                    hexagons = h3.k_ring(center_hex, rings)
                    
                    # Create features
                    features = []
                    
                    for hex_id in hexagons:
                        # Get boundary as polygon
                        boundary = h3.h3_to_geo_boundary(hex_id, geo_json=True)
                        
                        # Create feature
                        features.append({
                            "type": "Feature",
                            "properties": {
                                "hex_id": hex_id,
                                "resolution": resolution
                            },
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [boundary]
                            }
                        })
                    
                    # Create GeoJSON
                    grid_geojson = {
                        "type": "FeatureCollection",
                        "features": features
                    }
                    
                    logger.info(f"Created hexagonal grid with {len(features)} cells")
                    return grid_geojson
                    
                except ImportError:
                    logger.warning("H3 library not available, falling back to square grid")
                    return self.create_grid(bounds, cell_size, 'square')
            else:
                logger.warning(f"Unknown cell type: {cell_type}, using square grid")
                return self.create_grid(bounds, cell_size, 'square')
        except Exception as e:
            logger.error(f"Error creating grid: {str(e)}")
            return {"type": "FeatureCollection", "features": []}
    
    def create_voronoi(self, 
                     geojson_data: Dict[str, Any], 
                     clip_to_bounds: bool = True) -> Dict[str, Any]:
        """
        Create Voronoi diagram from point features
        
        Args:
            geojson_data: GeoJSON data with point features
            clip_to_bounds: Whether to clip Voronoi cells to a bounding rectangle
            
        Returns:
            GeoJSON Voronoi diagram
        """
        try:
            # Convert to GeoDataFrame
            gdf = self.geojson_to_geodataframe(geojson_data)
            
            if gdf.empty:
                return geojson_data
            
            # Check if geometries are points
            if not all(geom.geom_type == 'Point' for geom in gdf.geometry):
                logger.warning("Voronoi diagram requires point geometries")
                return geojson_data
            
            # Extract points
            points = np.array([(geom.x, geom.y) for geom in gdf.geometry])
            
            # Create Voronoi diagram
            from scipy.spatial import Voronoi
            vor = Voronoi(points)
            
            # Create Voronoi polygons
            from shapely.geometry import Polygon
            
            # Get bounds for clipping
            if clip_to_bounds:
                minx, miny, maxx, maxy = gdf.total_bounds
                # Add some padding
                padding = 0.1 * max(maxx - minx, maxy - miny)
                bounds = (minx - padding, miny - padding, maxx + padding, maxy + padding)
                clip_polygon = Polygon([
                    (bounds[0], bounds[1]),
                    (bounds[2], bounds[1]),
                    (bounds[2], bounds[3]),
                    (bounds[0], bounds[3]),
                    (bounds[0], bounds[1])
                ])
            
            # Create polygons
            voronio_polygons = []
            
            for i, region_idx in enumerate(vor.point_region):
                region = vor.regions[region_idx]
                
                # Skip infinite regions
                if -1 in region:
                    continue
                
                # Get region vertices
                polygon_vertices = [vor.vertices[i] for i in region]
                
                # Create Polygon
                if len(polygon_vertices) > 2:  # Need at least 3 points for a polygon
                    voronoi_polygon = Polygon(polygon_vertices)
                    
                    # Clip to bounds if requested
                    if clip_to_bounds:
                        voronoi_polygon = voronoi_polygon.intersection(clip_polygon)
                    
                    # Get properties from original point
                    properties = gdf.iloc[i].to_dict()
                    if 'geometry' in properties:
                        del properties['geometry']
                    
                    # Add polygon to result
                    voronio_polygons.append({
                        "type": "Feature",
                        "properties": properties,
                        "geometry": mapping(voronoi_polygon)
                    })
            
            # Create GeoJSON
            voronoi_geojson = {
                "type": "FeatureCollection",
                "features": voronio_polygons
            }
            
            logger.info(f"Created Voronoi diagram with {len(voronio_polygons)} cells")
            return voronoi_geojson
            
        except Exception as e:
            logger.error(f"Error creating Voronoi diagram: {str(e)}")
            return geojson_data
    
    def create_heatmap_grid(self, 
                         point_geojson: Dict[str, Any],
                         value_property: Optional[str] = None,
                         cell_size: float = 0.01,
                         kernel_radius: int = 3) -> Dict[str, Any]:
        """
        Create a heatmap grid from point data
        
        Args:
            point_geojson: GeoJSON with point features
            value_property: Optional property to use as intensity value
            cell_size: Cell size in degrees
            kernel_radius: Radius for kernel density estimation
            
        Returns:
            GeoJSON grid with heatmap values
        """
        try:
            # Convert to GeoDataFrame
            gdf = self.geojson_to_geodataframe(point_geojson)
            
            if gdf.empty:
                return point_geojson
            
            # Check if geometries are points
            if not all(geom.geom_type == 'Point' for geom in gdf.geometry):
                logger.warning("Heatmap requires point geometries")
                return point_geojson
            
            # Get bounds
            minx, miny, maxx, maxy = gdf.total_bounds
            
            # Create grid
            grid_geojson = self.create_grid((minx, miny, maxx, maxy), cell_size, 'square')
            grid_gdf = self.geojson_to_geodataframe(grid_geojson)
            
            # Count points in each grid cell
            grid_gdf['count'] = 0
            grid_gdf['value'] = 0.0
            
            # Process points
            for idx, point in gdf.iterrows():
                # Find which grid cell contains this point
                contains_mask = grid_gdf.geometry.contains(point.geometry)
                
                if contains_mask.any():
                    cell_idx = contains_mask.idxmax()
                    
                    # Increment count
                    grid_gdf.at[cell_idx, 'count'] += 1
                    
                    # Add value if value_property specified
                    if value_property and value_property in point:
                        try:
                            value = float(point[value_property])
                            grid_gdf.at[cell_idx, 'value'] += value
                        except (ValueError, TypeError):
                            pass
            
            # Apply kernel smoothing
            if kernel_radius > 0:
                from scipy.ndimage import gaussian_filter
                
                # Reshape to 2D grid for filtering
                nx = len(grid_gdf['cell_x'].unique())
                ny = len(grid_gdf['cell_y'].unique())
                
                # Use count or value
                if value_property:
                    # Create 2D grid of values
                    grid_values = np.zeros((ny, nx))
                    for idx, cell in grid_gdf.iterrows():
                        x = int(cell['cell_x'])
                        y = int(cell['cell_y'])
                        if x < nx and y < ny:
                            grid_values[y, x] = cell['value']
                    
                    # Apply gaussian filter
                    smoothed = gaussian_filter(grid_values, sigma=kernel_radius)
                    
                    # Put back into GeoDataFrame
                    for idx, cell in grid_gdf.iterrows():
                        x = int(cell['cell_x'])
                        y = int(cell['cell_y'])
                        if x < nx and y < ny:
                            grid_gdf.at[idx, 'heatmap'] = smoothed[y, x]
                else:
                    # Create 2D grid of counts
                    grid_counts = np.zeros((ny, nx))
                    for idx, cell in grid_gdf.iterrows():
                        x = int(cell['cell_x'])
                        y = int(cell['cell_y'])
                        if x < nx and y < ny:
                            grid_counts[y, x] = cell['count']
                    
                    # Apply gaussian filter
                    smoothed = gaussian_filter(grid_counts, sigma=kernel_radius)
                    
                    # Put back into GeoDataFrame
                    for idx, cell in grid_gdf.iterrows():
                        x = int(cell['cell_x'])
                        y = int(cell['cell_y'])
                        if x < nx and y < ny:
                            grid_gdf.at[idx, 'heatmap'] = smoothed[y, x]
            else:
                # No smoothing, just use count or value
                if value_property:
                    grid_gdf['heatmap'] = grid_gdf['value']
                else:
                    grid_gdf['heatmap'] = grid_gdf['count']
            
            # Convert back to GeoJSON
            heatmap_geojson = self.geodataframe_to_geojson(grid_gdf)
            
            logger.info(f"Created heatmap grid with {len(grid_gdf)} cells")
            return heatmap_geojson
            
        except Exception as e:
            logger.error(f"Error creating heatmap grid: {str(e)}")
            return point_geojson