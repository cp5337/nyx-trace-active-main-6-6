"""
WhiteboxTools Integration Module
----------------------------
This module provides advanced geospatial analysis functionality
using the WhiteboxTools library for the NyxTrace platform.
"""

import os
import json
import logging
import tempfile
import pandas as pd
import geopandas as gpd
import numpy as np
from typing import Dict, List, Any, Union, Optional, Tuple
from pathlib import Path
import whitebox

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('whitebox_tools')


class WhiteboxIntegration:
    """
    WhiteboxTools integration for advanced geospatial analysis
    
    This class provides methods for:
    - Terrain analysis
    - Hydrological modeling
    - LiDAR data processing
    - Image analysis
    - And more geospatial operations
    """
    
    def __init__(self, whitebox_dir: Optional[str] = None):
        """
        Initialize WhiteboxTools integration
        
        Args:
            whitebox_dir: Directory containing WhiteboxTools executable (optional)
        """
        self.wbt = whitebox.WhiteboxTools()
        
        # Set WhiteboxTools executable directory if provided
        if whitebox_dir:
            if os.path.exists(whitebox_dir):
                self.wbt.set_whitebox_dir(whitebox_dir)
                logger.info(f"WhiteboxTools directory set to: {whitebox_dir}")
            else:
                logger.warning(f"Provided WhiteboxTools directory does not exist: {whitebox_dir}")
        
        # Log version information
        logger.info(f"WhiteboxTools version: {self.wbt.version()}")
        
        # Create a temporary working directory
        self.temp_dir = tempfile.mkdtemp()
        logger.info(f"Created temporary directory for WhiteboxTools: {self.temp_dir}")
    
    def __del__(self):
        """Clean up temporary files on destruction"""
        try:
            import shutil
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                logger.info(f"Removed temporary directory: {self.temp_dir}")
        except Exception as e:
            logger.error(f"Error cleaning up temporary directory: {str(e)}")
    
    def list_tools(self, keywords: Optional[List[str]] = None) -> List[Dict[str, str]]:
        """
        List available WhiteboxTools tools
        
        Args:
            keywords: Optional list of keywords to filter tools
            
        Returns:
            List of dictionaries with tool information
        """
        # Get tool listing from WhiteboxTools
        tools_json = self.wbt.list_tools()
        tools = json.loads(tools_json)
        
        # Filter by keywords if provided
        if keywords:
            filtered_tools = []
            for tool in tools:
                # Check if any keyword is in the tool name or description
                if any(keyword.lower() in tool['name'].lower() or 
                      keyword.lower() in tool['description'].lower() 
                      for keyword in keywords):
                    filtered_tools.append(tool)
            return filtered_tools
        else:
            return tools
    
    def generate_dem_derivatives(self, 
                               dem_file: str, 
                               output_dir: str,
                               derivatives: List[str] = ['slope', 'aspect', 'hillshade', 'curvature']) -> Dict[str, str]:
        """
        Generate common derivatives from a Digital Elevation Model (DEM)
        
        Args:
            dem_file: Path to input DEM file
            output_dir: Directory for output files
            derivatives: List of derivatives to generate
            
        Returns:
            Dictionary mapping derivative names to output file paths
        """
        # Check if input file exists
        if not os.path.exists(dem_file):
            logger.error(f"Input DEM file does not exist: {dem_file}")
            return {}
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize results dictionary
        results = {}
        
        # Generate derivatives
        for derivative in derivatives:
            derivative = derivative.lower()
            output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(dem_file))[0]}_{derivative}.tif")
            
            try:
                if derivative == 'slope':
                    self.wbt.slope(
                        dem_file, 
                        output_file, 
                        units="degrees"
                    )
                    results['slope'] = output_file
                
                elif derivative == 'aspect':
                    self.wbt.aspect(
                        dem_file, 
                        output_file
                    )
                    results['aspect'] = output_file
                
                elif derivative == 'hillshade':
                    self.wbt.hillshade(
                        dem_file, 
                        output_file
                    )
                    results['hillshade'] = output_file
                
                elif derivative == 'curvature':
                    self.wbt.plan_curvature(
                        dem_file, 
                        output_file
                    )
                    results['curvature'] = output_file
                
                elif derivative == 'twi':
                    # Topographic Wetness Index - needs flow accumulation first
                    flow_acc_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(dem_file))[0]}_flow_acc.tif")
                    self.wbt.d8_flow_accumulation(
                        dem_file,
                        flow_acc_file
                    )
                    
                    self.wbt.wetness_index(
                        dem_file,
                        flow_acc_file,
                        output_file
                    )
                    results['twi'] = output_file
                
                elif derivative == 'viewshed':
                    # Would need observer points for a proper viewshed
                    # Using the center point as an example
                    center_file = os.path.join(self.temp_dir, "center_point.shp")
                    # Create a point at the center of the DEM (placeholder)
                    # In a real system, this would use actual observer points
                    self.wbt.viewshed(
                        dem_file,
                        center_file,
                        output_file
                    )
                    results['viewshed'] = output_file
                
                else:
                    logger.warning(f"Unknown derivative type: {derivative}")
                    continue
                
                logger.info(f"Generated {derivative} from DEM: {output_file}")
                
            except Exception as e:
                logger.error(f"Error generating {derivative}: {str(e)}")
                continue
        
        return results
    
    def calculate_viewshed(self, 
                         dem_file: str, 
                         observer_points: Union[str, gpd.GeoDataFrame], 
                         output_file: str,
                         observer_height: float = 1.7,
                         max_distance: float = 10000) -> bool:
        """
        Calculate viewshed from observer points
        
        Args:
            dem_file: Path to input DEM file
            observer_points: Path to observer points shapefile or GeoDataFrame
            output_file: Path to output viewshed raster
            observer_height: Height of observer in meters
            max_distance: Maximum viewing distance in meters
            
        Returns:
            True if successful, False otherwise
        """
        # Check if input DEM exists
        if not os.path.exists(dem_file):
            logger.error(f"Input DEM file does not exist: {dem_file}")
            return False
        
        # Handle observer points
        points_file = observer_points
        temp_file = None
        
        # If observer_points is a GeoDataFrame, save it as a temporary shapefile
        if isinstance(observer_points, gpd.GeoDataFrame):
            temp_file = os.path.join(self.temp_dir, "observer_points.shp")
            try:
                observer_points.to_file(temp_file)
                points_file = temp_file
                logger.info(f"Saved observer points to temporary file: {temp_file}")
            except Exception as e:
                logger.error(f"Error saving observer points to temporary file: {str(e)}")
                return False
        elif not os.path.exists(points_file):
            logger.error(f"Observer points file does not exist: {points_file}")
            return False
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        try:
            # Calculate viewshed
            self.wbt.viewshed(
                dem_file,
                points_file,
                output_file,
                height=observer_height,
                max_dist=max_distance
            )
            
            logger.info(f"Calculated viewshed from {points_file} to {output_file}")
            return True
        
        except Exception as e:
            logger.error(f"Error calculating viewshed: {str(e)}")
            return False
    
    def extract_watersheds(self, 
                          dem_file: str, 
                          output_dir: str) -> Dict[str, str]:
        """
        Extract watersheds from a DEM
        
        Args:
            dem_file: Path to input DEM file
            output_dir: Directory for output files
            
        Returns:
            Dictionary mapping output names to file paths
        """
        # Check if input file exists
        if not os.path.exists(dem_file):
            logger.error(f"Input DEM file does not exist: {dem_file}")
            return {}
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize results dictionary
        results = {}
        
        try:
            # Fill depressions in the DEM
            filled_dem = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(dem_file))[0]}_filled.tif")
            self.wbt.fill_depressions(
                dem_file,
                filled_dem
            )
            results['filled_dem'] = filled_dem
            
            # Calculate D8 flow direction
            flow_dir = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(dem_file))[0]}_flow_dir.tif")
            self.wbt.d8_flow_direction(
                filled_dem,
                flow_dir
            )
            results['flow_direction'] = flow_dir
            
            # Calculate D8 flow accumulation
            flow_acc = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(dem_file))[0]}_flow_acc.tif")
            self.wbt.d8_flow_accumulation(
                filled_dem,
                flow_acc
            )
            results['flow_accumulation'] = flow_acc
            
            # Extract streams
            streams = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(dem_file))[0]}_streams.tif")
            self.wbt.extract_streams(
                flow_acc,
                streams,
                threshold=1000  # This threshold would need to be adjusted based on the DEM
            )
            results['streams'] = streams
            
            # Find stream junctions
            junctions = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(dem_file))[0]}_junctions.tif")
            self.wbt.find_stream_junctions(
                streams,
                junctions
            )
            results['junctions'] = junctions
            
            # Extract watersheds
            watersheds = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(dem_file))[0]}_watersheds.tif")
            self.wbt.watershed(
                flow_dir,
                junctions,
                watersheds
            )
            results['watersheds'] = watersheds
            
            # Convert watersheds to vector
            watersheds_vector = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(dem_file))[0]}_watersheds.shp")
            self.wbt.raster_to_vector_polygons(
                watersheds,
                watersheds_vector
            )
            results['watersheds_vector'] = watersheds_vector
            
            logger.info(f"Extracted watersheds from DEM: {watersheds_vector}")
            
            return results
        
        except Exception as e:
            logger.error(f"Error extracting watersheds: {str(e)}")
            return results  # Return partial results if some steps completed successfully
    
    def perform_lidar_analysis(self, 
                             lidar_file: str, 
                             output_dir: str) -> Dict[str, str]:
        """
        Perform analysis on LiDAR data
        
        Args:
            lidar_file: Path to input LiDAR file (LAS/LAZ)
            output_dir: Directory for output files
            
        Returns:
            Dictionary mapping output names to file paths
        """
        # Check if input file exists
        if not os.path.exists(lidar_file):
            logger.error(f"Input LiDAR file does not exist: {lidar_file}")
            return {}
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize results dictionary
        results = {}
        
        try:
            # Create a bare-earth DEM
            dem = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(lidar_file))[0]}_dem.tif")
            self.wbt.lidar_tin_gridding(
                lidar_file,
                dem,
                resolution=1.0,
                exclude_cls="3,4,5,6,7"  # Exclude non-ground points
            )
            results['dem'] = dem
            
            # Create a Digital Surface Model (DSM)
            dsm = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(lidar_file))[0]}_dsm.tif")
            self.wbt.lidar_tin_gridding(
                lidar_file,
                dsm,
                resolution=1.0,
                exclude_cls=""  # Include all points
            )
            results['dsm'] = dsm
            
            # Calculate Canopy Height Model (CHM)
            chm = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(lidar_file))[0]}_chm.tif")
            self.wbt.subtract(
                dsm,
                dem,
                chm
            )
            results['chm'] = chm
            
            # Extract building footprints
            buildings = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(lidar_file))[0]}_buildings.tif")
            self.wbt.lidar_segmentation(
                lidar_file,
                buildings,
                exclude_cls="1,2,3,5,6,7,9",  # Keep only building points (class 6)
                resolution=1.0
            )
            results['buildings'] = buildings
            
            # Convert buildings to vector
            buildings_vector = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(lidar_file))[0]}_buildings.shp")
            self.wbt.raster_to_vector_polygons(
                buildings,
                buildings_vector
            )
            results['buildings_vector'] = buildings_vector
            
            logger.info(f"Performed LiDAR analysis: {lidar_file}")
            
            return results
        
        except Exception as e:
            logger.error(f"Error performing LiDAR analysis: {str(e)}")
            return results  # Return partial results if some steps completed successfully
    
    def create_visibility_index(self, 
                              dem_file: str, 
                              output_file: str,
                              observer_height: float = 1.7,
                              sample_size: int = 100) -> bool:
        """
        Create a visibility index map from a DEM
        
        This shows which areas are most visible from random observer points
        
        Args:
            dem_file: Path to input DEM file
            output_file: Path to output visibility index raster
            observer_height: Height of observer in meters
            sample_size: Number of random observer points to use
            
        Returns:
            True if successful, False otherwise
        """
        # Check if input file exists
        if not os.path.exists(dem_file):
            logger.error(f"Input DEM file does not exist: {dem_file}")
            return False
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        try:
            # Create visibility index
            self.wbt.visibility_index(
                dem_file,
                output_file,
                height=observer_height,
                res_factor=5,
                observer_density=sample_size
            )
            
            logger.info(f"Created visibility index: {output_file}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error creating visibility index: {str(e)}")
            return False
    
    def perform_cost_distance_analysis(self, 
                                     dem_file: str, 
                                     source_points: Union[str, gpd.GeoDataFrame], 
                                     output_file: str) -> bool:
        """
        Perform cost-distance analysis from source points
        
        Args:
            dem_file: Path to input DEM file
            source_points: Path to source points shapefile or GeoDataFrame
            output_file: Path to output cost-distance raster
            
        Returns:
            True if successful, False otherwise
        """
        # Check if input DEM exists
        if not os.path.exists(dem_file):
            logger.error(f"Input DEM file does not exist: {dem_file}")
            return False
        
        # Handle source points
        points_file = source_points
        temp_file = None
        
        # If source_points is a GeoDataFrame, save it as a temporary shapefile
        if isinstance(source_points, gpd.GeoDataFrame):
            temp_file = os.path.join(self.temp_dir, "source_points.shp")
            try:
                source_points.to_file(temp_file)
                points_file = temp_file
                logger.info(f"Saved source points to temporary file: {temp_file}")
            except Exception as e:
                logger.error(f"Error saving source points to temporary file: {str(e)}")
                return False
        elif not os.path.exists(points_file):
            logger.error(f"Source points file does not exist: {points_file}")
            return False
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        try:
            # First, calculate the slope from the DEM
            slope_file = os.path.join(self.temp_dir, "slope.tif")
            self.wbt.slope(
                dem_file,
                slope_file,
                units="degrees"
            )
            
            # Then calculate the cost surface using the slope
            cost_file = os.path.join(self.temp_dir, "cost.tif")
            self.wbt.slope_to_cost(
                slope_file,
                cost_file
            )
            
            # Finally, calculate the cost distance from the source points
            self.wbt.cost_distance(
                cost_file,
                points_file,
                output_file
            )
            
            logger.info(f"Performed cost-distance analysis: {output_file}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error performing cost-distance analysis: {str(e)}")
            return False
    
    def calculate_least_cost_path(self, 
                                cost_surface: str, 
                                source_point: Tuple[float, float], 
                                destination_point: Tuple[float, float], 
                                output_file: str) -> bool:
        """
        Calculate least cost path between two points
        
        Args:
            cost_surface: Path to cost surface raster
            source_point: Source point coordinates (x, y)
            destination_point: Destination point coordinates (x, y)
            output_file: Path to output least cost path shapefile
            
        Returns:
            True if successful, False otherwise
        """
        # Check if input cost surface exists
        if not os.path.exists(cost_surface):
            logger.error(f"Input cost surface file does not exist: {cost_surface}")
            return False
        
        # Create source and destination point files
        source_file = os.path.join(self.temp_dir, "source.shp")
        dest_file = os.path.join(self.temp_dir, "destination.shp")
        
        try:
            # Create source point
            source_gdf = gpd.GeoDataFrame({
                'geometry': [gpd.points_from_xy([source_point[0]], [source_point[1]])[0]]
            }, crs="EPSG:4326")
            source_gdf.to_file(source_file)
            
            # Create destination point
            dest_gdf = gpd.GeoDataFrame({
                'geometry': [gpd.points_from_xy([destination_point[0]], [destination_point[1]])[0]]
            }, crs="EPSG:4326")
            dest_gdf.to_file(dest_file)
            
            # Calculate cost-distance from source
            cost_dist_file = os.path.join(self.temp_dir, "cost_distance.tif")
            self.wbt.cost_distance(
                cost_surface,
                source_file,
                cost_dist_file
            )
            
            # Calculate least-cost path from destination to source
            self.wbt.cost_pathway(
                cost_dist_file,
                dest_file,
                output_file
            )
            
            logger.info(f"Calculated least cost path: {output_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error calculating least cost path: {str(e)}")
            return False
    
    def calculate_travel_time(self, 
                            dem_file: str, 
                            source_points: Union[str, gpd.GeoDataFrame], 
                            output_file: str,
                            speed: float = 5.0) -> bool:
        """
        Calculate travel time from source points
        
        Args:
            dem_file: Path to input DEM file
            source_points: Path to source points shapefile or GeoDataFrame
            output_file: Path to output travel time raster
            speed: Walking speed in km/h
            
        Returns:
            True if successful, False otherwise
        """
        # Check if input DEM exists
        if not os.path.exists(dem_file):
            logger.error(f"Input DEM file does not exist: {dem_file}")
            return False
        
        # Handle source points
        points_file = source_points
        temp_file = None
        
        # If source_points is a GeoDataFrame, save it as a temporary shapefile
        if isinstance(source_points, gpd.GeoDataFrame):
            temp_file = os.path.join(self.temp_dir, "source_points.shp")
            try:
                source_points.to_file(temp_file)
                points_file = temp_file
                logger.info(f"Saved source points to temporary file: {temp_file}")
            except Exception as e:
                logger.error(f"Error saving source points to temporary file: {str(e)}")
                return False
        elif not os.path.exists(points_file):
            logger.error(f"Source points file does not exist: {points_file}")
            return False
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        try:
            # First, calculate the slope from the DEM
            slope_file = os.path.join(self.temp_dir, "slope.tif")
            self.wbt.slope(
                dem_file,
                slope_file,
                units="degrees"
            )
            
            # Convert slope to travel time
            cost_file = os.path.join(self.temp_dir, "travel_time_cost.tif")
            self.wbt.slope_to_travel_time(
                slope_file,
                cost_file,
                speed=speed  # km/h
            )
            
            # Calculate travel time from source points
            self.wbt.cost_distance(
                cost_file,
                points_file,
                output_file
            )
            
            logger.info(f"Calculated travel time: {output_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error calculating travel time: {str(e)}")
            return False
    
    def detect_terrain_features(self, 
                              dem_file: str, 
                              output_dir: str) -> Dict[str, str]:
        """
        Detect terrain features from a DEM
        
        Args:
            dem_file: Path to input DEM file
            output_dir: Directory for output files
            
        Returns:
            Dictionary mapping feature names to output file paths
        """
        # Check if input file exists
        if not os.path.exists(dem_file):
            logger.error(f"Input DEM file does not exist: {dem_file}")
            return {}
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize results dictionary
        results = {}
        
        try:
            # Detect valleys
            valleys = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(dem_file))[0]}_valleys.tif")
            self.wbt.multi_scale_topographic_position_image(
                dem_file,
                valleys,
                min_scale=1,
                max_scale=100,
                step=10,
                feature_type="valley"
            )
            results['valleys'] = valleys
            
            # Detect ridges
            ridges = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(dem_file))[0]}_ridges.tif")
            self.wbt.multi_scale_topographic_position_image(
                dem_file,
                ridges,
                min_scale=1,
                max_scale=100,
                step=10,
                feature_type="ridge"
            )
            results['ridges'] = ridges
            
            # Detect peaks
            peaks = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(dem_file))[0]}_peaks.tif")
            self.wbt.find_peaks(
                dem_file,
                peaks
            )
            results['peaks'] = peaks
            
            # Convert peaks to vector
            peaks_vector = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(dem_file))[0]}_peaks.shp")
            self.wbt.raster_to_vector_points(
                peaks,
                peaks_vector
            )
            results['peaks_vector'] = peaks_vector
            
            # Extract flat areas
            flats = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(dem_file))[0]}_flats.tif")
            self.wbt.classify_slope_facets(
                dem_file,
                flats,
                output_scaling="classifications"
            )
            results['flats'] = flats
            
            logger.info(f"Detected terrain features from DEM: {dem_file}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error detecting terrain features: {str(e)}")
            return results  # Return partial results if some steps completed successfully