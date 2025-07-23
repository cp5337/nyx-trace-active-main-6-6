"""
Geospatial Components Module
============================
Core components for the refactored geospatial intelligence system.
These components provide modular, focused functionality for different
aspects of geospatial analysis and processing.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class GeospatialPluginManager:
    """
    Manages geospatial plugins and their integration
    
    This class handles the discovery, loading, and management
    of geospatial intelligence plugins.
    """
    
    def __init__(self):
        self.plugins = {}
        self.available_plugins = []
    
    def get_available_plugins(self) -> List[Dict[str, Any]]:
        """Get list of available plugins"""
        return [
            {
                "name": "Distance Calculator",
                "description": "Calculate distances between geographic points",
                "version": "1.0.0",
                "status": "active"
            },
            {
                "name": "Hotspot Analyzer",
                "description": "Identify areas of high activity",
                "version": "1.0.0",
                "status": "active"
            },
            {
                "name": "Spatial Joiner",
                "description": "Join spatial datasets",
                "version": "1.0.0",
                "status": "active"
            }
        ]

class GeospatialDataProcessor:
    """
    Processes and validates geospatial data
    
    This class handles data loading, validation, and preprocessing
    for geospatial analysis.
    """
    
    def __init__(self):
        self.sample_data = None
    
    def process_uploaded_file(self, file) -> pd.DataFrame:
        """Process uploaded geospatial data file"""
        try:
            if file.name.endswith('.csv'):
                data = pd.read_csv(file)
            elif file.name.endswith('.json'):
                data = pd.read_json(file)
            elif file.name.endswith('.geojson'):
                data = pd.read_json(file)
            else:
                raise ValueError("Unsupported file format")
            
            # Validate and clean data
            data = self._validate_geospatial_data(data)
            return data
            
        except Exception as e:
            logger.error(f"Error processing file: {e}")
            st.error(f"Error processing file: {e}")
            return pd.DataFrame()
    
    def get_sample_data(self) -> pd.DataFrame:
        """Get sample geospatial data for testing"""
        if self.sample_data is None:
            self.sample_data = self._generate_sample_data()
        return self.sample_data
    
    def calculate_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate basic statistics for geospatial data"""
        if data.empty:
            return {
                'total_points': 0,
                'coverage_area': 0.0,
                'density': 0.0,
                'time_range': 'N/A'
            }
        
        stats = {
            'total_points': len(data),
            'coverage_area': self._calculate_coverage_area(data),
            'density': len(data) / max(self._calculate_coverage_area(data), 1),
            'time_range': self._calculate_time_range(data)
        }
        
        return stats
    
    def export_to_csv(self, results: Dict[str, Any]) -> str:
        """Export results to CSV format"""
        df = pd.DataFrame(results)
        return df.to_csv(index=False)
    
    def export_to_geojson(self, results: Dict[str, Any]) -> str:
        """Export results to GeoJSON format"""
        import json
        
        geojson = {
            "type": "FeatureCollection",
            "features": []
        }
        
        # Convert results to GeoJSON format
        for i, result in enumerate(results):
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [result.get('lon', 0), result.get('lat', 0)]
                },
                "properties": result
            }
            geojson["features"].append(feature)
        
        return json.dumps(geojson, indent=2)
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate HTML report from results"""
        html = f"""
        <html>
        <head>
            <title>Geospatial Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 10px; }}
                .section {{ margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Geospatial Analysis Report</h1>
                <p>Generated on: {pd.Timestamp.now()}</p>
            </div>
            
            <div class="section">
                <h2>Analysis Summary</h2>
                <p>Total records analyzed: {len(results)}</p>
            </div>
            
            <div class="section">
                <h2>Results</h2>
                <table>
                    <tr>
                        <th>Record</th>
                        <th>Latitude</th>
                        <th>Longitude</th>
                        <th>Value</th>
                    </tr>
        """
        
        for i, result in enumerate(results):
            html += f"""
                    <tr>
                        <td>{i+1}</td>
                        <td>{result.get('lat', 'N/A')}</td>
                        <td>{result.get('lon', 'N/A')}</td>
                        <td>{result.get('value', 'N/A')}</td>
                    </tr>
            """
        
        html += """
                </table>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _validate_geospatial_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean geospatial data"""
        # Check for required columns
        required_cols = ['lat', 'lon']
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            st.warning(f"Missing required columns: {missing_cols}")
            return pd.DataFrame()
        
        # Remove rows with invalid coordinates
        data = data.dropna(subset=['lat', 'lon'])
        
        # Validate coordinate ranges
        data = data[
            (data['lat'] >= -90) & (data['lat'] <= 90) &
            (data['lon'] >= -180) & (data['lon'] <= 180)
        ]
        
        return data
    
    def _generate_sample_data(self) -> pd.DataFrame:
        """Generate sample geospatial data"""
        np.random.seed(42)
        
        # Generate random points around San Francisco
        n_points = 100
        lat_center, lon_center = 37.7749, -122.4194
        
        lats = np.random.normal(lat_center, 0.1, n_points)
        lons = np.random.normal(lon_center, 0.1, n_points)
        values = np.random.exponential(1.0, n_points)
        
        data = pd.DataFrame({
            'lat': lats,
            'lon': lons,
            'value': values,
            'timestamp': pd.date_range('2024-01-01', periods=n_points, freq='H')
        })
        
        return data
    
    def _calculate_coverage_area(self, data: pd.DataFrame) -> float:
        """Calculate approximate coverage area in km²"""
        if data.empty:
            return 0.0
        
        lat_range = data['lat'].max() - data['lat'].min()
        lon_range = data['lon'].max() - data['lon'].min()
        
        # Approximate area calculation (simplified)
        area_km2 = lat_range * lon_range * 111 * 111  # Rough conversion
        return max(area_km2, 0.01)  # Minimum area
    
    def _calculate_time_range(self, data: pd.DataFrame) -> str:
        """Calculate time range of data"""
        if 'timestamp' not in data.columns:
            return 'N/A'
        
        start_time = data['timestamp'].min()
        end_time = data['timestamp'].max()
        
        if pd.isna(start_time) or pd.isna(end_time):
            return 'N/A'
        
        duration = end_time - start_time
        return f"{start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')} ({duration.days} days)"

class GeospatialVisualizer:
    """
    Creates visualizations for geospatial data
    
    This class handles the creation of various types of
    geospatial visualizations including maps, charts, and graphs.
    """
    
    def __init__(self):
        self.base_map = None
    
    def create_heatmap(self, data: pd.DataFrame, results: Dict[str, Any]) -> Any:
        """Create a heatmap visualization"""
        import folium
        
        # Create base map
        center_lat = data['lat'].mean()
        center_lon = data['lon'].mean()
        m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
        
        # Add heatmap layer
        heat_data = [[row['lat'], row['lon'], row['value']] 
                    for _, row in data.iterrows()]
        
        folium.plugins.HeatMap(heat_data).add_to(m)
        
        return m
    
    def create_marker_cluster(self, data: pd.DataFrame) -> Any:
        """Create a marker cluster visualization"""
        import folium
        
        # Create base map
        center_lat = data['lat'].mean()
        center_lon = data['lon'].mean()
        m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
        
        # Add marker cluster
        marker_cluster = folium.plugins.MarkerCluster().add_to(m)
        
        for _, row in data.iterrows():
            folium.Marker(
                location=[row['lat'], row['lon']],
                popup=f"Value: {row['value']:.2f}"
            ).add_to(marker_cluster)
        
        return m
    
    def create_network_graph(self, data: pd.DataFrame, results: Dict[str, Any]) -> Any:
        """Create a network graph visualization"""
        import folium
        
        # Create base map
        center_lat = data['lat'].mean()
        center_lon = data['lon'].mean()
        m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
        
        # Add network connections (simplified)
        for i in range(len(data) - 1):
            point1 = data.iloc[i]
            point2 = data.iloc[i + 1]
            
            folium.PolyLine(
                locations=[[point1['lat'], point1['lon']], 
                          [point2['lat'], point2['lon']]],
                color='blue',
                weight=2,
                opacity=0.7
            ).add_to(m)
        
        return m
    
    def create_choropleth_map(self, data: pd.DataFrame, results: Dict[str, Any]) -> Any:
        """Create a choropleth map visualization"""
        import folium
        
        # Create base map
        center_lat = data['lat'].mean()
        center_lon = data['lon'].mean()
        m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
        
        # Add choropleth layer (simplified)
        for _, row in data.iterrows():
            color = self._get_color_for_value(row['value'])
            
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=10,
                color=color,
                fill=True,
                popup=f"Value: {row['value']:.2f}"
            ).add_to(m)
        
        return m
    
    def _get_color_for_value(self, value: float) -> str:
        """Get color based on value"""
        if value < 0.5:
            return 'green'
        elif value < 1.0:
            return 'yellow'
        elif value < 2.0:
            return 'orange'
        else:
            return 'red'

class GeospatialAnalyzer:
    """
    Performs geospatial analysis operations
    
    This class provides various geospatial analysis capabilities
    including hotspot analysis, spatial joins, and distance calculations.
    """
    
    def __init__(self):
        self.analysis_methods = {
            'Hotspot Analysis': self._perform_hotspot_analysis,
            'Spatial Join': self._perform_spatial_join,
            'Distance Calculation': self._perform_distance_calculation,
            'Network Analysis': self._perform_network_analysis
        }
    
    def perform_analysis(self, data: pd.DataFrame, analysis_type: str) -> Dict[str, Any]:
        """Perform the specified type of geospatial analysis"""
        if analysis_type not in self.analysis_methods:
            st.error(f"Unknown analysis type: {analysis_type}")
            return {}
        
        try:
            method = self.analysis_methods[analysis_type]
            results = method(data)
            return results
        except Exception as e:
            logger.error(f"Error performing {analysis_type}: {e}")
            st.error(f"Error performing analysis: {e}")
            return {}
    
    def _perform_hotspot_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform hotspot analysis"""
        if data.empty:
            return {}
        
        # Calculate density-based hotspots
        hotspots = []
        
        # Group data into grid cells
        data['grid_lat'] = (data['lat'] * 10).round() / 10
        data['grid_lon'] = (data['lon'] * 10).round() / 10
        
        grid_counts = data.groupby(['grid_lat', 'grid_lon']).size().reset_index(name='count')
        
        # Identify hotspots (cells with high counts)
        threshold = grid_counts['count'].quantile(0.8)
        hotspots_data = grid_counts[grid_counts['count'] >= threshold]
        
        for _, row in hotspots_data.iterrows():
            hotspots.append({
                'lat': row['grid_lat'],
                'lon': row['grid_lon'],
                'intensity': row['count'],
                'type': 'hotspot'
            })
        
        return {
            'analysis_type': 'Hotspot Analysis',
            'hotspots': hotspots,
            'total_hotspots': len(hotspots),
            'threshold': threshold
        }
    
    def _perform_spatial_join(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform spatial join analysis"""
        if data.empty:
            return {}
        
        # Simplified spatial join (nearest neighbor)
        joined_data = []
        
        for i, point1 in data.iterrows():
            nearest_point = None
            min_distance = float('inf')
            
            for j, point2 in data.iterrows():
                if i != j:
                    distance = self._calculate_distance(
                        point1['lat'], point1['lon'],
                        point2['lat'], point2['lon']
                    )
                    
                    if distance < min_distance:
                        min_distance = distance
                        nearest_point = point2
            
            if nearest_point is not None:
                joined_data.append({
                    'point1_lat': point1['lat'],
                    'point1_lon': point1['lon'],
                    'point2_lat': nearest_point['lat'],
                    'point2_lon': nearest_point['lon'],
                    'distance': min_distance
                })
        
        return {
            'analysis_type': 'Spatial Join',
            'joined_pairs': joined_data,
            'total_joins': len(joined_data)
        }
    
    def _perform_distance_calculation(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform distance calculation analysis"""
        if data.empty:
            return {}
        
        distances = []
        
        # Calculate distances between consecutive points
        for i in range(len(data) - 1):
            point1 = data.iloc[i]
            point2 = data.iloc[i + 1]
            
            distance = self._calculate_distance(
                point1['lat'], point1['lon'],
                point2['lat'], point2['lon']
            )
            
            distances.append({
                'from_lat': point1['lat'],
                'from_lon': point1['lon'],
                'to_lat': point2['lat'],
                'to_lon': point2['lon'],
                'distance_km': distance
            })
        
        return {
            'analysis_type': 'Distance Calculation',
            'distances': distances,
            'total_distances': len(distances),
            'average_distance': np.mean([d['distance_km'] for d in distances]) if distances else 0
        }
    
    def _perform_network_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform network analysis"""
        if data.empty:
            return {}
        
        # Simplified network analysis
        network_nodes = []
        network_edges = []
        
        # Create nodes from data points
        for i, point in data.iterrows():
            network_nodes.append({
                'id': i,
                'lat': point['lat'],
                'lon': point['lon'],
                'value': point['value']
            })
        
        # Create edges between nearby points
        for i in range(len(data)):
            for j in range(i + 1, len(data)):
                point1 = data.iloc[i]
                point2 = data.iloc[j]
                
                distance = self._calculate_distance(
                    point1['lat'], point1['lon'],
                    point2['lat'], point2['lon']
                )
                
                # Connect points within 1km
                if distance < 1.0:
                    network_edges.append({
                        'from_node': i,
                        'to_node': j,
                        'distance': distance
                    })
        
        return {
            'analysis_type': 'Network Analysis',
            'nodes': network_nodes,
            'edges': network_edges,
            'total_nodes': len(network_nodes),
            'total_edges': len(network_edges)
        }
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points in kilometers"""
        import math
        
        # Haversine formula
        R = 6371  # Earth's radius in kilometers
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = (math.sin(delta_lat / 2) ** 2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c 