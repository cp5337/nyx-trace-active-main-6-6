"""
Mapbox Integration Module
----------------------
This module provides integration with Mapbox for advanced
geospatial visualization in the NyxTrace platform.
"""

import os
import json
import logging
import pandas as pd
import geopandas as gpd
import numpy as np
from typing import Dict, List, Any, Union, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pydeck as pdk

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('mapbox_integration')


class MapboxIntegration:
    """
    Mapbox integration for advanced geospatial visualization
    
    This class provides methods for:
    - Creating interactive maps with Mapbox
    - Visualizing various geospatial data types
    - Adding custom layers and controls
    """
    
    def __init__(self, access_token: Optional[str] = None):
        """
        Initialize Mapbox integration
        
        Args:
            access_token: Mapbox access token (optional, can use environment variable)
        """
        # Try to get token from environment if not provided
        self.access_token = access_token or os.environ.get('MAPBOX_ACCESS_TOKEN')
        
        # Warn if no token is available
        if not self.access_token:
            logger.warning("No Mapbox access token provided. Some functionality may be limited.")
        
        # Set default map styles
        self.map_styles = {
            'dark': 'mapbox://styles/mapbox/dark-v10',
            'satellite': 'mapbox://styles/mapbox/satellite-streets-v11',
            'light': 'mapbox://styles/mapbox/light-v10',
            'outdoors': 'mapbox://styles/mapbox/outdoors-v11',
            'streets': 'mapbox://styles/mapbox/streets-v11'
        }
        
        # Set default colors for different threat types
        self.threat_colors = {
            'cyber': '#FF4B4B',      # Red
            'physical': '#FFA15A',   # Orange
            'hybrid': '#FF00FF',     # Purple
            'cartel': '#FFFF00',     # Yellow
            'terrorism': '#FF0000',  # Bright red
            'nation_state': '#0000FF', # Blue
            'other': '#FFFFFF'       # White
        }
        
        logger.info("Mapbox integration initialized")
    
    def create_threat_map(self, 
                         data: pd.DataFrame, 
                         map_style: str = 'dark',
                         center: Optional[List[float]] = None,
                         zoom: int = 3) -> go.Figure:
        """
        Create a threat map with Mapbox
        
        Args:
            data: DataFrame with lat, lon, and threat data
            map_style: Mapbox style ('dark', 'satellite', 'light', etc.)
            center: Map center [longitude, latitude]
            zoom: Initial zoom level
            
        Returns:
            Plotly figure with Mapbox map
        """
        if data.empty:
            logger.warning("No data provided for threat map")
            fig = go.Figure(go.Scattermapbox())
            fig.update_layout(
                mapbox={
                    'style': self.map_styles.get(map_style, self.map_styles['dark']),
                    'accesstoken': self.access_token,
                    'zoom': zoom,
                    'center': {'lon': 0, 'lat': 0} if center is None else {'lon': center[0], 'lat': center[1]}
                },
                margin={"r":0, "t":0, "l":0, "b":0}
            )
            return fig
        
        # Check required columns
        required_cols = ['lat', 'lon']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            logger.warning(f"Missing required columns for threat map: {missing_cols}")
            return go.Figure()
        
        # Get threat type column if available
        threat_type_col = None
        for col in ['threat_type', 'type', 'category']:
            if col in data.columns:
                threat_type_col = col
                break
        
        # Get severity column if available
        severity_col = None
        for col in ['severity', 'impact', 'weight', 'magnitude']:
            if col in data.columns:
                severity_col = col
                break
        
        # Set default center if not provided
        if center is None:
            center = [data['lon'].mean(), data['lat'].mean()]
        
        # Create figure with scattermapbox
        if threat_type_col is not None:
            # Create map with color by threat type
            fig = px.scatter_mapbox(
                data,
                lat="lat",
                lon="lon",
                color=threat_type_col,
                size=severity_col if severity_col else None,
                size_max=20,
                zoom=zoom,
                color_discrete_map=self.threat_colors,
                hover_data=data.columns.tolist(),
                mapbox_style=self.map_styles.get(map_style, self.map_styles['dark']),
                opacity=0.8
            )
        else:
            # Create map without color differentiation
            fig = px.scatter_mapbox(
                data,
                lat="lat",
                lon="lon",
                size=severity_col if severity_col else None,
                size_max=20,
                zoom=zoom,
                hover_data=data.columns.tolist(),
                mapbox_style=self.map_styles.get(map_style, self.map_styles['dark']),
                opacity=0.8
            )
        
        # Update layout
        fig.update_layout(
            mapbox={
                'accesstoken': self.access_token,
                'center': {'lon': center[0], 'lat': center[1]}
            },
            margin={"r":0, "t":0, "l":0, "b":0}
        )
        
        return fig
    
    def create_advanced_threat_map(self, 
                                 data: pd.DataFrame, 
                                 areas_gdf: Optional[gpd.GeoDataFrame] = None,
                                 infrastructure_gdf: Optional[gpd.GeoDataFrame] = None,
                                 heatmap_data: Optional[pd.DataFrame] = None,
                                 map_style: str = 'dark',
                                 center: Optional[List[float]] = None,
                                 zoom: int = 3) -> go.Figure:
        """
        Create an advanced threat map with multiple layers
        
        Args:
            data: DataFrame with lat, lon, and threat data
            areas_gdf: GeoDataFrame with polygon areas (optional)
            infrastructure_gdf: GeoDataFrame with infrastructure lines (optional)
            heatmap_data: DataFrame for heatmap layer (optional)
            map_style: Mapbox style ('dark', 'satellite', 'light', etc.)
            center: Map center [longitude, latitude]
            zoom: Initial zoom level
            
        Returns:
            Plotly figure with advanced Mapbox map
        """
        if data.empty:
            logger.warning("No data provided for advanced threat map")
            fig = go.Figure(go.Scattermapbox())
            fig.update_layout(
                mapbox={
                    'style': self.map_styles.get(map_style, self.map_styles['dark']),
                    'accesstoken': self.access_token,
                    'zoom': zoom,
                    'center': {'lon': 0, 'lat': 0} if center is None else {'lon': center[0], 'lat': center[1]}
                },
                margin={"r":0, "t":0, "l":0, "b":0}
            )
            return fig
        
        # Check required columns
        required_cols = ['lat', 'lon']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            logger.warning(f"Missing required columns for advanced threat map: {missing_cols}")
            return go.Figure()
        
        # Get threat type column if available
        threat_type_col = None
        for col in ['threat_type', 'type', 'category']:
            if col in data.columns:
                threat_type_col = col
                break
        
        # Get severity column if available
        severity_col = None
        for col in ['severity', 'impact', 'weight', 'magnitude']:
            if col in data.columns:
                severity_col = col
                break
        
        # Set default center if not provided
        if center is None:
            center = [data['lon'].mean(), data['lat'].mean()]
        
        # Create base map
        fig = go.Figure()
        
        # Add infrastructure layer if provided
        if infrastructure_gdf is not None and not infrastructure_gdf.empty:
            # Convert GeoDataFrame to GeoJSON
            infrastructure_json = json.loads(infrastructure_gdf.to_json())
            
            # Add lines for infrastructure
            for feature in infrastructure_json['features']:
                if feature['geometry']['type'] == 'LineString':
                    coords = feature['geometry']['coordinates']
                    lon = [coord[0] for coord in coords]
                    lat = [coord[1] for coord in coords]
                    
                    fig.add_trace(go.Scattermapbox(
                        mode='lines',
                        lon=lon,
                        lat=lat,
                        line=dict(width=2, color='rgba(255, 255, 255, 0.5)'),
                        name=feature['properties'].get('name', 'Infrastructure'),
                        hoverinfo='name',
                        showlegend=False
                    ))
        
        # Add areas layer if provided
        if areas_gdf is not None and not areas_gdf.empty:
            # Convert GeoDataFrame to GeoJSON
            areas_json = json.loads(areas_gdf.to_json())
            
            # Add polygons for areas
            for feature in areas_json['features']:
                if feature['geometry']['type'] == 'Polygon':
                    # Extract coordinates and properties
                    coords = feature['geometry']['coordinates'][0]
                    lon = [coord[0] for coord in coords]
                    lat = [coord[1] for coord in coords]
                    
                    # Get area properties
                    area_name = feature['properties'].get('name', 'Area of Interest')
                    area_type = feature['properties'].get('type', 'general')
                    
                    # Set color based on area type
                    color = self.threat_colors.get(area_type, 'rgba(255, 165, 0, 0.1)')
                    
                    # Add fill area
                    fig.add_trace(go.Scattermapbox(
                        mode='lines',
                        lon=lon,
                        lat=lat,
                        fill='toself',
                        fillcolor=color.replace(')', ', 0.2)') if not color.startswith('rgba') else color,
                        line=dict(width=1, color=color),
                        name=area_name,
                        hoverinfo='name'
                    ))
        
        # Add heatmap layer if provided
        if heatmap_data is not None and not heatmap_data.empty:
            # Check required columns
            required_heatmap_cols = ['lat', 'lon']
            intensity_col = None
            
            for col in ['intensity', 'weight', 'value', 'activity_level']:
                if col in heatmap_data.columns:
                    intensity_col = col
                    break
            
            if all(col in heatmap_data.columns for col in required_heatmap_cols):
                fig.add_trace(go.Densitymapbox(
                    lat=heatmap_data['lat'],
                    lon=heatmap_data['lon'],
                    z=heatmap_data[intensity_col] if intensity_col else None,
                    radius=10,
                    colorscale='YlOrRd',
                    showscale=False,
                    hoverinfo='none',
                    name='Activity Heatmap',
                    below='traces'
                ))
        
        # Add threat points layer
        if threat_type_col is not None:
            # Get unique threat types
            threat_types = data[threat_type_col].unique()
            
            # Add a trace for each threat type
            for threat_type in threat_types:
                type_data = data[data[threat_type_col] == threat_type]
                
                # Get color for this threat type
                color = self.threat_colors.get(threat_type.lower(), '#FFFFFF')
                
                # Create hover text
                hover_text = []
                for _, row in type_data.iterrows():
                    text = f"<b>{row.get('location', 'Unknown Location')}</b><br>"
                    for col in type_data.columns:
                        if col not in ['lat', 'lon', 'geometry']:
                            text += f"{col}: {row[col]}<br>"
                    hover_text.append(text)
                
                # Add scatter trace
                fig.add_trace(go.Scattermapbox(
                    mode='markers',
                    lon=type_data['lon'],
                    lat=type_data['lat'],
                    marker=dict(
                        size=type_data[severity_col] * 5 if severity_col else 10,
                        color=color,
                        opacity=0.8
                    ),
                    text=hover_text,
                    hoverinfo='text',
                    name=f"{threat_type} Threat"
                ))
        else:
            # Create hover text
            hover_text = []
            for _, row in data.iterrows():
                text = f"<b>{row.get('location', 'Unknown Location')}</b><br>"
                for col in data.columns:
                    if col not in ['lat', 'lon', 'geometry']:
                        text += f"{col}: {row[col]}<br>"
                hover_text.append(text)
            
            # Add all threats as a single trace
            fig.add_trace(go.Scattermapbox(
                mode='markers',
                lon=data['lon'],
                lat=data['lat'],
                marker=dict(
                    size=data[severity_col] * 5 if severity_col else 10,
                    color='#FF4B4B',
                    opacity=0.8
                ),
                text=hover_text,
                hoverinfo='text',
                name='Threats'
            ))
        
        # Update layout with map configuration
        fig.update_layout(
            mapbox=dict(
                accesstoken=self.access_token,
                style=self.map_styles.get(map_style, self.map_styles['dark']),
                center=dict(lon=center[0], lat=center[1]),
                zoom=zoom
            ),
            margin={"r":0, "t":0, "l":0, "b":0},
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(0,0,0,0.5)"
            ),
            uirevision='constant',  # Preserve view on updates
            # Add layer controls
            updatemenus=[
                # Map style selector
                dict(
                    buttons=[
                        dict(
                            args=[{"mapbox.style": self.map_styles['dark']}],
                            label="Dark",
                            method="relayout"
                        ),
                        dict(
                            args=[{"mapbox.style": self.map_styles['satellite']}],
                            label="Satellite",
                            method="relayout"
                        ),
                        dict(
                            args=[{"mapbox.style": self.map_styles['light']}],
                            label="Light",
                            method="relayout"
                        ),
                        dict(
                            args=[{"mapbox.style": self.map_styles['outdoors']}],
                            label="Terrain",
                            method="relayout"
                        )
                    ],
                    direction="down",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.01,
                    xanchor="left",
                    y=0.05,
                    yanchor="bottom",
                    bgcolor="rgba(0,0,0,0.5)"
                )
            ]
        )
        
        return fig
    
    def create_pydeck_map(self, 
                        data: pd.DataFrame, 
                        layer_type: str = 'scatter',
                        map_style: str = 'dark') -> pdk.Deck:
        """
        Create a PyDeck map for advanced visualization
        
        Args:
            data: DataFrame with lat, lon, and other data
            layer_type: Layer type ('scatter', 'heatmap', 'hexagon', etc.)
            map_style: Map style ('dark', 'satellite', 'light', etc.)
            
        Returns:
            PyDeck Deck object
        """
        try:
            # Check required columns
            required_cols = ['lat', 'lon']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                logger.warning(f"Missing required columns for PyDeck map: {missing_cols}")
                return pdk.Deck()
            
            # Convert DataFrame to format needed by PyDeck
            data_dict = data.to_dict(orient='records')
            
            # Set initial view state
            view_state = pdk.ViewState(
                latitude=data['lat'].mean(),
                longitude=data['lon'].mean(),
                zoom=5,
                pitch=0
            )
            
            # Set map style
            if map_style == 'dark':
                style = pdk.map_styles.DARK
            elif map_style == 'satellite':
                style = pdk.map_styles.SATELLITE
            elif map_style == 'light':
                style = pdk.map_styles.LIGHT
            else:
                style = pdk.map_styles.DARK
            
            # Create layer based on type
            if layer_type == 'scatter':
                # Scatter plot layer
                layer = pdk.Layer(
                    'ScatterplotLayer',
                    data=data_dict,
                    get_position=['lon', 'lat'],
                    get_color=[255, 0, 0, 160],
                    get_radius=100,
                    pickable=True
                )
            
            elif layer_type == 'heatmap':
                # Heatmap layer
                layer = pdk.Layer(
                    'HeatmapLayer',
                    data=data_dict,
                    get_position=['lon', 'lat'],
                    opacity=0.8,
                    threshold=0.05,
                    get_weight='weight' if 'weight' in data.columns else 1
                )
            
            elif layer_type == 'hexagon':
                # Hexagon layer
                layer = pdk.Layer(
                    'HexagonLayer',
                    data=data_dict,
                    get_position=['lon', 'lat'],
                    radius=1000,
                    elevation_scale=10,
                    elevation_range=[0, 1000],
                    extruded=True,
                    coverage=0.8,
                    pickable=True
                )
            
            elif layer_type == 'path':
                # Path layer (requires properly structured data)
                layer = pdk.Layer(
                    'PathLayer',
                    data=data_dict,
                    get_path='path',
                    get_width=5,
                    get_color=[255, 0, 0],
                    pickable=True
                )
            
            elif layer_type == 'polygon':
                # Polygon layer (requires properly structured data)
                layer = pdk.Layer(
                    'PolygonLayer',
                    data=data_dict,
                    get_polygon='polygon',
                    filled=True,
                    extruded=True,
                    wireframe=True,
                    get_elevation='elevation' if 'elevation' in data.columns else 0,
                    get_fill_color=[255, 165, 0, 80],
                    get_line_color=[255, 255, 255],
                    pickable=True
                )
            
            else:
                # Default to scatter layer
                layer = pdk.Layer(
                    'ScatterplotLayer',
                    data=data_dict,
                    get_position=['lon', 'lat'],
                    get_color=[255, 0, 0, 160],
                    get_radius=100,
                    pickable=True
                )
            
            # Create tooltip
            tooltip = {
                "html": "<b>{location}</b><br>{description}",
                "style": {
                    "backgroundColor": "steelblue",
                    "color": "white"
                }
            }
            
            # Create deck
            deck = pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                map_style=style,
                tooltip=tooltip
            )
            
            return deck
            
        except Exception as e:
            logger.error(f"Error creating PyDeck map: {str(e)}")
            return pdk.Deck()
    
    def create_3d_terrain_map(self, 
                             data: pd.DataFrame, 
                             terrain_url: Optional[str] = None) -> pdk.Deck:
        """
        Create a 3D terrain map with PyDeck
        
        Args:
            data: DataFrame with lat, lon, and other data
            terrain_url: URL to terrain tileset (optional)
            
        Returns:
            PyDeck Deck object
        """
        try:
            # Check required columns
            required_cols = ['lat', 'lon']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                logger.warning(f"Missing required columns for 3D terrain map: {missing_cols}")
                return pdk.Deck()
            
            # Convert DataFrame to format needed by PyDeck
            data_dict = data.to_dict(orient='records')
            
            # Set initial view state
            view_state = pdk.ViewState(
                latitude=data['lat'].mean(),
                longitude=data['lon'].mean(),
                zoom=5,
                pitch=45,
                bearing=0
            )
            
            # Create scatter layer for data points
            scatter_layer = pdk.Layer(
                'ScatterplotLayer',
                data=data_dict,
                get_position=['lon', 'lat'],
                get_color=[255, 0, 0, 160],
                get_radius=100,
                pickable=True
            )
            
            # Create terrain layer
            layers = [scatter_layer]
            
            if terrain_url is not None:
                terrain_layer = pdk.Layer(
                    'TerrainLayer',
                    elevation_decoder={
                        "rScaler": 2,
                        "gScaler": 0,
                        "bScaler": 0,
                        "offset": 0
                    },
                    terrain_image=terrain_url,
                    wireframe=False,
                    material=False
                )
                layers.append(terrain_layer)
            else:
                # Use default Mapbox terrain if no URL provided
                terrain_layer = pdk.Layer(
                    'TerrainLayer',
                    elevation_decoder={
                        "rScaler": 2,
                        "gScaler": 0,
                        "bScaler": 0,
                        "offset": 0
                    },
                    wireframe=False,
                    material=False
                )
                layers.append(terrain_layer)
            
            # Create tooltip
            tooltip = {
                "html": "<b>{location}</b><br>{description}",
                "style": {
                    "backgroundColor": "steelblue",
                    "color": "white"
                }
            }
            
            # Create deck
            deck = pdk.Deck(
                layers=layers,
                initial_view_state=view_state,
                map_style=pdk.map_styles.SATELLITE,
                tooltip=tooltip
            )
            
            return deck
            
        except Exception as e:
            logger.error(f"Error creating 3D terrain map: {str(e)}")
            return pdk.Deck()
    
    def create_time_lapse_map(self, 
                            data: pd.DataFrame, 
                            time_column: str,
                            map_style: str = 'dark') -> go.Figure:
        """
        Create a time-lapse map with Mapbox
        
        Args:
            data: DataFrame with lat, lon, time, and other data
            time_column: Column with time information
            map_style: Mapbox style ('dark', 'satellite', 'light', etc.)
            
        Returns:
            Plotly figure with time-lapse controls
        """
        # Check required columns
        required_cols = ['lat', 'lon', time_column]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            logger.warning(f"Missing required columns for time-lapse map: {missing_cols}")
            return go.Figure()
        
        # Get threat type column if available
        threat_type_col = None
        for col in ['threat_type', 'type', 'category']:
            if col in data.columns:
                threat_type_col = col
                break
        
        # Convert time column to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(data[time_column]):
            try:
                data[time_column] = pd.to_datetime(data[time_column])
            except:
                logger.warning(f"Could not convert {time_column} to datetime")
                return go.Figure()
        
        # Sort by time
        data = data.sort_values(by=time_column)
        
        # Get unique time periods
        time_periods = data[time_column].unique()
        
        # Set map center
        center = [data['lon'].mean(), data['lat'].mean()]
        
        # Create figure with frames
        fig = go.Figure()
        
        # Add frames for each time period
        frames = []
        for time_period in time_periods:
            # Filter data for this time period
            period_data = data[data[time_column] == time_period]
            
            # Create frame for this time period
            if threat_type_col is not None:
                # Get unique threat types for this period
                threat_types = period_data[threat_type_col].unique()
                
                # Create data for each threat type
                frame_data = []
                for threat_type in threat_types:
                    type_data = period_data[period_data[threat_type_col] == threat_type]
                    
                    # Create hover text
                    hover_text = []
                    for _, row in type_data.iterrows():
                        text = f"<b>{row.get('location', 'Unknown Location')}</b><br>"
                        text += f"Time: {row[time_column]}<br>"
                        for col in type_data.columns:
                            if col not in ['lat', 'lon', 'geometry', time_column]:
                                text += f"{col}: {row[col]}<br>"
                        hover_text.append(text)
                    
                    # Add data for this threat type
                    frame_data.append(
                        go.Scattermapbox(
                            mode='markers',
                            lon=type_data['lon'],
                            lat=type_data['lat'],
                            marker=dict(
                                size=10,
                                color=self.threat_colors.get(threat_type.lower(), '#FFFFFF'),
                                opacity=0.8
                            ),
                            text=hover_text,
                            hoverinfo='text',
                            name=f"{threat_type} Threat"
                        )
                    )
                
                # Create frame with data for all threat types
                frames.append(
                    go.Frame(
                        data=frame_data,
                        name=str(time_period)
                    )
                )
            else:
                # Create hover text
                hover_text = []
                for _, row in period_data.iterrows():
                    text = f"<b>{row.get('location', 'Unknown Location')}</b><br>"
                    text += f"Time: {row[time_column]}<br>"
                    for col in period_data.columns:
                        if col not in ['lat', 'lon', 'geometry', time_column]:
                            text += f"{col}: {row[col]}<br>"
                    hover_text.append(text)
                
                # Create frame with all data
                frames.append(
                    go.Frame(
                        data=[
                            go.Scattermapbox(
                                mode='markers',
                                lon=period_data['lon'],
                                lat=period_data['lat'],
                                marker=dict(
                                    size=10,
                                    color='#FF4B4B',
                                    opacity=0.8
                                ),
                                text=hover_text,
                                hoverinfo='text',
                                name='Threats'
                            )
                        ],
                        name=str(time_period)
                    )
                )
        
        # Add initial empty data
        if threat_type_col is not None:
            # Add empty traces for each threat type
            threat_types = data[threat_type_col].unique()
            for threat_type in threat_types:
                fig.add_trace(
                    go.Scattermapbox(
                        mode='markers',
                        lon=[],
                        lat=[],
                        marker=dict(
                            size=10,
                            color=self.threat_colors.get(threat_type.lower(), '#FFFFFF'),
                            opacity=0.8
                        ),
                        hoverinfo='text',
                        name=f"{threat_type} Threat"
                    )
                )
        else:
            # Add empty trace for all threats
            fig.add_trace(
                go.Scattermapbox(
                    mode='markers',
                    lon=[],
                    lat=[],
                    marker=dict(
                        size=10,
                        color='#FF4B4B',
                        opacity=0.8
                    ),
                    hoverinfo='text',
                    name='Threats'
                )
            )
        
        # Add frames to figure
        fig.frames = frames
        
        # Update layout with map configuration
        fig.update_layout(
            mapbox=dict(
                accesstoken=self.access_token,
                style=self.map_styles.get(map_style, self.map_styles['dark']),
                center=dict(lon=center[0], lat=center[1]),
                zoom=3
            ),
            margin={"r":0, "t":0, "l":0, "b":0},
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(0,0,0,0.5)"
            ),
            updatemenus=[
                # Animation controls
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[
                                None,
                                dict(
                                    frame=dict(duration=1000, redraw=True),
                                    fromcurrent=True,
                                    transition=dict(duration=500),
                                    mode="immediate"
                                )
                            ]
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[
                                [None],
                                dict(
                                    frame=dict(duration=0, redraw=False),
                                    mode="immediate",
                                    transition=dict(duration=0)
                                )
                            ]
                        )
                    ],
                    direction="left",
                    pad=dict(r=10, t=10),
                    x=0.1,
                    xanchor="right",
                    y=0.1,
                    yanchor="top"
                ),
                # Map style selector
                dict(
                    buttons=[
                        dict(
                            args=[{"mapbox.style": self.map_styles['dark']}],
                            label="Dark",
                            method="relayout"
                        ),
                        dict(
                            args=[{"mapbox.style": self.map_styles['satellite']}],
                            label="Satellite",
                            method="relayout"
                        ),
                        dict(
                            args=[{"mapbox.style": self.map_styles['light']}],
                            label="Light",
                            method="relayout"
                        ),
                        dict(
                            args=[{"mapbox.style": self.map_styles['outdoors']}],
                            label="Terrain",
                            method="relayout"
                        )
                    ],
                    direction="down",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.01,
                    xanchor="left",
                    y=0.05,
                    yanchor="bottom",
                    bgcolor="rgba(0,0,0,0.5)"
                )
            ],
            # Add slider for time selection
            sliders=[
                dict(
                    active=0,
                    steps=[
                        dict(
                            method="animate",
                            args=[
                                [str(time_period)],
                                dict(
                                    frame=dict(duration=300, redraw=True),
                                    mode="immediate",
                                    transition=dict(duration=300)
                                )
                            ],
                            label=str(time_period)
                        )
                        for time_period in time_periods
                    ],
                    x=0.1,
                    xanchor="left",
                    y=0,
                    yanchor="top",
                    currentvalue=dict(
                        font=dict(size=12),
                        prefix="Time: ",
                        visible=True,
                        xanchor="right"
                    ),
                    transition=dict(duration=300, easing="cubic-in-out"),
                    pad=dict(b=10, t=50),
                    len=0.9,
                    minorticklen=0,
                    bgcolor="rgba(0,0,0,0.5)"
                )
            ]
        )
        
        return fig