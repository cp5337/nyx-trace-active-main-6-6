"""
Heatmap Visualization Module
---------------------------
This module provides advanced heatmap visualization capabilities
for geospatial data.
"""
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import streamlit as st
import sys
import os

# Import theme settings directly
try:
    from utils import get_theme_settings
except ImportError:
    # Fallback for module imports
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from utils import get_theme_settings


def create_heatmap(df, intensity_column="Activity_Level", radius=25, blur=10, 
                   zoom=3, center_lat=39.8283, center_lon=-98.5795, height=600):
    """
    Create an advanced heatmap visualization based on location data
    
    Args:
        df: DataFrame containing location data with Latitude and Longitude columns
        intensity_column: Column name to use for heatmap intensity values
        radius: Radius of influence for each point in the heatmap (in pixels)
        blur: Amount of blur to apply to the heatmap (in pixels)
        zoom: Initial zoom level for the map
        center_lat: Center latitude for the map
        center_lon: Center longitude for the map
        height: Height of the map in pixels
        
    Returns:
        Plotly figure with heatmap visualization
    """
    # Get theme settings
    theme = get_theme_settings()
    
    # Validate input data
    if not all(col in df.columns for col in ['Latitude', 'Longitude', intensity_column]):
        raise ValueError(f"DataFrame must contain 'Latitude', 'Longitude' and '{intensity_column}' columns")
    
    # Normalize intensity values to 0-1 range for better visualization
    if df[intensity_column].min() != df[intensity_column].max():
        df = df.copy()
        df['intensity_normalized'] = (df[intensity_column] - df[intensity_column].min()) / \
                                    (df[intensity_column].max() - df[intensity_column].min())
    else:
        df = df.copy()
        df['intensity_normalized'] = 1.0
        
    # Create base map
    fig = go.Figure()
    
    # Add heatmap layer
    fig.add_trace(go.Densitymapbox(
        lat=df['Latitude'],
        lon=df['Longitude'],
        z=df['intensity_normalized'],
        radius=radius,
        colorscale='Viridis',
        colorbar=dict(
            title=intensity_column,
            titleside='right',
            title_font=dict(color='white' if theme['is_dark'] else 'black')
        ),
        hoverinfo='none'
    ))
    
    # Add marker layer for locations
    fig.add_trace(go.Scattermapbox(
        lat=df['Latitude'],
        lon=df['Longitude'],
        mode='markers',
        marker=dict(
            size=8,
            color='white' if theme['is_dark'] else 'black',
            opacity=0.7
        ),
        text=df['City'] if 'City' in df.columns else None,
        hoverinfo='text'
    ))
    
    # Configure layout
    fig.update_layout(
        mapbox=dict(
            style=theme['mapbox_style'],
            center=dict(lat=center_lat, lon=center_lon),
            zoom=zoom
        ),
        margin=dict(r=0, t=0, l=0, b=0),
        height=height,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def create_time_based_heatmap(df, date_column='Date', intensity_column='Activity_Level',
                             animation_frame=None, show_slider=True, height=600):
    """
    Create a time-based heatmap visualization that shows how intensity changes over time
    
    Args:
        df: DataFrame containing time series location data
        date_column: Column containing date information 
        intensity_column: Column with intensity values for the heatmap
        animation_frame: Column to use for animation frames (typically a time-based column)
        show_slider: Whether to show the time slider control
        height: Height of the visualization in pixels
        
    Returns:
        Plotly figure with animated time-based heatmap
    """
    # Get theme settings
    theme = get_theme_settings()
    
    # Validate input data
    required_cols = ['Latitude', 'Longitude', date_column, intensity_column]
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"DataFrame missing required columns: {missing}")
    
    # If animation_frame is None but we have a date column, use that for animation
    if animation_frame is None and date_column in df.columns:
        # Convert dates to string format for animation
        if pd.api.types.is_datetime64_any_dtype(df[date_column]):
            df = df.copy()
            df['date_str'] = df[date_column].dt.strftime('%Y-%m-%d')
            animation_frame = 'date_str'
        else:
            animation_frame = date_column
    
    # Create the animated heatmap
    fig = px.density_mapbox(
        df,
        lat='Latitude',
        lon='Longitude',
        z=intensity_column,
        animation_frame=animation_frame,
        mapbox_style=theme['mapbox_style'],
        radius=20,
        color_continuous_scale='Viridis',
        range_color=[df[intensity_column].min(), df[intensity_column].max()],
        hover_name='City' if 'City' in df.columns else None,
        height=height
    )
    
    # Configure layout
    fig.update_layout(
        mapbox=dict(
            center=dict(lat=39.8283, lon=-98.5795),
            zoom=3
        ),
        margin=dict(r=0, t=0, l=0, b=0),
        template=theme['template']
    )
    
    # Configure animation settings
    if animation_frame is not None:
        fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 500
        fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 300
        
        # Hide slider if requested
        if not show_slider:
            fig.layout.sliders[0].visible = False
    
    return fig


def create_clustered_heatmap(df, n_clusters=5, intensity_column='Activity_Level',
                            zoom=3, height=600):
    """
    Create a heatmap with clustered data points for better visualization
    
    Args:
        df: DataFrame containing location data
        n_clusters: Number of clusters to create
        intensity_column: Column with intensity values
        zoom: Initial zoom level for the map
        height: Height of the visualization in pixels
        
    Returns:
        Plotly figure with clustered heatmap
    """
    from sklearn.cluster import KMeans
    
    # Get theme settings
    theme = get_theme_settings()
    
    # Check if we have enough data points for clustering
    if len(df) < n_clusters:
        # Fall back to regular heatmap if not enough points
        return create_heatmap(df, intensity_column=intensity_column, zoom=zoom, height=height)
    
    # Extract coordinates for clustering
    coords = df[['Latitude', 'Longitude']].values
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df = df.copy()
    df['Cluster'] = kmeans.fit_predict(coords)
    
    # Get cluster centers
    cluster_centers = kmeans.cluster_centers_
    
    # Create a figure with both the points and the cluster centers
    fig = px.scatter_mapbox(
        df,
        lat='Latitude',
        lon='Longitude',
        color='Cluster',
        size=intensity_column if intensity_column in df.columns else None,
        hover_name='City' if 'City' in df.columns else None,
        zoom=zoom,
        height=height,
        mapbox_style=theme['mapbox_style'],
        opacity=0.7
    )
    
    # Add the cluster centers
    for i, center in enumerate(cluster_centers):
        fig.add_trace(go.Scattermapbox(
            lat=[center[0]],
            lon=[center[1]],
            mode='markers',
            marker=dict(
                size=20,
                color=f'rgba({i*50 % 255}, {(i*80) % 255}, {(i*120) % 255}, 0.8)',
                symbol='circle'
            ),
            name=f'Cluster {i+1} Center'
        ))
    
    # Configure layout
    fig.update_layout(
        mapbox=dict(
            center=dict(lat=39.8283, lon=-98.5795),
            zoom=zoom
        ),
        margin=dict(r=0, t=0, l=0, b=0),
        template=theme['template']
    )
    
    return fig