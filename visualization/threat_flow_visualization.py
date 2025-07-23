"""
Threat Flow Visualization Module
-------------------------------
This module provides components for visualizing threat flows between
source and target entities (threat actors and victims).
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Union, Optional, Tuple
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('threat_flow_visualization')


class ThreatFlowVisualizer:
    """
    Visualizer for cyber threat flows between source and target entities
    """
    
    def __init__(self, map_style: str = "carto-darkmatter"):
        """
        Initialize threat flow visualizer
        
        Args:
            map_style: Mapbox style for the base map
        """
        self.map_style = map_style
        self.map_token = os.environ.get('MAPBOX_ACCESS_TOKEN', None)
        self.colors = {
            'attack': 'red',
            'reconnaissance': 'orange',
            'data_exfiltration': 'purple',
            'lateral_movement': 'yellow',
            'command_and_control': 'blue'
        }
    
    def plot_threat_flows(self, 
                         flow_data: pd.DataFrame, 
                         title: str = "Cyber Threat Flows",
                         height: int = 600) -> go.Figure:
        """
        Create a map visualization of threat flows
        
        Args:
            flow_data: DataFrame with threat flow data
                - source, target: entity names
                - source_lat, source_lon: source coordinates
                - target_lat, target_lon: target coordinates
                - weight: flow weight (thickness)
                - type: flow type (color)
                - title, date: flow metadata
            title: Plot title
            height: Plot height in pixels
            
        Returns:
            Plotly figure with threat flows
        """
        if flow_data.empty:
            # Return empty figure if no data
            fig = go.Figure()
            fig.update_layout(
                title=title,
                height=height,
                annotations=[
                    dict(
                        text="No threat flow data available",
                        showarrow=False,
                        xref="paper",
                        yref="paper",
                        x=0.5,
                        y=0.5,
                        font=dict(size=20)
                    )
                ]
            )
            return fig
        
        # First create base map
        fig = go.Figure()
        
        # Add lines for flows
        for idx, row in flow_data.iterrows():
            # Calculate curve midpoint (with some randomness)
            mid_lat = (row['source_lat'] + row['target_lat']) / 2
            mid_lon = (row['source_lon'] + row['target_lon']) / 2
            
            # Add random offset for curve height
            # Use a specific seed for each flow pair to ensure consistent curves
            np.random.seed(hash(f"{row['source']}-{row['target']}") % 2**32)
            lat_offset = np.random.uniform(0.5, 3.0) * (1 if row['source_lat'] > row['target_lat'] else -1)
            lon_offset = np.random.uniform(0.5, 3.0) * (1 if row['source_lon'] > row['target_lon'] else -1)
            
            mid_lat += lat_offset
            mid_lon += lon_offset
            
            # Create the curved line
            line_color = self.colors.get(row['type'], 'red')
            weight_scaled = min(max(1, row['weight'] * 1.5), 10)  # Scale thickness
            
            # Add flow path as curved line
            fig.add_trace(
                go.Scattergeo(
                    lon=[row['source_lon'], mid_lon, row['target_lon']],
                    lat=[row['source_lat'], mid_lat, row['target_lat']],
                    mode='lines',
                    line=dict(
                        width=weight_scaled,
                        color=line_color,
                        dash='solid'
                    ),
                    opacity=0.7,
                    name=f"{row['source']} → {row['target']}",
                    hoverinfo='text',
                    text=f"{row['source']} → {row['target']}<br>{row['title']}<br>{row['date'].strftime('%Y-%m-%d')}"
                )
            )
            
            # Add animated marker moving along the path
            # Divide the path into segments for animation
            num_points = 20
            lats = []
            lons = []
            
            # Generate bezier curve points
            t_vals = np.linspace(0, 1, num_points)
            for t in t_vals:
                # Quadratic bezier curve formula
                lat = (1-t)**2 * row['source_lat'] + 2*(1-t)*t * mid_lat + t**2 * row['target_lat']
                lon = (1-t)**2 * row['source_lon'] + 2*(1-t)*t * mid_lon + t**2 * row['target_lon']
                lats.append(lat)
                lons.append(lon)
            
            # Create animation frames
            for i in range(num_points):
                fig.add_trace(
                    go.Scattergeo(
                        lon=[lons[i]],
                        lat=[lats[i]],
                        mode='markers',
                        marker=dict(
                            size=8,
                            color=line_color,
                            opacity=0.8
                        ),
                        hoverinfo='skip',
                        visible=(i == 0)  # Only first point visible initially
                    )
                )
        
        # Add source nodes (origins)
        sources = flow_data[['source', 'source_lat', 'source_lon']].drop_duplicates()
        
        fig.add_trace(
            go.Scattergeo(
                lon=sources['source_lon'],
                lat=sources['source_lat'],
                text=sources['source'],
                mode='markers',
                marker=dict(
                    size=15,
                    color='yellow',
                    opacity=0.8,
                    line=dict(
                        width=1,
                        color='black'
                    )
                ),
                name='Threat Actors',
                hoverinfo='text'
            )
        )
        
        # Add target nodes (destinations)
        targets = flow_data[['target', 'target_lat', 'target_lon']].drop_duplicates()
        
        fig.add_trace(
            go.Scattergeo(
                lon=targets['target_lon'],
                lat=targets['target_lat'],
                text=targets['target'],
                mode='markers',
                marker=dict(
                    size=12,
                    color='red',
                    opacity=0.8,
                    line=dict(
                        width=1,
                        color='black'
                    ),
                    symbol='diamond'
                ),
                name='Targets',
                hoverinfo='text'
            )
        )
        
        # Set up layout
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=20)
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=height,
            margin=dict(l=0, r=0, t=50, b=0),
            geo=dict(
                scope='world',
                projection_type='natural earth',
                showland=True,
                landcolor='rgb(50, 50, 50)',
                oceancolor='rgb(30, 30, 30)',
                showocean=True,
                showcountries=True,
                countrycolor='rgb(100, 100, 100)',
                showframe=False,
                showcoastlines=True,
                coastlinecolor='rgb(100, 100, 100)',
                bgcolor='rgba(0,0,0,0)'
            ),
            updatemenus=[
                dict(
                    type="buttons",
                    buttons=[
                        dict(
                            label="Play",
                            method="animate", 
                            args=[None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}]
                        )
                    ],
                    direction="left",
                    pad={"r": 10, "t": 10},
                    showactive=False,
                    x=0.1,
                    xanchor="right",
                    y=0,
                    yanchor="top"
                )
            ],
            # Add animation frames
            # Create frames for each point in the animation
            sliders=[
                {
                    "steps": [
                        {
                            "args": [
                                [f"frame-{i}"],
                                {"frame": {"duration": 100, "redraw": True}, "mode": "immediate"}
                            ],
                            "label": f"{i}",
                            "method": "animate"
                        }
                        for i in range(num_points)
                    ],
                    "active": 0,
                    "yanchor": "top",
                    "xanchor": "left",
                    "currentvalue": {
                        "visible": True,
                        "prefix": "Frame: ",
                        "xanchor": "right"
                    },
                    "transition": {"duration": 100},
                    "pad": {"b": 10, "t": 50},
                    "len": 0.9,
                    "x": 0.1,
                    "y": 0
                }
            ]
        )
        
        # Add the Mapbox layout if token available
        if self.map_token:
            fig.update_layout(
                mapbox=dict(
                    style=self.map_style,
                    accesstoken=self.map_token,
                    zoom=1.5
                )
            )
        
        # Create animation frames
        frames = []
        
        # Calculate number of traces per frame
        num_flow_traces = len(flow_data) * 2  # Lines + animation points
        num_node_traces = 2  # Source and target nodes
        
        for i in range(num_points):
            frame_data = []
            
            # Include all base flow lines
            for j in range(len(flow_data)):
                frame_data.append(fig.data[j])  # Flow lines
            
            # Include animation points with visibility
            for j in range(len(flow_data)):
                # Calculate the index for the animation point
                point_index = len(flow_data) + j * num_points + i
                
                # If index is within bounds, update visibility
                if point_index < len(fig.data) - num_node_traces:
                    trace = fig.data[point_index]
                    # Make a copy and update visibility
                    visible_trace = trace
                    frame_data.append(visible_trace)
                
            # Add source and target nodes (always visible)
            frame_data.append(fig.data[-2])  # Source nodes
            frame_data.append(fig.data[-1])  # Target nodes
            
            frames.append(go.Frame(data=frame_data, name=f"frame-{i}"))
        
        fig.frames = frames
        
        return fig
    
    def create_threat_actor_chart(self, 
                               flow_data: pd.DataFrame, 
                               title: str = "Threat Actors by Activity",
                               height: int = 400) -> go.Figure:
        """
        Create a bar chart of threat actors by activity level
        
        Args:
            flow_data: DataFrame with threat flow data
            title: Chart title
            height: Chart height in pixels
            
        Returns:
            Plotly figure with threat actor activity
        """
        if flow_data.empty:
            # Return empty figure if no data
            fig = go.Figure()
            fig.update_layout(
                title=title,
                height=height,
                annotations=[
                    dict(
                        text="No threat actor data available",
                        showarrow=False,
                        xref="paper",
                        yref="paper",
                        x=0.5,
                        y=0.5,
                        font=dict(size=20)
                    )
                ]
            )
            return fig
        
        # Aggregate data by source (threat actor)
        actor_data = flow_data.groupby('source')['weight'].sum().reset_index()
        actor_data = actor_data.sort_values('weight', ascending=True)
        
        # Create horizontal bar chart
        fig = go.Figure()
        
        fig.add_trace(
            go.Bar(
                y=actor_data['source'],
                x=actor_data['weight'],
                orientation='h',
                marker=dict(
                    color='rgba(255, 165, 0, 0.8)',
                    line=dict(
                        color='rgba(255, 165, 0, 1.0)',
                        width=1
                    )
                )
            )
        )
        
        # Customize layout
        fig.update_layout(
            title=title,
            xaxis_title="Activity Level",
            yaxis_title="Threat Actor",
            height=height,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False)
        )
        
        return fig
    
    def create_target_distribution(self, 
                               flow_data: pd.DataFrame, 
                               title: str = "Target Distribution",
                               height: int = 400) -> go.Figure:
        """
        Create a pie chart of target distribution
        
        Args:
            flow_data: DataFrame with threat flow data
            title: Chart title
            height: Chart height in pixels
            
        Returns:
            Plotly figure with target distribution
        """
        if flow_data.empty:
            # Return empty figure if no data
            fig = go.Figure()
            fig.update_layout(
                title=title,
                height=height,
                annotations=[
                    dict(
                        text="No target data available",
                        showarrow=False,
                        xref="paper",
                        yref="paper",
                        x=0.5,
                        y=0.5,
                        font=dict(size=20)
                    )
                ]
            )
            return fig
        
        # Aggregate data by target
        target_data = flow_data.groupby('target')['weight'].sum().reset_index()
        
        # Create pie chart
        fig = go.Figure(
            go.Pie(
                labels=target_data['target'],
                values=target_data['weight'],
                hole=0.4,
                marker=dict(
                    colors=['#FF6B6B', '#4ECDC4', '#FFA938', '#25CED1', '#F8961E', '#F94144']
                )
            )
        )
        
        # Customize layout
        fig.update_layout(
            title=title,
            height=height,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            )
        )
        
        return fig
    
    def render_dashboard(self, flow_data: pd.DataFrame) -> None:
        """
        Render a complete threat intelligence dashboard using Streamlit
        
        Args:
            flow_data: DataFrame with threat flow data
        """
        st.markdown("## Cyber Threat Intelligence Dashboard")
        
        # Add date range selector (if flow_data has dates)
        if not flow_data.empty and 'date' in flow_data.columns:
            min_date = flow_data['date'].min().date()
            max_date = flow_data['date'].max().date()
            
            col1, col2 = st.columns(2)
            
            with col1:
                start_date = st.date_input("Start Date", min_date, key="tf_start_date")
            
            with col2:
                end_date = st.date_input("End Date", max_date, key="tf_end_date")
            
            # Filter data by date range
            mask = (flow_data['date'].dt.date >= start_date) & (flow_data['date'].dt.date <= end_date)
            filtered_data = flow_data[mask]
        else:
            filtered_data = flow_data
        
        # Display flow map
        st.plotly_chart(
            self.plot_threat_flows(
                filtered_data, 
                title="Global Cyber Threat Activity",
                height=600
            ),
            use_container_width=True
        )
        
        # Activity metrics
        if not filtered_data.empty:
            # Create metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                num_actors = filtered_data['source'].nunique()
                st.metric("Threat Actors", num_actors)
            
            with col2:
                num_targets = filtered_data['target'].nunique()
                st.metric("Targeted Entities", num_targets)
            
            with col3:
                num_flows = len(filtered_data)
                st.metric("Threat Flows", num_flows)
            
            with col4:
                total_weight = filtered_data['weight'].sum()
                st.metric("Total Activity", f"{total_weight:.1f}")
        
        # Add additional charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(
                self.create_threat_actor_chart(filtered_data),
                use_container_width=True
            )
        
        with col2:
            st.plotly_chart(
                self.create_target_distribution(filtered_data),
                use_container_width=True
            )
        
        # Add table of recent activity
        if not filtered_data.empty:
            st.markdown("### Recent Threat Activity")
            
            display_df = filtered_data.sort_values('date', ascending=False).head(10)
            display_df = display_df[['source', 'target', 'type', 'weight', 'title', 'date']]
            
            # Rename columns for display
            display_df.columns = ['Threat Actor', 'Target', 'Attack Type', 'Severity', 'Description', 'Date']
            
            # Format date
            display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
            
            # Add styling
            def color_severity(val):
                if val >= 5:
                    return 'background-color: rgba(255,0,0,0.5)'
                elif val >= 3:
                    return 'background-color: rgba(255,165,0,0.5)'
                else:
                    return 'background-color: rgba(255,255,0,0.3)'
            
            # Apply styling
            styled_df = display_df.style.map(
                color_severity, 
                subset=['Severity']
            )
            
            st.dataframe(styled_df, use_container_width=True)
        
        # Add analysis insights
        st.markdown("### Threat Analysis Insights")
        
        if not filtered_data.empty:
            # Generate some simple insights
            top_actor = filtered_data.groupby('source')['weight'].sum().idxmax()
            top_target = filtered_data.groupby('target')['weight'].sum().idxmax()
            
            insights = f"""
            - The most active threat actor is **{top_actor}** with significant activity across multiple targets.
            - **{top_target}** is the most targeted entity in this time period.
            - There are **{num_flows}** distinct threat flows from **{num_actors}** threat actors targeting **{num_targets}** entities.
            """
            
            st.markdown(insights)
            
            # Add recommendations
            st.markdown("### Defensive Recommendations")
            
            recommendations = """
            1. Implement additional monitoring for communications with identified threat actor infrastructure
            2. Deploy enhanced security controls for the most targeted systems
            3. Update threat intelligence feeds with newly identified indicators
            4. Conduct targeted threat hunting for specific TTPs observed
            5. Share intelligence with partner organizations via ISAC/ISAO channels
            """
            
            st.markdown(recommendations)
        else:
            st.info("No threat intelligence data available for analysis. Please adjust filters or import data.")