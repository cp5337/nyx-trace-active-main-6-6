"""
Threat Intelligence Dashboard
----------------------------
Provides visualization and analysis of cyber threat intelligence data,
including threat actor activities, IOCs, and bulletins.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os
from pathlib import Path

# Add parent directory to path to ensure imports work correctly
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

# Set page config
st.set_page_config(
    page_title="NyxTrace - Threat Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("Threat Intelligence Dashboard")
    
    # Initialize session state
    if 'threat_intel_initialized' not in st.session_state:
        st.session_state['threat_intel_initialized'] = True
        st.session_state['flow_data'] = pd.DataFrame()
        st.session_state['ioc_data'] = []
        st.session_state['bulletins'] = []
    
    # API Key input
    otx_api_key = st.text_input("AlienVault OTX API Key", value="", type="password")
    
    if otx_api_key:
        st.session_state['otx_api_key'] = otx_api_key
        st.success("API Key configured")
    
    # Display header and introduction
    st.markdown("""
    This dashboard displays threat intelligence data from AlienVault OTX and other sources.
    Monitor threat actor activities, review indicators of compromise, and analyze threat bulletins.
    """)
    
    # Load sample data
    if st.session_state['flow_data'].empty:
        try:
            from data.sample_threat_intel import generate_sample_flow_data, generate_sample_iocs, generate_sample_bulletins
            st.session_state['flow_data'] = generate_sample_flow_data()
            st.session_state['ioc_data'] = generate_sample_iocs()
            st.session_state['bulletins'] = generate_sample_bulletins()
            st.info("Using sample data for demonstration. Connect to AlienVault OTX for real threat intelligence.")
        except ImportError as e:
            st.error(f"Could not load sample threat intelligence data: {e}")
    
    # Threat map visualization
    st.header("Global Threat Activity Map")
    
    # Create a simplified map using the flow data
    if not st.session_state['flow_data'].empty:
        df = st.session_state['flow_data']
        
        # Create map
        fig = px.scatter_mapbox(
            df, 
            lat="target_lat",
            lon="target_lon", 
            color="weight",
            size="weight",
            size_max=15,
            zoom=1.5,
            color_continuous_scale=["green", "yellow", "red"],
            mapbox_style="carto-darkmatter",
            hover_name="target",
            hover_data=["source", "date", "type", "sector"]
        )
        
        # Add connecting lines between source and target
        for _, row in df.iterrows():
            fig.add_trace(
                go.Scattermapbox(
                    lat=[row['source_lat'], row['target_lat']],
                    lon=[row['source_lon'], row['target_lon']],
                    mode='lines',
                    line=dict(width=1, color='rgba(255, 255, 255, 0.2)'),
                    hoverinfo='none',
                    showlegend=False
                )
            )
        
        # Update layout
        fig.update_layout(
            margin={"r":0,"t":0,"l":0,"b":0},
            coloraxis_showscale=False,
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No flow data available to display map")
    
    # Split into columns for metrics and table
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Threat Flow Analysis")
        
        if not st.session_state['flow_data'].empty:
            df = st.session_state['flow_data']
            
            # Display dataframe with most recent attacks
            st.subheader("Recent Activities")
            recent_df = df.sort_values('date', ascending=False).head(10)
            st.dataframe(
                recent_df[['date', 'source', 'target', 'type', 'sector', 'confidence']],
                use_container_width=True
            )
            
            # Display breakdown by attack type
            st.subheader("Attack Types")
            attack_counts = df['type'].value_counts().reset_index()
            attack_counts.columns = ['Attack Type', 'Count']
            
            fig = px.bar(
                attack_counts,
                x='Attack Type',
                y='Count',
                color='Attack Type',
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            
            fig.update_layout(
                xaxis_title="",
                yaxis_title="Number of Attacks",
                height=300,
                margin=dict(l=20, r=20, t=20, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.header("Threat Metrics")
        
        # Calculate metrics
        if not st.session_state['flow_data'].empty:
            df = st.session_state['flow_data']
            
            # Count unique values
            actor_count = df['source'].nunique()
            target_count = df['target'].nunique()
            incident_count = len(df)
            
            # Calculate most targeted sector
            if 'sector' in df.columns:
                top_sector = df['sector'].value_counts().idxmax()
            else:
                top_sector = "Unknown"
            
            # Calculate average confidence
            if 'confidence' in df.columns:
                confidence_map = {'High': 3, 'Medium': 2, 'Low': 1}
                if all(conf in confidence_map for conf in df['confidence']):
                    avg_confidence = sum(confidence_map[conf] for conf in df['confidence']) / len(df)
                    avg_confidence = f"{avg_confidence:.1f}/3.0"
                else:
                    avg_confidence = "N/A"
            else:
                avg_confidence = "N/A"
            
            # Display metrics
            metrics = {
                "Threat Actors": actor_count,
                "Target Entities": target_count,
                "Total Incidents": incident_count,
                "Top Targeted Sector": top_sector,
                "Average Confidence": avg_confidence
            }
            
            for name, value in metrics.items():
                st.metric(name, value)
    
    # Tabs for IOCs and Bulletins
    tab1, tab2 = st.tabs(["Indicators of Compromise", "Threat Bulletins"])
    
    with tab1:
        st.header("Indicators of Compromise (IOCs)")
        
        if st.session_state['ioc_data']:
            # Create DataFrame for display
            iocs_df = pd.DataFrame(st.session_state['ioc_data'])
            
            # Add filter for IOC type
            if 'type' in iocs_df.columns:
                ioc_types = ['All'] + sorted(iocs_df['type'].unique().tolist())
                selected_type = st.selectbox("Filter by Type", ioc_types)
                
                if selected_type != 'All':
                    filtered_iocs = iocs_df[iocs_df['type'] == selected_type]
                else:
                    filtered_iocs = iocs_df
                
                # Display the filtered dataframe
                st.dataframe(filtered_iocs, use_container_width=True)
            else:
                st.dataframe(iocs_df, use_container_width=True)
        else:
            st.info("No IOC data available.")
    
    with tab2:
        st.header("Threat Intelligence Bulletins")
        
        if st.session_state['bulletins']:
            for bulletin in st.session_state['bulletins']:
                with st.expander(f"{bulletin.get('title', 'Untitled')} - {bulletin.get('date', 'No date')}"):
                    st.markdown(f"**Severity:** {bulletin.get('severity', 'Unknown')}")
                    st.markdown(f"**Category:** {bulletin.get('category', 'Uncategorized')}")
                    st.markdown(bulletin.get('description', 'No description available.'))
                    
                    if 'recommendations' in bulletin and bulletin['recommendations']:
                        st.markdown("**Recommendations:**")
                        for rec in bulletin['recommendations']:
                            st.markdown(f"- {rec}")
        else:
            st.info("No threat bulletins available.")

if __name__ == "__main__":
    main()