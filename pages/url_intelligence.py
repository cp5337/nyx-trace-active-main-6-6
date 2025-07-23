"""
URL Intelligence Dashboard
----------------------
Interactive dashboard for analyzing URL contexts, websites, 
and their connections to other intelligence sources.
"""

import asyncio
import datetime
import io
import json
import logging
import os
import time
import uuid
from typing import Dict, Any, List, Optional, Tuple, Union

import streamlit as st

# Set page configuration
# Must be the first Streamlit command
st.set_page_config(
    page_title="URL Intelligence | NyxTrace",
    page_icon="🌐",
    layout="wide"
)

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# Import our components
from collectors.url_context_collector import URLContextCollector
from processors.url_context_processor import URLContextProcessor
from processors.geo_resolver import GeospatialResolver
from processors.osint_data_processor import OSINTDataProcessor
from visualizers.url_context_card import URLContextCardVisualizer

# Initialize logger
logger = logging.getLogger(__name__)

# Create our component instances
url_collector = URLContextCollector()
url_processor = URLContextProcessor()
geo_resolver = GeospatialResolver()
osint_processor = OSINTDataProcessor()
url_visualizer = URLContextCardVisualizer()

def main():
    """Main function to run the URL Intelligence dashboard"""
    
    # Display page header
    st.markdown("""
    # URL Intelligence
    Analyze websites and online resources with instant context, metadata analysis, and intelligence connections.
    """)
    
    # Create tabs for different functionalities
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 URL Context Card", 
        "🗺️ Geospatial Analysis", 
        "🔗 Network Connections",
        "📊 Historical Analysis"
    ])
    
    with tab1:
        render_url_context_tab()
        
    with tab2:
        render_geospatial_tab()
        
    with tab3:
        render_network_tab()
        
    with tab4:
        render_historical_tab()

def render_url_context_tab():
    """Render the URL Context Card tab"""
    
    st.header("One-Click URL Context Card")
    st.markdown("""
    Enter a URL to get an instant snapshot with intelligent metadata analysis and emoji-based summaries.
    """)
    
    # URL input
    url = st.text_input("Website URL", placeholder="https://example.com")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        take_screenshot = st.checkbox("Take screenshot", value=True)
        
    with col2:
        if st.button("Analyze URL", type="primary"):
            if url:
                with st.spinner("Analyzing URL..."):
                    # Collect URL data
                    url_data = asyncio.run(collect_url_data(url, take_screenshot))
                    
                    if url_data:
                        # Process URL data
                        processed_data = asyncio.run(process_url_data(url_data))
                        
                        # Store in session state for other tabs
                        st.session_state.url_data = url_data
                        st.session_state.processed_url_data = processed_data
                        
                        # Visualize the context card
                        url_visualizer.visualize(processed_data)
                        
                        # Show export options
                        url_visualizer.render_export_options(processed_data)
                    else:
                        st.error("Failed to collect data from the URL. Please check the URL and try again.")
            else:
                st.warning("Please enter a URL.")
                
    # Recent URLs section
    st.subheader("Recent URLs")
    
    # Show some sample URLs
    sample_urls = [
        "https://www.bbc.com/news",
        "https://www.reuters.com",
        "https://www.theguardian.com/world",
        "https://apnews.com"
    ]
    
    url_cols = st.columns(4)
    
    for i, sample_url in enumerate(sample_urls):
        with url_cols[i]:
            if st.button(f"📄 {sample_url}", key=f"sample_url_{i}"):
                with st.spinner(f"Analyzing {sample_url}..."):
                    # Collect URL data
                    url_data = asyncio.run(collect_url_data(sample_url, take_screenshot))
                    
                    if url_data:
                        # Process URL data
                        processed_data = asyncio.run(process_url_data(url_data))
                        
                        # Store in session state for other tabs
                        st.session_state.url_data = url_data
                        st.session_state.processed_url_data = processed_data
                        
                        # Visualize the context card
                        url_visualizer.visualize(processed_data)
                        
                        # Show export options
                        url_visualizer.render_export_options(processed_data)
                    else:
                        st.error(f"Failed to collect data from {sample_url}. Please try again.")

def render_geospatial_tab():
    """Render the Geospatial Analysis tab"""
    
    st.header("Geospatial Intelligence from URLs")
    st.markdown("""
    Extract location information from URLs and visualize it on an interactive map.
    """)
    
    # Check if we have URL data in session state
    if not hasattr(st.session_state, 'url_data') or not st.session_state.url_data:
        st.info("Please analyze a URL in the URL Context Card tab first.")
        return
        
    # Get URL data
    url_data = st.session_state.url_data
    
    # Process for geospatial data
    with st.spinner("Extracting location information..."):
        geo_data = asyncio.run(process_geo_data(url_data))
        
    if not geo_data or not geo_data.get("geocoded_locations"):
        st.info("No location information found in the URL content.")
        return
        
    # Display geospatial data
    st.subheader(f"Locations mentioned in {url_data.get('domain', 'the website')}")
    
    # Create map data
    geocoded_locations = geo_data.get("geocoded_locations", [])
    
    if geocoded_locations:
        # Create dataframe for map
        map_data = pd.DataFrame([
            {
                "name": loc.get("query", "Unknown"),
                "address": loc.get("address", ""),
                "lat": loc.get("latitude"),
                "lon": loc.get("longitude")
            }
            for loc in geocoded_locations
            if "latitude" in loc and "longitude" in loc
        ])
        
        # Display map
        st.map(map_data)
        
        # Display location table
        st.subheader("Extracted Locations")
        st.dataframe(map_data)
        
    # Display region information if available
    regions = geo_data.get("regions", [])
    if regions:
        st.subheader("Detected Regions")
        
        for i, region in enumerate(regions):
            st.markdown(f"**Region {i+1}: {region.get('name', 'Unnamed Region')}**")
            st.markdown(f"- Type: {region.get('type', 'Unknown')}")
            st.markdown(f"- Points: {region.get('points', 0)}")
            
            center = region.get("center", {})
            if center:
                st.markdown(f"- Center: Lat {center.get('latitude', 0):.4f}, Lon {center.get('longitude', 0):.4f}")

def render_network_tab():
    """Render the Network Connections tab"""
    
    st.header("URL Intelligence Network")
    st.markdown("""
    View connections between the URL content and OSINT data sources.
    """)
    
    # Check if we have URL data in session state
    if not hasattr(st.session_state, 'url_data') or not st.session_state.url_data:
        st.info("Please analyze a URL in the URL Context Card tab first.")
        return
        
    # Get URL data
    url_data = st.session_state.url_data
    
    # Process for OSINT connections
    with st.spinner("Finding connections to OSINT data..."):
        osint_data = asyncio.run(process_osint_connections(url_data))
        
    if not osint_data or not osint_data.get("related_entities"):
        st.info("No connections found to OSINT data sources.")
        
        # Show sample OSINT file upload option
        st.subheader("Upload OSINT Data")
        st.markdown("""
        Upload an OSINT dataset (Excel or CSV) to analyze connections with this URL.
        """)
        
        osint_file = st.file_uploader("Upload OSINT dataset", type=["xlsx", "csv"])
        
        if osint_file:
            # Save the uploaded file
            file_path = os.path.join("data", osint_file.name)
            os.makedirs("data", exist_ok=True)
            
            with open(file_path, "wb") as f:
                f.write(osint_file.getbuffer())
                
            # Process the dataset
            with st.spinner("Processing OSINT dataset..."):
                result = osint_processor.load_osint_dataset(file_path)
                
                st.success(f"OSINT dataset loaded with {len(result.get('entities', []))} entities.")
                
                # Re-process URL with this dataset
                osint_data = asyncio.run(process_osint_connections(url_data))
                
                if osint_data and osint_data.get("related_entities"):
                    st.success(f"Found {len(osint_data['related_entities'])} connections!")
                else:
                    st.info("No connections found between the URL and the dataset.")
        
        return
        
    # Display connections
    st.subheader(f"Connections to {url_data.get('domain', 'the website')}")
    
    # Create network visualization
    # For simplicity, just show related entities for now
    related_entities = osint_data.get("related_entities", [])
    
    if related_entities:
        # Create dataframe for display
        entity_df = pd.DataFrame([
            {
                "Type": entity.get("type", "Unknown").title(),
                "Name": entity.get("attributes", {}).get("name", "Unnamed"),
                "Source": entity.get("source", {}).get("type", "Unknown")
            }
            for entity in related_entities
        ])
        
        # Display entity table
        st.dataframe(entity_df)
        
        # Display connection graph
        st.subheader("Entity Network")
        st.info("Interactive network graph will be displayed here.")
        
    # Related datasets
    related_datasets = osint_data.get("related_datasets", [])
    if related_datasets:
        st.subheader("Related OSINT Datasets")
        
        for dataset in related_datasets:
            st.markdown(f"- **{os.path.basename(dataset.get('file_path', 'Unknown'))}**: {dataset.get('match_count', 0)} matches")

def render_historical_tab():
    """Render the Historical Analysis tab"""
    
    st.header("URL History and Changes")
    st.markdown("""
    Track changes in URL content over time and analyze historical patterns.
    """)
    
    # Check if we have URL data in session state
    if not hasattr(st.session_state, 'url_data') or not st.session_state.url_data:
        st.info("Please analyze a URL in the URL Context Card tab first.")
        return
        
    # Get URL data
    url_data = st.session_state.url_data
    
    # For now, just show placeholder content
    st.info("Historical analysis will be implemented in a future version.")
    
    # Sample visualization
    st.subheader(f"Content Changes for {url_data.get('domain', 'the website')}")
    
    # Create sample data
    dates = pd.date_range(end=pd.Timestamp.now(), periods=10, freq='D')
    changes = [5, 8, 3, 12, 7, 4, 9, 15, 6, 10]
    
    # Create dataframe
    df = pd.DataFrame({
        'Date': dates,
        'Content Changes': changes
    })
    
    # Create bar chart
    fig = px.bar(
        df, 
        x='Date', 
        y='Content Changes',
        title=f"Content Changes for {url_data.get('domain', 'the website')}",
        labels={'Content Changes': 'Number of Changes', 'Date': 'Date'}
    )
    
    st.plotly_chart(fig, use_container_width=True)

async def collect_url_data(url: str, take_screenshot: bool = True) -> Optional[Dict[str, Any]]:
    """
    Collect data from a URL
    
    Args:
        url: URL to collect data from
        take_screenshot: Whether to take a screenshot
        
    Returns:
        Collected URL data or None if collection failed
    """
    try:
        # Create collection parameters
        from core.interfaces.collectors import CollectorParams
        
        params = CollectorParams(
            targets=[url],
            priority="Medium",
            custom_params={
                "take_screenshot": take_screenshot
            }
        )
        
        # Collect URL data
        url_data = await url_collector.collect_url_context(url, params)
        
        if url_data:
            return url_data.to_dict()
        return None
        
    except Exception as e:
        logger.error(f"Error collecting URL data for {url}: {str(e)}")
        return None

async def process_url_data(url_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process collected URL data
    
    Args:
        url_data: Raw URL data to process
        
    Returns:
        Processed URL data
    """
    try:
        # Create EEI for processing
        from core.interfaces.collectors import EEI
        
        eei = EEI(
            eei_id=f"eei-url-{uuid.uuid4()}",
            source_id="url_context_collector",
            collection_time=datetime.datetime.now(),
            data=url_data,
            confidence=0.8,
            priority="Medium",
            metadata={
                "url": url_data.get("url", ""),
                "domain": url_data.get("domain", "")
            }
        )
        
        # Process the EEI
        processed = await url_processor.process_single(eei)
        
        return processed.processed_data
        
    except Exception as e:
        logger.error(f"Error processing URL data: {str(e)}")
        return url_data  # Return original data if processing fails

async def process_geo_data(url_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process URL data for geospatial information
    
    Args:
        url_data: URL data to process
        
    Returns:
        Geospatial data extracted from the URL
    """
    try:
        # Create EEI for processing
        from core.interfaces.collectors import EEI
        
        eei = EEI(
            eei_id=f"eei-url-geo-{uuid.uuid4()}",
            source_id="url_context_collector",
            collection_time=datetime.datetime.now(),
            data=url_data,
            confidence=0.8,
            priority="Medium",
            metadata={
                "url": url_data.get("url", ""),
                "domain": url_data.get("domain", "")
            }
        )
        
        # Process the EEI with geo resolver
        processed = await geo_resolver.process_single(eei)
        
        return processed.processed_data
        
    except Exception as e:
        logger.error(f"Error processing geospatial data: {str(e)}")
        return {"geocoded_locations": [], "regions": []}

async def process_osint_connections(url_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process URL data for connections to OSINT data
    
    Args:
        url_data: URL data to process
        
    Returns:
        OSINT connections data
    """
    try:
        # Create EEI for processing
        from core.interfaces.collectors import EEI
        
        eei = EEI(
            eei_id=f"eei-url-osint-{uuid.uuid4()}",
            source_id="url_context_collector",
            collection_time=datetime.datetime.now(),
            data=url_data,
            confidence=0.8,
            priority="Medium",
            metadata={
                "url": url_data.get("url", ""),
                "domain": url_data.get("domain", "")
            }
        )
        
        # Process the EEI with OSINT processor
        processed = await osint_processor.process_single(eei)
        
        return processed.processed_data
        
    except Exception as e:
        logger.error(f"Error processing OSINT connections: {str(e)}")
        return {"related_entities": [], "related_datasets": [], "connections": []}

if __name__ == "__main__":
    main()