"""
Threat Intelligence Dashboard
---------------------------
This module provides a comprehensive dashboard for cyber threat intelligence
visualization and analysis, with a focus on AlienVault OTX data and source-target
flow visualization.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, List, Any, Union, Optional, Tuple

# Import data source
from data_sources.plugins.alienvault_otx_source import AlienVaultOTXSource

# Import visualization
from visualization.threat_flow_visualization import ThreatFlowVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('threat_intelligence')

# Page configuration
st.set_page_config(
    page_title="Cyber Threat Intelligence | NyxTrace",
    page_icon="🛡️",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .threat-header {
        color: #FF5757;
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 15px;
        border-bottom: 2px solid #FF5757;
        padding-bottom: 5px;
    }
    
    .threat-subheader {
        color: #FFA500;
        font-size: 20px;
        font-weight: bold;
        margin-top: 25px;
        margin-bottom: 10px;
    }
    
    .data-source-badge {
        background-color: #333;
        color: #FFF;
        padding: 4px 10px;
        border-radius: 15px;
        font-size: 14px;
        margin-right: 10px;
    }
    
    .alert-high {
        background-color: rgba(255, 87, 87, 0.2);
        border-left: 4px solid #FF5757;
        padding: 10px;
        margin-bottom: 10px;
    }
    
    .alert-medium {
        background-color: rgba(255, 165, 0, 0.2);
        border-left: 4px solid #FFA500;
        padding: 10px;
        margin-bottom: 10px;
    }
    
    .alert-low {
        background-color: rgba(255, 255, 0, 0.1);
        border-left: 4px solid #FFFF00;
        padding: 10px;
        margin-bottom: 10px;
    }
    
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 5px;
    }
    
    .status-active {
        background-color: #00FF00;
    }
    
    .status-warning {
        background-color: #FFA500;
    }
    
    .status-inactive {
        background-color: #FF0000;
    }
</style>
""", unsafe_allow_html=True)


def render_threat_intel_header():
    """Render the threat intelligence dashboard header"""
    
    # Header with status
    st.markdown('<div class="threat-header">CYBER THREAT INTELLIGENCE</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([2, 3, 1])
    
    with col1:
        st.markdown("""
        <div>
            <span class="data-source-badge">AlienVault OTX</span>
            <span class="data-source-badge">MISP</span>
            <span class="data-source-badge">MITRE ATT&CK</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center;">
            <span style="font-size: 16px; color: #CCC;">Global Threat Level: </span>
            <span style="font-size: 18px; color: #FFA500; font-weight: bold;">ELEVATED</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align: right;">
            <div><span class="status-indicator status-active"></span> OTX Feed</div>
            <div><span class="status-indicator status-warning"></span> TAXII Feed</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")


def initialize_session_state():
    """Initialize session state variables for the threat intelligence page"""
    
    # Initialize OTX source if not already done
    if 'otx_source' not in st.session_state:
        st.session_state.otx_source = AlienVaultOTXSource()
    
    # Initialize flow visualizer if not already done
    if 'flow_visualizer' not in st.session_state:
        st.session_state.flow_visualizer = ThreatFlowVisualizer()
    
    # Initialize other state variables
    if 'flow_data' not in st.session_state:
        st.session_state.flow_data = pd.DataFrame()
    
    if 'last_otx_query' not in st.session_state:
        st.session_state.last_otx_query = None
    
    if 'last_otx_update' not in st.session_state:
        st.session_state.last_otx_update = None


def show_api_key_input():
    """Show API key input form if not configured"""
    
    st.markdown('<div class="threat-subheader">API Configuration</div>', unsafe_allow_html=True)
    
    # Check if API key is in environment
    api_key = os.environ.get('ALIENVAULT_OTX_API_KEY')
    
    if not api_key:
        st.warning("AlienVault OTX API key not configured. Please enter your API key:")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            api_key = st.text_input("AlienVault OTX API Key", 
                                  type="password", 
                                  placeholder="Enter your API key here",
                                  help="Get your API key from https://otx.alienvault.com/api")
        
        with col2:
            if st.button("Save API Key", use_container_width=True):
                if api_key:
                    # In a real production application, this would be saved securely
                    # For this demo, we'll store in session state
                    st.session_state.otx_api_key = api_key
                    
                    # Configure the OTX source
                    st.session_state.otx_source.configure({
                        'api_key': api_key
                    })
                    
                    st.success("API key saved successfully!")
                    st.experimental_rerun()
                else:
                    st.error("Please enter a valid API key.")
        
        # Show example data option
        st.markdown("**Don't have an API key?**")
        if st.button("Use Sample Data"):
            # Load sample data
            st.session_state.flow_data = load_sample_flow_data()
            st.session_state.last_otx_update = datetime.now()
            st.experimental_rerun()
        
        return False
    else:
        # Configure the OTX source with environment variable
        st.session_state.otx_source.configure({
            'api_key': api_key
        })
        return True


def load_sample_flow_data():
    """Load sample flow data for demonstration"""
    try:
        # Try to import the more comprehensive sample data
        from data.sample_threat_intel import generate_sample_flow_data
        return generate_sample_flow_data()
    except (ImportError, ModuleNotFoundError):
        # Fallback to basic sample data
        data = {
            'source': [
                'APT28', 'APT29', 'APT41', 'Lazarus Group', 'APT33', 
                'APT10', 'Sandworm', 'APT28', 'Kimsuky', 'APT41'
            ],
            'target': [
                'United States', 'United States', 'United States', 'South Korea', 'United States',
                'Japan', 'Ukraine', 'Ukraine', 'South Korea', 'Taiwan'
            ],
            'source_lat': [
                55.7558, 55.7558, 39.9042, 39.0392, 35.6892,
                39.9042, 55.7558, 55.7558, 39.0392, 39.9042
            ],
            'source_lon': [
                37.6173, 37.6173, 116.4074, 125.7625, 51.3890,
                116.4074, 37.6173, 37.6173, 125.7625, 116.4074
            ],
            'target_lat': [
                38.8951, 38.8951, 38.8951, 37.5665, 38.8951,
                35.6762, 50.4501, 50.4501, 37.5665, 25.0330
            ],
            'target_lon': [
                -77.0364, -77.0364, -77.0364, 126.9780, -77.0364,
                139.6503, 30.5234, 30.5234, 126.9780, 121.5654
            ],
            'weight': [
                3, 2, 4, 2, 1,
                2, 3, 2, 1, 3
            ],
            'type': [
                'attack', 'reconnaissance', 'data_exfiltration', 'attack', 'reconnaissance',
                'lateral_movement', 'attack', 'attack', 'command_and_control', 'data_exfiltration'
            ],
            'title': [
                'Spear-phishing campaign targeting government agencies',
                'Reconnaissance of critical infrastructure',
                'Data exfiltration from defense contractors',
                'Destructive malware targeting financial sector',
                'Infrastructure scanning of energy companies',
                'Supply chain compromise affecting tech companies',
                'Power grid attack targeting energy infrastructure',
                'Election interference campaign',
                'Command and control infrastructure established',
                'Intellectual property theft campaign'
            ],
            'date': [
                datetime.now() - timedelta(days=2),
                datetime.now() - timedelta(days=5),
                datetime.now() - timedelta(days=3),
                datetime.now() - timedelta(days=7),
                datetime.now() - timedelta(days=10),
                datetime.now() - timedelta(days=12),
                datetime.now() - timedelta(days=4),
                datetime.now() - timedelta(days=8),
                datetime.now() - timedelta(days=15),
                datetime.now() - timedelta(days=1)
            ]
        }
        
        return pd.DataFrame(data)


def load_and_filter_data():
    """Load and filter threat intelligence data"""
    
    st.markdown('<div class="threat-subheader">Intelligence Filters</div>', unsafe_allow_html=True)
    
    # Create filter controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Location filter
        locations = [
            "All Locations",
            "United States",
            "Russia",
            "China",
            "North Korea",
            "Iran",
            "Ukraine",
            "Taiwan",
            "Japan",
            "South Korea",
            "Israel"
        ]
        selected_location = st.selectbox("Target Location", locations)
        
        # Convert "All Locations" to None for the API
        location_filter = None if selected_location == "All Locations" else selected_location
    
    with col2:
        # Date range filter
        today = datetime.now().date()
        start_date = st.date_input("Start Date", today - timedelta(days=30), key="threat_intel_start_date")
        start_datetime = datetime.combine(start_date, datetime.min.time())
    
    with col3:
        # End date filter
        end_date = st.date_input("End Date", today, key="threat_intel_end_date")
        end_datetime = datetime.combine(end_date, datetime.max.time())
    
    # Add fetch button
    col1, col2 = st.columns([1, 3])
    
    with col1:
        fetch_data = st.button("Fetch Intelligence Data", use_container_width=True, type="primary")
    
    # Check if we should load data
    query_params = {
        'location': location_filter,
        'start_date': start_datetime,
        'end_date': end_datetime
    }
    
    if fetch_data or (st.session_state.last_otx_query != query_params and st.session_state.otx_source.test_connection()):
        with st.spinner("Fetching threat intelligence data..."):
            try:
                # Get flow data
                flow_data = st.session_state.otx_source.get_source_target_flows(
                    location=location_filter,
                    start_date=start_datetime,
                    end_date=end_datetime
                )
                
                st.session_state.flow_data = flow_data
                st.session_state.last_otx_query = query_params
                st.session_state.last_otx_update = datetime.now()
                
                if flow_data.empty:
                    st.warning("No threat intelligence data found for the selected filters. Using sample data.")
                    st.session_state.flow_data = load_sample_flow_data()
                
            except Exception as e:
                st.error(f"Error fetching threat intelligence data: {str(e)}")
                if st.session_state.flow_data.empty:
                    st.session_state.flow_data = load_sample_flow_data()
    
    # Show last update time
    if st.session_state.last_otx_update:
        with col2:
            st.markdown(f"*Last updated: {st.session_state.last_otx_update.strftime('%Y-%m-%d %H:%M:%S')}*")
    
    # Return current data
    return st.session_state.flow_data


def render_threat_dashboard(flow_data):
    """Render the threat intelligence dashboard"""
    
    # Use the ThreatFlowVisualizer to render the dashboard
    visualizer = st.session_state.flow_visualizer
    visualizer.render_dashboard(flow_data)


def render_ioc_table():
    """Render a table of indicators of compromise"""
    
    st.markdown('<div class="threat-subheader">Indicators of Compromise (IOCs)</div>', unsafe_allow_html=True)
    
    # Create sample IOCs if needed
    if not hasattr(st.session_state, 'iocs') or st.session_state.iocs is None:
        try:
            # Try to import the more comprehensive sample data
            from data.sample_threat_intel import generate_sample_iocs
            st.session_state.iocs = generate_sample_iocs()
        except (ImportError, ModuleNotFoundError):
            # Fallback to basic sample data
            st.session_state.iocs = [
                {
                    "type": "IP", 
                    "value": "185.193.38.24", 
                    "description": "C2 server for APT28 campaign", 
                    "confidence": "High",
                    "first_seen": "2023-04-15"
                },
                {
                    "type": "Domain", 
                    "value": "securedataupdates.com", 
                    "description": "Phishing domain used by APT29", 
                    "confidence": "Medium",
                    "first_seen": "2023-05-02"
                },
                {
                    "type": "Hash", 
                    "value": "8f92fce1c18348a1c91bb5adcd7add7fb97a7e7a", 
                    "description": "Dropper malware SHA1 hash", 
                    "confidence": "High",
                    "first_seen": "2023-05-10"
                },
                {
                    "type": "URL", 
                    "value": "https://legitimate-looking-site.com/update.php", 
                    "description": "Malware distribution URL", 
                    "confidence": "Medium",
                    "first_seen": "2023-04-28"
                },
                {
                    "type": "Email", 
                    "value": "security-updates@microsoft-verify.com", 
                    "description": "Phishing sender address", 
                    "confidence": "High",
                    "first_seen": "2023-05-05"
                }
            ]
    
    # Create dataframe
    ioc_df = pd.DataFrame(st.session_state.iocs)
    
    # Apply styling based on confidence
    def style_confidence(val):
        if val == "High":
            return 'background-color: rgba(255,0,0,0.3)'
        elif val == "Medium":
            return 'background-color: rgba(255,165,0,0.3)'
        else:
            return 'background-color: rgba(255,255,0,0.2)'
    
    # Show table
    st.dataframe(
        ioc_df.style.map(style_confidence, subset=['confidence']),
        use_container_width=True
    )
    
    # Add controls for IOC actions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.button("Export IOCs", use_container_width=True)
    
    with col2:
        st.button("Share with ISAC", use_container_width=True)
    
    with col3:
        st.button("Deploy to Security Tools", use_container_width=True)


def render_threat_bulletins():
    """Render a list of threat bulletins"""
    
    st.markdown('<div class="threat-subheader">Threat Bulletins</div>', unsafe_allow_html=True)
    
    # Create sample bulletins if needed
    if not hasattr(st.session_state, 'bulletins') or st.session_state.bulletins is None:
        try:
            # Try to import the more comprehensive sample data
            from data.sample_threat_intel import generate_sample_bulletins
            st.session_state.bulletins = generate_sample_bulletins()
        except (ImportError, ModuleNotFoundError):
            # Fallback to basic sample data
            st.session_state.bulletins = [
                {
                    "level": "high",
                    "title": "APT28 Campaign Targeting Critical Infrastructure",
                    "date": "2023-05-07",
                    "summary": "New spear-phishing campaign targeting energy sector employees with malicious attachments that deploy custom malware."
                },
                {
                    "level": "medium",
                    "title": "Ransomware Activity Increasing in Financial Sector",
                    "date": "2023-05-05",
                    "summary": "Multiple financial institutions reporting increased reconnaissance activity consistent with pre-ransomware deployment tactics."
                },
                {
                    "level": "low",
                    "title": "New Vulnerability in Popular VPN Solution",
                    "date": "2023-05-03",
                    "summary": "Vulnerability allows for potential unauthorized access. No active exploitation observed in the wild yet."
                }
            ]
    
    # Render bulletins
    for bulletin in st.session_state.bulletins:
        with st.container():
            st.markdown(f"""
            <div class="alert-{bulletin['level']}">
                <strong>{bulletin['title']}</strong><br>
                <small>{bulletin['date']}</small><br>
                {bulletin['summary']}
            </div>
            """, unsafe_allow_html=True)


def main():
    """Main function to run the threat intelligence dashboard"""
    
    # Initialize session state
    initialize_session_state()
    
    # Render header
    render_threat_intel_header()
    
    # Check if API key is configured
    api_configured = show_api_key_input()
    
    if api_configured or 'flow_data' in st.session_state and not st.session_state.flow_data.empty:
        # Load and filter data
        flow_data = load_and_filter_data()
        
        # Show main threat dashboard
        render_threat_dashboard(flow_data)
        
        # Show tabs for additional information
        tab1, tab2 = st.tabs(["Indicators of Compromise", "Threat Bulletins"])
        
        with tab1:
            render_ioc_table()
        
        with tab2:
            render_threat_bulletins()


# For Streamlit pages, we need to call main() directly
main()