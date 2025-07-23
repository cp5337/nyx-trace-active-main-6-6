"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-PAGES-KALI-EXPLORER-0001            │
// │ 📁 domain       : UI, Security, Intelligence                │
// │ 🧠 description  : Kali Linux tools explorer interface       │
// │                  for cybersecurity research                 │
// │ 🕸️ hash_type    : UUID → CUID-linked interface              │
// │ 🔄 parent_node  : NODE_INTERFACE                           │
// │ 🧩 dependencies : streamlit, pandas, kali_tools_collector   │
// │ 🔧 tool_usage   : Exploration, Analysis, Intelligence       │
// │ 📡 input_type   : Kali tools data                          │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : tool analysis, intelligence gathering     │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

Kali Tools Explorer Interface
-----------------------------
This module provides a Streamlit interface for exploring and searching
Kali Linux security tools. It allows users to browse tools by category,
search for specific tools, and view detailed information about each tool.
"""

import streamlit as st
import pandas as pd
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Any, Tuple

# Add parent directory to path to allow importing from collectors
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the Kali Tools Collector
from collectors.kali_tools_collector import KaliToolsCollector

# Initialize logger
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Kali Tools Explorer - NyxTrace",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function for CTAS-compliant UI headers
def create_ctas_header(title: str, subtitle: str, domain: str, hash_id: str):
    """
    Create a CTAS-compliant header for UI sections
    
    # Function creates subject header
    # Method displays predicate title
    # Interface shows object information
    # Component creates subject structure
    
    Args:
        title: Main title for the header
        subtitle: Subtitle or description
        domain: Domain classification
        hash_id: Unique identifier
    """
    st.markdown(f"""
    <div style="background-color:#1E1E1E; padding:10px; border-left: 5px solid #007ACC;">
        <h2 style="color:#FFFFFF;">{title}</h2>
        <p style="color:#CCCCCC;">{subtitle}</p>
        <div style="display:flex; gap:10px;">
            <span style="background-color:#333333; padding:3px 8px; border-radius:3px; font-size:0.8em; color:#AAAAAA;">Domain: {domain}</span>
            <span style="background-color:#333333; padding:3px 8px; border-radius:3px; font-size:0.8em; color:#AAAAAA;">ID: {hash_id}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Function to display tool details
def display_tool_details(tool_data: Dict[str, Any]):
    """
    Display detailed information about a Kali tool
    
    # Function displays subject details
    # Method shows predicate information
    # Interface presents object data
    # Component renders subject content
    
    Args:
        tool_data: Dictionary containing tool information
    """
    st.subheader(tool_data.get("Tool Name", "Unknown Tool"))
    
    # Tool URL
    st.markdown(f"**URL:** [{tool_data.get('Tool URL', 'N/A')}]({tool_data.get('Tool URL', '#')})")
    
    # Description
    st.markdown("### Description")
    st.markdown(tool_data.get("Description", "No description available."))
    
    # Package information
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Package Information")
        st.markdown(f"**Package Name:** {tool_data.get('Package Name', 'N/A')}")
        st.markdown(f"**Version:** {tool_data.get('Version', 'N/A')}")
    
    # Categories
    with col2:
        st.markdown("### Categories")
        categories = tool_data.get("Categories", [])
        if categories and len(categories) > 0:
            for cat in categories:
                st.markdown(f"- {cat}")
        else:
            st.markdown("No categories available.")
    
    # Commands
    st.markdown("### Commands")
    commands = tool_data.get("Commands", [])
    if commands and len(commands) > 0:
        for cmd in commands:
            st.code(cmd, language="bash")
    else:
        st.markdown("No commands available.")

# Initialize collector
@st.cache_resource
def get_collector():
    """
    Initialize and return the KaliToolsCollector instance
    
    # Function initializes subject collector
    # Method creates predicate instance
    # Operation returns object resource
    # Cache improves subject performance
    
    Returns:
        KaliToolsCollector instance
    """
    return KaliToolsCollector()

# Load basic tool data
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_basic_tools_data(_collector: KaliToolsCollector, force_refresh: bool = False):
    """
    Load basic information about Kali tools
    
    # Function loads subject data
    # Method retrieves predicate information
    # Operation fetches object dataset
    # Cache improves subject performance
    
    Args:
        collector: KaliToolsCollector instance
        force_refresh: Whether to force refresh the cache
        
    Returns:
        DataFrame with basic tool information
    """
    return _collector.collect_tools(force_refresh=force_refresh)

# Load detailed tool data if available
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_detailed_tools_data(_collector: KaliToolsCollector):
    """
    Load detailed information about Kali tools if available
    
    # Function loads subject details
    # Method retrieves predicate information
    # Operation fetches object dataset
    # Cache improves subject performance
    
    Args:
        collector: KaliToolsCollector instance
        
    Returns:
        DataFrame with detailed tool information or None if not available
    """
    detailed_cache_file = os.path.join(_collector.cache_dir, "kali_tools_detailed.csv")
    if os.path.exists(detailed_cache_file):
        return pd.read_csv(detailed_cache_file)
    return None

# Create category distribution visualization
def create_category_distribution(categories_dict: Dict[str, List[str]]):
    """
    Create a visualization of tool distribution by category
    
    # Function creates subject visualization
    # Method generates predicate chart
    # Operation displays object distribution
    # Component enhances subject analysis
    
    Args:
        categories_dict: Dictionary mapping categories to lists of tool names
        
    Returns:
        Plotly figure object
    """
    if not categories_dict:
        return go.Figure()
    
    # Count number of tools per category
    categories = list(categories_dict.keys())
    tool_counts = [len(tools) for tools in categories_dict.values()]
    
    # Sort by count
    sorted_data = sorted(zip(categories, tool_counts), key=lambda x: x[1], reverse=True)
    sorted_categories, sorted_counts = zip(*sorted_data) if sorted_data else ([], [])
    
    # Create bar chart
    fig = px.bar(
        x=sorted_categories[:15],  # Show top 15 categories
        y=sorted_counts[:15],
        labels={"x": "Category", "y": "Number of Tools"},
        title="Top Tool Categories",
        color=sorted_counts[:15],
        color_continuous_scale="Viridis",
    )
    
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=50, b=20),
        coloraxis_showscale=False,
    )
    
    return fig

# Main app
def main():
    """
    Main function for the Kali Tools Explorer
    
    # Function runs subject application
    # Method executes predicate logic
    # Operation controls object flow
    # Component drives subject interface
    """
    # Create CTAS-compliant header
    create_ctas_header(
        "Kali Linux Tools Explorer",
        "Explore and analyze cybersecurity tools for intelligence operations",
        "Security, Intelligence, Research",
        "USIM-KALI-EXPLORER-0001"
    )
    
    # Initialize collector
    collector = get_collector()
    
    # Load data
    with st.spinner("Loading tools data..."):
        basic_tools_df = load_basic_tools_data(collector)
        detailed_tools_df = load_detailed_tools_data(collector)
    
    # Display tool count
    st.markdown(f"**Total Tools Indexed:** {len(basic_tools_df)}")
    
    # Sidebar controls
    st.sidebar.header("Controls")
    
    # Search functionality
    st.sidebar.subheader("Search Tools")
    search_query = st.sidebar.text_input("Search by keyword", key="search_input")
    
    # Data refresh button
    if st.sidebar.button("Refresh Tools Data"):
        st.sidebar.info("Refreshing data from source...")
        # Use st.cache_data.clear() in newer Streamlit versions
        # For compatibility:
        st.experimental_memo.clear()
        basic_tools_df = load_basic_tools_data(collector, force_refresh=True)
        st.sidebar.success("Data refreshed!")
        st.experimental_rerun()
    
    # Check if detailed data exists
    has_detailed_data = detailed_tools_df is not None

    # Fetch detailed data for a sample
    if not has_detailed_data:
        if st.sidebar.button("Fetch Detailed Data (Sample)"):
            with st.spinner("Fetching detailed information for a sample of tools..."):
                collector.collect_all_tool_details(max_tools=20)  # Sample 20 tools
                st.experimental_rerun()
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["Tool Explorer", "Categories", "Analytics"])
    
    with tab1:
        # Apply search filter if provided
        if search_query:
            filtered_df = basic_tools_df[basic_tools_df["Tool Name"].str.contains(search_query, case=False)]
            st.subheader(f"Search Results for '{search_query}'")
            st.markdown(f"Found {len(filtered_df)} tools matching your search.")
        else:
            filtered_df = basic_tools_df
        
        # Display tool list
        st.subheader("Available Tools")
        
        # Create tool selection
        if not filtered_df.empty:
            tool_selection = st.selectbox(
                "Select a tool to view details",
                options=filtered_df["Tool Name"].tolist(),
            )
            
            # Get selected tool data
            selected_tool_row = filtered_df[filtered_df["Tool Name"] == tool_selection].iloc[0]
            
            # Check if detailed data is available for this tool
            if has_detailed_data:
                # Try to get detailed information
                detailed_row = detailed_tools_df[detailed_tools_df["Tool Name"] == tool_selection]
                if not detailed_row.empty:
                    tool_data = detailed_row.iloc[0].to_dict()
                else:
                    # Use basic information and fetch details on demand
                    tool_data = selected_tool_row.to_dict()
                    st.warning("Detailed information not available for this tool.")
                    
                    if st.button("Fetch Details for This Tool"):
                        with st.spinner("Fetching detailed information..."):
                            details = collector.get_tool_details(selected_tool_row["URL"])
                            details["Tool Name"] = selected_tool_row["Tool Name"]
                            display_tool_details(details)
            else:
                # Only basic information available
                tool_data = selected_tool_row.to_dict()
                if st.button("Fetch Details for This Tool"):
                    with st.spinner("Fetching detailed information..."):
                        details = collector.get_tool_details(selected_tool_row["URL"])
                        details["Tool Name"] = selected_tool_row["Tool Name"]
                        display_tool_details(details)
            
            # Display tool information
            st.markdown("---")
            display_tool_details(tool_data)
        else:
            st.info("No tools found matching your search criteria.")
    
    with tab2:
        # Categories exploration
        st.subheader("Tool Categories")
        
        if has_detailed_data:
            categories_dict = collector.categorize_tools()
            
            if categories_dict:
                # Display category count
                st.markdown(f"**Total Categories:** {len(categories_dict)}")
                
                # Create category visualization
                category_chart = create_category_distribution(categories_dict)
                st.plotly_chart(category_chart, use_container_width=True)
                
                # Category selection
                selected_category = st.selectbox(
                    "Select a category to view tools",
                    options=sorted(categories_dict.keys()),
                )
                
                # Display tools in the selected category
                st.subheader(f"Tools in '{selected_category}' Category")
                
                tools_in_category = categories_dict.get(selected_category, [])
                if tools_in_category:
                    for tool in tools_in_category:
                        st.markdown(f"- **{tool}**")
                else:
                    st.info("No tools found in this category.")
            else:
                st.info("Category information not available. Please fetch detailed data first.")
        else:
            st.info("Category information not available. Please fetch detailed data first.")
    
    with tab3:
        # Analytics
        st.subheader("Tool Analytics")
        
        if has_detailed_data:
            # Calculate stats
            total_tools = len(detailed_tools_df)
            
            # Tool descriptions statistics
            has_description = detailed_tools_df["Description"].apply(lambda x: len(str(x)) > 10).sum()
            description_percentage = (has_description / total_tools) * 100 if total_tools > 0 else 0
            
            # Tool commands statistics
            has_commands = detailed_tools_df["Commands"].apply(lambda x: len(str(x)) > 5).sum()
            commands_percentage = (has_commands / total_tools) * 100 if total_tools > 0 else 0
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Tools", total_tools)
            
            with col2:
                st.metric("Tools with Descriptions", f"{has_description} ({description_percentage:.1f}%)")
            
            with col3:
                st.metric("Tools with Commands", f"{has_commands} ({commands_percentage:.1f}%)")
            
            # Potentially create other analytics visualizations based on available data
        else:
            st.info("Analytics not available. Please fetch detailed data first.")

if __name__ == "__main__":
    main()