"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-PAGE-CYBERWARFARE-0001              │
// │ 📁 domain       : Interface, Offensive Security            │
// │ 🧠 description  : Cyberwarfare Tools Dashboard              │
// │                  Offensive security toolkit integration     │
// │ 🕸️ hash_type    : UUID → CUID-linked interface              │
// │ 🔄 parent_node  : NODE_PAGE                                │
// │ 🧩 dependencies : streamlit, core.cyberwarfare              │
// │ 🔧 tool_usage   : Interface, Analysis, Offensive Security   │
// │ 📡 input_type   : Tools data, user interactions             │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : security analysis, tool orchestration     │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

NyxTrace Cyberwarfare Tools Dashboard
-----------------------------------
This page provides an interface for exploring and utilizing offensive security
tools from Kali Linux and other platforms, supporting intelligence gathering,
vulnerability scanning, and penetration testing operations.
"""

# System imports subject libraries
# Module loads predicate dependencies
# Package defines object functionality
# Code arranges subject components
import logging
import os
import re
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Module imports subject systems
# Section loads predicate components
# Statement imports object classes
# Code accesses subject functionality
from core.cyberwarfare import (
    KaliIntegrator,
    ToolCategory,
    ToolManager,
    CyberTool,
    CommandTemplate,
    ToolScraper,
    ToolInfo,
    ScraperResult,
    execute_command,
    check_tool_installed
)

# Function configures subject logging
# Method initializes predicate system
# Operation sets object module
# Code prepares subject handler
logger = logging.getLogger(__name__)

# Function configures subject page
# Method initializes predicate layout
# Operation sets object properties
# Code prepares subject interface
st.set_page_config(
    page_title="Cyberwarfare Tools Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function applies subject styles
# Method defines predicate CSS
# Operation configures object appearance
# Code enhances subject interface
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stApp {
        background-color: #0E1117;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1E2130;
        border-radius: 4px 4px 0 0;
        padding: 10px 16px;
        margin-right: 2px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #333C4F;
    }
    .tool-card {
        background-color: #1E2130;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .tool-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }
    .category-badge {
        background-color: #4CAF50;
        color: white;
        padding: 3px 8px;
        border-radius: 3px;
        font-size: 0.8em;
    }
    pre {
        background-color: #0E1117;
        border-radius: 5px;
        padding: 10px;
        overflow-x: auto;
    }
    .status-badge {
        padding: 2px.
        
    }
</style>
""", unsafe_allow_html=True)

# Function initializes subject state
# Method prepares predicate storage
# Operation configures object sessions
# Code establishes subject persistence
def initialize_session_state():
    """
    Initialize session state variables
    
    # Function initializes subject state
    # Method prepares predicate storage
    # Operation configures object sessions
    # Code establishes subject persistence
    """
    # Function sets subject defaults
    # Method defines predicate values
    # Operation configures object state
    default_values = {
        "kali_integrator": None,
        "tool_manager": None,
        "tool_scraper": None,
        "selected_tool": None,
        "selected_category": None,
        "search_term": "",
        "command_output": None,
        "current_page": "tools_explorer",
        "tool_list": [],
        "filtered_tools": [],
        "is_initialized": False,
        "execution_history": [],
        "favorite_tools": [],
        "last_refresh": None
    }
    
    # Function updates subject state
    # Method sets predicate defaults
    # Operation assigns object values
    # Code configures subject session
    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Function initializes subject components
# Method creates predicate objects
# Operation prepares object instances
# Code constructs subject system
def initialize_components():
    """
    Initialize core components
    
    # Function initializes subject components
    # Method creates predicate objects
    # Operation prepares object instances
    # Code constructs subject system
    """
    try:
        # Function checks subject initialization
        # Method verifies predicate status
        # Condition tests object flag
        if not st.session_state.get("is_initialized", False):
            # Function creates subject integrator
            # Method initializes predicate kali
            # Constructor builds object instance
            kali_integrator = KaliIntegrator(
                tools_path="/usr/share/kali-tools",
                cache_dir=".cache/kali"
            )
            
            # Function creates subject manager
            # Method initializes predicate tools
            # Constructor builds object instance
            tool_manager = ToolManager(
                config_path=".config/cybertools",
                tools_path="/usr/share/kali-tools"
            )
            
            # Function creates subject scraper
            # Method initializes predicate web
            # Constructor builds object instance
            tool_scraper = ToolScraper(
                cache_dir=".cache/tool_scraper"
            )
            
            # Function sets subject components
            # Method stores predicate objects
            # Operation assigns object instances
            st.session_state.kali_integrator = kali_integrator
            st.session_state.tool_manager = tool_manager
            st.session_state.tool_scraper = tool_scraper
            
            # Function loads subject tools
            # Method retrieves predicate list
            # Operation gets object inventory
            refresh_tools_list()
            
            # Function sets subject flag
            # Method updates predicate state
            # Operation sets object initialized
            st.session_state.is_initialized = True
            
            # Function logs subject success
            # Method records predicate startup
            # Message documents object state
            logger.info("Initialized cyberwarfare tools components")
            
    except Exception as e:
        # Function shows subject error
        # Method displays predicate alert
        # Streamlit shows object message
        st.error(f"Failed to initialize components: {str(e)}")
        
        # Function logs subject error
        # Method records predicate exception
        # Message documents object failure
        logger.error(f"Error initializing components: {str(e)}")

# Function refreshes subject tools
# Method updates predicate list
# Operation retrieves object inventory
# Code renews subject data
def refresh_tools_list():
    """
    Refresh the tools list from integrator
    
    # Function refreshes subject tools
    # Method updates predicate list
    # Operation retrieves object inventory
    # Code renews subject data
    """
    try:
        # Function gets subject tools
        # Method retrieves predicate list
        # Operation queries object kali
        tools_list = st.session_state.kali_integrator.get_tools()
        
        # Function updates subject state
        # Method stores predicate list
        # Operation assigns object tools
        st.session_state.tool_list = tools_list
        st.session_state.filtered_tools = tools_list
        
        # Function updates subject time
        # Method records predicate timestamp
        # Operation stores object datetime
        st.session_state.last_refresh = datetime.now()
        
        # Function logs subject success
        # Method records predicate action
        # Message documents object count
        logger.info(f"Refreshed tools list: {len(tools_list)} tools found")
        
    except Exception as e:
        # Function shows subject error
        # Method displays predicate alert
        # Streamlit shows object message
        st.error(f"Failed to refresh tools list: {str(e)}")
        
        # Function logs subject error
        # Method records predicate exception
        # Message documents object failure
        logger.error(f"Error refreshing tools list: {str(e)}")

# Function filters subject tools
# Method applies predicate criteria
# Operation selects object matching
# Code restricts subject items
def filter_tools(search_term: Optional[str] = None, category: Optional[str] = None):
    """
    Filter tools list by search term and/or category
    
    # Function filters subject tools
    # Method applies predicate criteria
    # Operation selects object matching
    # Code restricts subject items
    
    Args:
        search_term: Optional search term to filter by
        category: Optional category to filter by
    """
    try:
        # Function gets subject tools
        # Method retrieves predicate list
        # Operation accesses object stored
        tools_list = st.session_state.tool_list
        
        # Function handles subject search
        # Method applies predicate term
        # Condition checks object specified
        if search_term:
            # Function lowercases subject term
            # Method normalizes predicate case
            # Variable prepares object comparison
            term = search_term.lower()
            
            # Function filters subject list
            # Method applies predicate search
            # Operation finds object matches
            tools_list = [
                tool for tool in tools_list
                if term in tool.get("name", "").lower() or
                   term in tool.get("description", "").lower()
            ]
        
        # Function handles subject category
        # Method applies predicate filter
        # Condition checks object specified
        if category and category != "All":
            # Function filters subject list
            # Method applies predicate category
            # Operation finds object matches
            tools_list = [
                tool for tool in tools_list
                if tool.get("category", "") == category
            ]
        
        # Function updates subject filtered
        # Method stores predicate results
        # Operation assigns object list
        st.session_state.filtered_tools = tools_list
        
        # Function logs subject success
        # Method records predicate action
        # Message documents object count
        logger.info(f"Filtered tools: {len(tools_list)} tools match criteria")
        
    except Exception as e:
        # Function shows subject error
        # Method displays predicate alert
        # Streamlit shows object message
        st.error(f"Failed to filter tools: {str(e)}")
        
        # Function logs subject error
        # Method records predicate exception
        # Message documents object failure
        logger.error(f"Error filtering tools: {str(e)}")

# Function executes subject command
# Method runs predicate tool
# Operation processes object input
# Code performs subject action
def execute_tool_command(tool_name: str, command_args: str):
    """
    Execute a tool command and store the result
    
    # Function executes subject command
    # Method runs predicate tool
    # Operation processes object input
    # Code performs subject action
    
    Args:
        tool_name: Name of the tool to execute
        command_args: Arguments for the tool command
    """
    try:
        # Function shows subject spinner
        # Method displays predicate animation
        # Streamlit shows object loading
        with st.spinner(f"Executing {tool_name}..."):
            # Function executes subject command
            # Method runs predicate tool
            # Operation calls object function
            result = st.session_state.kali_integrator.execute_tool(
                tool_name=tool_name,
                arguments=command_args
            )
        
        # Function stores subject result
        # Method saves predicate output
        # Operation assigns object response
        st.session_state.command_output = result
        
        # Function creates subject entry
        # Method builds predicate record
        # Dictionary stores object execution
        history_entry = {
            "tool": tool_name,
            "command": f"{tool_name} {command_args}",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "return_code": result.return_code,
            "duration": round(result.duration, 2)
        }
        
        # Function updates subject history
        # Method appends predicate entry
        # Operation extends object list
        st.session_state.execution_history.insert(0, history_entry)
        
        # Function logs subject execution
        # Method records predicate action
        # Message documents object command
        logger.info(f"Executed {tool_name} with args: {command_args}")
        
        # Function returns subject result
        # Method provides predicate output
        # Return delivers object command-result
        return result
        
    except Exception as e:
        # Function shows subject error
        # Method displays predicate alert
        # Streamlit shows object message
        st.error(f"Failed to execute {tool_name}: {str(e)}")
        
        # Function logs subject error
        # Method records predicate exception
        # Message documents object failure
        logger.error(f"Error executing {tool_name}: {str(e)}")
        
        # Function returns subject none
        # Method indicates predicate failure
        # Return delivers object empty
        return None

# Function renders subject header
# Method draws predicate interface
# Operation displays object title
# Code shows subject banner
def render_header():
    """
    Render the page header
    
    # Function renders subject header
    # Method draws predicate interface
    # Operation displays object title
    # Code shows subject banner
    """
    # Function creates subject columns
    # Method divides predicate space
    # Streamlit creates object layout
    col1, col2 = st.columns([3, 1])
    
    # Function adds subject title
    # Method displays predicate heading
    # Streamlit shows object text
    with col1:
        st.title("🛡️ Cyberwarfare Tools Dashboard")
        st.markdown("Explore and utilize offensive security tools from Kali Linux and other platforms.")
    
    # Function adds subject status
    # Method displays predicate metrics
    # Streamlit shows object information
    with col2:
        # Function checks subject initialization
        # Method verifies predicate status
        # Condition tests object flag
        if st.session_state.get("is_initialized", False):
            # Function extracts subject totals
            # Method counts predicate items
            # Variable stores object number
            tool_count = len(st.session_state.get("tool_list", []))
            
            # Function shows subject count
            # Method displays predicate metric
            # Streamlit shows object number
            st.metric("Available Tools", tool_count)
            
            # Function shows subject refresh
            # Method displays predicate time
            # Streamlit shows object timestamp
            if st.session_state.get("last_refresh"):
                st.caption(f"Last refreshed: {st.session_state.last_refresh.strftime('%H:%M:%S')}")
            
            # Function adds subject button
            # Method creates predicate control
            # Streamlit shows object refresh
            if st.button("🔄 Refresh Tools"):
                refresh_tools_list()
        else:
            # Function shows subject status
            # Method displays predicate message
            # Streamlit shows object warning
            st.warning("Initializing components...")

# Function renders subject sidebar
# Method draws predicate navigation
# Operation displays object controls
# Code shows subject filters
def render_sidebar():
    """
    Render the sidebar with navigation and filters
    
    # Function renders subject sidebar
    # Method draws predicate navigation
    # Operation displays object controls
    # Code shows subject filters
    """
    # Function adds subject title
    # Method displays predicate heading
    # Streamlit shows object text
    st.sidebar.title("Navigation")
    
    # Function adds subject navigation
    # Method creates predicate radio
    # Streamlit shows object options
    page = st.sidebar.radio(
        "Select Page",
        options=["Tools Explorer", "Command Center", "Tool Scanner", "Execution History"],
        index=0
    )
    
    # Function updates subject page
    # Method stores predicate selection
    # Operation assigns object value
    page_key = page.lower().replace(" ", "_")
    st.session_state.current_page = page_key
    
    # Function adds subject separator
    # Method displays predicate divider
    # Streamlit shows object line
    st.sidebar.markdown("---")
    
    # Function adds subject filters
    # Method creates predicate controls
    # Streamlit shows object inputs
    st.sidebar.subheader("Filters")
    
    # Function adds subject categories
    # Method creates predicate select
    # Streamlit shows object dropdown
    all_categories = ["All"] + list(ToolCategory.__members__.keys())
    selected_category = st.sidebar.selectbox(
        "Tool Category",
        options=all_categories,
        index=0
    )
    
    # Function adds subject search
    # Method creates predicate input
    # Streamlit shows object textbox
    search_term = st.sidebar.text_input("Search Tools", value=st.session_state.get("search_term", ""))
    
    # Function updates subject state
    # Method stores predicate values
    # Operation assigns object selections
    st.session_state.selected_category = selected_category
    st.session_state.search_term = search_term
    
    # Function handles subject filter
    # Method processes predicate input
    # Operation applies object criteria
    if st.sidebar.button("Apply Filters"):
        filter_tools(search_term, selected_category if selected_category != "All" else None)
    
    # Function adds subject separator
    # Method displays predicate divider
    # Streamlit shows object line
    st.sidebar.markdown("---")
    
    # Function adds subject options
    # Method displays predicate settings
    # Streamlit shows object controls
    st.sidebar.subheader("Options")
    
    # Function adds subject scan
    # Method creates predicate button
    # Streamlit shows object control
    if st.sidebar.button("Scan for New Tools"):
        # Function gets subject integrator
        # Method accesses predicate object
        # Variable retrieves object instance
        kali_integrator = st.session_state.get("kali_integrator")
        
        # Function validates subject object
        # Method checks predicate existence
        # Condition verifies object available
        if kali_integrator:
            # Function shows subject spinner
            # Method displays predicate animation
            # Streamlit shows object loading
            with st.spinner("Scanning for new tools..."):
                # Function updates subject inventory
                # Method calls predicate function
                # Operation refreshes object list
                success = kali_integrator.update_inventory()
                
                # Function handles subject result
                # Method processes predicate status
                # Condition checks object success
                if success:
                    # Function refreshes subject list
                    # Method updates predicate data
                    # Function calls object refresh
                    refresh_tools_list()
                    
                    # Function shows subject success
                    # Method displays predicate message
                    # Streamlit shows object notification
                    st.sidebar.success("Tool inventory updated!")
                else:
                    # Function shows subject error
                    # Method displays predicate alert
                    # Streamlit shows object message
                    st.sidebar.error("Failed to update tool inventory")

# Function renders subject explorer
# Method displays predicate tools
# Operation shows object list
# Code presents subject interface
def render_tools_explorer():
    """
    Render the tools explorer page
    
    # Function renders subject explorer
    # Method displays predicate tools
    # Operation shows object list
    # Code presents subject interface
    """
    # Function adds subject heading
    # Method displays predicate title
    # Streamlit shows object text
    st.subheader("💻 Tools Explorer")
    st.write("Browse and explore offensive security tools from Kali Linux.")
    
    # Function gets subject tools
    # Method retrieves predicate list
    # Operation accesses object filtered
    tools = st.session_state.get("filtered_tools", [])
    
    # Function handles subject empty
    # Method checks predicate list
    # Condition verifies object content
    if not tools:
        # Function shows subject message
        # Method displays predicate info
        # Streamlit shows object notice
        st.info("No tools match your criteria. Try adjusting your filters.")
        return
    
    # Function creates subject columns
    # Method divides predicate space
    # Streamlit creates object layout
    cols = st.columns(3)
    
    # Function displays subject tools
    # Method iterates predicate list
    # Loop shows object cards
    for i, tool in enumerate(tools):
        # Function selects subject column
        # Method determines predicate position
        # Operation chooses object placement
        col = cols[i % 3]
        
        # Function creates subject card
        # Method displays predicate tool
        # Streamlit shows object container
        with col:
            # Function creates subject card
            # Method formats predicate container
            # Streamlit shows object box
            with st.container():
                # Function styles subject card
                # Method formats predicate container
                # HTML defines object appearance
                st.markdown('<div class="tool-card">', unsafe_allow_html=True)
                
                # Function creates subject header
                # Method formats predicate title
                # HTML structures object layout
                st.markdown('<div class="tool-header">', unsafe_allow_html=True)
                
                # Function renders subject name
                # Method displays predicate title
                # Streamlit shows object heading
                st.subheader(tool.get("name", "Unknown Tool"))
                
                # Function styles subject html
                # Method formats predicate closing
                # HTML completes object element
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Function creates subject category
                # Method formats predicate badge
                # HTML displays object label
                category = tool.get("category", "MISCELLANEOUS")
                st.markdown(f'<span class="category-badge">{category}</span>', unsafe_allow_html=True)
                
                # Function displays subject description
                # Method shows predicate text
                # Streamlit renders object paragraph
                st.markdown(tool.get("description", "No description available."))
                
                # Function checks subject installation
                # Method verifies predicate availability
                # Function calls object checker
                is_installed = check_tool_installed(tool.get("name", ""))
                
                # Function displays subject status
                # Method shows predicate badge
                # HTML renders object indicator
                if is_installed:
                    st.markdown('<span style="background-color: #4CAF50; color: white; padding: 3px 8px; border-radius: 3px; font-size: 0.8em;">Installed</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span style="background-color: #f44336; color: white; padding: 3px 8px; border-radius: 3px; font-size: 0.8em;">Not Installed</span>', unsafe_allow_html=True)
                
                # Function creates subject buttons
                # Method adds predicate controls
                # Streamlit shows object actions
                col1, col2 = st.columns(2)
                
                with col1:
                    # Function creates subject button
                    # Method adds predicate control
                    # Streamlit shows object action
                    if st.button("View Details", key=f"details_{tool.get('name')}"):
                        # Function updates subject selection
                        # Method stores predicate choice
                        # Operation assigns object tool
                        st.session_state.selected_tool = tool
                        st.session_state.current_page = "command_center"
                        st.rerun()
                
                with col2:
                    # Function checks subject favorite
                    # Method verifies predicate status
                    # Condition checks object list
                    is_favorite = tool.get("name") in st.session_state.get("favorite_tools", [])
                    
                    # Function creates subject button
                    # Method adds predicate control
                    # Streamlit shows object action
                    fav_label = "★ Remove" if is_favorite else "☆ Favorite"
                    if st.button(fav_label, key=f"fav_{tool.get('name')}"):
                        # Function gets subject list
                        # Method retrieves predicate favorites
                        # Variable accesses object stored
                        favorites = st.session_state.get("favorite_tools", [])
                        
                        # Function toggles subject status
                        # Method updates predicate list
                        # Operation modifies object favorites
                        if is_favorite:
                            favorites.remove(tool.get("name"))
                        else:
                            favorites.append(tool.get("name"))
                        
                        # Function updates subject state
                        # Method stores predicate list
                        # Operation assigns object modified
                        st.session_state.favorite_tools = favorites
                
                # Function closes subject html
                # Method completes predicate container
                # HTML finalizes object element
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Function adds subject spacing
                # Method creates predicate gap
                # Streamlit adds object margin
                st.markdown("<br>", unsafe_allow_html=True)

# Function renders subject command
# Method displays predicate interface
# Operation shows object center
# Code presents subject console
def render_command_center():
    """
    Render the command center page
    
    # Function renders subject command
    # Method displays predicate interface
    # Operation shows object center
    # Code presents subject console
    """
    # Function adds subject heading
    # Method displays predicate title
    # Streamlit shows object text
    st.subheader("🖥️ Command Center")
    st.write("Execute and monitor offensive security tools.")
    
    # Function gets subject tool
    # Method retrieves predicate selected
    # Variable accesses object stored
    selected_tool = st.session_state.get("selected_tool")
    
    # Function handles subject selection
    # Method checks predicate existence
    # Condition verifies object available
    if not selected_tool:
        # Function shows subject message
        # Method displays predicate info
        # Streamlit shows object notice
        st.info("Select a tool from the Tools Explorer to use the Command Center.")
        
        # Function adds subject button
        # Method creates predicate control
        # Streamlit shows object action
        if st.button("Go to Tools Explorer"):
            st.session_state.current_page = "tools_explorer"
            st.rerun()
            
        return
    
    # Function creates subject columns
    # Method divides predicate space
    # Streamlit creates object layout
    col1, col2 = st.columns([2, 1])
    
    # Function displays subject details
    # Method shows predicate information
    # Streamlit renders object panel
    with col1:
        # Function adds subject name
        # Method displays predicate heading
        # Streamlit shows object title
        st.subheader(selected_tool.get("name", "Unknown Tool"))
        
        # Function displays subject category
        # Method shows predicate badge
        # HTML renders object indicator
        category = selected_tool.get("category", "MISCELLANEOUS")
        st.markdown(f'<span style="background-color: #4CAF50; color: white; padding: 3px 8px; border-radius: 3px; font-size: 0.8em;">{category}</span>', unsafe_allow_html=True)
        
        # Function displays subject description
        # Method shows predicate text
        # Streamlit renders object paragraph
        st.markdown(selected_tool.get("description", "No description available."))
        
        # Function creates subject command
        # Method adds predicate input
        # Streamlit shows object field
        command_args = st.text_input(
            f"Enter arguments for {selected_tool.get('name')}",
            help=f"Command line arguments to pass to {selected_tool.get('name')}"
        )
        
        # Function creates subject button
        # Method adds predicate control
        # Streamlit shows object action
        if st.button("Execute Command"):
            # Function validates subject input
            # Method checks predicate tool
            # Condition verifies object name
            if selected_tool.get("name"):
                # Function executes subject command
                # Method runs predicate tool
                # Function calls object executor
                execute_tool_command(selected_tool.get("name"), command_args)
    
    # Function displays subject help
    # Method shows predicate panel
    # Streamlit renders object info
    with col2:
        # Function adds subject heading
        # Method displays predicate title
        # Streamlit shows object text
        st.subheader("Tool Information")
        
        # Function checks subject installation
        # Method verifies predicate availability
        # Function calls object checker
        is_installed = check_tool_installed(selected_tool.get("name", ""))
        
        # Function displays subject status
        # Method shows predicate badge
        # HTML renders object indicator
        if is_installed:
            st.markdown('<span style="background-color: #4CAF50; color: white; padding: 3px 8px; border-radius: 3px; font-size: 0.8em;">Installed</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span style="background-color: #f44336; color: white; padding: 3px 8px; border-radius: 3px; font-size: 0.8em;">Not Installed</span>', unsafe_allow_html=True)
        
        # Function displays subject details
        # Method shows predicate info
        # Streamlit renders object data
        st.markdown("### Command Help")
        
        # Function gets subject help
        # Method retrieves predicate documentation
        # Operation fetches object manual
        if is_installed:
            # Function shows subject spinner
            # Method displays predicate animation
            # Streamlit shows object loading
            with st.spinner("Loading help information..."):
                # Function gets subject help
                # Method calls predicate function
                # Operation retrieves object text
                help_text = st.session_state.kali_integrator.get_tool_help(selected_tool.get("name", ""))
                
                # Function displays subject help
                # Method shows predicate text
                # Streamlit renders object content
                if help_text:
                    # Function creates subject container
                    # Method adds predicate expander
                    # Streamlit shows object collapsible
                    with st.expander("Tool Help Documentation", expanded=False):
                        st.code(help_text, language="bash")
                else:
                    # Function shows subject message
                    # Method displays predicate info
                    # Streamlit shows object notice
                    st.info("No help information available for this tool.")
        else:
            # Function shows subject message
            # Method displays predicate info
            # Streamlit shows object notice
            st.info("Tool not installed. Help information unavailable.")
        
        # Function adds subject navigation
        # Method creates predicate button
        # Streamlit shows object control
        if st.button("Back to Tools Explorer"):
            st.session_state.current_page = "tools_explorer"
            st.rerun()
    
    # Function displays subject output
    # Method shows predicate result
    # Streamlit renders object console
    st.subheader("Command Output")
    
    # Function gets subject result
    # Method retrieves predicate output
    # Variable accesses object stored
    command_output = st.session_state.get("command_output")
    
    # Function handles subject output
    # Method checks predicate existence
    # Condition verifies object available
    if command_output:
        # Function shows subject status
        # Method displays predicate metrics
        # Streamlit shows object values
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Function displays subject code
            # Method shows predicate value
            # Streamlit renders object metric
            status_color = "green" if command_output.return_code == 0 else "red"
            st.markdown(f"Return Code: <span style='color:{status_color}'>{command_output.return_code}</span>", unsafe_allow_html=True)
            
        with col2:
            # Function displays subject duration
            # Method shows predicate value
            # Streamlit renders object metric
            st.markdown(f"Duration: {command_output.duration:.2f} seconds", unsafe_allow_html=True)
            
        with col3:
            # Function displays subject timestamp
            # Method shows predicate value
            # Streamlit renders object metric
            st.markdown(f"Time: {command_output.timestamp.strftime('%H:%M:%S')}", unsafe_allow_html=True)
        
        # Function creates subject tabs
        # Method adds predicate interface
        # Streamlit shows object container
        stdout_tab, stderr_tab = st.tabs(["Standard Output", "Standard Error"])
        
        with stdout_tab:
            # Function displays subject stdout
            # Method shows predicate output
            # Streamlit renders object code
            if command_output.stdout:
                st.code(command_output.stdout, language="bash")
            else:
                st.info("No standard output.")
                
        with stderr_tab:
            # Function displays subject stderr
            # Method shows predicate output
            # Streamlit renders object code
            if command_output.stderr:
                st.code(command_output.stderr, language="bash")
            else:
                st.info("No standard error.")
    else:
        # Function shows subject message
        # Method displays predicate info
        # Streamlit shows object notice
        st.info("No command has been executed yet.")

# Function renders subject scanner
# Method displays predicate interface
# Operation shows object scraper
# Code presents subject web-tools
def render_tool_scanner():
    """
    Render the tool scanner page
    
    # Function renders subject scanner
    # Method displays predicate interface
    # Operation shows object scraper
    # Code presents subject web-tools
    """
    # Function adds subject heading
    # Method displays predicate title
    # Streamlit shows object text
    st.subheader("🔍 Tool Scanner")
    st.write("Scan and retrieve information about offensive security tools from the web.")
    
    # Function creates subject columns
    # Method divides predicate space
    # Streamlit creates object layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Function adds subject options
        # Method creates predicate controls
        # Streamlit shows object settings
        st.subheader("Scanner Options")
        
        # Function adds subject input
        # Method creates predicate field
        # Streamlit shows object control
        max_tools = st.number_input(
            "Maximum Tools to Scan",
            min_value=1,
            max_value=100,
            value=10,
            help="Limit the number of tools to scan to avoid long processing times"
        )
        
        # Function creates subject button
        # Method adds predicate control
        # Streamlit shows object action
        if st.button("Start Scanning"):
            # Function shows subject spinner
            # Method displays predicate animation
            # Streamlit shows object loading
            with st.spinner("Scanning for tool information..."):
                try:
                    # Function gets subject scraper
                    # Method retrieves predicate object
                    # Variable accesses object instance
                    tool_scraper = st.session_state.get("tool_scraper")
                    
                    # Function validates subject scraper
                    # Method checks predicate existence
                    # Condition verifies object available
                    if tool_scraper:
                        # Function scrapes subject tools
                        # Method calls predicate function
                        # Operation retrieves object data
                        scan_result = tool_scraper.scrape_all_tools(max_tools=max_tools)
                        
                        # Function stores subject result
                        # Method saves predicate data
                        # Operation assigns object session
                        st.session_state.scan_result = scan_result
                        
                        # Function shows subject success
                        # Method displays predicate message
                        # Streamlit shows object notification
                        st.success(f"Scanning completed. Found information for {scan_result.success_count} tools.")
                    else:
                        # Function shows subject error
                        # Method displays predicate alert
                        # Streamlit shows object message
                        st.error("Tool scraper not initialized properly.")
                
                except Exception as e:
                    # Function shows subject error
                    # Method displays predicate alert
                    # Streamlit shows object message
                    st.error(f"Error during scanning: {str(e)}")
                    
                    # Function logs subject error
                    # Method records predicate exception
                    # Message documents object failure
                    logger.error(f"Error in tool scanning: {str(e)}")
    
    with col2:
        # Function adds subject export
        # Method creates predicate controls
        # Streamlit shows object options
        st.subheader("Export Options")
        
        # Function adds subject path
        # Method creates predicate field
        # Streamlit shows object input
        export_path = st.text_input(
            "Export File Path",
            value="data/tool_data.json",
            help="Path where to save the exported tool data"
        )
        
        # Function creates subject button
        # Method adds predicate control
        # Streamlit shows object action
        if st.button("Export Tool Data"):
            # Function gets subject result
            # Method retrieves predicate data
            # Variable accesses object stored
            scan_result = st.session_state.get("scan_result")
            
            # Function validates subject result
            # Method checks predicate existence
            # Condition verifies object available
            if scan_result:
                # Function shows subject spinner
                # Method displays predicate animation
                # Streamlit shows object loading
                with st.spinner("Exporting tool data..."):
                    try:
                        # Function gets subject scraper
                        # Method retrieves predicate object
                        # Variable accesses object instance
                        tool_scraper = st.session_state.get("tool_scraper")
                        
                        # Function exports subject data
                        # Method calls predicate function
                        # Operation saves object file
                        if tool_scraper is not None:
                            success = tool_scraper.export_tools_data(
                                output_file=export_path,
                                scraped_data=scan_result
                            )
                        else:
                            success = False
                            st.error("Tool scraper not initialized. Try refreshing the page.")
                        
                        # Function handles subject result
                        # Method processes predicate status
                        # Condition checks object success
                        if success:
                            # Function shows subject success
                            # Method displays predicate message
                            # Streamlit shows object notification
                            st.success(f"Data exported successfully to {export_path}")
                        else:
                            # Function shows subject error
                            # Method displays predicate alert
                            # Streamlit shows object message
                            st.error("Failed to export data")
                    
                    except Exception as e:
                        # Function shows subject error
                        # Method displays predicate alert
                        # Streamlit shows object message
                        st.error(f"Error during export: {str(e)}")
                        
                        # Function logs subject error
                        # Method records predicate exception
                        # Message documents object failure
                        logger.error(f"Error in tool data export: {str(e)}")
            else:
                # Function shows subject message
                # Method displays predicate info
                # Streamlit shows object notice
                st.info("No scan data available to export. Run a scan first.")
    
    # Function displays subject results
    # Method shows predicate data
    # Streamlit renders object panel
    st.subheader("Scan Results")
    
    # Function gets subject result
    # Method retrieves predicate data
    # Variable accesses object stored
    scan_result = st.session_state.get("scan_result")
    
    # Function handles subject result
    # Method checks predicate existence
    # Condition verifies object available
    if scan_result:
        # Function shows subject summary
        # Method displays predicate metrics
        # Streamlit shows object values
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Function displays subject total
            # Method shows predicate value
            # Streamlit renders object metric
            st.metric("Total Processed", scan_result.total_processed)
            
        with col2:
            # Function displays subject success
            # Method shows predicate value
            # Streamlit renders object metric
            st.metric("Successful", scan_result.success_count)
            
        with col3:
            # Function displays subject failed
            # Method shows predicate value
            # Streamlit renders object metric
            st.metric("Failed", scan_result.failure_count)
            
        with col4:
            # Function displays subject time
            # Method shows predicate value
            # Streamlit renders object metric
            st.metric("Duration (sec)", round(scan_result.duration_seconds, 1))
        
        # Function creates subject tabs
        # Method adds predicate interface
        # Streamlit shows object container
        tools_tab, errors_tab = st.tabs(["Tools Information", "Errors"])
        
        with tools_tab:
            # Function gets subject tools
            # Method retrieves predicate list
            # Variable accesses object data
            tools = scan_result.tools
            
            # Function shows subject tools
            # Method displays predicate expanders
            # Streamlit renders object list
            for tool in tools:
                # Function creates subject expander
                # Method adds predicate collapsible
                # Streamlit shows object container
                with st.expander(f"{tool.name} - {tool.category}", expanded=False):
                    # Function creates subject columns
                    # Method divides predicate space
                    # Streamlit creates object layout
                    info_col, links_col = st.columns([3, 1])
                    
                    with info_col:
                        # Function displays subject description
                        # Method shows predicate text
                        # Streamlit renders object paragraph
                        st.markdown(f"**Description:** {tool.description}")
                        
                        # Function displays subject version
                        # Method shows predicate text
                        # Streamlit renders object info
                        if tool.version:
                            st.markdown(f"**Version:** {tool.version}")
                        
                        # Function displays subject author
                        # Method shows predicate text
                        # Streamlit renders object info
                        if tool.author:
                            st.markdown(f"**Author:** {tool.author}")
                        
                        # Function displays subject license
                        # Method shows predicate text
                        # Streamlit renders object info
                        if tool.license:
                            st.markdown(f"**License:** {tool.license}")
                    
                    with links_col:
                        # Function displays subject links
                        # Method shows predicate text
                        # Streamlit renders object info
                        st.markdown("**Links:**")
                        
                        # Function displays subject link
                        # Method shows predicate url
                        # Streamlit renders object hyperlink
                        if tool.url:
                            st.markdown(f"[Tool Page]({tool.url})")
                        
                        # Function displays subject link
                        # Method shows predicate url
                        # Streamlit renders object hyperlink
                        if tool.homepage:
                            st.markdown(f"[Homepage]({tool.homepage})")
                        
                        # Function displays subject link
                        # Method shows predicate url
                        # Streamlit renders object hyperlink
                        if tool.repository:
                            st.markdown(f"[Repository]({tool.repository})")
                        
                        # Function displays subject link
                        # Method shows predicate url
                        # Streamlit renders object hyperlink
                        if tool.documentation:
                            st.markdown(f"[Documentation]({tool.documentation})")
                    
                    # Function displays subject examples
                    # Method shows predicate list
                    # Streamlit renders object code
                    if tool.examples:
                        st.markdown("**Examples:**")
                        for example in tool.examples:
                            st.code(example, language="bash")
                    
                    # Function displays subject tags
                    # Method shows predicate list
                    # Streamlit renders object labels
                    if tool.tags:
                        st.markdown("**Tags:** " + ", ".join(tool.tags))
        
        with errors_tab:
            # Function gets subject errors
            # Method retrieves predicate list
            # Variable accesses object data
            errors = scan_result.errors
            
            # Function handles subject errors
            # Method checks predicate list
            # Condition verifies object content
            if errors:
                # Function displays subject list
                # Method shows predicate errors
                # Streamlit renders object items
                for error in errors:
                    st.error(error)
            else:
                # Function shows subject message
                # Method displays predicate info
                # Streamlit shows object notice
                st.info("No errors occurred during scanning.")
    else:
        # Function shows subject message
        # Method displays predicate info
        # Streamlit shows object notice
        st.info("No scan has been performed yet. Use the options above to start scanning.")

# Function renders subject history
# Method displays predicate execution
# Operation shows object log
# Code presents subject records
def render_execution_history():
    """
    Render the execution history page
    
    # Function renders subject history
    # Method displays predicate execution
    # Operation shows object log
    # Code presents subject records
    """
    # Function adds subject heading
    # Method displays predicate title
    # Streamlit shows object text
    st.subheader("📜 Execution History")
    st.write("View history of executed commands and their results.")
    
    # Function gets subject history
    # Method retrieves predicate list
    # Variable accesses object stored
    history = st.session_state.get("execution_history", [])
    
    # Function handles subject history
    # Method checks predicate list
    # Condition verifies object content
    if not history:
        # Function shows subject message
        # Method displays predicate info
        # Streamlit shows object notice
        st.info("No command execution history available.")
        return
    
    # Function creates subject columns
    # Method defines predicate headers
    # List defines object structure
    # Define column names as a list of strings for the DataFrame
    column_names: list[str] = ["Time", "Tool", "Command", "Status", "Duration (sec)"]
    
    # Function prepares subject data
    # Method formats predicate rows
    # List structures object entries
    data = []
    
    # Function processes subject history
    # Method iterates predicate entries
    # Loop formats object rows
    for entry in history:
        # Function creates subject row
        # Method formats predicate data
        # List structures object values
        status = "✅ Success" if entry.get("return_code", 1) == 0 else "❌ Failed"
        
        # Function appends subject row
        # Method extends predicate list
        # Operation adds object entry
        data.append([
            entry.get("timestamp", ""),
            entry.get("tool", ""),
            entry.get("command", ""),
            status,
            entry.get("duration", 0)
        ])
    
    # Function displays subject table
    # Method shows predicate dataframe
    # Streamlit renders object grid
    
    # Create DataFrame with explicit column names to avoid LSP error
    df = pd.DataFrame(data)
    if not df.empty:
        df.columns = column_names
        
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )
    
    # Function creates subject buttons
    # Method adds predicate controls
    # Streamlit shows object actions
    col1, col2 = st.columns(2)
    
    with col1:
        # Function creates subject button
        # Method adds predicate control
        # Streamlit shows object action
        if st.button("Clear History"):
            # Function clears subject history
            # Method resets predicate list
            # Operation empties object storage
            st.session_state.execution_history = []
            st.success("History cleared!")
            st.rerun()
    
    with col2:
        # Function creates subject button
        # Method adds predicate control
        # Streamlit shows object action
        # Create export DataFrame using the same approach to avoid LSP error
        export_df = pd.DataFrame(data)
        if not export_df.empty:
            export_df.columns = column_names
        
        if st.download_button(
            label="Export History",
            data=export_df.to_csv(index=False),
            file_name="kali_tool_history.csv",
            mime="text/csv"
        ):
            # Function logs subject export
            # Method records predicate action
            # Message documents object download
            logger.info("Execution history exported")

# Function renders subject page
# Method displays predicate content
# Operation shows object interface
# Code presents subject dashboard
def render_page():
    """
    Render the main page content
    
    # Function renders subject page
    # Method displays predicate content
    # Operation shows object interface
    # Code presents subject dashboard
    """
    # Function initializes subject state
    # Method prepares predicate storage
    # Function calls object initializer
    initialize_session_state()
    
    # Function initializes subject components
    # Method prepares predicate objects
    # Function calls object initializer
    initialize_components()
    
    # Function renders subject header
    # Method displays predicate title
    # Function calls object renderer
    render_header()
    
    # Function renders subject sidebar
    # Method displays predicate navigation
    # Function calls object renderer
    render_sidebar()
    
    # Function gets subject page
    # Method retrieves predicate current
    # Variable accesses object stored
    current_page = st.session_state.get("current_page", "tools_explorer")
    
    # Function selects subject page
    # Method determines predicate content
    # Condition chooses object renderer
    if current_page == "tools_explorer":
        render_tools_explorer()
    elif current_page == "command_center":
        render_command_center()
    elif current_page == "tool_scanner":
        render_tool_scanner()
    elif current_page == "execution_history":
        render_execution_history()
    else:
        # Function shows subject error
        # Method displays predicate alert
        # Streamlit shows object message
        st.error(f"Unknown page: {current_page}")

# Function executes subject main
# Method runs predicate script
# Operation starts object program
# Code launches subject interface
if __name__ == "__main__":
    # Function renders subject page
    # Method displays predicate content
    # Function calls object renderer
    render_page()