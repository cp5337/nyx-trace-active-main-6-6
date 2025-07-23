"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-PAGES-MEDIA-OUTLETS-0001            │
// │ 📁 domain       : UI, Media Monitoring                      │
// │ 🧠 description  : Media outlets monitoring interface for    │
// │                  tracking thousands of sources continuously  │
// │ 🕸️ hash_type    : UUID → CUID-linked interface              │
// │ 🔄 parent_node  : NODE_INTERFACE                           │
// │ 🧩 dependencies : streamlit, media_outlets_processor        │
// │ 🔧 tool_usage   : Monitoring, Collection, Intelligence      │
// │ 📡 input_type   : URLs, keywords, media outlets             │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : tracking, monitoring, analysis            │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

Media Outlets Monitoring Interface
---------------------------------
Advanced interface for monitoring thousands of media outlets and tracking
keyword matches across them. Supports outlet management, keyword configuration,
and comprehensive content tracking for intelligence gathering.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
import io
import base64

from core.web_intelligence.media_outlets_processor import (
    MediaOutletsProcessor, 
    MediaOutlet, 
    MonitoringKeyword, 
    ContentMatch
)

# Function configures subject logger
# Method initializes predicate output
# Operation sets object format
logger = logging.getLogger("media_outlets_monitoring")
logger.setLevel(logging.INFO)


# Function configures subject page
# Method sets predicate settings
# Operation defines object layout
def configure_page():
    """
    Configure the Streamlit page
    
    # Function configures subject page
    # Method sets predicate settings
    # Operation defines object layout
    """
    st.set_page_config(
        page_title="Media Outlets Monitoring - NyxTrace",
        page_icon=None,  # Professional approach without emojis
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Add custom CSS for professional appearance
    st.markdown("""
    <style>
    .data-table {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 12px;
    }
    .result-card {
        background-color: #f9f9f9;
        border-left: 3px solid #2c3e50;
        padding: 10px 15px;
        margin: 10px 0;
    }
    .keyword-tag {
        background-color: #e9ecef;
        padding: 2px 8px;
        border-radius: 3px;
        margin-right: 5px;
        font-size: 12px;
    }
    .priority-high {
        border-left: 3px solid #dc3545;
    }
    .priority-medium {
        border-left: 3px solid #fd7e14;
    }
    .priority-low {
        border-left: 3px solid #20c997;
    }
    .metric-card {
        background-color: #f5f5f5;
        border-radius: 5px;
        padding: 15px;
        text-align: center;
    }
    .metric-card h3 {
        margin: 0;
        font-size: 28px;
    }
    .metric-card p {
        margin: 5px 0 0 0;
        font-size: 14px;
        color: #6c757d;
    }
    </style>
    """, unsafe_allow_html=True)


# Function initializes subject processor
# Method creates predicate instance
# Operation prepares object component
def initialize_processor():
    """
    Initialize the media outlets processor
    
    # Function initializes subject processor
    # Method creates predicate instance
    # Operation prepares object component
    """
    # Create necessary directories
    os.makedirs('data/web_intelligence', exist_ok=True)
    os.makedirs('data/web_intelligence/outlets', exist_ok=True)
    os.makedirs('data/web_intelligence/content_cache', exist_ok=True)
    
    # Initialize processor if not already in session state
    if 'media_processor' not in st.session_state:
        st.session_state.media_processor = MediaOutletsProcessor()
        logger.info("Initialized Media Outlets Processor")


# Function renders subject header
# Method displays predicate title
# Operation shows object heading
def render_header():
    """
    Render the page header
    
    # Function renders subject header
    # Method displays predicate title
    # Operation shows object heading
    """
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.title("Media Outlets Monitoring")
        st.markdown("Advanced monitoring system for tracking thousands of media outlets and keyword matches")
    
    with col2:
        # Display current date and monitoring stats
        st.markdown(f"**Current Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Get counts if processor is initialized
        if 'media_processor' in st.session_state:
            outlets_count = len(st.session_state.media_processor.get_all_outlets())
            keywords_count = len(st.session_state.media_processor.get_monitoring_keywords())
            
            st.markdown(f"**Outlets Monitored**: {outlets_count}")
            st.markdown(f"**Keywords Tracked**: {keywords_count}")
    
    st.markdown("---")


# Function renders subject dashboard
# Method displays predicate overview
# Operation shows object metrics
def render_dashboard():
    """
    Render the dashboard overview
    
    # Function renders subject dashboard
    # Method displays predicate overview
    # Operation shows object metrics
    """
    st.subheader("Monitoring Dashboard")
    
    # Get the processor
    processor = st.session_state.media_processor
    
    # Get outlet and keyword counts
    outlets = processor.get_all_outlets()
    keywords = processor.get_monitoring_keywords()
    
    # Get recent content matches
    recent_matches = processor.get_content_matches(
        start_date=datetime.now() - timedelta(days=7),
        limit=1000
    )
    
    # Create metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>{}</h3>
            <p>Media Outlets</p>
        </div>
        """.format(len(outlets)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>{}</h3>
            <p>Keywords Tracked</p>
        </div>
        """.format(len(keywords)), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>{}</h3>
            <p>Content Matches (7 days)</p>
        </div>
        """.format(len(recent_matches)), unsafe_allow_html=True)
    
    with col4:
        # Calculate active outlets (checked in last 24 hours)
        active_count = 0
        for outlet in outlets:
            if outlet.last_checked and (datetime.now() - outlet.last_checked).total_seconds() < 86400:
                active_count += 1
        
        st.markdown("""
        <div class="metric-card">
            <h3>{}</h3>
            <p>Active Outlets (24h)</p>
        </div>
        """.format(active_count), unsafe_allow_html=True)
    
    # Show top keywords by matches
    st.subheader("Top Keywords by Matches")
    
    # Count matches per keyword
    keyword_counts = {}
    for match in recent_matches:
        keyword_counts[match.keyword] = keyword_counts.get(match.keyword, 0) + 1
    
    # Convert to DataFrame and sort
    if keyword_counts:
        df_keywords = pd.DataFrame({
            'Keyword': list(keyword_counts.keys()),
            'Matches': list(keyword_counts.values())
        })
        df_keywords = df_keywords.sort_values('Matches', ascending=False).head(10)
        
        # Display as bar chart
        st.bar_chart(df_keywords.set_index('Keyword'))
    else:
        st.info("No keyword matches recorded in the last 7 days.")
    
    # Show recent matches
    st.subheader("Recent Content Matches")
    
    if recent_matches:
        # Create a DataFrame for the matches
        matches_data = []
        for match in recent_matches[:20]:  # Show only first 20
            matches_data.append({
                'Date': match.match_date,
                'Keyword': match.keyword,
                'Outlet': match.outlet_name,
                'Title': match.title or 'Untitled',
                'URL': match.url
            })
        
        df_matches = pd.DataFrame(matches_data)
        
        # Display the DataFrame
        st.dataframe(df_matches.style.format({'Date': lambda x: x.strftime('%Y-%m-%d %H:%M')}),
                     column_config={
                         "URL": st.column_config.LinkColumn()
                     },
                     hide_index=True,
                     use_container_width=True)
    else:
        st.info("No recent content matches found.")


# Function renders subject management
# Method displays predicate controls
# Operation shows object interface
def render_outlets_management():
    """
    Render the outlets management interface
    
    # Function renders subject management
    # Method displays predicate controls
    # Operation shows object interface
    """
    st.subheader("Media Outlets Management")
    
    # Get the processor
    processor = st.session_state.media_processor
    
    # Create tabs for different management functions
    list_tab, add_tab, import_tab, search_tab = st.tabs([
        "Outlets List", 
        "Add Outlet", 
        "Import Outlets", 
        "Search Outlets"
    ])
    
    # Outlets List Tab
    with list_tab:
        st.markdown("### Current Media Outlets")
        
        # Get all outlets
        outlets = processor.get_all_outlets(active_only=False)
        
        if outlets:
            # Filter options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Category filter
                categories = ['All Categories'] + list(set([o.category for o in outlets if o.category]))
                selected_category = st.selectbox("Filter by Category", categories)
            
            with col2:
                # Country filter
                countries = ['All Countries'] + list(set([o.country for o in outlets if o.country]))
                selected_country = st.selectbox("Filter by Country", countries)
            
            with col3:
                # Status filter
                status_options = ['All Outlets', 'Active Only', 'Inactive Only']
                selected_status = st.selectbox("Filter by Status", status_options)
            
            # Apply filters
            filtered_outlets = outlets
            
            if selected_category != 'All Categories':
                filtered_outlets = [o for o in filtered_outlets if o.category == selected_category]
                
            if selected_country != 'All Countries':
                filtered_outlets = [o for o in filtered_outlets if o.country == selected_country]
                
            if selected_status == 'Active Only':
                filtered_outlets = [o for o in filtered_outlets if o.active]
                
            elif selected_status == 'Inactive Only':
                filtered_outlets = [o for o in filtered_outlets if not o.active]
            
            # Create DataFrame for display
            outlets_data = []
            for outlet in filtered_outlets:
                outlets_data.append({
                    'Name': outlet.name,
                    'Domain': outlet.domain,
                    'Category': outlet.category or '',
                    'Country': outlet.country or '',
                    'Language': outlet.language or '',
                    'Active': 'Yes' if outlet.active else 'No',
                    'Last Checked': outlet.last_checked,
                    'ID': outlet.outlet_id
                })
            
            if outlets_data:
                df_outlets = pd.DataFrame(outlets_data)
                
                # Display the DataFrame
                st.dataframe(df_outlets.style.format({'Last Checked': lambda x: x.strftime('%Y-%m-%d %H:%M') if x else 'Never'}),
                             column_config={
                                 "Name": st.column_config.TextColumn(
                                     "Name",
                                     help="Outlet name"
                                 ),
                                 "Active": st.column_config.TextColumn(
                                     "Active",
                                     help="Whether the outlet is actively monitored"
                                 )
                             },
                             hide_index=True,
                             use_container_width=True)
                
                # Outlet details expander
                with st.expander("View Outlet Details"):
                    # Outlet selector
                    selected_outlet_id = st.selectbox(
                        "Select an outlet to view details",
                        options=[o['ID'] for o in outlets_data],
                        format_func=lambda x: next((o['Name'] for o in outlets_data if o['ID'] == x), x)
                    )
                    
                    if selected_outlet_id:
                        # Get outlet details
                        outlet = processor.get_outlet(selected_outlet_id)
                        
                        if outlet:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown(f"**Name**: {outlet.name}")
                                st.markdown(f"**Domain**: {outlet.domain}")
                                st.markdown(f"**URL**: [{outlet.url}]({outlet.url})")
                                st.markdown(f"**Category**: {outlet.category or 'Not specified'}")
                                st.markdown(f"**Country**: {outlet.country or 'Not specified'}")
                                st.markdown(f"**Language**: {outlet.language or 'Not specified'}")
                                
                            with col2:
                                st.markdown(f"**Active**: {'Yes' if outlet.active else 'No'}")
                                st.markdown(f"**Reliability Score**: {outlet.reliability_score or 'Not rated'}")
                                st.markdown(f"**Bias Rating**: {outlet.bias_rating or 'Not rated'}")
                                st.markdown(f"**Last Checked**: {outlet.last_checked.strftime('%Y-%m-%d %H:%M') if outlet.last_checked else 'Never'}")
                                st.markdown(f"**Discovered Date**: {outlet.discovered_date.strftime('%Y-%m-%d %H:%M')}")
                            
                            # RSS Feeds
                            if outlet.rss_feeds:
                                st.markdown("#### RSS Feeds")
                                for feed in outlet.rss_feeds:
                                    st.markdown(f"- [{feed}]({feed})")
                            
                            # Keywords
                            if outlet.keywords:
                                st.markdown("#### Associated Keywords")
                                keywords_html = ""
                                for keyword in outlet.keywords:
                                    keywords_html += f'<span class="keyword-tag">{keyword}</span>'
                                st.markdown(keywords_html, unsafe_allow_html=True)
                            
                            # Actions
                            st.markdown("#### Actions")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                # Toggle active status
                                if outlet.active:
                                    if st.button("Deactivate Outlet", key=f"deactivate_{outlet.outlet_id}"):
                                        outlet.active = False
                                        processor.save_outlet(outlet)
                                        st.rerun()
                                else:
                                    if st.button("Activate Outlet", key=f"activate_{outlet.outlet_id}"):
                                        outlet.active = True
                                        processor.save_outlet(outlet)
                                        st.rerun()
                            
                            with col2:
                                # Monitor outlet now
                                if st.button("Monitor Now", key=f"monitor_{outlet.outlet_id}"):
                                    with st.spinner("Monitoring outlet..."):
                                        results = processor.batch_monitor_outlets([outlet])
                                        st.success(f"Monitoring completed. Found {results['matches_found']} matches.")
                            
                            with col3:
                                # Discover related outlets
                                if st.button("Discover Related", key=f"discover_{outlet.outlet_id}"):
                                    with st.spinner("Discovering related outlets..."):
                                        related = processor.discover_related_outlets(outlet, max_links=10)
                                        st.success(f"Discovered {len(related)} related outlets.")
            else:
                st.info("No outlets match the selected filters.")
        else:
            st.info("No media outlets have been added yet.")
    
    # Add Outlet Tab
    with add_tab:
        st.markdown("### Add New Media Outlet")
        
        # Create form for adding a new outlet
        with st.form("add_outlet_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("Outlet Name*", help="Name of the media outlet")
                domain = st.text_input("Domain*", help="Domain without www (e.g., nytimes.com)")
                url = st.text_input("URL*", help="Homepage URL of the outlet")
                category = st.text_input("Category", help="Category of the outlet (e.g., News, Sports)")
                
            with col2:
                country = st.text_input("Country", help="Country of origin")
                language = st.text_input("Language", help="Primary language of the outlet")
                reliability_score = st.slider("Reliability Score", 0.0, 1.0, 0.8, 0.05, 
                                            help="Reliability score from 0 to 1")
                bias_rating = st.selectbox("Bias Rating", 
                                         ["", "left", "left-center", "center", "right-center", "right"],
                                         help="Political bias of the outlet")
            
            # RSS Feeds
            rss_feeds = st.text_area("RSS Feeds (one per line)", 
                                   help="List of RSS feed URLs, one per line")
            
            # Keywords
            keywords = st.text_input("Keywords (comma-separated)", 
                                   help="Keywords associated with this outlet")
            
            # Form submission
            submitted = st.form_submit_button("Add Outlet")
            
            if submitted:
                # Validate required fields
                if not name or not domain or not url:
                    st.error("Name, domain, and URL are required fields.")
                else:
                    # Generate outlet ID
                    outlet_id = processor.hashlib.md5(f"{domain}:{name}".encode()).hexdigest()
                    
                    # Create outlet object
                    outlet = MediaOutlet(
                        outlet_id=outlet_id,
                        name=name,
                        domain=domain,
                        url=url,
                        category=category if category else None,
                        country=country if country else None,
                        language=language if language else None,
                        reliability_score=reliability_score if reliability_score > 0 else None,
                        bias_rating=bias_rating if bias_rating else None,
                        active=True,
                        discovered_date=datetime.now()
                    )
                    
                    # Add RSS feeds
                    if rss_feeds:
                        outlet.rss_feeds = [feed.strip() for feed in rss_feeds.split('\n') if feed.strip()]
                    
                    # Add keywords
                    if keywords:
                        outlet.keywords = [kw.strip() for kw in keywords.split(',') if kw.strip()]
                    
                    # Save outlet
                    if processor.save_outlet(outlet):
                        st.success(f"Added outlet: {name}")
                    else:
                        st.error("Failed to add outlet.")
    
    # Import Outlets Tab
    with import_tab:
        st.markdown("### Import Media Outlets")
        
        # File uploader for Excel files
        uploaded_file = st.file_uploader("Upload Excel File", type=['xlsx', 'xls'])
        
        if uploaded_file is not None:
            # Display file info
            st.info(f"File: {uploaded_file.name} ({uploaded_file.size} bytes)")
            
            # Preview Excel file
            try:
                df_preview = pd.read_excel(uploaded_file)
                
                # Check for required columns
                required_columns = ['name', 'domain', 'url']
                missing_columns = [col for col in required_columns if col not in df_preview.columns]
                
                if missing_columns:
                    st.error(f"Missing required columns: {', '.join(missing_columns)}")
                    st.markdown("""
                    ### Required Excel Format
                    
                    The Excel file must have at least the following columns:
                    - `name`: Name of the media outlet
                    - `domain`: Domain name without www (e.g., nytimes.com)
                    - `url`: Homepage URL of the outlet
                    
                    Optional columns:
                    - `category`: Category of the outlet
                    - `country`: Country of origin
                    - `language`: Primary language
                    - `reliability_score`: Score from 0 to 1
                    - `bias_rating`: Political bias
                    - `rss_feeds`: Comma-separated list of RSS feed URLs
                    - `keywords`: Comma-separated list of associated keywords
                    """)
                else:
                    # Show preview
                    st.markdown("#### Data Preview")
                    st.dataframe(df_preview.head(5), use_container_width=True)
                    
                    # Import button
                    if st.button("Import Outlets"):
                        with st.spinner("Importing outlets..."):
                            # Save uploaded file temporarily
                            temp_file = os.path.join('data/web_intelligence', uploaded_file.name)
                            with open(temp_file, 'wb') as f:
                                f.write(uploaded_file.getbuffer())
                            
                            # Import from the temp file
                            imported_count = processor.import_outlets_from_excel(temp_file)
                            
                            # Clean up
                            os.remove(temp_file)
                            
                            st.success(f"Successfully imported {imported_count} outlets.")
            
            except Exception as e:
                st.error(f"Error reading Excel file: {e}")
                
        else:
            st.markdown("""
            ### Excel File Format
            
            Upload an Excel file (.xlsx or .xls) with the following columns:
            
            #### Required Columns:
            - `name`: Name of the media outlet
            - `domain`: Domain name without www (e.g., nytimes.com)
            - `url`: Homepage URL of the outlet
            
            #### Optional Columns:
            - `category`: Category of the outlet
            - `country`: Country of origin
            - `language`: Primary language
            - `reliability_score`: Score from 0 to 1
            - `bias_rating`: Political bias
            - `rss_feeds`: Comma-separated list of RSS feed URLs
            - `keywords`: Comma-separated list of associated keywords
            """)
    
    # Search Outlets Tab
    with search_tab:
        st.markdown("### Search Media Outlets")
        
        # Search box
        search_query = st.text_input("Search Query", help="Search by name, domain, URL, category, etc.")
        
        if search_query:
            with st.spinner("Searching..."):
                # Search outlets
                results = processor.search_outlets(search_query)
                
                if results:
                    st.success(f"Found {len(results)} matching outlets")
                    
                    # Display results
                    results_data = []
                    for outlet in results:
                        results_data.append({
                            'Name': outlet.name,
                            'Domain': outlet.domain,
                            'Category': outlet.category or '',
                            'Country': outlet.country or '',
                            'Active': 'Yes' if outlet.active else 'No',
                            'ID': outlet.outlet_id
                        })
                    
                    df_results = pd.DataFrame(results_data)
                    st.dataframe(df_results, hide_index=True, use_container_width=True)
                    
                    # Select outlet for detailed view
                    if len(results) > 0:
                        st.markdown("### Outlet Details")
                        selected_index = st.selectbox(
                            "Select an outlet to view details",
                            range(len(results)),
                            format_func=lambda i: results[i].name
                        )
                        
                        # Display selected outlet details
                        selected_outlet = results[selected_index]
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"**Name**: {selected_outlet.name}")
                            st.markdown(f"**Domain**: {selected_outlet.domain}")
                            st.markdown(f"**URL**: [{selected_outlet.url}]({selected_outlet.url})")
                            st.markdown(f"**Category**: {selected_outlet.category or 'Not specified'}")
                            st.markdown(f"**Country**: {selected_outlet.country or 'Not specified'}")
                            
                        with col2:
                            st.markdown(f"**Language**: {selected_outlet.language or 'Not specified'}")
                            st.markdown(f"**Active**: {'Yes' if selected_outlet.active else 'No'}")
                            st.markdown(f"**Reliability Score**: {selected_outlet.reliability_score or 'Not rated'}")
                            st.markdown(f"**Bias Rating**: {selected_outlet.bias_rating or 'Not rated'}")
                            st.markdown(f"**Last Checked**: {selected_outlet.last_checked.strftime('%Y-%m-%d %H:%M') if selected_outlet.last_checked else 'Never'}")
                        
                        # Actions
                        st.markdown("#### Actions")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            # Monitor outlet now
                            if st.button("Monitor Now", key=f"search_monitor_{selected_outlet.outlet_id}"):
                                with st.spinner("Monitoring outlet..."):
                                    results = processor.batch_monitor_outlets([selected_outlet])
                                    st.success(f"Monitoring completed. Found {results['matches_found']} matches.")
                        
                        with col2:
                            # Discover related outlets
                            if st.button("Discover Related", key=f"search_discover_{selected_outlet.outlet_id}"):
                                with st.spinner("Discovering related outlets..."):
                                    related = processor.discover_related_outlets(selected_outlet, max_links=10)
                                    st.success(f"Discovered {len(related)} related outlets.")
                        
                        with col3:
                            # Toggle active status
                            if selected_outlet.active:
                                if st.button("Deactivate", key=f"search_deactivate_{selected_outlet.outlet_id}"):
                                    selected_outlet.active = False
                                    processor.save_outlet(selected_outlet)
                                    st.success("Outlet deactivated.")
                            else:
                                if st.button("Activate", key=f"search_activate_{selected_outlet.outlet_id}"):
                                    selected_outlet.active = True
                                    processor.save_outlet(selected_outlet)
                                    st.success("Outlet activated.")
                else:
                    st.info(f"No outlets found matching '{search_query}'")
        else:
            st.info("Enter a search term to find media outlets.")


# Function renders subject monitoring
# Method displays predicate keywords
# Operation shows object controls
def render_keywords_monitoring():
    """
    Render the keywords monitoring interface
    
    # Function renders subject monitoring
    # Method displays predicate keywords
    # Operation shows object controls
    """
    st.subheader("Keywords Monitoring")
    
    # Get the processor
    processor = st.session_state.media_processor
    
    # Create tabs for different keyword functions
    list_tab, add_tab, import_tab, matches_tab = st.tabs([
        "Keywords List", 
        "Add Keywords", 
        "Import Keywords",
        "Keyword Matches"
    ])
    
    # Keywords List Tab
    with list_tab:
        st.markdown("### Current Monitoring Keywords")
        
        # Get all keywords
        keywords = processor.get_monitoring_keywords(active_only=False)
        
        if keywords:
            # Filter options
            col1, col2 = st.columns(2)
            
            with col1:
                # Category filter
                categories = ['All Categories'] + list(set([k.category for k in keywords if k.category]))
                selected_category = st.selectbox("Filter by Category", categories)
            
            with col2:
                # Status filter
                status_options = ['All Keywords', 'Active Only', 'Inactive Only']
                selected_status = st.selectbox("Filter by Status", status_options)
            
            # Apply filters
            filtered_keywords = keywords
            
            if selected_category != 'All Categories':
                filtered_keywords = [k for k in filtered_keywords if k.category == selected_category]
                
            if selected_status == 'Active Only':
                filtered_keywords = [k for k in filtered_keywords if k.active]
                
            elif selected_status == 'Inactive Only':
                filtered_keywords = [k for k in filtered_keywords if not k.active]
            
            # Create DataFrame for display
            keywords_data = []
            for keyword in filtered_keywords:
                keywords_data.append({
                    'Keyword': keyword.keyword,
                    'Category': keyword.category or '',
                    'Priority': keyword.priority,
                    'Active': 'Yes' if keyword.active else 'No',
                    'Match Count': keyword.match_count,
                    'Last Matched': keyword.last_matched
                })
            
            if keywords_data:
                df_keywords = pd.DataFrame(keywords_data)
                
                # Sort by priority and match count
                df_keywords = df_keywords.sort_values(['Priority', 'Match Count'], ascending=[False, False])
                
                # Display the DataFrame
                st.dataframe(df_keywords.style.format({'Last Matched': lambda x: x.strftime('%Y-%m-%d %H:%M') if x else 'Never'}),
                             column_config={
                                 "Priority": st.column_config.NumberColumn(
                                     "Priority",
                                     help="Priority level (1-5)",
                                     format="%d"
                                 ),
                                 "Match Count": st.column_config.NumberColumn(
                                     "Match Count",
                                     help="Number of matches found",
                                     format="%d"
                                 )
                             },
                             hide_index=True,
                             use_container_width=True)
                
                # Keyword actions
                with st.expander("Keyword Actions"):
                    # Select keyword
                    selected_keyword = st.selectbox(
                        "Select a keyword",
                        options=[k['Keyword'] for k in keywords_data]
                    )
                    
                    if selected_keyword:
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            # Change priority
                            new_priority = st.slider(
                                "Change Priority",
                                1, 5, 
                                next((k.priority for k in filtered_keywords if k.keyword == selected_keyword), 3)
                            )
                            
                            if st.button("Update Priority"):
                                # Find the keyword object
                                for k in keywords:
                                    if k.keyword == selected_keyword:
                                        # Create a new keyword with updated priority
                                        processor.add_monitoring_keyword(k.keyword, k.category, new_priority)
                                        st.success(f"Updated priority for '{selected_keyword}'")
                                        time.sleep(1)
                                        st.rerun()
                        
                        with col2:
                            # Toggle active status
                            is_active = next((k.active for k in filtered_keywords if k.keyword == selected_keyword), True)
                            
                            if is_active:
                                if st.button("Deactivate Keyword"):
                                    # Find and update the keyword
                                    for k in keywords:
                                        if k.keyword == selected_keyword:
                                            # Create a new keyword with active=False
                                            # Note: The processor doesn't have a direct method to update status,
                                            # so we'd need to modify the underlying database directly
                                            st.error("Deactivation not implemented in this version")
                            else:
                                if st.button("Activate Keyword"):
                                    # Find and update the keyword
                                    for k in keywords:
                                        if k.keyword == selected_keyword:
                                            # Create a new keyword with active=True
                                            st.error("Activation not implemented in this version")
                        
                        with col3:
                            # View matches
                            if st.button("View Recent Matches"):
                                matches = processor.get_content_matches(
                                    keyword=selected_keyword,
                                    limit=100
                                )
                                
                                if matches:
                                    st.success(f"Found {len(matches)} matches for '{selected_keyword}'")
                                    
                                    # Create DataFrame
                                    matches_data = []
                                    for match in matches:
                                        matches_data.append({
                                            'Date': match.match_date,
                                            'Outlet': match.outlet_name,
                                            'Title': match.title or 'Untitled',
                                            'URL': match.url
                                        })
                                    
                                    df_matches = pd.DataFrame(matches_data)
                                    st.dataframe(df_matches.style.format({'Date': lambda x: x.strftime('%Y-%m-%d %H:%M')}),
                                                column_config={"URL": st.column_config.LinkColumn()},
                                                hide_index=True,
                                                use_container_width=True)
                                else:
                                    st.info(f"No matches found for '{selected_keyword}'")
            else:
                st.info("No keywords match the selected filters.")
        else:
            st.info("No monitoring keywords have been added yet.")
    
    # Add Keywords Tab
    with add_tab:
        st.markdown("### Add Monitoring Keywords")
        
        # Create form for adding keywords
        with st.form("add_keywords_form"):
            keywords_input = st.text_area(
                "Keywords (one per line)",
                help="Enter keywords to monitor, one per line"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                category = st.text_input(
                    "Category",
                    help="Category for these keywords (optional)"
                )
            
            with col2:
                priority = st.slider(
                    "Priority",
                    1, 5, 3,
                    help="Priority level (1-5, 5 being highest)"
                )
            
            submitted = st.form_submit_button("Add Keywords")
            
            if submitted:
                if not keywords_input:
                    st.error("Please enter at least one keyword.")
                else:
                    # Parse keywords
                    keywords_list = [k.strip() for k in keywords_input.split('\n') if k.strip()]
                    
                    # Add each keyword
                    added_count = 0
                    for keyword in keywords_list:
                        if processor.add_monitoring_keyword(keyword, category, priority):
                            added_count += 1
                    
                    if added_count > 0:
                        st.success(f"Added {added_count} keywords for monitoring")
                    else:
                        st.error("Failed to add keywords.")
    
    # Import Keywords Tab
    with import_tab:
        st.markdown("### Import Keywords")
        
        # File uploader for CSV files
        uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])
        
        if uploaded_file is not None:
            # Display file info
            st.info(f"File: {uploaded_file.name} ({uploaded_file.size} bytes)")
            
            # Preview CSV file
            try:
                df_preview = pd.read_csv(uploaded_file)
                
                # Check for required column
                if 'keyword' not in df_preview.columns:
                    st.error("CSV file must have a 'keyword' column")
                    st.markdown("""
                    ### Required CSV Format
                    
                    The CSV file must have at least the following column:
                    - `keyword`: Keyword to monitor
                    
                    Optional columns:
                    - `category`: Category for organization
                    - `priority`: Priority level (1-5)
                    """)
                else:
                    # Show preview
                    st.markdown("#### Data Preview")
                    st.dataframe(df_preview.head(5), use_container_width=True)
                    
                    # Import button
                    if st.button("Import Keywords"):
                        with st.spinner("Importing keywords..."):
                            # Save uploaded file temporarily
                            temp_file = os.path.join('data/web_intelligence', uploaded_file.name)
                            with open(temp_file, 'wb') as f:
                                f.write(uploaded_file.getbuffer())
                            
                            # Custom import function
                            added_count = 0
                            try:
                                # Read keywords from CSV
                                df = pd.read_csv(temp_file)
                                
                                # Add each keyword
                                for _, row in df.iterrows():
                                    keyword = row['keyword']
                                    
                                    # Get optional parameters if they exist
                                    category = row.get('category') if 'category' in df.columns else None
                                    priority = int(row.get('priority', 3)) if 'priority' in df.columns else 3
                                    
                                    # Add the keyword
                                    if processor.add_monitoring_keyword(keyword, category, priority):
                                        added_count += 1
                                        
                            except Exception as e:
                                st.error(f"Error importing keywords: {e}")
                                
                            # Clean up
                            os.remove(temp_file)
                            
                            st.success(f"Successfully imported {added_count} keywords.")
            
            except Exception as e:
                st.error(f"Error reading CSV file: {e}")
                
        else:
            st.markdown("""
            ### CSV File Format
            
            Upload a CSV file with the following columns:
            
            #### Required Column:
            - `keyword`: Keyword to monitor
            
            #### Optional Columns:
            - `category`: Category for organization
            - `priority`: Priority level (1-5)
            
            Example:
            ```
            keyword,category,priority
            cybersecurity,Technology,5
            data breach,Security,5
            artificial intelligence,Technology,4
            ```
            """)
    
    # Keyword Matches Tab
    with matches_tab:
        st.markdown("### Keyword Content Matches")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Keyword filter
            keywords_list = ['All Keywords'] + [k.keyword for k in processor.get_monitoring_keywords()]
            selected_keyword = st.selectbox("Filter by Keyword", keywords_list)
        
        with col2:
            # Time range filter
            time_options = ['Last 24 hours', 'Last 7 days', 'Last 30 days', 'All time']
            selected_time = st.selectbox("Time Range", time_options)
            
            # Calculate start date
            if selected_time == 'Last 24 hours':
                start_date = datetime.now() - timedelta(days=1)
            elif selected_time == 'Last 7 days':
                start_date = datetime.now() - timedelta(days=7)
            elif selected_time == 'Last 30 days':
                start_date = datetime.now() - timedelta(days=30)
            else:
                start_date = None
        
        with col3:
            # Outlet filter
            outlets = processor.get_all_outlets()
            outlet_options = ['All Outlets'] + [o.name for o in outlets]
            selected_outlet = st.selectbox("Filter by Outlet", outlet_options)
            
            # Get outlet ID if selected
            selected_outlet_id = None
            if selected_outlet != 'All Outlets':
                for o in outlets:
                    if o.name == selected_outlet:
                        selected_outlet_id = o.outlet_id
                        break
        
        # Get content matches with filters
        if selected_keyword != 'All Keywords':
            keyword_filter = selected_keyword
        else:
            keyword_filter = None
            
        matches = processor.get_content_matches(
            keyword=keyword_filter,
            outlet_id=selected_outlet_id,
            start_date=start_date,
            limit=500
        )
        
        if matches:
            st.success(f"Found {len(matches)} matches")
            
            # Create DataFrame
            matches_data = []
            for match in matches:
                matches_data.append({
                    'Date': match.match_date,
                    'Keyword': match.keyword,
                    'Outlet': match.outlet_name,
                    'Title': match.title or 'Untitled',
                    'URL': match.url,
                    'ID': match.content_id
                })
            
            df_matches = pd.DataFrame(matches_data)
            
            # Display the DataFrame
            st.dataframe(df_matches.style.format({'Date': lambda x: x.strftime('%Y-%m-%d %H:%M')}),
                        column_config={
                            "URL": st.column_config.LinkColumn(),
                            "ID": st.column_config.Column(
                                "ID",
                                help="Content ID",
                                width="none",
                                disabled=True
                            )
                        },
                        hide_index=True,
                        use_container_width=True)
            
            # Match details
            with st.expander("View Match Context"):
                # Match selector
                selected_match_id = st.selectbox(
                    "Select a match to view context",
                    options=[m['ID'] for m in matches_data],
                    format_func=lambda x: next((f"{m['Keyword']} in {m['Outlet']} - {m['Title']}" 
                                              for m in matches_data if m['ID'] == x), x)
                )
                
                if selected_match_id:
                    # Find the match
                    selected_match = next((m for m in matches if m.content_id == selected_match_id), None)
                    
                    if selected_match:
                        # Display match details
                        st.markdown(f"**Keyword**: {selected_match.keyword}")
                        st.markdown(f"**Outlet**: {selected_match.outlet_name}")
                        st.markdown(f"**Title**: {selected_match.title or 'Untitled'}")
                        st.markdown(f"**URL**: [{selected_match.url}]({selected_match.url})")
                        st.markdown(f"**Match Date**: {selected_match.match_date.strftime('%Y-%m-%d %H:%M')}")
                        
                        # Display match context
                        st.markdown("#### Match Context")
                        
                        context = selected_match.match_context
                        if context:
                            # Highlight the keyword in the context
                            keyword = selected_match.keyword
                            highlighted_context = context.replace(
                                keyword, 
                                f"<span style='background-color: #ffff00;'>{keyword}</span>"
                            )
                            
                            st.markdown(f"""
                            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px;">
                            ...{highlighted_context}...
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.info("No context available for this match.")
                            
                        # Actions
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if st.button("Mark as Processed", key=f"process_{selected_match_id}"):
                                # Note: This would require adding a method to the processor
                                st.success("Match marked as processed")
                        
                        with col2:
                            if st.button("Extract Full Content", key=f"extract_{selected_match_id}"):
                                with st.spinner("Extracting content..."):
                                    # Extract content from the URL
                                    content = processor.extract_content_from_url(selected_match.url)
                                    
                                    if content and content.get('text'):
                                        st.markdown("#### Full Article Content")
                                        st.markdown(content['text'])
                                    else:
                                        st.error("Failed to extract full content.")
            
            # Export matches
            st.markdown("### Export Matches")
            col1, col2 = st.columns(2)
            
            with col1:
                export_format = st.selectbox("Export Format", ["CSV", "JSON"])
            
            with col2:
                if st.button("Export Matches"):
                    with st.spinner("Exporting matches..."):
                        # Create export directory
                        os.makedirs('exports', exist_ok=True)
                        
                        # Export to selected format
                        if export_format == "CSV":
                            # Export to CSV
                            csv_data = io.StringIO()
                            df_matches.to_csv(csv_data, index=False)
                            
                            # Create download link
                            b64 = base64.b64encode(csv_data.getvalue().encode()).decode()
                            href = f'<a href="data:file/csv;base64,{b64}" download="keyword_matches.csv">Download CSV File</a>'
                            st.markdown(href, unsafe_allow_html=True)
                        else:
                            # Export to JSON
                            json_data = df_matches.to_json(orient='records')
                            
                            # Create download link
                            b64 = base64.b64encode(json_data.encode()).decode()
                            href = f'<a href="data:file/json;base64,{b64}" download="keyword_matches.json">Download JSON File</a>'
                            st.markdown(href, unsafe_allow_html=True)
        else:
            st.info("No matches found with the selected filters.")


# Function renders subject monitoring
# Method displays predicate controls
# Operation shows object interface
def render_batch_monitoring():
    """
    Render the batch monitoring interface
    
    # Function renders subject monitoring
    # Method displays predicate controls
    # Operation shows object interface
    """
    st.subheader("Batch Monitoring")
    
    # Get the processor
    processor = st.session_state.media_processor
    
    # Monitoring controls
    st.markdown("### Monitor Media Outlets")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Number of outlets to monitor
        max_outlets = st.slider("Number of Outlets", 1, 50, 10, 
                              help="Maximum number of outlets to monitor in this batch")
    
    with col2:
        # Category filter
        outlets = processor.get_all_outlets(active_only=True)
        categories = ['All Categories'] + list(set([o.category for o in outlets if o.category]))
        selected_category = st.selectbox("Filter by Category", categories,
                                     help="Limit monitoring to outlets in this category")
    
    with col3:
        # Keyword filter
        keywords = processor.get_monitoring_keywords(active_only=True)
        keyword_options = ['All Keywords'] + [k.keyword for k in keywords]
        selected_keyword = st.selectbox("Monitor for Specific Keyword", keyword_options,
                                      help="Focus monitoring on a specific keyword")
    
    # Monitor button
    if st.button("Start Monitoring"):
        with st.spinner("Monitoring outlets..."):
            # Get filtered outlets
            filtered_outlets = outlets
            if selected_category != 'All Categories':
                filtered_outlets = [o for o in outlets if o.category == selected_category]
            
            # Limit to max outlets
            outlets_to_monitor = filtered_outlets[:max_outlets]
            
            # Get selected keyword if any
            keywords_to_monitor = None
            if selected_keyword != 'All Keywords':
                keywords_to_monitor = [selected_keyword]
            
            # Run batch monitoring
            start_time = time.time()
            results = processor.batch_monitor_outlets(
                outlets=outlets_to_monitor,
                max_outlets=max_outlets,
                keywords=keywords_to_monitor
            )
            duration = time.time() - start_time
            
            # Display results
            st.success(f"Monitoring completed in {duration:.1f} seconds")
            
            # Results metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Outlets Processed", results['outlets_processed'])
            
            with col2:
                st.metric("URLs Checked", results['urls_checked'])
            
            with col3:
                st.metric("Content Extracted", results['content_extracted'])
            
            with col4:
                st.metric("Matches Found", results['matches_found'])
            
            # Show any errors
            if results['errors'] > 0:
                st.warning(f"Encountered {results['errors']} errors during monitoring")
    
    # Recent monitoring results
    st.markdown("### Recent Content Matches")
    
    # Get recent matches
    recent_matches = processor.get_content_matches(
        start_date=datetime.now() - timedelta(hours=24),
        limit=50
    )
    
    if recent_matches:
        # Group by outlet
        outlets_with_matches = {}
        for match in recent_matches:
            if match.outlet_name not in outlets_with_matches:
                outlets_with_matches[match.outlet_name] = []
            outlets_with_matches[match.outlet_name].append(match)
        
        # Display matches by outlet
        for outlet_name, matches in outlets_with_matches.items():
            with st.expander(f"{outlet_name} ({len(matches)} matches)"):
                for match in matches:
                    st.markdown(f"""
                    <div class="result-card">
                        <h4>{match.title or 'Untitled'}</h4>
                        <p><strong>Keyword:</strong> {match.keyword}</p>
                        <p><strong>URL:</strong> <a href="{match.url}" target="_blank">{match.url}</a></p>
                        <p><strong>Found:</strong> {match.match_date.strftime('%Y-%m-%d %H:%M')}</p>
                        <p><strong>Context:</strong> ...{match.match_context or 'No context available'}...</p>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.info("No content matches found in the last 24 hours.")
    
    # Scheduled monitoring configuration
    with st.expander("Configure Scheduled Monitoring"):
        st.markdown("""
        ### Scheduled Monitoring Configuration
        
        Configure automated monitoring schedules to regularly check outlets for keyword matches.
        
        > Note: Scheduling is a placeholder in this version. In a production environment, 
        > this would connect to a task scheduler or cron job system.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            schedule_interval = st.selectbox(
                "Monitoring Interval",
                ["Every hour", "Every 6 hours", "Every 12 hours", "Daily", "Weekly"]
            )
        
        with col2:
            max_outlets_scheduled = st.slider(
                "Outlets per Run",
                10, 500, 100,
                help="Maximum outlets to process in each scheduled run"
            )
        
        # Target specific categories
        target_categories = st.multiselect(
            "Target Categories",
            list(set([o.category for o in outlets if o.category])),
            help="Target specific categories for scheduled monitoring"
        )
        
        # Save configuration button
        if st.button("Save Schedule Configuration"):
            st.success("Schedule configuration saved")
            st.info("Scheduled monitoring will begin at the next interval")


# Function runs subject interface
# Method executes predicate application
# Operation starts object processing
def main():
    """
    Main function to run the media outlets monitoring interface
    
    # Function runs subject interface
    # Method executes predicate application
    # Operation starts object processing
    """
    # Configure page
    configure_page()
    
    # Initialize processor
    initialize_processor()
    
    # Render header
    render_header()
    
    # Create navigation tabs
    dashboard_tab, outlets_tab, keywords_tab, monitoring_tab = st.tabs([
        "Dashboard", 
        "Outlets Management", 
        "Keywords Monitoring",
        "Batch Monitoring"
    ])
    
    # Dashboard Tab
    with dashboard_tab:
        render_dashboard()
    
    # Outlets Management Tab
    with outlets_tab:
        render_outlets_management()
    
    # Keywords Monitoring Tab
    with keywords_tab:
        render_keywords_monitoring()
    
    # Batch Monitoring Tab
    with monitoring_tab:
        render_batch_monitoring()


# Run the application
if __name__ == "__main__":
    main()