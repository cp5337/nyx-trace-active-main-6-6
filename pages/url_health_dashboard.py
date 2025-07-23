"""
URL Health Dashboard
-------------------
Real-time link reliability and accessibility metrics for URLs used in the application.

This dashboard provides insights into the health and availability of data sources,
with a focus on monitoring web scraping endpoints and ensuring consistent data access.
"""
import streamlit as st
import pandas as pd
import time
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from utils.url_health_monitor import URLHealthMonitor, URLHealthStatus
import json
import re


# Initialize session state for URL health monitor
if 'url_health_monitor' not in st.session_state:
    st.session_state.url_health_monitor = URLHealthMonitor()

if 'custom_urls' not in st.session_state:
    st.session_state.custom_urls = []
    
if 'last_check_time' not in st.session_state:
    st.session_state.last_check_time = None
    
# Helper function to validate URL
def is_valid_url(url):
    url_pattern = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ipv4
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return bool(url_pattern.match(url))

# Function to add custom URL
def add_custom_url():
    if st.session_state.new_url and is_valid_url(st.session_state.new_url):
        if st.session_state.new_url not in st.session_state.custom_urls:
            st.session_state.custom_urls.append(st.session_state.new_url)
        st.session_state.new_url = ""

def run_health_check():
    """Run health check on all URLs"""
    urls_to_check = []
    
    # Add custom URLs
    urls_to_check.extend(st.session_state.custom_urls)
    
    # Add data source URLs if available
    try:
        if 'data_source_urls' in st.session_state:
            urls_to_check.extend(st.session_state.data_source_urls)
    except Exception as e:
        st.error(f"Error accessing data source URLs: {str(e)}")
    
    # Remove duplicates
    urls_to_check = list(set(urls_to_check))
    
    if urls_to_check:
        with st.spinner("Checking URL health..."):
            monitor = st.session_state.url_health_monitor
            monitor.check_urls_health(urls_to_check)
            st.session_state.last_check_time = datetime.now()
            
            # Update the data sources URLs in session state
            if 'data_source_urls' not in st.session_state:
                st.session_state.data_source_urls = []
    else:
        st.warning("No URLs to check. Add custom URLs or configure data sources.")

# Main dashboard UI
st.title("URL Health Dashboard")
st.write("Monitor the health and accessibility of data source URLs in real-time.")

# Sidebar for configuration
with st.sidebar:
    st.header("URL Health Configuration")
    
    # Add custom URLs
    st.subheader("Custom URLs")
    st.text_input("Add URL", key="new_url", on_change=add_custom_url)
    
    # Display and manage custom URLs
    if st.session_state.custom_urls:
        st.write("Current custom URLs:")
        for i, url in enumerate(st.session_state.custom_urls):
            cols = st.columns([4, 1])
            cols[0].text(url)
            if cols[1].button("🗑️", key=f"delete_{i}"):
                st.session_state.custom_urls.pop(i)
                st.experimental_rerun()
    
    # Run health check button
    st.button("Run Health Check Now", on_click=run_health_check)
    
    # Auto-refresh settings
    st.subheader("Auto-Refresh")
    auto_refresh = st.checkbox("Enable auto-refresh", value=False)
    if auto_refresh:
        refresh_interval = st.slider("Refresh interval (minutes)", 1, 60, 5)
        
        # Initialize auto-refresh time if needed
        if 'next_refresh' not in st.session_state:
            st.session_state.next_refresh = datetime.now() + timedelta(minutes=refresh_interval)
        
        # Display countdown timer
        now = datetime.now()
        if now >= st.session_state.next_refresh:
            run_health_check()
            st.session_state.next_refresh = now + timedelta(minutes=refresh_interval)
        
        time_remaining = (st.session_state.next_refresh - now).total_seconds()
        st.write(f"Next refresh in: {int(time_remaining // 60)}m {int(time_remaining % 60)}s")
    else:
        if 'next_refresh' in st.session_state:
            del st.session_state.next_refresh

# Main content
tabs = st.tabs(["Health Status", "Response Times", "Availability", "Detailed View"])

with tabs[0]:
    st.header("URL Health Status")
    
    # Last check time
    if st.session_state.last_check_time:
        st.write(f"Last checked: {st.session_state.last_check_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Summary metrics
    monitor = st.session_state.url_health_monitor
    summary = monitor.get_status_summary()
    
    # Display metrics in columns
    cols = st.columns(5)
    cols[0].metric("Healthy", summary[URLHealthStatus.STATUS_OK])
    cols[1].metric("Slow", summary[URLHealthStatus.STATUS_SLOW])
    cols[2].metric("Error", summary[URLHealthStatus.STATUS_ERROR])
    cols[3].metric("Timeout", summary[URLHealthStatus.STATUS_TIMEOUT])
    cols[4].metric("Invalid", summary[URLHealthStatus.STATUS_INVALID])
    
    # Summary pie chart
    if any(summary.values()):
        summary_chart = monitor.create_summary_chart()
        st.plotly_chart(summary_chart, use_container_width=True)
    else:
        st.info("No URL health data available. Run a health check to see results.")

with tabs[1]:
    st.header("Response Times")
    
    # Response time chart
    if monitor.url_status:
        # Allow user to select which URLs to display
        all_urls = list(monitor.url_status.keys())
        selected_urls = st.multiselect("Select URLs to display", all_urls, default=all_urls[:5] if len(all_urls) > 5 else all_urls)
        
        if selected_urls:
            response_chart = monitor.create_response_time_chart(selected_urls)
            st.plotly_chart(response_chart, use_container_width=True)
        else:
            st.info("Select at least one URL to display response times.")
    else:
        st.info("No response time data available. Run a health check to see results.")

with tabs[2]:
    st.header("URL Availability")
    
    # Availability chart
    if monitor.url_status:
        availability_chart = monitor.create_availability_chart()
        st.plotly_chart(availability_chart, use_container_width=True)
    else:
        st.info("No availability data available. Run a health check to see results.")

with tabs[3]:
    st.header("Detailed View")
    
    # Detailed table of URL status
    if monitor.url_status:
        df = monitor.to_dataframe()
        
        # Format the DataFrame for display
        if not df.empty:
            # Convert timestamps to readable format
            df['last_checked'] = pd.to_datetime(df['last_checked']).dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Format response time
            df['response_time'] = df['response_time'].apply(lambda x: f"{x:.2f}s" if pd.notnull(x) else "N/A")
            
            # Add styling based on status
            def highlight_status(val):
                color_map = {
                    URLHealthStatus.STATUS_OK: 'background-color: #00CC96; color: white',
                    URLHealthStatus.STATUS_SLOW: 'background-color: #FFA15A; color: white',
                    URLHealthStatus.STATUS_ERROR: 'background-color: #EF553B; color: white',
                    URLHealthStatus.STATUS_TIMEOUT: 'background-color: #AB63FA; color: white',
                    URLHealthStatus.STATUS_INVALID: 'background-color: #636EFA; color: white'
                }
                return color_map.get(val, '')
            
            # Display the styled table
            st.dataframe(
                df.style.applymap(highlight_status, subset=['status']),
                use_container_width=True
            )
        else:
            st.info("No detailed data available.")
    else:
        st.info("No URL data available. Run a health check to see results.")

# Run an initial health check if we have URLs but haven't checked them yet
if (st.session_state.custom_urls or 
    ('data_source_urls' in st.session_state and st.session_state.data_source_urls)) and not st.session_state.last_check_time:
    run_health_check()