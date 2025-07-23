"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-URLHEALTH-PAGE-0001                 │
// │ 📁 domain       : OSINT, Web Intelligence, Connectivity    │
// │ 🧠 description  : OSINT URL Health Monitoring Dashboard    │
// │                  with domain visualization and extraction tools │
// │ 🕸️ hash_type    : UUID → CUID-linked                       │
// │ 🔄 parent_node  : NODE_URL_INTERFACE                       │
// │ 🧩 dependencies : streamlit, plotly, networkx              │
// │ 🔧 tool_usage   : Dashboard, OSINT Analysis, Intel Collection │
// │ 📡 input_type   : URLs, text/html, raw URL text            │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : domain-centered intelligence aggregation │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘
OSINT URL Health Monitoring
--------------------------
Real-time link reliability and accessibility metrics for monitoring URLs from OSINT investigations and external data sources.

This dashboard provides:
1. Health status of URLs discovered during OSINT investigations
2. Monitoring of essential external data source endpoints
3. Alerts for inaccessible or problematic links
4. Historical availability tracking of critical resources
5. Domain relationship visualization with network graphing
6. URL extraction capability from text content
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
from typing import Dict, List, Set


# Initialize session state for URL health monitor
if 'url_health_monitor' not in st.session_state:
    st.session_state.url_health_monitor = URLHealthMonitor()

if 'custom_urls' not in st.session_state:
    st.session_state.custom_urls = []
    
if 'last_check_time' not in st.session_state:
    st.session_state.last_check_time = None
    
# Helper function to validate URL
def is_valid_url(url):
    """
    Validate if a string is a properly formatted URL
    
    Args:
        url (str): String to check if it's a valid URL
        
    Returns:
        bool: True if valid URL, False otherwise
    """
    # Check if it's a string first
    if not isinstance(url, str):
        return False
        
    # Regex pattern for URL validation
    url_pattern = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ipv4
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    # Check if URL matches pattern
    return bool(url_pattern.match(url))
    
    # Additional validation could include:
    # - Try to parse with urlparse
    # - Check length
    # - Check for prohibited characters

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
st.title("OSINT URL Health Monitoring")
st.write("Monitor the health and accessibility of URLs discovered during OSINT investigations and external data sources.")

# Sidebar for configuration
with st.sidebar:
    st.header("URL Health Configuration")
    
    # Add custom URLs
    st.subheader("Custom URLs")
    st.text_input("Add URL", key="new_url", on_change=add_custom_url)
    
    # Extract URLs from text
    st.subheader("Extract URLs from Text")
    url_text = st.text_area("Paste text containing URLs", height=100, 
                           help="Paste article text, notes, or any content containing URLs")
    
    if st.button("Extract URLs"):
        if url_text:
            extracted_urls = st.session_state.url_health_monitor.extract_urls_from_text(url_text)
            if extracted_urls:
                st.success(f"Found {len(extracted_urls)} URLs in the text")
                for url in extracted_urls:
                    if url not in st.session_state.custom_urls:
                        st.session_state.custom_urls.append(url)
                st.rerun()
            else:
                st.warning("No valid URLs found in the text")
    
    # Display and manage custom URLs
    if st.session_state.custom_urls:
        st.write("Current custom URLs:")
        for i, url in enumerate(st.session_state.custom_urls):
            cols = st.columns([4, 1])
            cols[0].text(url)
            if cols[1].button("🗑️", key=f"delete_{i}"):
                st.session_state.custom_urls.pop(i)
                st.rerun()
    
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
tabs = st.tabs(["Health Status", "Response Times", "Availability", "Domain Analysis", "Network Visualization", "Detailed View"])

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
    st.header("Domain Analysis")
    
    # Domain-based analysis of URLs
    if monitor.url_status:
        # Get all URLs
        all_urls = list(monitor.url_status.keys())
        
        # Group URLs by domain
        domain_groups = monitor.group_urls_by_domain(all_urls)
        
        # Domain distribution chart
        domain_counts = {domain: len(urls) for domain, urls in domain_groups.items()}
        domain_df = pd.DataFrame({
            'Domain': list(domain_counts.keys()), 
            'URL Count': list(domain_counts.values())
        }).sort_values('URL Count', ascending=False)
        
        # Domain metrics
        st.subheader("Domain Summary")
        cols = st.columns(3)
        cols[0].metric("Total Domains", len(domain_groups))
        cols[1].metric("Top Domain", domain_df['Domain'].iloc[0] if not domain_df.empty else "N/A")
        cols[2].metric("URLs in Top Domain", domain_df['URL Count'].iloc[0] if not domain_df.empty else 0)
        
        # Domain distribution chart
        st.subheader("Domain Distribution")
        fig = px.bar(domain_df, 
                    x='Domain', 
                    y='URL Count',
                    color='URL Count',
                    color_continuous_scale='Viridis',
                    title="URLs by Domain")
        st.plotly_chart(fig, use_container_width=True)
        
        # Domain health summary
        st.subheader("Domain Health Status")
        domain_health = {}
        
        for domain, urls in domain_groups.items():
            status_counts = {
                URLHealthStatus.STATUS_OK: 0,
                URLHealthStatus.STATUS_SLOW: 0,
                URLHealthStatus.STATUS_ERROR: 0,
                URLHealthStatus.STATUS_TIMEOUT: 0,
                URLHealthStatus.STATUS_INVALID: 0
            }
            
            for url in urls:
                if url in monitor.url_status:
                    status = monitor.url_status[url].status
                    status_counts[status] += 1
            
            domain_health[domain] = status_counts
        
        # Create domain health dataframe
        health_data = []
        for domain, counts in domain_health.items():
            health_data.append({
                'Domain': domain,
                'OK': counts[URLHealthStatus.STATUS_OK],
                'Slow': counts[URLHealthStatus.STATUS_SLOW],
                'Error': counts[URLHealthStatus.STATUS_ERROR],
                'Timeout': counts[URLHealthStatus.STATUS_TIMEOUT],
                'Invalid': counts[URLHealthStatus.STATUS_INVALID],
                'Total': sum(counts.values())
            })
        
        health_df = pd.DataFrame(health_data).sort_values('Total', ascending=False)
        
        # Display domain health table
        st.dataframe(health_df, use_container_width=True)
        
        # Domain details expander
        st.subheader("Domain Details")
        for domain, urls in domain_groups.items():
            with st.expander(f"{domain} ({len(urls)} URLs)"):
                for url in urls:
                    status = "Unknown"
                    if url in monitor.url_status:
                        status = monitor.url_status[url].status
                    
                    status_color = {
                        URLHealthStatus.STATUS_OK: "green",
                        URLHealthStatus.STATUS_SLOW: "orange",
                        URLHealthStatus.STATUS_ERROR: "red",
                        URLHealthStatus.STATUS_TIMEOUT: "purple",
                        URLHealthStatus.STATUS_INVALID: "blue"
                    }.get(status, "gray")
                    
                    st.markdown(f"* <span style='color:{status_color}'>{status}</span>: {url}", unsafe_allow_html=True)
    else:
        st.info("No URL data available for domain analysis. Run a health check to see results.")

with tabs[4]:
    st.header("URL Network Visualization")
    
    if monitor.url_status:
        # Get all URLs
        all_urls = list(monitor.url_status.keys())
        
        # Configuration options
        st.subheader("Network Visualization Options")
        col1, col2 = st.columns(2)
        
        with col1:
            # Option to include health status coloring
            include_status = st.checkbox("Color by health status", value=True)
            
        with col2:
            # Set maximum number of URLs to visualize
            max_urls = st.slider("Maximum URLs to visualize", 10, 100, 50,
                                 help="Larger values may slow down the visualization")
        
        # Sample URLs if we have too many
        urls_to_visualize = all_urls
        if len(all_urls) > max_urls:
            urls_to_visualize = all_urls[:max_urls]
            st.info(f"Showing network visualization for {max_urls} out of {len(all_urls)} URLs. Use the slider to adjust.")

        # Visualize the URL network    
        with st.spinner("Generating network visualization..."):
            network_fig = monitor.create_url_network_graph(urls_to_visualize, include_status=include_status)
            st.plotly_chart(network_fig, use_container_width=True)

        # URL root extraction feature
        st.subheader("Dynamic URL Root Extractor")
        url_text = st.text_area("Enter a URL or text containing URLs", 
                               height=100,
                               help="Enter a single URL or paste text containing multiple URLs to extract and visualize their relationships")
        
        if st.button("Extract and Visualize"):
            if url_text:
                # If it's a single URL
                if is_valid_url(url_text):
                    extracted_urls = [url_text]
                else:
                    # Try to extract URLs from text
                    extracted_urls = monitor.extract_urls_from_text(url_text)
                
                if extracted_urls:
                    st.success(f"Found {len(extracted_urls)} URLs for visualization")
                    
                    # Visualize the extracted URLs
                    with st.spinner("Generating visualization for extracted URLs..."):
                        extracted_fig = monitor.create_url_network_graph(extracted_urls, include_status=False)
                        st.plotly_chart(extracted_fig, use_container_width=True)
                    
                    # Show the extracted URLs
                    with st.expander("View Extracted URLs"):
                        for i, url in enumerate(extracted_urls):
                            st.write(f"{i+1}. {url}")
                    
                    # Option to add these URLs to monitoring
                    if st.button("Add Extracted URLs to Monitoring"):
                        for url in extracted_urls:
                            if url not in st.session_state.custom_urls:
                                st.session_state.custom_urls.append(url)
                        st.success(f"Added {len(extracted_urls)} URLs to monitoring")
                        st.rerun()
                else:
                    st.warning("No valid URLs found in the input text")
    else:
        st.info("No URL data available for network visualization. Run a health check to see results.")

with tabs[5]:
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