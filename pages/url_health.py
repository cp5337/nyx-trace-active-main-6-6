"""
URL Health Dashboard
------------------
Real-time link reliability and accessibility metrics for monitoring 
the health and availability of important web resources.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
from urllib.parse import urlparse
import time
import json
import re
import os
from pathlib import Path
import sys

# Add parent directory to path to ensure imports work correctly
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

# Set page config
st.set_page_config(
    page_title="NyxTrace - URL Health Dashboard",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .url-card {
        background-color: rgba(49, 51, 63, 0.7);
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .url-title {
        font-weight: bold;
        margin-bottom: 5px;
    }
    .url-status-healthy {
        color: #4CAF50;
        font-weight: bold;
    }
    .url-status-warning {
        color: #FFC107;
        font-weight: bold;
    }
    .url-status-error {
        color: #F44336;
        font-weight: bold;
    }
    .metric-container {
        background-color: rgba(49, 51, 63, 0.7);
        border-radius: 5px;
        padding: 10px;
        text-align: center;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
    }
    .metric-label {
        font-size: 14px;
        color: #CCC;
    }
</style>
""", unsafe_allow_html=True)

# Helper Functions
def check_url_health(url, timeout=5):
    """
    Check the health of a URL by sending a request and measuring response time and status
    
    Args:
        url: The URL to check
        timeout: Request timeout in seconds
        
    Returns:
        Dictionary with health metrics
    """
    start_time = time.time()
    result = {
        "url": url,
        "timestamp": datetime.now(),
        "status_code": None,
        "response_time": None,
        "is_accessible": False,
        "error": None,
        "content_type": None,
        "content_length": None,
        "domain": urlparse(url).netloc
    }
    
    try:
        response = requests.get(url, timeout=timeout, allow_redirects=True)
        end_time = time.time()
        
        result["status_code"] = response.status_code
        result["response_time"] = (end_time - start_time) * 1000  # convert to ms
        result["is_accessible"] = 200 <= response.status_code < 400
        result["content_type"] = response.headers.get("Content-Type", "Unknown")
        result["content_length"] = len(response.content)
        
        if 300 <= response.status_code < 400:
            result["redirect_url"] = response.url
            
    except requests.exceptions.Timeout:
        result["error"] = "Timeout"
    except requests.exceptions.ConnectionError:
        result["error"] = "Connection Error"
    except requests.exceptions.TooManyRedirects:
        result["error"] = "Too Many Redirects"
    except requests.exceptions.RequestException as e:
        result["error"] = str(e)
    
    return result

def batch_check_urls(urls):
    """
    Check the health of multiple URLs
    
    Args:
        urls: List of URLs to check
        
    Returns:
        DataFrame with health metrics for all URLs
    """
    results = []
    
    with st.spinner(f"Checking health of {len(urls)} URLs..."):
        progress_bar = st.progress(0)
        
        for i, url in enumerate(urls):
            result = check_url_health(url)
            results.append(result)
            progress_bar.progress((i + 1) / len(urls))
    
    return pd.DataFrame(results)

def save_url_health_history(results_df, history_file="data/url_health_history.json"):
    """
    Save URL health check results to history file
    
    Args:
        results_df: DataFrame with health check results
        history_file: Path to history file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(history_file), exist_ok=True)
    
    # Convert DataFrame to dict records
    current_results = results_df.to_dict(orient="records")
    
    # Load existing history if available
    history = []
    if os.path.exists(history_file):
        try:
            with open(history_file, "r") as f:
                history = json.load(f)
        except (json.JSONDecodeError, IOError):
            # If file is corrupted, start with empty history
            history = []
    
    # Convert datetime to string for JSON serialization
    for result in current_results:
        if isinstance(result.get("timestamp"), datetime):
            result["timestamp"] = result["timestamp"].isoformat()
    
    # Append new results
    history.extend(current_results)
    
    # Save history
    with open(history_file, "w") as f:
        json.dump(history, f)

def load_url_health_history(history_file="data/url_health_history.json"):
    """
    Load URL health check history
    
    Args:
        history_file: Path to history file
        
    Returns:
        DataFrame with URL health history
    """
    if not os.path.exists(history_file):
        return pd.DataFrame()
    
    try:
        with open(history_file, "r") as f:
            history = json.load(f)
        
        df = pd.DataFrame(history)
        
        # Convert timestamp string to datetime
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        return df
    except (json.JSONDecodeError, IOError):
        return pd.DataFrame()

def get_url_health_status(response_time, status_code, error=None):
    """
    Determine the health status of a URL based on response time and status code
    
    Args:
        response_time: Response time in ms
        status_code: HTTP status code
        error: Error message if any
        
    Returns:
        Tuple of (status, description, color)
    """
    if error:
        return ("Error", error, "#F44336")
    
    if status_code is None:
        return ("Unknown", "Unable to determine status", "#9E9E9E")
    
    if 200 <= status_code < 300:
        if response_time < 500:
            return ("Healthy", f"Fast response ({response_time:.0f}ms)", "#4CAF50")
        elif response_time < 1000:
            return ("Good", f"Good response ({response_time:.0f}ms)", "#8BC34A")
        elif response_time < 2000:
            return ("Slow", f"Slow response ({response_time:.0f}ms)", "#FFC107")
        else:
            return ("Very Slow", f"Very slow response ({response_time:.0f}ms)", "#FF9800")
    
    if 300 <= status_code < 400:
        return ("Redirect", f"Redirects to another URL (Code: {status_code})", "#03A9F4")
    
    if 400 <= status_code < 500:
        return ("Client Error", f"Client-side error (Code: {status_code})", "#F44336")
    
    if 500 <= status_code < 600:
        return ("Server Error", f"Server-side error (Code: {status_code})", "#F44336")
    
    return ("Unknown", f"Unknown status (Code: {status_code})", "#9E9E9E")

def main():
    """Main function for the URL Health Dashboard"""
    
    st.title("URL Health Dashboard")
    st.markdown("""
    Monitor the health and accessibility of important web resources in real-time.
    Track response times, status codes, and identify potential issues with critical URLs.
    """)
    
    # Initialize session state for URL management
    if "urls" not in st.session_state:
        st.session_state.urls = [
            "https://www.google.com",
            "https://www.github.com",
            "https://www.wikipedia.org"
        ]
    
    if "url_results" not in st.session_state:
        st.session_state.url_results = pd.DataFrame()
    
    if "history_loaded" not in st.session_state:
        st.session_state.url_history = load_url_health_history()
        st.session_state.history_loaded = True
    
    # Create tabs for different dashboard sections
    tab1, tab2, tab3 = st.tabs(["URL Monitor", "Health Trends", "Configuration"])
    
    with tab1:
        # URL Monitor Tab
        st.header("URL Health Monitor")
        
        # Action buttons
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("Check All URLs", key="check_all", use_container_width=True):
                if st.session_state.urls:
                    results_df = batch_check_urls(st.session_state.urls)
                    st.session_state.url_results = results_df
                    
                    # Save results to history
                    save_url_health_history(results_df)
                    
                    # Reload history
                    st.session_state.url_history = load_url_health_history()
                else:
                    st.warning("No URLs to check. Add URLs in the Configuration tab.")
        
        with col2:
            if st.button("Clear Results", key="clear_results", use_container_width=True):
                st.session_state.url_results = pd.DataFrame()
        
        # Health summary metrics
        if not st.session_state.url_results.empty:
            st.markdown("### Health Summary")
            
            results_df = st.session_state.url_results
            
            # Calculate summary metrics
            total_urls = len(results_df)
            healthy_urls = sum((results_df["status_code"] >= 200) & (results_df["status_code"] < 400))
            error_urls = sum((results_df["status_code"] >= 400) | (results_df["status_code"].isna()))
            avg_response_time = results_df["response_time"].mean() if "response_time" in results_df else 0
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-value">{total_urls}</div>
                    <div class="metric-label">Total URLs</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-value" style="color: #4CAF50;">{healthy_urls}</div>
                    <div class="metric-label">Healthy URLs</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-value" style="color: #F44336;">{error_urls}</div>
                    <div class="metric-label">Problem URLs</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-value" style="color: #03A9F4;">{avg_response_time:.0f} ms</div>
                    <div class="metric-label">Avg Response Time</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Display detailed URL health cards
            st.markdown("### URL Health Details")
            
            for _, row in results_df.iterrows():
                url = row["url"]
                status_code = row["status_code"]
                response_time = row["response_time"]
                error = row["error"]
                content_type = row["content_type"]
                content_length = row["content_length"]
                
                # Get status color and description
                status, description, color = get_url_health_status(response_time, status_code, error)
                
                # Create URL card
                st.markdown(f"""
                <div class="url-card">
                    <div class="url-title">{url}</div>
                    <div style="color: {color}; font-weight: bold;">{status}: {description}</div>
                    <div>Status Code: {status_code if status_code else 'N/A'}</div>
                    <div>Content Type: {content_type if content_type else 'N/A'}</div>
                    <div>Content Size: {(f"{content_length/1024:.1f} KB" if content_length else 'N/A')}</div>
                    <div>Last Check: {row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Click 'Check All URLs' to see health metrics for the configured URLs.")
    
    with tab2:
        # Health Trends Tab
        st.header("URL Health Trends")
        
        if st.session_state.url_history.empty:
            st.info("No historical data available. Check some URLs first to collect data.")
        else:
            # Get historical data
            history_df = st.session_state.url_history
            
            # Add filter for URLs
            available_urls = sorted(history_df["url"].unique())
            selected_urls = st.multiselect(
                "Select URLs to display trends", 
                options=available_urls,
                default=available_urls[:3] if len(available_urls) > 3 else available_urls
            )
            
            if not selected_urls:
                st.warning("Please select at least one URL to display trends.")
            else:
                # Filter data based on selected URLs
                filtered_df = history_df[history_df["url"].isin(selected_urls)]
                
                # Response time trends
                st.subheader("Response Time Trends")
                
                # Group by URL and timestamp
                trend_data = filtered_df.groupby(["url", "timestamp"]).agg({
                    "response_time": "mean",
                    "is_accessible": "mean"  # Will give percentage of accessible checks
                }).reset_index()
                
                # Create line chart
                fig = px.line(
                    trend_data, 
                    x="timestamp", 
                    y="response_time", 
                    color="url",
                    labels={"response_time": "Response Time (ms)", "timestamp": "Date"},
                    title="Response Time Trends"
                )
                
                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Response Time (ms)",
                    legend_title="URL",
                    height=400,
                    margin=dict(l=20, r=20, t=30, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Availability trends
                st.subheader("Availability Trends")
                
                # Create availability chart
                fig = px.line(
                    trend_data, 
                    x="timestamp", 
                    y="is_accessible", 
                    color="url",
                    labels={"is_accessible": "Availability (%)", "timestamp": "Date"},
                    title="URL Availability Trends"
                )
                
                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Availability",
                    legend_title="URL",
                    height=400,
                    margin=dict(l=20, r=20, t=30, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', range=[0, 1])
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Status code distribution
                st.subheader("Status Code Distribution")
                
                # Count status codes by URL
                status_counts = filtered_df.groupby(["url", "status_code"]).size().reset_index(name="count")
                
                # Create status code distribution chart
                fig = px.bar(
                    status_counts, 
                    x="status_code", 
                    y="count", 
                    color="url",
                    barmode="group",
                    labels={"status_code": "HTTP Status Code", "count": "Occurrences"},
                    title="Status Code Distribution"
                )
                
                fig.update_layout(
                    xaxis_title="HTTP Status Code",
                    yaxis_title="Number of Occurrences",
                    legend_title="URL",
                    height=400,
                    margin=dict(l=20, r=20, t=30, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed statistics table
                st.subheader("URL Health Statistics")
                
                # Calculate statistics by URL
                stats_df = filtered_df.groupby("url").agg({
                    "response_time": ["mean", "min", "max", "std"],
                    "is_accessible": "mean",
                    "timestamp": ["min", "max", "count"]
                }).reset_index()
                
                # Flatten multi-level columns
                stats_df.columns = [
                    "_".join(col).strip("_") for col in stats_df.columns.values
                ]
                
                # Rename columns for clarity
                stats_df = stats_df.rename(columns={
                    "response_time_mean": "Avg Response (ms)",
                    "response_time_min": "Min Response (ms)",
                    "response_time_max": "Max Response (ms)",
                    "response_time_std": "Std Dev (ms)",
                    "is_accessible_mean": "Availability (%)",
                    "timestamp_min": "First Check",
                    "timestamp_max": "Last Check",
                    "timestamp_count": "Total Checks"
                })
                
                # Format numeric columns
                for col in ["Avg Response (ms)", "Min Response (ms)", "Max Response (ms)", "Std Dev (ms)"]:
                    if col in stats_df.columns:
                        stats_df[col] = stats_df[col].round(1)
                
                # Format availability as percentage
                if "Availability (%)" in stats_df.columns:
                    stats_df["Availability (%)"] = (stats_df["Availability (%)"] * 100).round(1)
                
                # Display the table
                st.dataframe(stats_df, use_container_width=True)
    
    with tab3:
        # Configuration Tab
        st.header("URL Configuration")
        
        # URL Management
        st.subheader("Manage URLs")
        
        # Add new URL
        with st.form("add_url_form"):
            new_url = st.text_input("Add New URL", placeholder="https://example.com")
            
            col1, col2 = st.columns([1, 3])
            with col1:
                submit_button = st.form_submit_button("Add URL", use_container_width=True)
            
            if submit_button and new_url:
                # Validate URL
                if not new_url.startswith(("http://", "https://")):
                    new_url = "https://" + new_url
                
                if new_url not in st.session_state.urls:
                    st.session_state.urls.append(new_url)
                    st.success(f"Added URL: {new_url}")
                else:
                    st.warning("URL already exists in the list.")
        
        # URL list management
        st.subheader("URL List")
        
        if not st.session_state.urls:
            st.info("No URLs configured. Add URLs using the form above.")
        else:
            for i, url in enumerate(st.session_state.urls):
                col1, col2 = st.columns([5, 1])
                
                with col1:
                    st.text(url)
                
                with col2:
                    if st.button("Remove", key=f"remove_{i}", use_container_width=True):
                        st.session_state.urls.remove(url)
                        st.experimental_rerun()
        
        # Import/Export URLs
        st.subheader("Import/Export URLs")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export URLs
            if st.button("Export URLs", use_container_width=True):
                # Convert URLs to CSV format
                urls_df = pd.DataFrame({"url": st.session_state.urls})
                csv = urls_df.to_csv(index=False).encode("utf-8")
                
                # Create download button
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="url_list.csv",
                    mime="text/csv"
                )
        
        with col2:
            # Import URLs
            uploaded_file = st.file_uploader("Import URLs from CSV", type=["csv"])
            
            if uploaded_file is not None:
                try:
                    # Read CSV file
                    urls_df = pd.read_csv(uploaded_file)
                    
                    if "url" in urls_df.columns:
                        # Extract URLs
                        imported_urls = urls_df["url"].tolist()
                        
                        # Add new URLs
                        for url in imported_urls:
                            if url not in st.session_state.urls:
                                st.session_state.urls.append(url)
                        
                        st.success(f"Imported {len(imported_urls)} URLs from CSV.")
                    else:
                        st.error("CSV file must contain a 'url' column.")
                
                except Exception as e:
                    st.error(f"Error importing URLs: {str(e)}")
        
        # Advanced settings
        st.subheader("Advanced Settings")
        
        with st.expander("Health Check Settings"):
            col1, col2 = st.columns(2)
            
            with col1:
                timeout = st.number_input("Request Timeout (seconds)", min_value=1, max_value=30, value=5)
            
            with col2:
                check_interval = st.number_input("Scheduled Check Interval (minutes)", 
                                              min_value=5, max_value=1440, value=60)
                
            # These settings would be used in a production implementation with scheduled checks

if __name__ == "__main__":
    main()