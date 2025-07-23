"""
NyxTrace CTAS Command Center
---------------------------
Main entry point for the NyxTrace CTAS Command Center.
Provides access to the Convergent Threat Analysis System (CTAS),
specializing in geospatial intelligence and threat analysis.
"""

import streamlit as st

# Set page config must be the first Streamlit command
st.set_page_config(
    page_title="CTAS Command Center",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded",
)

import os
import sys
import logging
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Add current directory to path to ensure imports work correctly
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))


# Custom CSS for better styling
def apply_custom_css():
    # Load global CSS from file if it exists
    css_file = Path(__file__).parent / ".streamlit" / "style.css"
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
    # Additional custom CSS for specific components
    st.markdown(
        """
    <style>
        /* Main header styling */
        .main-header {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-weight: 600;
            color: #4D96FF;
            margin-bottom: 0.5rem;
            font-size: 2rem;
            letter-spacing: -0.5px;
        }
        
        /* Sidebar menu items */
        .sidebar-menu-item {
            font-size: 1.1rem;
            font-weight: 500;
            letter-spacing: 0.3px;
            text-transform: uppercase;
            margin: 0.5rem 0;
            padding: 0.5rem 0;
            border-left: 3px solid transparent;
            transition: all 0.2s ease;
        }
        
        .sidebar-menu-item:hover {
            border-left: 3px solid #4D96FF;
            padding-left: 0.5rem;
        }
        
        .sidebar-menu-item.active {
            border-left: 3px solid #4D96FF;
            padding-left: 0.5rem;
            color: #4D96FF;
        }
        
        /* Status indicator pill */
        .status-pill {
            display: inline-block;
            padding: 0.25rem 0.6rem;
            border-radius: 1rem;
            font-size: 0.8rem;
            font-weight: 600;
            margin-left: 0.5rem;
        }
        
        .status-normal {
            background-color: #00CC96;
            color: #FFFFFF;
        }
        
        .status-elevated {
            background-color: #FFCC29;
            color: #262730;
        }
        
        .status-high {
            background-color: #FF6B6B;
            color: #FFFFFF;
        }
        
        /* Section headers */
        .section-header {
            font-size: 1.5rem;
            font-weight: 600;
            margin: 1rem 0 0.5rem 0;
            color: #FAFAFA;
            border-bottom: 1px solid #444444;
            padding-bottom: 0.5rem;
        }
        
        /* Timeline styling */
        .timeline-container {
            border-left: 2px solid #444444;
            padding-left: 15px;
            margin-left: 5px;
        }
        
        /* Card styling */
        .info-card {
            background-color: #1A1C24;
            border-radius: 8px;
            padding: 1rem;
            border: 1px solid #333333;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            margin-bottom: 0.5rem;
        }
        
        /* Fix white space issues */
        h1, h2, h3, p, li, a {
            color: white !important;
            margin-bottom: 0.5rem !important;
        }
        
        /* Remove extra spacing */
        .css-1544g2n {
            padding-top: 1rem !important;
        }
        
        .stMarkdown p {
            margin-bottom: 0.5rem !important;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )


def create_sidebar():
    """Create and render the sidebar navigation"""
    with st.sidebar:
        # Removed image to fix layout issues
        st.markdown(
            "<h1 style='font-size: 1.5rem; margin-bottom: 2rem;'>CTAS COMMAND CENTER</h1>",
            unsafe_allow_html=True,
        )

        # System status indicator
        st.markdown(
            """
        <div style="margin-bottom: 2rem;">
            <span style="font-size: 0.9rem; color: #AAAAAA;">SYSTEM STATUS:</span>
            <span class="status-pill status-elevated">ELEVATED</span>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Navigation menu
        st.markdown(
            "<div class='sidebar-menu-item active'>📊 DASHBOARD</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='sidebar-menu-item'>🧩 NODE CARDS</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='sidebar-menu-item'>🌐 GEO INTELLIGENCE</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='sidebar-menu-item'>🔍 OSINT COLLECTOR</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='sidebar-menu-item'>📈 ANALYSIS TOOLS</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='sidebar-menu-item'>📝 REPORTS</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='sidebar-menu-item'>⚙️ SETTINGS</div>",
            unsafe_allow_html=True,
        )

        # Add more space between menu and user status
        st.markdown("<div style='height: 60px;'></div>", unsafe_allow_html=True)

        # User authentication status - not using absolute positioning
        st.markdown(
            """
        <div style="width: 100%; padding-right: 1rem; margin-top: 30px;">
            <div style="font-size: 0.8rem; color: #AAAAAA;">LOGGED IN AS:</div>
            <div style="font-size: 0.9rem; font-weight: 500;">ANALYST_ALPHA</div>
            <div style="font-size: 0.8rem; color: #00CC96;">● ONLINE</div>
        </div>
        """,
            unsafe_allow_html=True,
        )


def display_metrics():
    """Display key metrics in the dashboard with improved styling"""
    st.markdown(
        """
    <div style="display: flex; justify-content: space-between; margin-bottom: 1rem;">
        <div style="background-color: #1A1C24; border-radius: 8px; padding: 1rem; width: 23%; border: 1px solid #333333;">
            <div style="font-size: 0.8rem; color: #AAAAAA;">ACTIVE INCIDENTS</div>
            <div style="font-size: 1.8rem; font-weight: 600; color: #FAFAFA;">24</div>
            <div style="font-size: 0.8rem; color: #FF6B6B;">+3 from yesterday</div>
        </div>
        <div style="background-color: #1A1C24; border-radius: 8px; padding: 1rem; width: 23%; border: 1px solid #333333;">
            <div style="font-size: 0.8rem; color: #AAAAAA;">INTELLIGENCE SOURCES</div>
            <div style="font-size: 1.8rem; font-weight: 600; color: #FAFAFA;">16</div>
            <div style="font-size: 0.8rem; color: #00CC96;">All operational</div>
        </div>
        <div style="background-color: #1A1C24; border-radius: 8px; padding: 1rem; width: 23%; border: 1px solid #333333;">
            <div style="font-size: 0.8rem; color: #AAAAAA;">THREAT LEVEL</div>
            <div style="font-size: 1.8rem; font-weight: 600; color: #FFCC29;">ELEVATED</div>
            <div style="font-size: 0.8rem; color: #FFCC29;">Increased activity detected</div>
        </div>
        <div style="background-color: #1A1C24; border-radius: 8px; padding: 1rem; width: 23%; border: 1px solid #333333;">
            <div style="font-size: 0.8rem; color: #AAAAAA;">SYSTEM STATUS</div>
            <div style="font-size: 1.8rem; font-weight: 600; color: #00CC96;">OPERATIONAL</div>
            <div style="font-size: 0.8rem; color: #00CC96;">All systems nominal</div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def create_threat_map():
    """Create and display the global threat map with improved styling"""
    st.markdown(
        "<div class='section-header'>Global Threat Map</div>",
        unsafe_allow_html=True,
    )

    # Create a more comprehensive dataset for map
    map_data = pd.DataFrame(
        {
            "lat": [
                40.7128,
                34.0522,
                41.8781,
                37.7749,
                51.5074,
                35.6762,
                19.4326,
                48.8566,
                55.7558,
                25.2048,
                -33.8688,
                -23.5505,
                28.6139,
                1.3521,
                30.0444,
                -34.6037,
                -26.2041,
                31.2304,
                39.9042,
                52.5200,
            ],
            "lon": [
                -74.0060,
                -118.2437,
                -87.6298,
                -122.4194,
                -0.1278,
                139.6503,
                -99.1332,
                2.3522,
                37.6176,
                55.2708,
                151.2093,
                -46.6333,
                77.2090,
                103.8198,
                31.2357,
                -58.3816,
                28.0473,
                121.4737,
                116.4074,
                13.4050,
            ],
            "location": [
                "New York",
                "Los Angeles",
                "Chicago",
                "San Francisco",
                "London",
                "Tokyo",
                "Mexico City",
                "Paris",
                "Moscow",
                "Dubai",
                "Sydney",
                "São Paulo",
                "New Delhi",
                "Singapore",
                "Cairo",
                "Buenos Aires",
                "Johannesburg",
                "Shanghai",
                "Beijing",
                "Berlin",
            ],
            "threat_level": [
                0.8,
                0.6,
                0.4,
                0.7,
                0.5,
                0.3,
                0.9,
                0.4,
                0.7,
                0.5,
                0.4,
                0.8,
                0.6,
                0.3,
                0.7,
                0.5,
                0.6,
                0.4,
                0.6,
                0.3,
            ],
            "alert_count": [
                12,
                8,
                5,
                9,
                6,
                4,
                15,
                5,
                10,
                7,
                6,
                11,
                8,
                4,
                9,
                7,
                8,
                5,
                8,
                4,
            ],
            "incident_type": [
                "Cyber",
                "Physical",
                "Cyber",
                "Cyber",
                "Cyber",
                "Physical",
                "Cartel",
                "Cyber",
                "Cyber",
                "Physical",
                "Cyber",
                "Cartel",
                "Physical",
                "Cyber",
                "Physical",
                "Cartel",
                "Physical",
                "Cyber",
                "Cyber",
                "Cyber",
            ],
        }
    )

    # Optimize marker sizing for better visualization
    marker_sizes = [max(8, tl * 20) for tl in map_data["threat_level"]]

    # Create the map with enhanced visual elements
    fig = px.scatter_mapbox(
        map_data,
        lat="lat",
        lon="lon",
        hover_name="location",
        hover_data=["threat_level", "alert_count", "incident_type"],
        size=marker_sizes,
        zoom=1,
        color="threat_level",
        color_continuous_scale=["green", "yellow", "red"],
        mapbox_style="carto-darkmatter",
        opacity=0.8,
        custom_data=[
            "location",
            "threat_level",
            "alert_count",
            "incident_type",
        ],
    )

    # Customize hover template with incident type
    fig.update_traces(
        hovertemplate="<b>%{customdata[0]}</b><br>Threat Level: %{customdata[1]:.1f}<br>Alerts: %{customdata[2]}<br>Type: %{customdata[3]}<extra></extra>"
    )

    # Add glowing effect for high threat levels
    for i, row in map_data.iterrows():
        if row["threat_level"] > 0.6:
            fig.add_trace(
                go.Scattermapbox(
                    lat=[row["lat"]],
                    lon=[row["lon"]],
                    mode="markers",
                    marker=dict(
                        size=max(30, row["threat_level"] * 50),
                        color="rgba(255, 0, 0, 0.2)",
                        opacity=0.5,
                    ),
                    hoverinfo="none",
                    showlegend=False,
                )
            )

    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        coloraxis_showscale=True,
        coloraxis_colorbar=dict(
            title="Threat Level",
            tickvals=[0.2, 0.5, 0.8],
            ticktext=["Low", "Medium", "High"],
            len=0.5,
            yanchor="top",
            y=0.99,
            titleside="right",
        ),
        height=500,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(26,28,36,0.8)",
            bordercolor="#333333",
        ),
    )

    st.plotly_chart(fig, use_container_width=True)


def display_incident_timeline():
    """Display the incident timeline with improved styling"""
    st.markdown(
        "<div class='section-header'>Incident Timeline</div>",
        unsafe_allow_html=True,
    )

    # Enhanced timeline data with more comprehensive incident information
    current_date = datetime.now().strftime("%Y-%m-%d")
    timeline_data = [
        {
            "time": "08:30",
            "date": current_date,
            "event": "Cyber intrusion detected",
            "type": "cyber",
            "details": "Multiple failed authentication attempts followed by successful privilege escalation",
            "severity": "high",
            "location": "New York",
            "status": "Active",
            "team": "SOC Team A",
        },
        {
            "time": "10:15",
            "date": current_date,
            "event": "Physical security breach",
            "type": "physical",
            "details": "Unauthorized access to secure area detected on camera #14",
            "severity": "medium",
            "location": "London",
            "status": "Investigating",
            "team": "Security Team C",
        },
        {
            "time": "12:45",
            "date": current_date,
            "event": "Data exfiltration attempt",
            "type": "cyber",
            "details": "Large outbound data transfer detected from research database",
            "severity": "high",
            "location": "San Francisco",
            "status": "Containment",
            "team": "SOC Team B",
        },
        {
            "time": "14:20",
            "date": current_date,
            "event": "Suspicious network traffic",
            "type": "cyber",
            "details": "Unusual connection patterns from internal workstation to external IP",
            "severity": "medium",
            "location": "Chicago",
            "status": "Investigating",
            "team": "SOC Team A",
        },
        {
            "time": "16:10",
            "date": current_date,
            "event": "Unauthorized facility access",
            "type": "physical",
            "details": "Badge reader detected invalid credentials with repeated attempts",
            "severity": "medium",
            "location": "Tokyo",
            "status": "Resolved",
            "team": "Security Team B",
        },
        {
            "time": "17:35",
            "date": current_date,
            "event": "Social engineering attempt",
            "type": "human",
            "details": "Phone call to help desk requesting password reset without verification",
            "severity": "low",
            "location": "Berlin",
            "status": "Closed",
            "team": "SOC Team C",
        },
    ]

    st.markdown("<div class='timeline-container'>", unsafe_allow_html=True)

    for event in timeline_data:
        # Set color based on type and severity
        if event["type"] == "cyber":
            color = "#FF6B6B" if event["severity"] == "high" else "#FFCC29"
            icon = "🖥️"
        elif event["type"] == "physical":
            color = "#FA8072" if event["severity"] == "high" else "#FFA15A"
            icon = "🚪"
        else:
            color = "#4D96FF"
            icon = "👤"

        # Create status badge based on status
        status_color = {
            "Active": "#FF6B6B",  # Red
            "Investigating": "#FFCC29",  # Yellow
            "Containment": "#FFA500",  # Orange
            "Resolved": "#4CAF50",  # Green
            "Closed": "#808080",  # Gray
        }.get(event["status"], "#FFFFFF")

        # Create severity badge
        severity_badge = ""
        if event["severity"] == "high":
            severity_badge = "<span style='background-color: #FF6B6B; color: white; padding: 2px 6px; border-radius: 4px; font-size: 0.7rem; margin-left: 5px;'>HIGH</span>"
        elif event["severity"] == "medium":
            severity_badge = "<span style='background-color: #FFCC29; color: #262730; padding: 2px 6px; border-radius: 4px; font-size: 0.7rem; margin-left: 5px;'>MEDIUM</span>"
        else:
            severity_badge = "<span style='background-color: #4D96FF; color: white; padding: 2px 6px; border-radius: 4px; font-size: 0.7rem; margin-left: 5px;'>LOW</span>"

        # Create status badge
        status_badge = f"<span style='background-color: {status_color}; color: white; padding: 2px 6px; border-radius: 4px; font-size: 0.7rem; margin-left: 5px;'>{event['status'].upper()}</span>"

        # Create event entry with improved styling and expanded information
        st.markdown(
            f"""
        <div style="position: relative; margin-bottom: 25px;">
            <div style="position: absolute; left: -20px; top: 0; width: 10px; height: 10px; border-radius: 50%; background-color: {color}; border: 2px solid #1A1C24;"></div>
            <div style="background-color: #1A1C24; border-radius: 8px; padding: 12px; border-left: 3px solid {color};">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <div style="font-size: 0.8rem; color: rgba(255, 255, 255, 0.6);">{event['time']} {severity_badge}</div>
                    <div style="font-size: 0.8rem;">{status_badge}</div>
                </div>
                <div style="font-size: 1rem; font-weight: 500; margin-bottom: 4px;">{icon} {event['event']}</div>
                <div style="font-size: 0.8rem; color: rgba(255, 255, 255, 0.8); margin-bottom: 4px;">{event['details']}</div>
                <div style="display: flex; justify-content: space-between; margin-top: 8px; font-size: 0.75rem; color: rgba(255, 255, 255, 0.6);">
                    <div>📍 {event['location']}</div>
                    <div>👥 {event['team']}</div>
                </div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


def create_threat_categories_chart():
    """Create the cyber threat categories pie chart with improved styling"""
    # Sample data for cyber threat categories
    categories = [
        "Malware",
        "Phishing",
        "DDoS",
        "Insider",
        "Zero-day",
        "Ransomware",
    ]
    values = [42, 23, 15, 8, 12, 18]
    colors = ["#4D96FF", "#FF6B6B", "#00CC96", "#AB63FA", "#FFA15A", "#EF553B"]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=categories,
                values=values,
                hole=0.4,
                marker=dict(colors=colors, line=dict(color="#0E1117", width=1)),
                textinfo="label+percent",
                insidetextorientation="radial",
                hoverinfo="label+value+percent",
                textfont=dict(size=12, color="white"),
                rotation=45,
            )
        ]
    )

    fig.update_layout(
        title={
            "text": "Cyber Threat Categories",
            "y": 0.96,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": dict(size=18, color="#FAFAFA"),
        },
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(size=10),
        ),
        height=350,
        paper_bgcolor="#1A1C24",
        plot_bgcolor="#1A1C24",
        margin=dict(t=30, b=0, l=0, r=0),
        annotations=[
            dict(
                text="118<br>Total<br>Incidents",
                x=0.5,
                y=0.5,
                font=dict(size=14, color="white", family="Arial, sans-serif"),
                showarrow=False,
            )
        ],
    )

    return fig


def create_attack_trend_chart():
    """Create the attack trend line chart with improved styling"""
    # Sample data for cyber attack trend - use 'ME' instead of 'M' for month end frequency
    dates = pd.date_range(start="2023-01-01", periods=10, freq="ME")
    trend_data = pd.DataFrame(
        {
            "date": dates,
            "incidents": [15, 18, 22, 20, 25, 30, 28, 35, 32, 38],
            "severity": [
                3.2,
                3.5,
                3.7,
                3.4,
                3.9,
                4.1,
                3.8,
                4.3,
                4.0,
                4.5,
            ],  # Average severity score
        }
    )

    # Create a figure with secondary Y-axis
    fig = go.Figure()

    # Add incidents line
    fig.add_trace(
        go.Scatter(
            x=trend_data["date"],
            y=trend_data["incidents"],
            name="Incident Count",
            mode="lines+markers",
            line=dict(color="#4D96FF", width=3),
            marker=dict(size=8, color="#4D96FF", symbol="circle"),
            hovertemplate="<b>%{x|%b %Y}</b><br>Incidents: %{y}<extra></extra>",
        )
    )

    # Add severity line
    fig.add_trace(
        go.Scatter(
            x=trend_data["date"],
            y=trend_data["severity"],
            name="Avg. Severity (1-5)",
            mode="lines+markers",
            line=dict(color="#FF6B6B", width=2, dash="dash"),
            marker=dict(size=6, color="#FF6B6B", symbol="diamond"),
            yaxis="y2",
            hovertemplate="<b>%{x|%b %Y}</b><br>Severity: %{y:.1f}/5.0<extra></extra>",
        )
    )

    # Add a trend line
    z = np.polyfit(range(len(trend_data)), trend_data["incidents"], 1)
    p = np.poly1d(z)
    fig.add_trace(
        go.Scatter(
            x=trend_data["date"],
            y=p(range(len(trend_data))),
            name="Trend",
            mode="lines",
            line=dict(color="#FFCC29", width=2, dash="dot"),
            hoverinfo="skip",
        )
    )

    # Update layout with dual Y axes
    fig.update_layout(
        title={
            "text": "Attack Trend (2023-2024)",
            "y": 0.96,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": dict(size=18, color="#FAFAFA"),
        },
        xaxis=dict(
            title="Date",
            gridcolor="#262730",
            showgrid=True,
            tickformat="%b<br>%Y",
            dtick="M2",
        ),
        yaxis=dict(
            title="Incidents",
            gridcolor="#262730",
            showgrid=True,
            range=[0, max(trend_data["incidents"]) * 1.2],
        ),
        yaxis2=dict(
            title="Severity Score",
            range=[1, 5],
            overlaying="y",
            side="right",
            gridcolor="#262730",
            showgrid=False,
            zeroline=False,
        ),
        height=350,
        paper_bgcolor="#1A1C24",
        plot_bgcolor="#1A1C24",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        ),
        margin=dict(l=10, r=10, t=50, b=80),
        hovermode="x unified",
    )

    # Add a marker for the most recent data point
    fig.add_trace(
        go.Scatter(
            x=[trend_data["date"].iloc[-1]],
            y=[trend_data["incidents"].iloc[-1]],
            mode="markers",
            marker=dict(
                color="#FAFAFA",
                size=12,
                symbol="circle",
                line=dict(color="#4D96FF", width=2),
            ),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Add annotations for latest values
    fig.add_annotation(
        x=trend_data["date"].iloc[-1],
        y=trend_data["incidents"].iloc[-1] + 5,
        text=f"Latest: {trend_data['incidents'].iloc[-1]} incidents",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#4D96FF",
        font=dict(size=12, color="#FAFAFA"),
        align="center",
    )

    return fig


def display_threat_analysis():
    """Display the threat analysis tabs and charts with improved styling"""
    st.markdown(
        "<div class='section-header'>Threat Analysis</div>",
        unsafe_allow_html=True,
    )

    threat_tabs = st.tabs(
        ["Cyber Threats", "Physical Threats", "Cartel Activity"]
    )

    with threat_tabs[0]:
        col1, col2 = st.columns(2)

        with col1:
            fig = create_threat_categories_chart()
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = create_attack_trend_chart()
            st.plotly_chart(fig, use_container_width=True)

        # Additional information row
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                """
            <div class="info-card">
                <h4 style="margin-top: 0; font-size: 1rem;">Top Targeted Sectors</h4>
                <ol style="margin-left: 1rem; padding-left: 0;">
                    <li>Financial Services (32%)</li>
                    <li>Healthcare (24%)</li>
                    <li>Government (18%)</li>
                    <li>Energy (15%)</li>
                    <li>Technology (11%)</li>
                </ol>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                """
            <div class="info-card">
                <h4 style="margin-top: 0; font-size: 1rem;">Emerging Threats</h4>
                <ul style="margin-left: 1rem; padding-left: 0;">
                    <li>Supply chain compromise increasing</li>
                    <li>AI-powered phishing campaigns</li>
                    <li>Ransomware-as-a-Service expansion</li>
                </ul>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown(
                """
            <div class="info-card">
                <h4 style="margin-top: 0; font-size: 1rem;">Risk Vectors</h4>
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span>Remote Access:</span>
                    <span style="color: #FF6B6B;">HIGH</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span>Cloud Environments:</span>
                    <span style="color: #FFCC29;">MEDIUM</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span>IoT Devices:</span>
                    <span style="color: #FF6B6B;">HIGH</span>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    with threat_tabs[1]:
        # Physical threats tab
        st.markdown(
            """
        <div class="info-card" style="padding: 2rem; text-align: center;">
            <h3 style="color: #4D96FF;">Physical Threat Analysis</h3>
            <p>Detailed analysis of physical security threats will be displayed here.</p>
            <p>Coming in next update.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with threat_tabs[2]:
        # Cartel activity tab
        st.markdown(
            """
        <div class="info-card" style="padding: 2rem; text-align: center;">
            <h3 style="color: #4D96FF;">Cartel Activity Analysis</h3>
            <p>Detailed analysis of cartel activities and patterns will be displayed here.</p>
            <p>Coming in next update.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Add spacer at the bottom to prevent bunching
    st.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True)


# Main application
def main():
    # Apply custom CSS
    apply_custom_css()

    # Create sidebar
    create_sidebar()

    # Page header with enhanced styling
    st.markdown(
        "<h1 class='main-header'>NyxTrace CTAS Command Center</h1>",
        unsafe_allow_html=True,
    )

    # Display key metrics
    display_metrics()

    # Display map and timeline in columns
    col1, col2 = st.columns([3, 1])

    with col1:
        create_threat_map()

    with col2:
        display_incident_timeline()

    # Display threat analysis
    display_threat_analysis()


# Run the application
if __name__ == "__main__":
    main()
