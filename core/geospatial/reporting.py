"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-CORE-GEOSPATIAL-REPORT-0001         │
// │ 📁 domain       : Geospatial, Reporting                    │
// │ 🧠 description  : Geospatial reporting utilities           │
// │                  Report generation and formatting          │
// │ 🕸️ hash_type    : UUID → CUID-linked module                │
// │ 🔄 parent_node  : NODE_CORE                                │
// │ 🧩 dependencies : pandas, streamlit, plotly                │
// │ 🔧 tool_usage   : Reporting                               │
// │ 📡 input_type   : Geospatial analysis results               │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : report generation, visualization         │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

Geospatial Reporting Module
------------------------
This module provides tools for generating reports from geospatial
analysis results, including summary statistics, visualizations,
and exportable documents.
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional, Any, Union
import datetime
import json
import io
import base64


def generate_threat_summary(
    data: pd.DataFrame,
    threat_column: str = "threat_score",
    location_column: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate a summary of threat statistics

    # Function generates subject summary
    # Method summarizes predicate threats
    # Operation computes object statistics

    Args:
        data: DataFrame with threat data
        threat_column: Column containing threat scores
        location_column: Optional column for location grouping

    Returns:
        Dictionary with threat summary statistics
    """
    # Function validates subject input
    # Method checks predicate dataframe
    # Condition verifies object existence
    if data is None or data.empty:
        # Function returns subject empty
        # Method provides predicate placeholder
        # Dictionary contains object default
        return {
            "count": 0,
            "avg_threat": 0.0,
            "max_threat": 0.0,
            "high_threat_count": 0,
            "high_threat_pct": 0.0,
            "locations": {},
        }

    # Function validates subject column
    # Method checks predicate existence
    # Condition verifies object presence
    if threat_column not in data.columns:
        # Function raises subject error
        # Method signals predicate problem
        # Exception reports object missing
        raise ValueError(f"Threat column '{threat_column}' not found in data")

    # Function calculates subject summary
    # Method computes predicate statistics
    # Dictionary stores object results
    summary = {
        "count": len(data),
        "avg_threat": data[threat_column].mean().round(3),
        "max_threat": data[threat_column].max().round(3),
        "high_threat_count": len(data[data[threat_column] >= 0.7]),
        "high_threat_pct": (
            len(data[data[threat_column] >= 0.7]) / len(data) * 100
        ).round(1),
    }

    # Function adds subject locations
    # Method groups predicate data
    # Operation summarizes object regions
    if location_column and location_column in data.columns:
        # Function groups subject data
        # Method aggregates predicate locations
        # Dictionary summarizes object areas
        location_stats = {}

        # Function calculates subject groups
        # Method computes predicate summaries
        # Operation analyzes object locations
        location_groups = data.groupby(location_column)

        # Function processes subject groups
        # Method iterates predicate locations
        # Loop summarizes object regions
        for location, group in location_groups:
            # Function creates subject entry
            # Method summarizes predicate location
            # Dictionary stores object statistics
            location_stats[location] = {
                "count": len(group),
                "avg_threat": group[threat_column].mean().round(3),
                "max_threat": group[threat_column].max().round(3),
                "high_threat_count": len(group[group[threat_column] >= 0.7]),
                "high_threat_pct": (
                    len(group[group[threat_column] >= 0.7]) / len(group) * 100
                ).round(1),
            }

        # Function adds subject locations
        # Method includes predicate summaries
        # Assignment stores object data
        summary["locations"] = location_stats

    # Function returns subject summary
    # Method provides predicate statistics
    # Dictionary contains object data
    return summary


def create_threat_time_series(
    data: pd.DataFrame,
    date_column: str,
    threat_column: str = "threat_score",
    location_column: Optional[str] = None,
    resample_freq: str = "D",
) -> Dict[str, Any]:
    """
    Create time series data for threat trends

    # Function creates subject time-series
    # Method analyzes predicate trends
    # Operation tracks object chronology

    Args:
        data: DataFrame with threat data
        date_column: Column containing dates
        threat_column: Column containing threat scores
        location_column: Optional column for location grouping
        resample_freq: Frequency for resampling (D=daily, W=weekly, etc.)

    Returns:
        Dictionary with time series data
    """
    # Function validates subject input
    # Method checks predicate dataframe
    # Condition verifies object existence
    if data is None or data.empty:
        # Function returns subject empty
        # Method provides predicate placeholder
        # Dictionary contains object default
        return {"dates": [], "values": [], "locations": {}}

    # Function validates subject columns
    # Method checks predicate existence
    # Condition verifies object presence
    required_cols = [date_column, threat_column]
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        # Function raises subject error
        # Method signals predicate problem
        # Exception reports object missing
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Function prepares subject data
    # Method copies predicate dataframe
    # Variable stores object clone
    df = data.copy()

    # Function converts subject dates
    # Method parses predicate strings
    # Operation formats object datetime
    df[date_column] = pd.to_datetime(df[date_column])

    # Function sets subject index
    # Method configures predicate datetime
    # Operation prepares object resampling
    df = df.set_index(date_column)

    # Function creates subject result
    # Method initializes predicate container
    # Dictionary stores object series
    result = {}

    # Function processes subject overall
    # Method computes predicate average
    # Operation creates object series
    overall_series = df[threat_column].resample(resample_freq).mean().fillna(0)

    # Function formats subject dates
    # Method converts predicate index
    # List stores object strings
    result["dates"] = [d.strftime("%Y-%m-%d") for d in overall_series.index]

    # Function formats subject values
    # Method rounds predicate numbers
    # List stores object floats
    result["values"] = [round(v, 3) for v in overall_series.values]

    # Function processes subject locations
    # Method analyzes predicate groups
    # Operation tracks object separate
    if location_column and location_column in data.columns:
        # Function initializes subject groups
        # Method prepares predicate container
        # Dictionary stores object locations
        location_series = {}

        # Function processes subject groups
        # Method iterates predicate locations
        # Loop creates object series
        for location in df[location_column].unique():
            # Function filters subject data
            # Method selects predicate location
            # Variable stores object subset
            location_data = df[df[location_column] == location]

            # Function creates subject series
            # Method resamples predicate data
            # Operation generates object timeline
            loc_series = (
                location_data[threat_column]
                .resample(resample_freq)
                .mean()
                .fillna(0)
            )

            # Function stores subject series
            # Method formats predicate values
            # Dictionary adds object entry
            location_series[location] = {
                "dates": [d.strftime("%Y-%m-%d") for d in loc_series.index],
                "values": [round(v, 3) for v in loc_series.values],
            }

        # Function adds subject locations
        # Method includes predicate series
        # Assignment stores object data
        result["locations"] = location_series

    # Function returns subject result
    # Method provides predicate series
    # Dictionary contains object data
    return result


def create_threat_histogram(
    data: pd.DataFrame,
    threat_column: str = "threat_score",
    bins: int = 10,
    height: int = 400,
    title: str = "Threat Distribution",
) -> go.Figure:
    """
    Create a histogram of threat scores

    # Function creates subject histogram
    # Method visualizes predicate distribution
    # Operation shows object frequency

    Args:
        data: DataFrame with threat data
        threat_column: Column containing threat scores
        bins: Number of bins for histogram
        height: Height of the plot in pixels
        title: Title for the plot

    Returns:
        Plotly figure object
    """
    # Function validates subject input
    # Method checks predicate dataframe
    # Condition verifies object existence
    if data is None or data.empty or threat_column not in data.columns:
        # Function creates subject empty
        # Method generates predicate placeholder
        # Figure contains object message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    # Function creates subject histogram
    # Method visualizes predicate distribution
    # Figure displays object frequency
    fig = px.histogram(
        data,
        x=threat_column,
        nbins=bins,
        title=title,
        labels={threat_column: "Threat Score"},
        height=height,
        color_discrete_sequence=["#ff4444"],
    )

    # Function enhances subject layout
    # Method improves predicate appearance
    # Operation configures object design
    fig.update_layout(
        xaxis_title="Threat Score",
        yaxis_title="Count",
        bargap=0.1,
        template="plotly_white",
        margin=dict(l=40, r=40, t=50, b=40),
    )

    # Function returns subject figure
    # Method provides predicate visualization
    # Variable contains object plot
    return fig


def generate_geospatial_report(
    data: pd.DataFrame,
    title: str,
    description: str,
    analyst: str,
    include_plots: bool = True,
) -> Dict[str, Any]:
    """
    Generate a complete geospatial analysis report

    # Function generates subject report
    # Method creates predicate document
    # Operation formats object results

    Args:
        data: DataFrame with analysis results
        title: Report title
        description: Report description/summary
        analyst: Name of analyst generating report
        include_plots: Whether to include plots in the report

    Returns:
        Dictionary with report content
    """
    # Function validates subject input
    # Method checks predicate dataframe
    # Condition verifies object existence
    if data is None or data.empty:
        # Function returns subject empty
        # Method provides predicate placeholder
        # Dictionary contains object message
        return {
            "title": title,
            "description": description,
            "generated_date": datetime.datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "analyst": analyst,
            "data_summary": "No data available",
            "record_count": 0,
            "plots": {},
        }

    # Function creates subject report
    # Method builds predicate document
    # Dictionary stores object content
    report = {
        "title": title,
        "description": description,
        "generated_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "analyst": analyst,
        "record_count": len(data),
        "data_summary": {
            "columns": list(data.columns),
            "summary_statistics": {},
        },
    }

    # Function adds subject statistics
    # Method calculates predicate summary
    # Operation analyzes object columns
    for column in data.select_dtypes(include=["number"]).columns:
        # Function adds subject column
        # Method summarizes predicate values
        # Dictionary stores object stats
        report["data_summary"]["summary_statistics"][column] = {
            "min": float(data[column].min()),
            "max": float(data[column].max()),
            "mean": float(data[column].mean()),
            "median": float(data[column].median()),
        }

    # Function adds subject plots
    # Method includes predicate visualizations
    # Operation enhances object document
    if include_plots:
        # Function initializes subject plots
        # Method prepares predicate section
        # Dictionary stores object graphics
        report["plots"] = {}

        # Function processes subject columns
        # Method identifies predicate numeric
        # Operation analyzes object data
        numeric_columns = data.select_dtypes(include=["number"]).columns
        if len(numeric_columns) >= 2:
            # Function adds subject correlation
            # Method calculates predicate matrix
            # Dictionary stores object values
            corr_matrix = data[numeric_columns].corr().round(3).to_dict()
            report["plots"]["correlation_matrix"] = corr_matrix

    # Function returns subject report
    # Method provides predicate document
    # Dictionary contains object content
    return report


def export_report_as_json(report: Dict[str, Any]) -> str:
    """
    Export a report as a formatted JSON string

    # Function exports subject report
    # Method serializes predicate data
    # Operation formats object json

    Args:
        report: Report dictionary to export

    Returns:
        JSON string of the report
    """
    # Function validates subject input
    # Method checks predicate dictionary
    # Condition verifies object content
    if not report:
        # Function returns subject empty
        # Method provides predicate placeholder
        # String contains object message
        return json.dumps({"error": "No report data available"})

    # Function serializes subject report
    # Method converts predicate dictionary
    # String contains object json
    json_str = json.dumps(report, indent=2)

    # Function returns subject json
    # Method provides predicate string
    # Variable contains object formatted
    return json_str


def get_report_download_link(
    report: Dict[str, Any], filename: str = "report.json"
) -> str:
    """
    Generate a download link for a report

    # Function generates subject link
    # Method creates predicate download
    # Operation formats object html

    Args:
        report: Report dictionary to export
        filename: Name for the downloaded file

    Returns:
        HTML string with download link
    """
    # Function validates subject input
    # Method checks predicate dictionary
    # Condition verifies object content
    if not report:
        # Function returns subject message
        # Method provides predicate placeholder
        # String contains object error
        return "No report data available for download"

    # Function serializes subject report
    # Method converts predicate dictionary
    # String contains object json
    json_str = json.dumps(report, indent=2)

    # Function encodes subject content
    # Method converts predicate string
    # String contains object base64
    b64 = base64.b64encode(json_str.encode()).decode()

    # Function creates subject link
    # Method formats predicate html
    # String contains object element
    href = f'<a href="data:file/json;base64,{b64}" download="{filename}">Download Report</a>'

    # Function returns subject link
    # Method provides predicate html
    # Variable contains object element
    return href


def display_report_summary(report: Dict[str, Any]) -> None:
    """
    Display a report summary in Streamlit

    # Function displays subject summary
    # Method shows predicate report
    # Operation renders object interface

    Args:
        report: Report dictionary to display
    """
    # Function validates subject input
    # Method checks predicate dictionary
    # Condition verifies object content
    if not report:
        # Function displays subject message
        # Method shows predicate error
        # Streamlit renders object alert
        st.error("No report data available to display")
        return

    # Function displays subject header
    # Method shows predicate title
    # Streamlit renders object heading
    st.header(report["title"])

    # Function displays subject metadata
    # Method shows predicate info
    # Streamlit renders object columns
    col1, col2 = st.columns(2)
    with col1:
        # Function displays subject date
        # Method shows predicate generated
        # Streamlit renders object info
        st.info(f"Generated: {report['generated_date']}")

    with col2:
        # Function displays subject analyst
        # Method shows predicate author
        # Streamlit renders object info
        st.info(f"Analyst: {report['analyst']}")

    # Function displays subject description
    # Method shows predicate summary
    # Streamlit renders object text
    st.write(report["description"])

    # Function displays subject metrics
    # Method shows predicate statistics
    # Streamlit renders object counters
    st.subheader("Data Summary")
    metric_cols = st.columns(3)
    with metric_cols[0]:
        # Function displays subject count
        # Method shows predicate records
        # Streamlit renders object metric
        st.metric("Record Count", report["record_count"])

    with metric_cols[1]:
        # Function displays subject columns
        # Method shows predicate fields
        # Streamlit renders object metric
        column_count = len(report.get("data_summary", {}).get("columns", []))
        st.metric("Column Count", column_count)

    with metric_cols[2]:
        # Function displays subject statistics
        # Method shows predicate metrics
        # Streamlit renders object count
        stat_count = len(
            report.get("data_summary", {}).get("summary_statistics", {})
        )
        st.metric("Numeric Fields", stat_count)

    # Function displays subject statistics
    # Method shows predicate details
    # Streamlit renders object expander
    with st.expander("Summary Statistics"):
        # Function retrieves subject stats
        # Method extracts predicate data
        # Dictionary stores object values
        stats = report.get("data_summary", {}).get("summary_statistics", {})

        # Function checks subject existence
        # Method verifies predicate content
        # Condition tests object presence
        if stats:
            # Function displays subject table
            # Method shows predicate stats
            # Streamlit renders object dataframe
            stats_df = pd.DataFrame(stats).T
            st.dataframe(stats_df)
        else:
            # Function displays subject message
            # Method shows predicate notice
            # Streamlit renders object info
            st.info("No summary statistics available")

    # Function displays subject plots
    # Method shows predicate visualizations
    # Streamlit renders object expander
    if "plots" in report and report["plots"]:
        # Function displays subject section
        # Method shows predicate heading
        # Streamlit renders object title
        st.subheader("Visualizations")

        # Function displays subject correlation
        # Method shows predicate matrix
        # Condition checks object existence
        if "correlation_matrix" in report["plots"]:
            # Function displays subject matrix
            # Method shows predicate correlation
            # Streamlit renders object heatmap
            corr_data = report["plots"]["correlation_matrix"]
            corr_df = pd.DataFrame(corr_data)

            # Function creates subject heatmap
            # Method visualizes predicate correlation
            # Figure displays object matrix
            fig = px.imshow(
                corr_df,
                text_auto=True,
                color_continuous_scale="RdBu_r",
                title="Correlation Matrix",
                aspect="auto",
            )

            # Function displays subject figure
            # Method shows predicate plot
            # Streamlit renders object visualization
            st.plotly_chart(fig, use_container_width=True)

    # Function displays subject download
    # Method shows predicate link
    # Streamlit renders object html
    st.markdown(get_report_download_link(report), unsafe_allow_html=True)
