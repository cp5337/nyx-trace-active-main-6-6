"""
Advanced Charts Module
--------------------
This module provides advanced statistical visualization capabilities
beyond basic charts.
"""
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import streamlit as st
import sys
import os

# Import theme settings directly
try:
    from utils import get_theme_settings
except ImportError:
    # Fallback for module imports
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from utils import get_theme_settings


def create_activity_comparison_chart(df, cities=None, date_column='Date', value_column='Activity_Level'):
    """
    Create a comparative chart showing activity levels across multiple cities
    
    Args:
        df: DataFrame with time series data
        cities: List of cities to include (None for all)
        date_column: Column containing date values
        value_column: Column containing activity values
        
    Returns:
        Plotly figure with comparative visualization
    """
    # Get theme settings
    theme = get_theme_settings()
    
    # Filter data if cities specified
    if cities:
        df = df[df['City'].isin(cities)]
    
    # Group by city and date
    grouped = df.groupby(['City', pd.Grouper(key=date_column, freq='W')])[value_column].mean().reset_index()
    
    # Create the chart
    fig = px.line(
        grouped,
        x=date_column,
        y=value_column,
        color='City',
        line_shape='spline',
        title='Weekly Activity Comparison Across Locations',
        labels={
            date_column: 'Date',
            value_column: 'Activity Level',
            'City': 'Location'
        }
    )
    
    # Improve layout
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Average Activity Level',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=500,
        template=theme['template']
    )
    
    return fig


def create_statistical_summary_chart(df, city_column='City', value_column='Activity_Level'):
    """
    Create a statistical summary visualization with box plots and distribution curves
    
    Args:
        df: DataFrame with activity data
        city_column: Column containing city names
        value_column: Column containing values to analyze
        
    Returns:
        Plotly figure with statistical visualization
    """
    # Get theme settings
    theme = get_theme_settings()
    
    # Create figure with box plots and distribution
    fig = make_subplots(
        rows=1, 
        cols=2,
        subplot_titles=('Distribution by Location', 'Statistical Summary'),
        column_widths=[0.6, 0.4]
    )
    
    # Get color from theme for consistency
    box_color = '#4D96FF' if theme['is_dark'] else 'lightseagreen'
    bar_color = '#32CD32' if theme['is_dark'] else 'royalblue'
    
    # Add box plot
    fig.add_trace(
        go.Box(
            x=df[city_column],
            y=df[value_column],
            name='Activity Distribution',
            boxmean=True,
            notched=True,
            marker_color=box_color
        ),
        row=1, col=1
    )
    
    # Generate summary statistics
    summary = df.groupby(city_column)[value_column].agg([
        'mean', 'median', 'std', 'min', 'max', 'count'
    ]).reset_index()
    
    # Add bar chart for mean values with error bars
    fig.add_trace(
        go.Bar(
            x=summary[city_column],
            y=summary['mean'],
            error_y=dict(
                type='data',
                array=summary['std'],
                visible=True
            ),
            name='Mean Activity Level',
            marker_color=bar_color
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        title='Statistical Analysis of Activity Levels',
        height=500,
        showlegend=False,
        boxmode='group',
        template=theme['template']
    )
    
    # Update axes
    fig.update_xaxes(title_text='Location', row=1, col=1)
    fig.update_yaxes(title_text='Activity Level', row=1, col=1)
    fig.update_xaxes(title_text='Location', row=1, col=2)
    fig.update_yaxes(title_text='Mean Activity Level', row=1, col=2)
    
    return fig


def create_trend_analysis_chart(df, city=None, date_column='Date', value_column='Activity_Level'):
    """
    Create an advanced trend analysis chart with trend line and anomaly detection
    
    Args:
        df: DataFrame with time series data
        city: City to filter for (None for all cities aggregated)
        date_column: Column containing dates
        value_column: Column containing values to analyze
        
    Returns:
        Plotly figure with trend analysis
    """
    from scipy import stats
    
    # Get theme settings
    theme = get_theme_settings()
    
    # Filter data if city specified
    if city:
        df = df[df['City'] == city]
    
    # Ensure data is sorted by date
    df = df.sort_values(by=date_column)
    
    # Create figure
    fig = go.Figure()
    
    # Convert dates to ordinal values for regression
    if pd.api.types.is_datetime64_any_dtype(df[date_column]):
        date_ordinals = np.array([(d - pd.Timestamp("1970-01-01")).days for d in df[date_column]])
    else:
        date_ordinals = np.arange(len(df))
    
    # Define colors based on theme
    actual_color = '#4D96FF' if theme['is_dark'] else 'blue'
    ma_color = '#32CD32' if theme['is_dark'] else 'green'
    trend_color = '#FF6B6B' if theme['is_dark'] else 'red'
    anomaly_color = '#FF9F1C' if theme['is_dark'] else 'orange'
    anomaly_line_color = '#FF6B6B' if theme['is_dark'] else 'red'
    
    # Add actual data points
    fig.add_trace(go.Scatter(
        x=df[date_column],
        y=df[value_column],
        mode='lines+markers',
        name='Actual Activity',
        line=dict(color=actual_color, width=1),
        marker=dict(size=4),
        opacity=0.7
    ))
    
    # Calculate moving average
    window = 7  # 7-day window
    df['MA'] = df[value_column].rolling(window=window, center=True).mean()
    
    # Add moving average
    fig.add_trace(go.Scatter(
        x=df[date_column],
        y=df['MA'],
        mode='lines',
        name=f'{window}-Day Moving Average',
        line=dict(color=ma_color, width=2.5)
    ))
    
    # Linear regression for trend line
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        date_ordinals, df[value_column].fillna(df[value_column].mean())
    )
    
    # Generate trend line
    trend_y = intercept + slope * date_ordinals
    
    # Add trend line
    fig.add_trace(go.Scatter(
        x=df[date_column],
        y=trend_y,
        mode='lines',
        name=f'Trend (r²={r_value**2:.2f})',
        line=dict(color=trend_color, width=2, dash='dash')
    ))
    
    # Find anomalies (points that deviate significantly from trend)
    residuals = df[value_column] - trend_y
    std_residuals = np.std(residuals)
    threshold = 2 * std_residuals  # 2 standard deviations
    anomalies = df[abs(residuals) > threshold]
    
    # Add anomalies to chart
    if not anomalies.empty:
        fig.add_trace(go.Scatter(
            x=anomalies[date_column],
            y=anomalies[value_column],
            mode='markers',
            name='Anomalies',
            marker=dict(
                color=anomaly_color,
                size=10,
                symbol='circle-open',
                line=dict(width=2, color=anomaly_line_color)
            )
        ))
    
    # Update layout
    title = f"Trend Analysis for {city}" if city else "Trend Analysis for All Locations"
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Activity Level',
        height=500,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template=theme['template']
    )
    
    return fig