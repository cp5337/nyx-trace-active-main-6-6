"""
OSINT Visualization Module
------------------------
This module provides specialized visualization tools for OSINT data analysis,
focusing on emotional content, sentiment, and relationships.
"""
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import networkx as nx
import sys
import os

# Import theme settings directly
try:
    from utils import get_theme_settings
except ImportError:
    # Fallback for module imports
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from utils import get_theme_settings


def create_emotion_radar_chart(emotion_data: Dict[str, Dict[str, float]]) -> go.Figure:
    """
    Create a radar chart showing the distribution of emotions in the analyzed content
    
    Args:
        emotion_data: Dictionary with emotion percentages
            e.g., {'anger': {'percentage': 25}, 'fear': {'percentage': 30}, ...}
    
    Returns:
        Plotly figure with radar chart
    """
    # Get theme settings
    theme = get_theme_settings()
    
    categories = list(emotion_data.keys())
    values = [emotion_data[emotion]['percentage'] for emotion in categories]
    
    # Define colors based on theme
    line_color = 'rgb(31, 119, 180)' if not theme['is_dark'] else 'rgb(61, 149, 210)'
    fill_color = 'rgba(31, 119, 180, 0.5)' if not theme['is_dark'] else 'rgba(61, 149, 210, 0.5)'
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Emotional Distribution',
        line_color=line_color,
        fillcolor=fill_color
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(values) + 10]
            )
        ),
        title="Emotional Distribution in Content",
        showlegend=False,
        height=450,
        template=theme['template']
    )
    
    return fig


def create_entity_network(entities: List[str], content_df: pd.DataFrame, 
                        content_column: str = 'Content', 
                        max_entities: int = 10) -> go.Figure:
    """
    Create a network visualization showing relationships between entities
    based on their co-occurrence in content
    
    Args:
        entities: List of entities to include in the network
        content_df: DataFrame containing the content
        content_column: Column containing the textual content
        max_entities: Maximum number of entities to include
        
    Returns:
        Plotly figure with network visualization
    """
    # Get theme settings
    theme = get_theme_settings()
    
    if not entities or len(entities) < 2 or content_df.empty:
        # Create an empty figure if there's not enough data
        fig = go.Figure()
        fig.update_layout(
            title="Entity Network (Insufficient Data)",
            height=500,
            template=theme['template']
        )
        return fig
        
    # Limit to top N entities
    entities = entities[:max_entities]
    
    # Create a graph
    G = nx.Graph()
    
    # Add nodes for each entity
    for entity in entities:
        G.add_node(entity)
    
    # Calculate co-occurrence
    for i, entity1 in enumerate(entities):
        for entity2 in entities[i+1:]:
            # Count how many times both entities appear in the same content
            co_occurrences = 0
            for content in content_df[content_column]:
                if not isinstance(content, str):
                    continue
                if entity1 in content and entity2 in content:
                    co_occurrences += 1
            
            if co_occurrences > 0:
                G.add_edge(entity1, entity2, weight=co_occurrences)
    
    # Use networkx spring layout to position nodes
    pos = nx.spring_layout(G)
    
    # Extract node positions
    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
    
    # Define colors based on theme
    node_color = '#4D96FF' if theme['is_dark'] else 'skyblue'
    node_line_color = '#8FB0FF' if theme['is_dark'] else 'darkblue'
    edge_color = '#999999' if theme['is_dark'] else 'gray'
    
    # Create node trace
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=list(G.nodes()),
        textposition="top center",
        marker=dict(
            size=15,
            color=node_color,
            line=dict(width=2, color=node_line_color)
        ),
        name='Entities'
    )
    
    # Create edge traces
    edge_traces = []
    for edge in G.edges(data=True):
        node1, node2, data = edge
        x0, y0 = pos[node1]
        x1, y1 = pos[node2]
        weight = data.get('weight', 1)
        
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(width=weight, color=edge_color),
            hoverinfo='none',
            showlegend=False
        )
        edge_traces.append(edge_trace)
    
    # Combine all traces
    fig = go.Figure(data=edge_traces + [node_trace])
    
    # Update layout
    fig.update_layout(
        title="Entity Relationship Network",
        height=500,
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        template=theme['template']
    )
    
    return fig


def create_emotional_trends_chart(emotional_trends: Dict[str, Dict[str, List]], 
                                title: str = "Emotional Trends Over Time") -> go.Figure:
    """
    Create a line chart showing how emotions trend over time
    
    Args:
        emotional_trends: Dictionary with emotion trends data
            e.g., {'anger': {'dates': ['2023-01-01', ...], 'counts': [5, ...]}, ...}
        title: Chart title
        
    Returns:
        Plotly figure with line chart
    """
    # Get theme settings
    theme = get_theme_settings()
    
    fig = go.Figure()
    
    # Color mapping for emotions
    color_map = {
        'anger': 'red',
        'fear': 'purple',
        'excitement': 'green',
        'neutral': 'gray'
    }
    
    # Add a trace for each emotion
    for emotion, data in emotional_trends.items():
        dates = data.get('dates', [])
        counts = data.get('counts', [])
        
        if dates and counts and len(dates) == len(counts):
            fig.add_trace(go.Scatter(
                x=dates,
                y=counts,
                mode='lines+markers',
                name=emotion.capitalize(),
                line=dict(
                    color=color_map.get(emotion, 'blue'),
                    width=2
                )
            ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Frequency",
        height=450,
        hovermode="x unified",
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


def create_sentiment_heatmap(df: pd.DataFrame, location_column: str = 'Location', 
                           date_column: str = 'Date', 
                           sentiment_column: str = 'Sentiment') -> go.Figure:
    """
    Create a heatmap showing sentiment across locations and time
    
    Args:
        df: DataFrame with location, date and sentiment data
        location_column: Column containing location names
        date_column: Column containing dates
        sentiment_column: Column containing sentiment scores
        
    Returns:
        Plotly figure with heatmap
    """
    # Get theme settings
    theme = get_theme_settings()
    
    if df.empty or not all(col in df.columns for col in [location_column, date_column, sentiment_column]):
        # Create empty figure if data is missing
        fig = go.Figure()
        fig.update_layout(
            title="Sentiment Heatmap (Insufficient Data)",
            height=500,
            template=theme['template']
        )
        return fig
    
    # Ensure date is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    
    # Extract date component only
    df['date_str'] = df[date_column].dt.strftime('%Y-%m-%d')
    
    # Group by location and date, calculate average sentiment
    pivot_df = df.pivot_table(
        index=location_column,
        columns='date_str',
        values=sentiment_column,
        aggfunc='mean'
    ).fillna(0)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=pivot_df.values,
        x=pivot_df.columns,
        y=pivot_df.index,
        colorscale='RdBu',
        zmid=0,  # Center colorscale at 0 (neutral sentiment)
        colorbar=dict(
            title="Sentiment",
            titleside="right",
            titlefont=dict(size=14)
        )
    ))
    
    # Update layout
    fig.update_layout(
        title="Sentiment by Location Over Time",
        xaxis_title="Date",
        yaxis_title="Location",
        height=500,
        template=theme['template']
    )
    
    return fig


def create_key_phrases_wordcloud(key_phrases: List[str], width: int = 800, height: int = 400) -> go.Figure:
    """
    Create a simple word cloud visualization for key phrases
    
    Args:
        key_phrases: List of key phrases
        width: Width of the visualization
        height: Height of the visualization
        
    Returns:
        Plotly figure with word cloud visualization
    """
    import random
    
    # Get theme settings
    theme = get_theme_settings()
    
    if not key_phrases:
        # Create empty figure if no phrases
        fig = go.Figure()
        fig.update_layout(
            title="Key Phrases (No Data)",
            height=height,
            template=theme['template']
        )
        return fig
    
    # Count words in phrases
    word_counts = {}
    for phrase in key_phrases:
        if not isinstance(phrase, str):
            continue
            
        words = phrase.split()
        for word in words:
            # Clean the word
            word = word.strip().lower()
            word = ''.join(c for c in word if c.isalnum())
            
            if word and len(word) > 3:  # Ignore short words
                word_counts[word] = word_counts.get(word, 0) + 1
    
    # Create scatter plot to simulate word cloud
    words = list(word_counts.keys())
    counts = [word_counts[word] for word in words]
    
    # Normalize sizes
    min_count = min(counts) if counts else 1
    max_count = max(counts) if counts else 1
    sizes = [10 + (count - min_count) * 40 / (max_count - min_count) if max_count > min_count else 20 for count in counts]
    
    # Generate random positions
    np.random.seed(42)  # For reproducibility
    x_pos = np.random.uniform(-1, 1, len(words))
    y_pos = np.random.uniform(-1, 1, len(words))
    
    # Create figure
    fig = go.Figure()
    
    # Adjust color range based on theme
    base_r = 100 if theme['is_dark'] else 50
    base_g = 120 if theme['is_dark'] else 50
    base_b = 200 if theme['is_dark'] else 150
    
    # Add words as scatter points
    fig.add_trace(go.Scatter(
        x=x_pos,
        y=y_pos,
        mode='text',
        text=words,
        textfont=dict(
            size=sizes,
            color=[f'rgb({random.randint(base_r, base_r+100)}, {random.randint(base_g, base_g+100)}, {random.randint(base_b, base_b+50)})' 
                  for _ in range(len(words))]
        ),
        hoverinfo='text',
        hovertext=[f"{word}: {count}" for word, count in zip(words, counts)]
    ))
    
    # Update layout
    fig.update_layout(
        title="Key Phrases Word Cloud",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=height,
        width=width,
        template=theme['template']
    )
    
    return fig


def create_osint_dashboard(insights_data: Dict[str, Any], 
                         content_df: Optional[pd.DataFrame] = None) -> Dict[str, go.Figure]:
    """
    Create a full OSINT dashboard with multiple visualizations
    
    Args:
        insights_data: Dictionary with insights from OSINTAnalyzer
        content_df: Optional DataFrame with content data
        
    Returns:
        Dictionary of Plotly figures for the dashboard
    """
    figures = {}
    
    # Emotion radar chart
    if 'dominant_emotions' in insights_data and insights_data['dominant_emotions']:
        figures['emotion_radar'] = create_emotion_radar_chart(insights_data['dominant_emotions'])
    
    # Entity network
    if content_df is not None and 'key_entities' in insights_data and insights_data['key_entities']:
        figures['entity_network'] = create_entity_network(
            insights_data['key_entities'], 
            content_df
        )
    
    # Emotional trends
    if 'emotional_trends' in insights_data and insights_data['emotional_trends']:
        figures['emotional_trends'] = create_emotional_trends_chart(insights_data['emotional_trends'])
    
    # Sentiment heatmap
    if content_df is not None and 'Location' in content_df.columns and 'Date' in content_df.columns:
        figures['sentiment_heatmap'] = create_sentiment_heatmap(content_df)
    
    # Word cloud
    if content_df is not None and 'key_phrases' in content_df.columns:
        # Flatten all key phrases
        all_phrases = []
        for phrases in content_df['key_phrases']:
            if isinstance(phrases, list):
                all_phrases.extend(phrases)
        
        if all_phrases:
            figures['word_cloud'] = create_key_phrases_wordcloud(all_phrases)
    
    return figures