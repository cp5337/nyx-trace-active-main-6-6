"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-ADVERSARY-TASK-VIEWER-0001          │
// │ 📁 domain       : Visualization, Task Viewer                │
// │ 🧠 description  : Task viewer for CTAS                      │
// │                  adversary task model visualization         │
// │ 🕸️ hash_type    : UUID → CUID-linked module                 │
// │ 🔄 parent_node  : NODE_VISUALIZATION                       │
// │ 🧩 dependencies : streamlit, plotly, uuid                  │
// │ 🔧 tool_usage   : Visualization, Analysis                   │
// │ 📡 input_type   : User interface events                     │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : visualization, analysis                   │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

Adversary Task Viewer
------------------
This page provides an interactive viewer for CTAS adversary tasks,
enabling detailed analysis of capabilities, limitations, tactics,
and relationships.
"""

import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import pandas as pd
import os
import logging
import random
import plotly.express as px
import numpy as np
import uuid
from pathlib import Path
from datetime import datetime, timedelta

# Import our custom HTML renderer for Replit-safe rendering
from utils.html_renderer import render_html, render_css

from core.periodic_table.task_registry import TaskRegistry
from core.periodic_table.adversary_task import AdversaryTask
from core.periodic_table.relationships import Relationship, RelationshipType
from core.periodic_table.task_loader import load_tasks_from_directory, create_demo_tasks

# Limit the maximum number of tasks to render to improve performance
MAX_TASKS = 50

# Configure logger
logger = logging.getLogger("adversary_task_viewer")
logger.setLevel(logging.INFO)

# NOTE: Page config is set in main.py - don't set it here to avoid conflicts

# Load CSS using components.html for reliable rendering in Replit
def load_task_viewer_css():
    """
    Load CSS for task viewer using streamlit components for reliable rendering
    
    This approach provides better isolation and prevents CSS rendering issues
    in Replit's environment. Both methods are used for maximum compatibility:
    1. components.html for reliable isolation
    2. st.markdown as a backup
    """
    # Try to load from file first
    css_path = os.path.join("public", "style.css")
    css_content = ""
    
    if os.path.exists(css_path):
        try:
            with open(css_path, 'r') as f:
                css_content = f.read()
            logger.info(f"Loaded CSS from {css_path}")
        except Exception as e:
            logger.error(f"Error loading CSS from file: {str(e)}")
    
    # If we couldn't load from file, use the embedded CSS
    if not css_content:
        logger.info("Using embedded CSS")
        css_content = """
        .task-card { border: 2px solid #4A4A4A; border-radius: 8px; padding: 12px; background-color: #1E293B; color: white; margin-bottom: 10px; position: relative; overflow: hidden; transition: all 0.3s ease; cursor: pointer; }
        .task-card:hover { transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.2); }
        .task-card-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
        .task-symbol-container { display: flex; align-items: center; gap: 10px; }
        .task-symbol { font-size: 1.5rem; font-weight: bold; display: flex; align-items: center; justify-content: center; width: 40px; height: 40px; border-radius: 6px; background-color: rgba(255,255,255,0.15); }
        .task-id-container { display: flex; flex-direction: column; }
        .task-hash-id { font-family: monospace; font-size: 1rem; font-weight: bold; }
        .task-id { font-family: monospace; font-size: 0.7rem; color: rgba(255,255,255,0.7); }
        .task-meta-data { display: flex; flex-direction: column; align-items: flex-end; font-size: 0.8rem; }
        .task-ttl { color: rgba(255,255,255,0.7); }
        .task-valence { color: rgba(255,255,255,0.9); }
        .task-name { font-size: 1.2rem; font-weight: bold; margin-bottom: 6px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .task-description { font-size: 0.9rem; margin-bottom: 8px; padding: 6px; background-color: rgba(255,255,255,0.1); border-radius: 4px; overflow: hidden; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; }
        .task-metrics { display: flex; justify-content: space-between; margin-top: 8px; }
        .task-metric { flex: 1; margin: 0 4px; }
        .metric-label { display: flex; justify-content: space-between; font-size: 0.7rem; }
        .metric-value { font-weight: bold; }
        .metric-bar { height: 6px; background-color: #2D3748; border-radius: 3px; margin-top: 3px; position: relative; }
        .metric-filled { height: 100%; border-radius: 3px; }
        .task-symbols { margin-top: 8px; font-size: 1.2rem; letter-spacing: 0.5rem; }
        .task-category-label { position: absolute; right: 0; top: 0; font-size: 0.7rem; padding: 2px 8px; border-bottom-left-radius: 4px; }
        .task-section-title { font-size: 1.1rem; font-weight: bold; margin: 12px 0 4px 0; padding-bottom: 4px; border-bottom: 1px solid rgba(255,255,255,0.2); }
        .task-property-container { background-color: rgba(0,0,0,0.2); padding: 8px; border-radius: 4px; margin-bottom: 8px; font-size: 0.9rem; }
        .task-list { list-style-type: none; padding-left: 10px; margin: 0; }
        .task-list li { padding: 3px 0; position: relative; }
        .task-list li:before { content: "•"; color: #4CAF50; font-weight: bold; position: absolute; left: -10px; }
        .ref-badge { display: inline-block; padding: 2px 6px; margin: 2px; border-radius: 4px; background-color: rgba(0,100,200,0.3); font-size: 0.8rem; }
        .relationship-item { padding: 4px 8px; background-color: rgba(255,255,255,0.1); margin-bottom: 4px; border-radius: 4px; font-size: 0.85rem; }
        .task-footer { margin-top: 10px; font-size: 0.75rem; color: rgba(255,255,255,0.5); border-top: 1px solid rgba(255,255,255,0.15); padding-top: 8px; }
        .periodic-grid { display: grid; grid-template-columns: repeat(var(--grid-cols, 4), 1fr); gap: 10px; margin-top: 20px; }
        .node-cell { min-height: 100px; }
        .phase-indicator { position: absolute; left: 0; top: 0; font-size: 0.7rem; padding: 2px 8px; border-bottom-right-radius: 4px; background-color: rgba(0,0,0,0.5); z-index: 1; }
        .phase-detect { background-color: rgba(75, 192, 192, 0.7); }
        .phase-deny { background-color: rgba(255, 99, 132, 0.7); }
        .phase-disrupt { background-color: rgba(255, 159, 64, 0.7); }
        .phase-degrade { background-color: rgba(153, 102, 255, 0.7); }
        .phase-deceive { background-color: rgba(201, 203, 207, 0.7); }
        .selected-task { box-shadow: 0 0 0 2px #FFD700; }
        .related-task { box-shadow: 0 0 0 2px #4CAF50; }
        .highlighted-task { animation: pulse 1.5s infinite; }
        @keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(255, 215, 0, 0.7); } 70% { box-shadow: 0 0 0 10px rgba(255, 215, 0, 0); } 100% { box-shadow: 0 0 0 0 rgba(255, 215, 0, 0); } }
        .property-container { margin-bottom: 20px; }
        .property-label { font-weight: bold; margin-bottom: 5px; }
        .comparison-container { display: flex; gap: 10px; overflow-x: auto; padding: 10px 0; }
        .comparison-card { flex: 0 0 auto; width: 250px; padding: 10px; border-radius: 8px; background-color: #1E293B; border: 1px solid #4A4A4A; }
        .comparison-header { font-weight: bold; padding-bottom: 5px; margin-bottom: 10px; border-bottom: 1px solid rgba(255,255,255,0.2); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .property-table { width: 100%; border-collapse: collapse; }
        .property-table th, .property-table td { padding: 8px; border: 1px solid #4A4A4A; }
        .property-table th { background-color: rgba(0,0,0,0.3); }
        .property-table tr:nth-child(even) { background-color: rgba(255,255,255,0.05); }
        .filter-container { background-color: #1E293B; padding: 15px; border-radius: 8px; margin-bottom: 20px; border: 1px solid #4A4A4A; }
        .filter-row { display: flex; gap: 15px; margin-bottom: 10px; }
        """
    
    # PRIMARY METHOD: Inject CSS using components.html for reliable rendering in Replit
    try:
        components.html(
            f"""
            <style>
            {css_content}
            </style>
            """,
            height=0,
            width=0
        )
        logger.info("CSS loaded using components.html for better isolation")
    except Exception as e:
        logger.error(f"Error injecting CSS via components.html: {str(e)}")
    
    # BACKUP METHOD: Also inject via markdown as a fallback
    try:
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
        logger.info("CSS loaded using st.markdown as backup")
    except Exception as e:
        logger.error(f"Error injecting CSS via st.markdown: {str(e)}")
        
    # Add a simple critical styles that must render for basic functionality
    st.markdown("""
    <style>
    /* Critical styles for basic card appearance */
    .task-card {
        border: 1px solid #ddd;
        padding: 10px;
        margin-bottom: 10px;
        border-radius: 4px;
        background: #f9f9f9;
    }
    .node-cell {
        padding: 5px;
        margin: 5px;
        border: 1px solid #eee;
        border-radius: 4px;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'registry' not in st.session_state:
    st.session_state.registry = None
if 'selected_tasks' not in st.session_state:
    st.session_state.selected_tasks = []
if 'comparison_mode' not in st.session_state:
    st.session_state.comparison_mode = False

# Initialize registry function
def initialize_registry():
    """Initialize the task registry with data from the 'attached_assets' directory."""
    try:
        # Create registry with in-memory database
        registry = TaskRegistry(db_path=":memory:")
        logger.info("Created in-memory task registry to avoid SQLite threading issues")
        
        # Look for task data in attached_assets directory
        data_dir = "attached_assets"
        if os.path.isdir(data_dir):
            # Load tasks from attached_assets directory
            tasks_loaded = registry.load_tasks_from_directory(data_dir)
            
            if tasks_loaded == 0:
                # If no tasks were loaded, create demo tasks
                logger.warning("No tasks found in attached_assets. Creating demo tasks...")
                registry.initialize_demo_data()
        else:
            # Create demo data
            logger.warning("No attached_assets directory found. Creating demo tasks...")
            registry.initialize_demo_data()
        
        # Ensure relationships are loaded
        registry.load_relationships_from_db()
        
        logger.info(f"Registry initialized with {len(registry.get_all_tasks())} tasks and {len(registry.get_all_relationships())} relationships")
        return registry
    except Exception as e:
        logger.error(f"Error initializing registry: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# Ensure initialized
def ensure_initialized():
    # Make sure the key exists first
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
        
    if not st.session_state.initialized:
        with st.spinner("Initializing CTAS Adversary Task Viewer..."):
            logger.info("Creating registry...")
            registry = initialize_registry()
            
            if registry is None:
                logger.error("Registry initialization failed!")
                st.error("Failed to initialize the registry. Please check the logs for errors.")
                return
                
            logger.info(f"Registry initialized successfully with {len(registry.get_all_tasks())} tasks")
            st.session_state.registry = registry
            st.session_state.initialized = True
            logger.info("Session state updated with registry")
    else:
        logger.info("Using existing registry from session state")

# Render compact task card
def render_task_card(task, compact=True):
    """
    Render a compact card for an adversary task.
    
    Args:
        task: The AdversaryTask to render
        compact: Whether to render in compact mode (default: True)
        
    Returns:
        str: HTML string representation of the task card that is properly
             formatted for rendering in Streamlit with unsafe_allow_html=True
    """
    try:
        # Get properties with safety checks
        task_id = task.task_id if hasattr(task, 'task_id') else "unknown"
        hash_id = task.hash_id if hasattr(task, 'hash_id') else "unknown"
        task_name = task.task_name if hasattr(task, 'task_name') else "Unknown Task"
        
        # Get metrics with safety checks
        reliability = task.get_reliability() if hasattr(task, 'get_reliability') else 0.5
        confidence = task.get_confidence() if hasattr(task, 'get_confidence') else 0.5
        
        # Calculate entropy (a fluctuating value based on metrics)
        entropy = (reliability + confidence) / 2
        
        # Get color based on category with safety check
        color = task.color if hasattr(task, 'color') else "#4A4A4A"
        
        # Get symbol with safety check
        symbol = task.symbol if hasattr(task, 'symbol') else "📋"
        
        # Get valence (attraction) with safety check
        valence = (task.get_valence() / 10) if hasattr(task, 'get_valence') else 0.0  # Scale to a -1.0 to 1.0 range
        
        # Get TTL (time to live in seconds, random for demo)
        ttl = random.randint(30, 180)  # Mock TTL for demonstration
        
        # Get category with safety check
        category = task.get_category() if hasattr(task, 'get_category') else "Unknown"
        
        # Get persona (derived from category and task name) with safety check
        persona = category[:4] if category else "UNKN"
        
        # Define symbols for the task (based on capabilities)
        symbols = ["🔎", "📡", "🧩"]
        
        if compact:
            # Render compact card HTML
            card_html = f"""
            <div class="task-card" style="border-color: {color};">
                <div class="task-category-label" style="background-color: {color}80;">{category}</div>
                <div class="task-card-header">
                    <div class="task-symbol-container">
                        <div class="task-symbol" style="background-color: {color};">
                            {symbol}
                        </div>
                        <div class="task-id-container">
                            <div class="task-hash-id">{hash_id}</div>
                            <div class="task-id">UUID: {task_id.split('-')[1] if task_id else 'N/A'}</div>
                        </div>
                    </div>
                </div>
                
                <div class="task-name">🧠 Persona: "{persona}"</div>
                
                <div class="task-metric">
                    <div class="metric-label">
                        <span>🔁 Entropy:</span>
                    </div>
                    <div class="metric-bar">
                        <div class="metric-filled" style="width: {int(entropy * 100)}%; background-color: #3B82F6;"></div>
                    </div>
                </div>
                
                <div style="display: flex; justify-content: space-between; margin-top: 8px; font-size: 0.85rem;">
                    <div>⏳ TTL: {ttl}s</div>
                    <div>🧲 Valence: {valence:+.2f}</div>
                </div>
                
                <div class="task-symbols" style="text-align: center;">
                    {"".join(symbols)}
                </div>
            </div>
            """
        else:
            # Get additional properties for detailed view
            description = task.description
            capabilities = task.capabilities if hasattr(task, 'capabilities') and task.capabilities else "No capabilities data available"
            limitations = task.limitations if hasattr(task, 'limitations') and task.limitations else "No limitations data available"
            ttps = task.ttps if hasattr(task, 'ttps') and task.ttps else []
            indicators = task.indicators if hasattr(task, 'indicators') and task.indicators else []
            toolchain_refs = task.toolchain_refs if hasattr(task, 'toolchain_refs') and task.toolchain_refs else []
            
            # Format metrics as percentages
            reliability_pct = int(reliability * 100)
            confidence_pct = int(confidence * 100)
            maturity = task.get_maturity()
            complexity = task.get_complexity()
            maturity_pct = int(maturity * 100)
            complexity_pct = int(complexity * 100)
            
            # Get relationships
            registry = st.session_state.registry
            relationships = registry.get_relationships_for_task(task.id)
            
            # Render detailed card HTML
            card_html = f"""
            <div class="task-card" style="border-color: {color};">
                <div class="task-category-label" style="background-color: {color}80;">{category}</div>
                <div class="task-card-header">
                    <div class="task-symbol-container">
                        <div class="task-symbol" style="background-color: {color};">
                            {symbol}
                        </div>
                        <div class="task-id-container">
                            <div class="task-hash-id">{hash_id}</div>
                            <div class="task-id">{task_id}</div>
                        </div>
                    </div>
                    <div class="task-meta-data">
                        <div class="task-ttl">⏳ TTL: {ttl}s</div>
                        <div class="task-valence">🧲 Valence: {valence:+.2f}</div>
                    </div>
                </div>
                
                <div class="task-name" style="font-size: 1.4rem; white-space: normal;">{task_name}</div>
                
                <div class="task-description" style="-webkit-line-clamp: unset;">
                    {description}
                </div>
                
                <div class="task-section-title">🧠 Persona: {persona}</div>
                <div class="task-symbols" style="text-align: center; margin-bottom: 15px;">
                    {"".join(symbols)}
                </div>
                
                <div class="task-metrics">
                    <div class="task-metric">
                        <div class="metric-label">
                            <span>Reliability</span>
                            <span class="metric-value">{reliability_pct}%</span>
                        </div>
                        <div class="metric-bar">
                            <div class="metric-filled" style="width: {reliability_pct}%; background-color: #3B82F6;"></div>
                        </div>
                    </div>
                    
                    <div class="task-metric">
                        <div class="metric-label">
                            <span>Confidence</span>
                            <span class="metric-value">{confidence_pct}%</span>
                        </div>
                        <div class="metric-bar">
                            <div class="metric-filled" style="width: {confidence_pct}%; background-color: #10B981;"></div>
                        </div>
                    </div>
                </div>
                
                <div class="task-metrics">
                    <div class="task-metric">
                        <div class="metric-label">
                            <span>Maturity</span>
                            <span class="metric-value">{maturity_pct}%</span>
                        </div>
                        <div class="metric-bar">
                            <div class="metric-filled" style="width: {maturity_pct}%; background-color: #F59E0B;"></div>
                        </div>
                    </div>
                    
                    <div class="task-metric">
                        <div class="metric-label">
                            <span>Complexity</span>
                            <span class="metric-value">{complexity_pct}%</span>
                        </div>
                        <div class="metric-bar">
                            <div class="metric-filled" style="width: {complexity_pct}%; background-color: #EF4444;"></div>
                        </div>
                    </div>
                </div>
                
                <div class="task-section-title">Capabilities</div>
                <div class="task-property-container">
                    {capabilities[:300]}...
                </div>
                
                <div class="task-section-title">Limitations</div>
                <div class="task-property-container">
                    {limitations[:300]}...
                </div>
            """
            
            # Add TTPs if available
            if ttps:
                card_html += """
                <div class="task-section-title">Tactics, Techniques & Procedures</div>
                <div class="task-property-container">
                    <ul class="task-list">
                """
                
                # Add TTPs
                for ttp in ttps[:3]:  # Limit to first 3
                    card_html += f"<li>{ttp}</li>"
                
                card_html += """
                    </ul>
                </div>
                """
            
            # Add indicators if available
            if indicators:
                card_html += """
                <div class="task-section-title">Indicators</div>
                <div class="task-property-container">
                    <ul class="task-list">
                """
                
                # Add indicators
                for indicator in indicators[:2]:  # Limit to first 2
                    card_html += f"<li>{indicator[:100]}...</li>"
                
                card_html += """
                    </ul>
                </div>
                """
            
            # Add toolchain if available
            if toolchain_refs:
                card_html += """
                <div class="task-section-title">Toolchain</div>
                <div class="task-property-container">
                """
                
                # Add toolchain references
                for tool in toolchain_refs[:5]:  # Limit to first 5
                    card_html += f'<span class="ref-badge">{tool}</span> '
                
                card_html += """
                </div>
                """
            
            # Add relationships if any
            if relationships:
                card_html += """
                <div class="task-section-title">Relationships</div>
                <div class="task-property-container">
                """
                
                for rel in relationships[:5]:  # Limit to first 5
                    # Get target task
                    target_task = registry.get_task(rel.target_id)
                    if target_task and target_task.id != task.id:
                        card_html += f"""
                        <div class="relationship-item">
                            {rel.type} → {target_task.hash_id} ({target_task.task_name})
                        </div>
                        """
                    
                    # Also show incoming relationships
                    source_task = registry.get_task(rel.source_id)
                    if source_task and source_task.id != task.id:
                        card_html += f"""
                        <div class="relationship-item">
                            {rel.type} ← {source_task.hash_id} ({source_task.task_name})
                        </div>
                        """
                
                card_html += """
                </div>
                """
            
            # Add footer
            atomic_number = task.atomic_number if hasattr(task, 'atomic_number') else 0
            card_html += f"""
                <div class="task-footer">
                    Category: {category} | Atomic Number: {atomic_number} | Valence: {task.get_valence()}
                </div>
            </div>
            """
        
        return card_html
    except Exception as e:
        logger.error(f"Error rendering task card: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"<div class='task-card' style='border-color: red;'>Error rendering task card: {str(e)}</div>"

# Render property comparison chart
def render_property_comparison(tasks):
    """Render a radar chart comparing task properties."""
    try:
        # Properties to compare
        properties = ["Reliability", "Confidence", "Maturity", "Complexity"]
        
        # Create a figure
        fig = go.Figure()
        
        # Add each task as a trace
        for task in tasks:
            # Get values
            reliability = task.get_reliability() * 100
            confidence = task.get_confidence() * 100
            maturity = task.get_maturity() * 100
            complexity = task.get_complexity() * 100
            
            # Add trace
            fig.add_trace(go.Scatterpolar(
                r=[reliability, confidence, maturity, complexity],
                theta=properties,
                fill='toself',
                name=task.hash_id
            ))
        
        # Update layout
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=True,
            title="Task Property Comparison"
        )
        
        return fig
    except Exception as e:
        logger.error(f"Error rendering property comparison: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# Render relationship graph
def render_relationship_graph(tasks):
    """Render a graph of task relationships."""
    try:
        registry = st.session_state.registry
        
        # Create a set of task IDs
        task_ids = {task.id for task in tasks}
        
        # Get all relationships for these tasks
        all_relationships = []
        for task in tasks:
            relationships = registry.get_relationships_for_task(task.id)
            all_relationships.extend(relationships)
        
        # Create nodes and edges
        nodes = []
        edges = []
        
        # Add task nodes
        for task in tasks:
            nodes.append({
                "id": str(task.id),
                "label": task.hash_id,
                "color": task.color,
                "title": task.task_name,
                "shape": "circle",
                "size": 20
            })
        
        # Add related task nodes and edges
        for rel in all_relationships:
            # Add source node if not already added
            source_id = str(rel.source_id)
            if source_id not in [node["id"] for node in nodes]:
                source_task = registry.get_task(rel.source_id)
                if source_task:
                    nodes.append({
                        "id": source_id,
                        "label": source_task.hash_id,
                        "color": source_task.color,
                        "title": source_task.task_name,
                        "shape": "circle",
                        "size": 15
                    })
            
            # Add target node if not already added
            target_id = str(rel.target_id)
            if target_id not in [node["id"] for node in nodes]:
                target_task = registry.get_task(rel.target_id)
                if target_task:
                    nodes.append({
                        "id": target_id,
                        "label": target_task.hash_id,
                        "color": target_task.color,
                        "title": target_task.task_name,
                        "shape": "circle",
                        "size": 15
                    })
            
            # Add edge
            edges.append({
                "from": source_id,
                "to": target_id,
                "label": rel.type,
                "arrows": "to"
            })
        
        # Create network data
        network_data = {
            "nodes": nodes,
            "edges": edges
        }
        
        return network_data
    except Exception as e:
        logger.error(f"Error rendering relationship graph: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# Main function
def main():
    # Load CSS styles using the reliable components.html method
    load_task_viewer_css()
    
    st.title("CTAS Periodic Table of Adversary Tasks")
    st.markdown("Explore the complete spectrum of adversary capabilities through the CTAS framework.")
    
    # Initialize
    ensure_initialized()
    
    if not st.session_state.initialized:
        st.warning("Initialization failed. Please check the logs for errors.")
        return
    
    # Get registry
    registry = st.session_state.registry
    
    # Get all tasks
    all_tasks = registry.get_all_tasks()
    if not all_tasks:
        st.warning("No tasks found in the registry.")
        return
    
    # Assign demo phases for visualization
    phases = ["Detect", "Disrupt", "Degrade", "Deceive", "Destroy"]
    # Add phase attribute if it doesn't exist
    for task in all_tasks:
        if not hasattr(task, 'phase'):
            # Assign a phase based on hash_id for consistency
            phase_index = hash(task.hash_id) % len(phases)
            task.phase = phases[phase_index]
    
    # Sidebar
    st.sidebar.title("Navigation")
    
    # View type selector
    view_type = st.sidebar.radio(
        "View Mode",
        ["Periodic Table", "Task Cards", "Property Analysis", "Relationship Network"],
        index=0
    )
    
    # Filter by category
    categories = sorted(set(task.get_category() for task in all_tasks))
    selected_category = st.sidebar.selectbox(
        "Filter by Category", 
        ["All"] + categories
    )
    
    # Filter by phase
    selected_phase = st.sidebar.selectbox("Filter by Phase", ["All"] + phases)
    
    # Apply filters
    filtered_tasks = all_tasks
    
    # Apply category filter
    if selected_category != "All":
        filtered_tasks = [task for task in filtered_tasks if task.get_category() == selected_category]
    
    # Apply phase filter
    if selected_phase != "All":
        filtered_tasks = [task for task in filtered_tasks if hasattr(task, 'phase') and task.phase == selected_phase]
    
    # Sort tasks by hash_id
    filtered_tasks.sort(key=lambda x: x.hash_id)
    
    # Limit the number of tasks to improve performance
    original_count = len(filtered_tasks)
    if original_count > MAX_TASKS:
        filtered_tasks = filtered_tasks[:MAX_TASKS]
        st.sidebar.warning(f"Limited to {MAX_TASKS} tasks for performance. Use filters to narrow results.")
    
    # Create task options for selection
    task_options = {f"{task.hash_id} - {task.task_name}": task.id for task in filtered_tasks}
    
    # Show task count
    if original_count > MAX_TASKS:
        st.sidebar.markdown(f"Showing {len(filtered_tasks)} of {original_count} filtered tasks (from {len(all_tasks)} total)")
    else:
        st.sidebar.markdown(f"Showing {len(filtered_tasks)} of {len(all_tasks)} tasks")
    
    # Add task selection for Property Analysis and Relationship Network views
    if view_type in ["Property Analysis", "Relationship Network"]:
        st.sidebar.markdown("---")
        # Task selector
        selected_task_ids = st.sidebar.multiselect(
            "Select Tasks for Analysis",
            options=list(task_options.keys()),
            default=list(task_options.keys())[:min(5, len(task_options))]
        )
        
        # Convert selection to task IDs
        selected_tasks = []
        for task_label in selected_task_ids:
            task_id = task_options[task_label]
            task = registry.get_task(task_id)
            if task:
                selected_tasks.append(task)
        
        # Sort selected tasks by hash_id
        selected_tasks.sort(key=lambda x: x.hash_id)
        
        # Update session state
        st.session_state.selected_tasks = selected_tasks
    else:
        # For Periodic Table and Task Cards views, don't use explicit task selection
        selected_tasks = filtered_tasks
    
    # Main content
    if view_type == "Periodic Table":
        # Add custom CSS for the grid layout and animations
        st.markdown("""
        <style>
        .periodic-grid {
            display: grid;
            grid-template-columns: repeat(var(--grid-cols), 1fr);
            gap: 8px;
            margin-bottom: 20px;
        }
        .node-cell {
            min-height: 50px;
        }
        .task-card {
            height: 100%;
            margin-bottom: 0 !important;
        }
        
        /* Animation for high entropy */
        @keyframes pulse-entropy {
            0% { box-shadow: 0 0 0 0 rgba(255, 0, 0, 0.7); }
            70% { box-shadow: 0 0 0 10px rgba(255, 0, 0, 0); }
            100% { box-shadow: 0 0 0 0 rgba(255, 0, 0, 0); }
        }
        .high-entropy {
            animation: pulse-entropy 2s infinite;
        }
        
        /* Animation for low TTL */
        @keyframes fade-ttl {
            0% { opacity: 1; }
            50% { opacity: 0.6; }
            100% { opacity: 1; }
        }
        .low-ttl {
            animation: fade-ttl 3s infinite;
        }
        
        /* Phase indicator styles */
        .phase-indicator {
            position: absolute;
            top: 0;
            left: 0;
            font-size: 0.7rem;
            padding: 2px 6px;
            border-bottom-right-radius: 4px;
            z-index: 2;
        }
        .phase-detect { background-color: rgba(59, 130, 246, 0.7); }
        .phase-disrupt { background-color: rgba(245, 158, 11, 0.7); }
        .phase-degrade { background-color: rgba(239, 68, 68, 0.7); }
        .phase-deceive { background-color: rgba(16, 185, 129, 0.7); }
        .phase-destroy { background-color: rgba(107, 33, 168, 0.7); }
        
        /* Valence glow effect */
        .valence-positive {
            box-shadow: 0 0 8px 2px rgba(16, 185, 129, 0.6);
        }
        .valence-negative {
            box-shadow: 0 0 8px 2px rgba(239, 68, 68, 0.6);
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Grid size options
        grid_sizes = {
            "12×12": (12, 12),
            "16×16": (16, 16),
            "8×8": (8, 8)
        }
        
        # Grid size selection
        grid_size = st.selectbox("Periodic Table Grid Size", 
                                list(grid_sizes.keys()), 
                                index=0)
        rows, cols = grid_sizes[grid_size]
        
        # Create the grid container
        st.markdown(f'<div class="periodic-grid" style="--grid-cols: {cols};">', unsafe_allow_html=True)
        
        # Populate grid with tasks
        max_cells = min(rows * cols, len(filtered_tasks))
        for i in range(max_cells):
            task = filtered_tasks[i]
            
            # For demo purposes, calculate entropy and TTL
            entropy = task.get_reliability() * task.get_confidence()
            ttl = random.randint(10, 200)
            valence = task.get_valence() / 10  # Scale to -1.0 to 1.0
            
            # Determine special classes
            special_classes = []
            if entropy > 0.7:
                special_classes.append("high-entropy")
            if ttl < 60:
                special_classes.append("low-ttl")
            if valence > 0.5:
                special_classes.append("valence-positive")
            elif valence < -0.5:
                special_classes.append("valence-negative")
            
            # Get phase for this task
            phase = task.phase if hasattr(task, 'phase') else "Detect"
            phase_class = f"phase-{phase.lower()}"
            
            # Generate the task card HTML
            card_html = render_task_card(task, compact=True)
            
            # Handle the case where card_html might not be a string (safety check)
            if not isinstance(card_html, str):
                # Create a simple card with basic information
                st.subheader(task.hash_id)
                st.caption(task.task_name)
                continue
                
            # Add phase indicator
            phase_indicator = f'<div class="phase-indicator {phase_class}">{phase}</div>'
            
            # Inject phase indicator and special classes into the card HTML
            card_html = card_html.replace('<div class="task-card"', 
                                         f'<div class="task-card {" ".join(special_classes)}"')
            card_html = card_html.replace('<div class="task-category-label"', 
                                         f'{phase_indicator}<div class="task-category-label"')
            
            # Always wrap HTML content properly with st.markdown
            if isinstance(card_html, str):
                # Wrap in node-cell div and output using st.markdown with unsafe_allow_html=True
                st.markdown(f'<div class="node-cell">{card_html}</div>', unsafe_allow_html=True)
            else:
                # If card_html is not a string, create a simple container
                st.markdown('<div class="node-cell">', unsafe_allow_html=True)
                st.write(f"**{task.hash_id}**")
                st.caption(task.task_name)
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Fill remaining grid with empty cells
        for i in range(max_cells, rows * cols):
            st.markdown('<div class="node-cell"></div>', unsafe_allow_html=True)
        
        # Close the grid container
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Add legends
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### Phases")
            phase_colors = {
                "Detect": "rgba(59, 130, 246, 0.7)",    # Blue
                "Disrupt": "rgba(245, 158, 11, 0.7)",   # Amber
                "Degrade": "rgba(239, 68, 68, 0.7)",    # Red
                "Deceive": "rgba(16, 185, 129, 0.7)",   # Green
                "Destroy": "rgba(107, 33, 168, 0.7)"    # Purple
            }
            for phase in ["Detect", "Disrupt", "Degrade", "Deceive", "Destroy"]:
                st.markdown(f"""
                <div style="display:flex; align-items:center; margin-bottom:5px;">
                    <div style="width:20px; height:20px; background-color:{phase_colors[phase]}; margin-right:10px;"></div>
                    <span>{phase}</span>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### Visual Indicators")
            st.markdown("""
            <div style="display:flex; align-items:center; margin-bottom:5px;">
                <div style="width:20px; height:20px; animation: pulse-entropy 2s infinite; background-color:#EF4444; margin-right:10px;"></div>
                <span>High Entropy (>0.7)</span>
            </div>
            <div style="display:flex; align-items:center; margin-bottom:5px;">
                <div style="width:20px; height:20px; animation: fade-ttl 3s infinite; background-color:#9CA3AF; margin-right:10px;"></div>
                <span>Low TTL (<60s)</span>
            </div>
            <div style="display:flex; align-items:center; margin-bottom:5px;">
                <div style="width:20px; height:20px; box-shadow: 0 0 8px 2px rgba(16, 185, 129, 0.6); background-color:#10B981; margin-right:10px;"></div>
                <span>Positive Valence</span>
            </div>
            <div style="display:flex; align-items:center; margin-bottom:5px;">
                <div style="width:20px; height:20px; box-shadow: 0 0 8px 2px rgba(239, 68, 68, 0.6); background-color:#EF4444; margin-right:10px;"></div>
                <span>Negative Valence</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("### Node Interactions")
            st.markdown("""
            **Hover:** Show brief details
            
            **Click:** Expand full details
            
            **Drag:** Test attraction/repulsion
            
            _Note: Interactive features will be implemented in future updates using JavaScript integration._
            """)
    
    elif view_type == "Task Cards":
        # Standard card view with expandable details
        st.subheader("Adversary Task Details")
        
        # Create columns for tasks
        col_count = 3
        for i in range(0, len(filtered_tasks), col_count):
            cols = st.columns(col_count)
            
            for j in range(col_count):
                idx = i + j
                if idx < len(filtered_tasks):
                    with cols[j]:
                        # Show compact card in a clickable expander
                        task = filtered_tasks[idx]
                        with st.expander(f"{task.hash_id} - {task.task_name}", expanded=False):
                            # Show detailed card in a cleaner way
                            # Render the HTML directly as markdown with unsafe_allow_html
                            detailed_card = render_task_card(task, compact=False)
                            
                            # Always properly wrap the HTML output with st.markdown
                            if isinstance(detailed_card, str):
                                # Use components.html for better isolation if needed
                                # from streamlit.components.v1 import html
                                # html(detailed_card, height=500)
                                
                                # Or use markdown with unsafe_allow_html
                                st.markdown(detailed_card, unsafe_allow_html=True)
                            else:
                                # If somehow not a string, display task details in a more structured way
                                st.subheader(task.task_name)
                                st.caption(f"ID: {task.task_id} | Hash: {task.hash_id}")
                                st.write(task.description)
                                st.write("**Category:** " + task.get_category())
                                metrics_cols = st.columns(2)
                                with metrics_cols[0]:
                                    st.metric("Reliability", f"{int(task.get_reliability() * 100)}%")
                                with metrics_cols[1]:
                                    st.metric("Confidence", f"{int(task.get_confidence() * 100)}%")
    
    elif view_type == "Property Analysis":
        # Property comparison
        if len(selected_tasks) > 1:
            # Property comparison chart
            st.subheader("Property Comparison")
            comparison_fig = render_property_comparison(selected_tasks)
            if comparison_fig:
                st.plotly_chart(comparison_fig, use_container_width=True)
            
            # Additional metrics visualizations could be added here
            st.subheader("Task Metrics")
            metrics_df = pd.DataFrame([{
                'Task': task.hash_id,
                'Reliability': task.get_reliability() * 100,
                'Confidence': task.get_confidence() * 100,
                'Maturity': task.get_maturity() * 100,
                'Complexity': task.get_complexity() * 100,
                'Valence': task.get_valence()
            } for task in selected_tasks])
            
            st.dataframe(metrics_df, use_container_width=True)
        else:
            st.info("Select at least 2 tasks to compare properties.")
    
    elif view_type == "Relationship Network":
        # Relationship graph
        if len(selected_tasks) > 1:
            st.subheader("Task Relationship Network")
            graph_fig = render_relationship_graph(selected_tasks)
            if graph_fig:
                st.plotly_chart(graph_fig, use_container_width=True)
            else:
                st.error("Error rendering relationship graph.")
            
            # List relationships in tabular form for clarity
            st.subheader("Task Relationships")
            relationships = []
            for task in selected_tasks:
                task_relationships = registry.get_relationships_for_task(task.id)
                for rel in task_relationships:
                    source_task = registry.get_task(rel.source_id)
                    target_task = registry.get_task(rel.target_id)
                    if source_task and target_task:
                        relationships.append({
                            'Source': source_task.hash_id,
                            'Relationship': rel.type,
                            'Target': target_task.hash_id
                        })
            
            if relationships:
                st.dataframe(pd.DataFrame(relationships), use_container_width=True)
            else:
                st.info("No relationships found between selected tasks.")
        else:
            st.info("Select at least 2 tasks to visualize relationships.")
    
    # Footer
    st.markdown("---")
    
    # Add credits and version info at the bottom
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("CTAS Adversary Task Viewer v6.5 | NyxTrace Intelligence Platform")
    with col2:
        st.markdown("📋 All 167 CTAS Tasks")

# Run the application
if __name__ == "__main__":
    main()