"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-PERIODIC-NODE-CARD-0001             │
// │ 📁 domain       : Visualization, Node Cards                 │
// │ 🧠 description  : Node card viewer for the                  │
// │                  CTAS Periodic Table of Nodes               │
// │ 🕸️ hash_type    : UUID → CUID-linked module                 │
// │ 🔄 parent_node  : NODE_VISUALIZATION                       │
// │ 🧩 dependencies : streamlit, plotly                        │
// │ 🔧 tool_usage   : Visualization, Comparison                │
// │ 📡 input_type   : User interface events                     │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : visualization, comparison                 │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

Node Card Viewer
------------
This page provides an interactive node card viewer for the CTAS Periodic Table
of Nodes, enabling detailed comparison of element properties and relationships.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import uuid
import os
import json
from pathlib import Path
import random
from datetime import datetime, timedelta

from core.periodic_table.table import PeriodicTable, ElementSymbol
from core.periodic_table.registry import PeriodicTableRegistry
from core.periodic_table.element import Element, ElementProperty
from core.periodic_table.group import Group, Period, Category, GroupType, CATEGORY_COLORS
from core.periodic_table.relationships import Relationship, RelationshipType

# Set page config
st.set_page_config(
    page_title="CTAS Node Card Viewer",
    page_icon="🧩",
    layout="wide"
)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'registry' not in st.session_state:
    st.session_state.registry = None
if 'periodic_table' not in st.session_state:
    st.session_state.periodic_table = None
if 'selected_elements' not in st.session_state:
    st.session_state.selected_elements = []
if 'comparison_mode' not in st.session_state:
    st.session_state.comparison_mode = False

# Custom CSS for node cards
st.markdown("""
<style>
    .node-card {
        border: 3px solid #FF0000;
        border-radius: 10px;
        padding: 10px;
        background-color: #1E3A8A;
        color: white;
        margin-bottom: 15px;
        position: relative;
        overflow: hidden;
    }
    
    .node-header {
        display: flex;
        justify-content: space-between;
        margin-bottom: 5px;
    }
    
    .node-title {
        background-color: rgba(255,255,255,0.2);
        padding: 2px 5px;
        border-radius: 5px;
        font-weight: bold;
    }
    
    .node-ttl {
        font-size: 0.8rem;
        color: rgba(255,255,255,0.7);
    }
    
    .node-symbols {
        display: flex;
        justify-content: space-between;
        margin: 10px 0;
    }
    
    .node-symbol-left {
        background-color: rgba(0,100,255,0.5);
        width: 50px;
        height: 50px;
        border-radius: 8px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        font-weight: bold;
    }
    
    .node-symbol-center {
        background-color: rgba(255,255,255,0.9);
        color: black;
        padding: 5px;
        text-align: center;
        border-radius: 5px;
        flex-grow: 1;
        margin: 0 5px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .node-symbol-right {
        background-color: rgba(0,200,0,0.5);
        width: 50px;
        height: 50px;
        border-radius: 8px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        font-weight: bold;
    }
    
    .node-name {
        background-color: rgba(200,0,200,0.7);
        padding: 8px;
        text-align: center;
        margin: 10px 0;
        border-radius: 5px;
        font-weight: bold;
        animation: pulse 2s infinite;
    }
    
    .node-info {
        background-color: rgba(255,255,255,0.2);
        padding: 8px;
        text-align: left;
        margin: 10px 0;
        border-radius: 5px;
        font-size: 0.9em;
    }
    
    .node-stats {
        font-size: 0.8em;
        margin-top: 10px;
    }
    
    .node-stat-row {
        display: flex;
        justify-content: space-between;
        margin-bottom: 3px;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .main-title {
        text-align: center;
        font-size: 2.5rem;
        margin-bottom: 1rem;
        color: #1E3A8A;
    }
    
    .section-title {
        font-size: 1.5rem;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        color: #1E3A8A;
    }
    
    .property-meter {
        height: 8px;
        background-color: #e0e0e0;
        border-radius: 4px;
        margin-top: 2px;
        position: relative;
    }
    
    .property-value {
        height: 100%;
        border-radius: 4px;
        background-color: #4CAF50;
    }
    
    .comparison-container {
        display: flex;
        gap: 20px;
    }
    
    .comparison-card {
        flex: 1;
    }
    
    .comparison-title {
        text-align: center;
        font-weight: bold;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize registry function
def initialize_registry():
    """Initialize the registry with sample data for demonstration."""
    try:
        # Create registry with default in-memory database
        registry = PeriodicTableRegistry(db_path=":memory:")
        print("Created in-memory registry to avoid SQLite threading issues")
        
        # Create sample data for the demo
        # Create categories
        categories = ["ENTITY", "INFRASTRUCTURE", "CAPABILITY", "THREAT", 
                     "ACTOR", "LOCATION", "EVENT", "INTELLIGENCE",
                     "ALGORITHM", "RESOURCE", "RELATIONSHIP", "ATTRIBUTE"]
        
        for name in categories:
            color = CATEGORY_COLORS.get(name, "#CCCCCC")
            category = Category(
                name=name,
                symbol=name[:3].upper(),
                color=color,
                description=f"{name} classification category"
            )
            registry.add_category(category)
            
        # Create 10 groups (vertical columns)
        for i in range(1, 11):
            group = Group(
                name=f"Group {i}",
                symbol=f"G{i}",
                number=i,
                description=f"Vertical classification group {i}",
                type=GroupType.VERTICAL
            )
            registry.add_group(group)
        
        # Create 7 periods (horizontal rows)
        for i in range(1, 8):
            period = Period(
                name=f"Period {i}",
                number=i,
                description=f"Horizontal classification period {i}"
            )
            registry.add_period(period)
        
        # Get all groups, periods, and categories
        all_groups = registry.get_all_groups()
        all_periods = registry.get_all_periods()
        all_categories = registry.get_all_categories()
        
        # Create elements
        elements = []
        atomic_number = 1
        
        # Create one element for each period-group combination with varied categories
        for period in all_periods:
            for group in all_groups:
                if random.random() < 0.7:  # 70% chance to create an element
                    # Pick a random category
                    category = random.choice(all_categories)
                    
                    # Create symbol: first letter of category + atomic number
                    symbol = f"{category.name[0]}{atomic_number}"
                    
                    # Create element name: category + classification + number
                    name = f"{category.name.capitalize()} {atomic_number}"
                    
                    # Create element
                    element = Element(
                        atomic_number=atomic_number,
                        symbol=symbol,
                        name=name,
                        group_id=str(group.id) if hasattr(group, 'id') else None,
                        period_id=str(period.id) if hasattr(period, 'id') else None,
                        category_id=str(category.id) if hasattr(category, 'id') else None
                    )
                    
                    # Set properties
                    element.set_property(ElementProperty.RELIABILITY, random.uniform(0.3, 1.0))
                    element.set_property(ElementProperty.CONFIDENCE, random.uniform(0.3, 1.0))
                    element.set_property(ElementProperty.ACCESSIBILITY, random.uniform(0.3, 1.0))
                    element.set_property(ElementProperty.SENSITIVITY, random.randint(1, 10))
                    element.set_property(ElementProperty.COMPLEXITY, random.randint(1, 10))
                    element.set_property(ElementProperty.MATURITY, random.randint(1, 10))
                    element.set_property(ElementProperty.PRIORITY, random.randint(1, 10))
                    element.set_property(ElementProperty.COMPUTATION_COST, random.randint(1, 10))
                    element.set_property(ElementProperty.STABILITY, random.uniform(0.3, 1.0))
                    element.set_property(ElementProperty.DESCRIPTION, 
                                        f"This is a {category.name.lower()} element classified as {symbol}.")
                    
                    # Add discovery and update dates
                    discovery_date = datetime.now() - timedelta(days=random.randint(1, 365))
                    element.set_property(ElementProperty.DISCOVERY_DATE, discovery_date.isoformat())
                    element.set_property(ElementProperty.LAST_UPDATED, datetime.now().isoformat())
                    element.set_property(ElementProperty.TTL, random.randint(30, 365))
                    
                    # Add to registry
                    registry.add_element(element)
                    elements.append(element)
                    
                    atomic_number += 1
                    
        # Create relationships between elements
        relationship_types = [
            RelationshipType.CONNECTED_TO, 
            RelationshipType.ANALYZES,
            RelationshipType.DERIVED_FROM,
            RelationshipType.ENABLES,
            RelationshipType.INHIBITS,
            RelationshipType.LOCATED_AT,
            RelationshipType.CONTAINS,
            RelationshipType.CAUSES,
            RelationshipType.AFFECTED_BY,
            RelationshipType.SIMILAR_TO
        ]
        
        # Add random relationships
        for _ in range(min(50, len(elements) * 2)):
            source = random.choice(elements)
            target = random.choice(elements)
            
            # Avoid self-relationships
            if source.id != target.id:
                rel_type = random.choice(relationship_types)
                
                relationship = Relationship(
                    source_id=source.id,
                    target_id=target.id,
                    type=rel_type,
                    weight=random.uniform(0.1, 1.0),
                    confidence=random.uniform(0.3, 1.0),
                    bidirectional=random.random() < 0.3  # 30% chance of bidirectional
                )
                
                registry.add_relationship(relationship)
                
        # Dump all registry data to memory to avoid thread safety issues
        # This is a workaround for the SQLite thread safety problem
        all_elements = registry.get_all_elements()
        relationships = registry.get_all_relationships() if hasattr(registry, 'get_all_relationships') else []
        print(f"Created {len(all_elements)} elements and {len(relationships)} relationships in registry")
        return registry
    except Exception as e:
        print(f"Error initializing registry: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# Ensure initialized
def ensure_initialized():
    # Make sure the key exists first
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
        
    if not st.session_state.initialized:
        with st.spinner("Initializing Periodic Table of Nodes..."):
            print("Creating registry...")
            registry = initialize_registry()
            
            if registry is None:
                print("ERROR: Registry initialization failed!")
                st.error("Failed to initialize the registry. Please check the console for errors.")
                return
                
            print(f"Registry initialized successfully with {len(registry.get_all_elements())} elements")
            st.session_state.registry = registry
            st.session_state.periodic_table = PeriodicTable(registry)
            st.session_state.initialized = True
            print("Session state updated with registry and periodic table")
    else:
        print("Using existing registry from session state")

# Render custom node card
def render_node_card(element):
    """
    Render a custom node card for an element.
    
    Args:
        element: The element to render
    """
    # Handle both dictionary and object access
    # First try to access as Element object attributes
    try:
        name = element.name
        symbol = element.symbol
        atomic_number = element.atomic_number
        category_id = element.category_id
        
        # Properties via get_property method
        ttl = element.get_property(ElementProperty.TTL, "N/A")
        reliability = element.get_property(ElementProperty.RELIABILITY, 1.0)
        confidence = element.get_property(ElementProperty.CONFIDENCE, 1.0)
        accessibility = element.get_property(ElementProperty.ACCESSIBILITY, 1.0)
        maturity = element.get_property(ElementProperty.MATURITY, 5)
        complexity = element.get_property(ElementProperty.COMPLEXITY, 5)
        
    except AttributeError:
        # Fallback to dictionary access
        name = element.get('name', 'Unknown')
        symbol = element.get('short_name', element.get('symbol', '?'))
        atomic_number = element.get('atomic_number', 0)
        category_id = element.get('category_id', '')
        
        properties = element.get('properties', {})
        ttl = properties.get('ttl', "N/A")
        reliability = properties.get('reliability', 1.0)
        confidence = properties.get('confidence', 1.0)
        accessibility = properties.get('accessibility', 1.0)
        maturity = properties.get('maturity', 5)
        complexity = properties.get('complexity', 5)
    
    # Get category for color
    categories = st.session_state.registry.get_all_categories()
    
    # Try both ways to get category
    try:
        # First try object access
        category = next((c for c in categories if c.id == category_id), None)
        color = category.color if category else "#1f77b4"
    except AttributeError:
        # Fallback to dictionary access
        category = next((c for c in categories if c.get('id', '') == category_id), None)
        color = category.get('color', '#1f77b4') if category else "#1f77b4"
    
    # Format reliability as percentages
    reliability_pct = int(reliability * 100)
    confidence_pct = int(confidence * 100)
    
    # Render card HTML
    card_html = f"""
    <div class="node-card" style="background-color: {color};">
        <div class="node-header">
            <div class="node-title">{atomic_number}.0 {name}</div>
            <div class="node-ttl">TTL: {ttl}</div>
        </div>
        
        <div class="node-symbols">
            <div class="node-symbol-left">
                θ<br>-3
            </div>
            
            <div class="node-symbol-center">
                <div>Symbols</div>
                <div style="font-weight: bold;">α αΩ♥♠Δ</div>
            </div>
            
            <div class="node-symbol-right">
                Δ<br>6
            </div>
        </div>
        
        <div class="node-name">
            {name}<br>with color and<br>border incl<br>animation
        </div>
        
        <div class="node-info">
            Each element is a link to<br>additional info
        </div>
        
        <div class="node-stats">
            <div class="node-stat-row">
                <span>Reliability: {reliability_pct}%</span>
                <span>Confidence: {confidence_pct}%</span>
            </div>
            <div class="node-stat-row">
                <span>Complexity: {complexity}/10</span>
                <span>Maturity: {maturity}/10</span>
            </div>
        </div>
    </div>
    """
    
    return card_html

# Render property comparison
def render_property_comparison(elements, property_name, property_enum, max_value=1.0, scale_factor=1.0):
    """
    Render a comparison of a property across elements.
    
    Args:
        elements: List of elements to compare
        property_name: Display name of the property
        property_enum: ElementProperty enum value
        max_value: Maximum value of the property
        scale_factor: Factor to scale property value (for percentages)
    """
    st.markdown(f"### {property_name}")
    
    for element in elements:
        value = element.get_property(property_enum, 0)
        
        if max_value == 1.0:
            display_value = f"{value*scale_factor:.0f}%"
        else:
            display_value = f"{value}/{max_value}"
        
        percentage = (value / max_value) * 100
        
        st.markdown(f"""
        <div style="margin-bottom: 15px;">
            <div style="display: flex; justify-content: space-between;">
                <span>{element.name} ({element.symbol})</span>
                <span>{display_value}</span>
            </div>
            <div class="property-meter">
                <div class="property-value" style="width: {percentage}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Main function
def main():
    # Ensure initialized - this needs to happen BEFORE we reference the session state
    ensure_initialized()
    
    # Get reference to registry and periodic table from session state
    if 'registry' not in st.session_state or st.session_state.registry is None:
        # If registry is not in session state or is None, initialize it again
        print("Registry not found in session state, reinitializing...")
        st.session_state.initialized = False
        ensure_initialized()
        
    # Now get the registry and periodic table
    registry = st.session_state.registry
    periodic_table = st.session_state.periodic_table
    
    # Safety check
    if registry is None:
        st.error("Registry initialization failed. Please refresh the page.")
        st.stop()
    
    # Title
    st.markdown('<div class="main-title">CTAS Node Card Viewer</div>', unsafe_allow_html=True)
    
    # Description
    st.markdown("""
    This page allows you to view and compare node cards from the CTAS Periodic Table of Nodes. 
    Each node represents an intelligence entity with properties, capabilities, and relationships.
    """)
    
    # Main tabs
    tab1, tab2 = st.tabs(["Node Cards", "Node Comparison"])
    
    with tab1:
        st.markdown('<div class="section-title">Node Card Explorer</div>', unsafe_allow_html=True)
        
        # Filter controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Category filter
            categories = registry.get_all_categories()
            # Handle Category objects properly
            category_options = {c.name: c.id for c in categories}
            selected_category = st.selectbox("Category", ["All"] + list(category_options.keys()))
        
        with col2:
            # Group filter
            groups = registry.get_all_groups()
            # Handle Group objects properly
            group_options = {g.name: g.id for g in groups}
            selected_group = st.selectbox("Group", ["All"] + list(group_options.keys()))
        
        with col3:
            # Period filter
            periods = registry.get_all_periods()
            # Handle Period objects properly
            period_options = {p.name: p.id for p in periods}
            selected_period = st.selectbox("Period", ["All"] + list(period_options.keys()))
        
        # Get all elements
        all_elements = registry.get_all_elements()
        
        # Apply filters
        filtered_elements = all_elements
        
        if selected_category != "All":
            category_id = category_options[selected_category]
            # Try object access first, then dictionary access
            try:
                filtered_elements = [e for e in filtered_elements if e.category_id == category_id]
            except AttributeError:
                filtered_elements = [e for e in filtered_elements if e.get('category_id', '') == category_id]
        
        if selected_group != "All":
            group_id = group_options[selected_group]
            # Try object access first, then dictionary access
            try:
                filtered_elements = [e for e in filtered_elements if e.group_id == group_id]
            except AttributeError:
                filtered_elements = [e for e in filtered_elements if e.get('group_id', '') == group_id]
        
        if selected_period != "All":
            period_id = period_options[selected_period]
            # Try object access first, then dictionary access
            try:
                filtered_elements = [e for e in filtered_elements if e.period_id == period_id]
            except AttributeError:
                filtered_elements = [e for e in filtered_elements if e.get('period_id', '') == period_id]
        
        # Sort by atomic number
        # Try object access first, then dictionary access
        try:
            filtered_elements.sort(key=lambda e: e.atomic_number)
        except AttributeError:
            filtered_elements.sort(key=lambda e: e.get('atomic_number', 0))
        
        # Display elements in a grid
        if filtered_elements:
            # Controls for comparison
            st.markdown("### Select elements to compare")
            num_to_compare = st.slider("Number of elements to compare", 2, 4, 2)
            
            # Display node cards in columns
            cols = st.columns(3)
            
            for i, element in enumerate(filtered_elements):
                col_idx = i % 3
                
                with cols[col_idx]:
                    # Checkbox for comparison - handle both object and dictionary formats
                    try:
                        # Try object access first
                        element_id = element.id
                        element_symbol = element.symbol
                    except AttributeError:
                        # Fall back to dictionary access
                        element_id = element.get('id', str(uuid.uuid4()))
                        element_symbol = element.get('short_name', element.get('symbol', '?'))
                    
                    selected = st.checkbox(f"Compare {element_symbol}", key=f"compare_{element_id}")
                    if selected and element_id not in st.session_state.selected_elements:
                        st.session_state.selected_elements.append(element_id)
                    elif not selected and element_id in st.session_state.selected_elements:
                        st.session_state.selected_elements.remove(element_id)
                    
                    # Render node card
                    st.markdown(render_node_card(element), unsafe_allow_html=True)
                    
                    # Element details
                    with st.expander("Element Details"):
                        # Get element properties - try both object and dictionary access
                        try:
                            # Try as object first
                            atomic_number = element.atomic_number
                            name = element.name
                            category_id = element.category_id
                            group_id = element.group_id
                            period_id = element.period_id
                            
                            # Try to get category, group, period names
                            category_name = next((c.name for c in categories if c.id == category_id), "Unknown")
                            group_name = next((g.name for g in groups if g.id == group_id), "Unknown")
                            period_name = next((p.name for p in periods if p.id == period_id), "Unknown")
                            
                        except AttributeError:
                            # Fall back to dictionary access
                            atomic_number = element.get('atomic_number', 'N/A')
                            name = element.get('name', 'Unknown')
                            category_id = element.get('category_id', '')
                            group_id = element.get('group_id', '')
                            period_id = element.get('period_id', '')
                            
                            # Get category, group, period names from dictionaries
                            category_name = next((c.get("name") for c in categories if c.get("id") == category_id), "Unknown")
                            group_name = next((g.get("name") for g in groups if g.get("id") == group_id), "Unknown")
                            period_name = next((p.get("name") for p in periods if p.get("id") == period_id), "Unknown")
                        
                        st.markdown(f"""
                        **Atomic Number:** {atomic_number}  
                        **Name:** {name}  
                        **Symbol:** {element_symbol}  
                        **Category:** {category_name}  
                        **Group:** {group_name}  
                        **Period:** {period_name}  
                        """)
                        
                        # Display properties
                        st.markdown("**Properties:**")
                        
                        # Try both object and dictionary access for properties
                        try:
                            # Try to access properties via Element object method
                            from core.periodic_table.element import ElementProperty
                            props = [
                                ("Reliability", element.get_property(ElementProperty.RELIABILITY, 0.0)),
                                ("Confidence", element.get_property(ElementProperty.CONFIDENCE, 0.0)),
                                ("Accessibility", element.get_property(ElementProperty.ACCESSIBILITY, 0.0)),
                                ("Maturity", element.get_property(ElementProperty.MATURITY, 0)),
                                ("Complexity", element.get_property(ElementProperty.COMPLEXITY, 0)),
                                ("Priority", element.get_property(ElementProperty.PRIORITY, 0)),
                                ("Stability", element.get_property(ElementProperty.STABILITY, 0.0))
                            ]
                        except (AttributeError, ImportError):
                            # Fall back to dictionary access
                            properties = element.get('properties', {})
                            props = [
                                ("Reliability", properties.get('reliability', 0.0)),
                                ("Confidence", properties.get('confidence', 0.0)),
                                ("Accessibility", properties.get('accessibility', 0.0)),
                                ("Maturity", properties.get('maturity', 0)),
                                ("Complexity", properties.get('complexity', 0)),
                                ("Priority", properties.get('priority', 0)),
                                ("Stability", properties.get('stability', 0.0))
                            ]
                        
                        for name, value in props:
                            if name in ["Reliability", "Confidence", "Accessibility", "Stability"]:
                                st.markdown(f"* **{name}:** {value:.2f}")
                            else:
                                st.markdown(f"* **{name}:** {value}/10")
            
            # Button to compare selected elements
            if len(st.session_state.selected_elements) >= 2:
                if st.button(f"Compare Selected Elements ({len(st.session_state.selected_elements)})"):
                    st.session_state.comparison_mode = True
                    # Limit to the first num_to_compare elements
                    st.session_state.selected_elements = st.session_state.selected_elements[:num_to_compare]
                    # No need to use experimental_rerun, just switch to the tab in the next render
        else:
            st.info("No elements match the selected filters.")
    
    with tab2:
        st.markdown('<div class="section-title">Node Comparison</div>', unsafe_allow_html=True)
        
        # Check if we have elements to compare
        if len(st.session_state.selected_elements) >= 2:
            # Get selected elements
            selected_elements = [registry.get_element(element_id) 
                               for element_id in st.session_state.selected_elements]
            
            # Display comparison
            st.markdown("### Selected Elements for Comparison")
            
            # Display node cards in columns
            cols = st.columns(len(selected_elements))
            
            for i, element in enumerate(selected_elements):
                with cols[i]:
                    st.markdown(f"**{element.get('name', 'Unknown')} ({element.get('short_name', element.get('symbol', '?'))})**")
                    st.markdown(render_node_card(element), unsafe_allow_html=True)
            
            # Property comparisons
            st.markdown("### Property Comparison")
            
            # Compare reliability
            render_property_comparison(
                selected_elements, 
                "Reliability", 
                ElementProperty.RELIABILITY, 
                max_value=1.0, 
                scale_factor=100
            )
            
            # Compare confidence
            render_property_comparison(
                selected_elements, 
                "Confidence", 
                ElementProperty.CONFIDENCE, 
                max_value=1.0, 
                scale_factor=100
            )
            
            # Compare maturity
            render_property_comparison(
                selected_elements, 
                "Maturity", 
                ElementProperty.MATURITY, 
                max_value=10
            )
            
            # Compare complexity
            render_property_comparison(
                selected_elements, 
                "Complexity", 
                ElementProperty.COMPLEXITY, 
                max_value=10
            )
            
            # Compare accessibility
            render_property_comparison(
                selected_elements, 
                "Accessibility", 
                ElementProperty.ACCESSIBILITY, 
                max_value=1.0, 
                scale_factor=100
            )
            
            # Compare stability
            render_property_comparison(
                selected_elements, 
                "Stability", 
                ElementProperty.STABILITY, 
                max_value=1.0, 
                scale_factor=100
            )
            
            # Compare priority
            render_property_comparison(
                selected_elements, 
                "Priority", 
                ElementProperty.PRIORITY, 
                max_value=10
            )
            
            # Relationship comparison
            st.markdown("### Relationship Comparison")
            
            # Get relationships for each element
            element_relationships = {}
            for element in selected_elements:
                rels = registry.get_element_relationships(element.id)
                element_relationships[element.id] = rels
            
            # Create relationship summary
            for element in selected_elements:
                rels = element_relationships[element.id]
                rel_count = sum(len(connections) for connections in rels.values())
                
                st.markdown(f"**{element.name} ({element.symbol})** - {rel_count} relationships")
                
                # Group by relationship type
                if rels:
                    for rel_type, connections in rels.items():
                        if connections:
                            st.markdown(f"* **{rel_type}**: {len(connections)} connections")
                else:
                    st.markdown("* No relationships")
            
            # Button to clear comparison
            if st.button("Clear Comparison"):
                st.session_state.selected_elements = []
                st.session_state.comparison_mode = False
        else:
            st.info("Select at least two elements to compare from the Node Cards tab.")

# Run the app
if __name__ == "__main__":
    main()