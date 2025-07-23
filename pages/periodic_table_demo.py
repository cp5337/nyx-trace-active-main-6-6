"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-PERIODIC-TABLE-DEMO-0001            │
// │ 📁 domain       : Visualization, Demo                       │
// │ 🧠 description  : Demo page for the CTAS Periodic Table     │
// │                  of Nodes visualization                     │
// │ 🕸️ hash_type    : UUID → CUID-linked module                 │
// │ 🔄 parent_node  : NODE_VISUALIZATION                       │
// │ 🧩 dependencies : streamlit, plotly                        │
// │ 🔧 tool_usage   : Visualization, Demo                      │
// │ 📡 input_type   : User interface events                     │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : visualization, interaction                │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

Periodic Table Demo Page
---------------------
This page provides an interactive demonstration of the CTAS Periodic Table
of Nodes, allowing users to explore elements, relationships, and properties.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import uuid
import os
import json
from pathlib import Path
import random
from datetime import datetime, timedelta

from core.periodic_table.table import PeriodicTable
from core.periodic_table.registry import PeriodicTableRegistry
from core.periodic_table.element import Element, ElementProperty
from core.periodic_table.group import Group, Period, Category, GroupType, CATEGORY_COLORS
from core.periodic_table.relationships import Relationship, RelationshipType

# Set page config
st.set_page_config(
    page_title="CTAS Periodic Table of Nodes",
    page_icon="🔬",
    layout="wide"
)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'registry' not in st.session_state:
    st.session_state.registry = None
if 'periodic_table' not in st.session_state:
    st.session_state.periodic_table = None
if 'selected_element_id' not in st.session_state:
    st.session_state.selected_element_id = None

# Custom CSS
st.markdown("""
<style>
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
    .element-card {
        border: 2px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        margin: 5px;
    }
    .stats-container {
        display: flex;
        justify-content: space-around;
        flex-wrap: wrap;
        margin: 1rem 0;
    }
    .stat-card {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 15px;
        margin: 5px;
        text-align: center;
        flex: 1;
        min-width: 120px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stat-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1E3A8A;
    }
    .stat-label {
        font-size: 0.9rem;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-title">CTAS Periodic Table of Nodes</div>', unsafe_allow_html=True)

# Description
st.markdown("""
The CTAS Periodic Table of Nodes provides a structured classification system for intelligence entities,
enabling organized traversal and relationship mapping across different types of nodes. This interactive
visualization allows exploration of elements, their properties, and relationships.
""")

# Initialize the registry and periodic table
def initialize_registry():
    """Initialize the registry with sample data for demonstration."""
    try:
        # Create registry with in-memory database to avoid SQLite threading issues
        registry = PeriodicTableRegistry(db_path=":memory:")
        print("Created in-memory registry to avoid SQLite threading issues")
        
        # Create groups (vertical columns)
        groups = []
        for i in range(1, 11):
            group = Group(
                name=f"Group {i}",
                symbol=f"G{i}",
                number=i,
                description=f"Vertical classification group {i}",
                type=GroupType.VERTICAL
            )
            registry.add_group(group)
            groups.append(group)
        
        # Create periods (horizontal rows)
        periods = []
        for i in range(1, 8):
            period = Period(
                name=f"Period {i}",
                number=i,
                description=f"Horizontal classification period {i}"
            )
            registry.add_period(period)
            periods.append(period)
        
        # Create categories
        categories = []
        category_names = [
            "ENTITY", "INFRASTRUCTURE", "CAPABILITY", "THREAT", 
            "ACTOR", "LOCATION", "EVENT", "INTELLIGENCE",
            "ALGORITHM", "RESOURCE", "RELATIONSHIP", "ATTRIBUTE"
        ]
        
        for name in category_names:
            color = CATEGORY_COLORS.get(name, "#CCCCCC")
            category = Category(
                name=name,
                symbol=name[:3].upper(),
                color=color,
                description=f"{name} classification category"
            )
            registry.add_category(category)
            categories.append(category)
        
        # Create elements
        elements = []
        atomic_number = 1
        
        # Create one element for each period-group combination with varied categories
        for period in periods:
            for group in groups:
                if random.random() < 0.7:  # 70% chance to create an element
                    # Pick a random category
                    category = random.choice(categories)
                    
                    # Create symbol: first letter of category + atomic number
                    symbol = f"{category.name[0]}{atomic_number}"
                    
                    # Create element name: category + classification + number
                    name = f"{category.name.capitalize()} {atomic_number}"
                    
                    # Create element
                    element = Element(
                        atomic_number=atomic_number,
                        symbol=symbol,
                        name=name,
                        group_id=group.id,
                        period_id=period.id,
                        category_id=category.id
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
        
        # Add some random relationships
        for i in range(min(100, len(elements) * 3)):  # Create up to 3 relationships per element
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
        print(f"Created {len(all_elements)} elements and {len(registry.relationship_manager.relationships)} relationships in registry")
        return registry
    except Exception as e:
        print(f"Error initializing registry: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# Initialize on first run
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
    
    # Create a periodic table if it doesn't exist yet
    if 'periodic_table' not in st.session_state or st.session_state.periodic_table is None:
        # Import the necessary module and create a fallback table renderer
        try:
            from core.periodic_table.table_renderer import PeriodicTableRenderer
            st.session_state.periodic_table = PeriodicTableRenderer(registry)
        except ImportError:
            # Create a simple fallback function
            class SimpleTableRenderer:
                def __init__(self, registry):
                    self.registry = registry
                
                def create_plotly_table(self, width=800, height=600, color_by='category'):
                    import plotly.graph_objects as go
                    
                    # Get elements from registry
                    elements = self.registry.get_all_elements()
                    
                    # Create a simple grid layout
                    fig = go.Figure()
                    
                    # Add elements as boxes
                    for i, element in enumerate(elements):
                        row = i // 10  # 10 elements per row
                        col = i % 10
                        
                        # Extract properties
                        name = element.get('name', 'Unknown')
                        symbol = element.get('short_name', '?')
                        
                        # Add element box
                        fig.add_trace(go.Scatter(
                            x=[col, col, col+0.9, col+0.9, col],
                            y=[row, row+0.9, row+0.9, row, row],
                            mode='lines',
                            line=dict(color='white', width=1),
                            fill='toself',
                            fillcolor='rgba(31, 119, 180, 0.7)',
                            hoverinfo='text',
                            text=f"{name} ({symbol})",
                            showlegend=False
                        ))
                        
                        # Add element symbol
                        fig.add_annotation(
                            x=col+0.45,
                            y=row+0.45,
                            text=symbol,
                            showarrow=False,
                            font=dict(color='white', size=14)
                        )
                    
                    # Update layout
                    fig.update_layout(
                        width=width,
                        height=height,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        margin=dict(l=0, r=0, t=0, b=0),
                        xaxis=dict(
                            range=[-0.5, 10.5],
                            showgrid=False,
                            zeroline=False,
                            showticklabels=False
                        ),
                        yaxis=dict(
                            range=[-0.5, (len(elements) // 10) + 0.5],
                            showgrid=False,
                            zeroline=False,
                            showticklabels=False
                        )
                    )
                    
                    return fig
            
            st.session_state.periodic_table = SimpleTableRenderer(registry)
    
    periodic_table = st.session_state.periodic_table
    
    # Safety check
    if registry is None:
        st.error("Registry initialization failed. Please refresh the page.")
        st.stop()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Periodic Table", "Element Details", "Relationship Network"])
    
    # Tab 1: Periodic Table
    with tab1:
        st.markdown('<div class="section-title">Interactive Periodic Table</div>', unsafe_allow_html=True)
        
        # Color options
        color_options = {
            "Category": "category",
            "Reliability": "reliability",
            "Confidence": "confidence",
            "Maturity": "maturity",
            "Complexity": "complexity"
        }
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            color_by = st.selectbox("Color elements by", list(color_options.keys()))
            
            # Display statistics
            elements = registry.get_all_elements()
            categories = registry.get_all_categories()
            relationships = list(registry.relationship_manager.relationships.values())
            
            st.markdown('<div class="section-title">Statistics</div>', unsafe_allow_html=True)
            
            st.markdown("""
            <div class="stats-container">
                <div class="stat-card">
                    <div class="stat-value">{}</div>
                    <div class="stat-label">Elements</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{}</div>
                    <div class="stat-label">Categories</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{}</div>
                    <div class="stat-label">Relationships</div>
                </div>
            </div>
            """.format(len(elements), len(categories), len(relationships)), 
            unsafe_allow_html=True)
            
            # Display filters
            st.markdown('<div class="section-title">Filters</div>', unsafe_allow_html=True)
            
            # Category filter
            category_names = [c["name"] for c in categories]
            selected_categories = st.multiselect("Categories", category_names, default=category_names)
            
            # Property filters
            min_reliability = st.slider("Min Reliability", 0.0, 1.0, 0.0, 0.1)
            min_complexity = st.slider("Min Complexity", 1, 10, 1)
            max_complexity = st.slider("Max Complexity", 1, 10, 10)
        
        with col1:
            # Create the plotly figure
            fig = periodic_table.create_plotly_table(
                width=800, 
                height=600, 
                color_by=color_options[color_by]
            )
            
            # Display the plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Handle element selection
            selected_points = st.session_state.get('plotly_selected_points', None)
            if selected_points:
                point = selected_points[0]
                element_id = uuid.UUID(point['customdata'][0])
                st.session_state.selected_element_id = element_id
                
                # Display selected element
                element = registry.get_element(element_id)
                if element:
                    st.markdown(f"**Selected Element:** {element.name} ({element.symbol})")
    
    # Tab 2: Element Details
    with tab2:
        st.markdown('<div class="section-title">Element Details</div>', unsafe_allow_html=True)
        
        # If no element is selected, let user select one
        if not st.session_state.selected_element_id:
            elements = registry.get_all_elements()
            element_options = {f"{e.get('name', 'Unknown')} ({e.get('short_name', e.get('symbol', '?'))})": e.get('id', str(uuid.uuid4())) for e in elements}
            
            selected_element_name = st.selectbox(
                "Select an element", 
                list(element_options.keys())
            )
            
            if selected_element_name:
                st.session_state.selected_element_id = element_options[selected_element_name]
        
        # Display element details
        if st.session_state.selected_element_id:
            element_id = st.session_state.selected_element_id
            element = registry.get_element(element_id)
            
            if element:
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # Display element card - handle both object and dictionary formats
                    try:
                        # Try object attribute access first
                        name = element.name
                        symbol = element.symbol
                        atomic_number = element.atomic_number
                        category_id = element.category_id
                        group_id = element.group_id
                        period_id = element.period_id
                    except AttributeError:
                        # Fall back to dictionary access if needed
                        name = element.get('name', 'Unknown')
                        symbol = element.get('short_name', element.get('symbol', '?'))
                        atomic_number = element.get('atomic_number', 0)
                        category_id = element.get('category_id', '')
                        group_id = element.get('group_id', '')
                        period_id = element.get('period_id', '')
                    
                    st.markdown(f"### {name} ({symbol})")
                    
                    # Display basic info - handle both object and dictionary formats for categories, groups, periods
                    categories = registry.get_all_categories()
                    groups = registry.get_all_groups()
                    periods = registry.get_all_periods()
                    
                    # Try both access methods for categories, groups, periods
                    try:
                        # Try object access first
                        category_name = next((c.name for c in categories if c.id == category_id), "Unknown")
                        group_name = next((g.name for g in groups if g.id == group_id), "Unknown")
                        period_name = next((p.name for p in periods if p.id == period_id), "Unknown")
                    except AttributeError:
                        # Fall back to dictionary access
                        category_name = next((c.get('name', 'Unknown') for c in categories if c.get('id', '') == category_id), "Unknown")
                        group_name = next((g.get('name', 'Unknown') for g in groups if g.get('id', '') == group_id), "Unknown")
                        period_name = next((p.get('name', 'Unknown') for p in periods if p.get('id', '') == period_id), "Unknown")
                    
                    st.markdown(f"""
                    - **Atomic Number:** {atomic_number}
                    - **Symbol:** {symbol}
                    - **Category:** {category_name}
                    - **Group:** {group_name}
                    - **Period:** {period_name}
                    """)
                    
                    # Display properties
                    st.markdown("### Properties")
                    
                    prop_df = pd.DataFrame({
                        "Property": [
                            "Reliability", "Confidence", "Accessibility", 
                            "Maturity", "Complexity", "Priority", 
                            "Stability", "Computation Cost"
                        ],
                        "Value": [
                            f"{element.get_property(ElementProperty.RELIABILITY, 0.0):.2f}",
                            f"{element.get_property(ElementProperty.CONFIDENCE, 0.0):.2f}",
                            f"{element.get_property(ElementProperty.ACCESSIBILITY, 0.0):.2f}",
                            f"{element.get_property(ElementProperty.MATURITY, 0)}/10",
                            f"{element.get_property(ElementProperty.COMPLEXITY, 0)}/10",
                            f"{element.get_property(ElementProperty.PRIORITY, 0)}/10",
                            f"{element.get_property(ElementProperty.STABILITY, 0.0):.2f}",
                            f"{element.get_property(ElementProperty.COMPUTATION_COST, 0)}/10"
                        ]
                    })
                    
                    st.dataframe(prop_df)
                
                with col2:
                    # Display detail plot
                    fig = periodic_table.create_element_detail_plot(element_id, width=800, height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    
                # Display relationships
                st.markdown("### Relationships")
                relationships = registry.get_element_relationships(element_id)
                
                if relationships:
                    # Group by relationship type
                    for rel_type, connections in relationships.items():
                        if connections:
                            st.markdown(f"#### {rel_type}")
                            
                            # Create table of connections
                            data = []
                            for relationship, connected_id in connections:
                                connected_element = registry.get_element(connected_id)
                                if connected_element:
                                    data.append({
                                        "Element": f"{connected_element.name} ({connected_element.symbol})",
                                        "Weight": f"{relationship.weight:.2f}",
                                        "Confidence": f"{relationship.confidence:.2f}",
                                        "Bidirectional": "Yes" if relationship.bidirectional else "No"
                                    })
                            
                            if data:
                                st.dataframe(pd.DataFrame(data))
                else:
                    st.info("No relationships found for this element.")
    
    # Tab 3: Relationship Network
    with tab3:
        st.markdown('<div class="section-title">Relationship Network</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            # Select root element
            elements = registry.get_all_elements()
            element_options = {f"{e.name} ({e.symbol})": e.id for e in elements}
            
            selected_element_name = st.selectbox(
                "Select root element", 
                list(element_options.keys()),
                key="network_element"
            )
            
            if selected_element_name:
                root_element_id = element_options[selected_element_name]
            else:
                root_element_id = None
            
            # Select depth
            depth = st.slider("Relationship Depth", 1, 5, 2)
            
            # Select relationship type filter
            rel_types = [rt.name for rt in RelationshipType]
            selected_rel_types = st.multiselect("Relationship Types", rel_types, default=rel_types[:5])
        
        with col1:
            # Display network graph
            if root_element_id:
                fig = periodic_table.create_network_graph(
                    element_id=root_element_id,
                    max_depth=depth,
                    width=800,
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Select a root element to visualize the relationship network.")

# Run the app
if __name__ == "__main__":
    main()