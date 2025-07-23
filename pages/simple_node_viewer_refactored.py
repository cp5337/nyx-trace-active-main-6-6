"""
Simple Node Card Viewer
---------------------
A streamlined, lightweight implementation of the node card viewer
that doesn't rely on SQLite or complex database operations.
"""

import streamlit as st
import pandas as pd
import uuid
import json
import random
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="CTAS Node Card Viewer",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom styling
st.markdown("""
<style>
    .node-card {
        background-color: #2C3E50;
        border-radius: 8px;
        padding: 12px;
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
    
    .node-symbol {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
    }
    
    .node-name {
        font-size: 1rem;
        margin-top: 5px;
        text-align: center;
        font-weight: bold;
    }
    
    .node-description {
        font-size: 0.8rem;
        margin-top: 10px;
        color: rgba(255,255,255,0.8);
    }
    
    .node-properties {
        margin-top: 10px;
    }
    
    .property-row {
        display: flex;
        justify-content: space-between;
        margin-bottom: 3px;
        font-size: 0.8rem;
    }
    
    .property-name {
        color: rgba(255,255,255,0.7);
    }
    
    .property-value {
        font-weight: bold;
    }
    
    .property-bar {
        height: 5px;
        background-color: rgba(255,255,255,0.2);
        border-radius: 2px;
        margin-top: 2px;
        margin-bottom: 8px;
    }
    
    .property-fill {
        height: 100%;
        border-radius: 2px;
    }
</style>
""",
    unsafe_allow_html=True,
)


# Load sample data
def load_sample_nodes():
    """
    Load sample nodes from JSON files in attached_assets directory.
    This is a simple alternative to using SQLite.
    """
    nodes = []
    
    # Load from provided interview files if they exist
    for i in range(1, 12):
        num = str(i).zfill(3)
        file_path = Path(f"attached_assets/node_{num}_interview.json")
        
        if file_path.exists():
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    
                    # Clean and transform data
                    node = {
                        "id": str(uuid.uuid4()),
                        "name": data.get("name", f"Node {num}"),
                        "symbol": data.get("symbol", f"N{num}"),
                        "atomic_number": i,
                        "category": data.get("category", "ENTITY"),
                        "description": data.get("description", 
                                              "No description available"),
                        "properties": {
                            "reliability": random.uniform(0.7, 1.0),
                            "confidence": random.uniform(0.7, 1.0),
                            "complexity": random.randint(3, 10),
                            "maturity": random.randint(5, 10),
                            "priority": random.randint(1, 10)
                        }
                    }
                    nodes.append(node)
            except Exception as e:
                st.error(f"Error loading node {num}: {str(e)}")
    
    # If no nodes were loaded, create some sample ones
    if not nodes:
        categories = ["ENTITY", "INFRASTRUCTURE", "CAPABILITY", "THREAT", 
                     "ACTOR", "LOCATION", "EVENT", "INTELLIGENCE"]
        
        for i in range(1, 15):
            category = random.choice(categories)
            nodes.append({
                "id": str(uuid.uuid4()),
                "name": f"{category} {i}",
                "symbol": f"{category[0]}{i}",
                "atomic_number": i,
                "category": category,
                "description": f"Sample {category.lower()} node for demonstration",
                "properties": {
                    "reliability": random.uniform(0.7, 1.0),
                    "confidence": random.uniform(0.7, 1.0),
                    "complexity": random.randint(3, 10),
                    "maturity": random.randint(5, 10),
                    "priority": random.randint(1, 10)
                }
            })
    
    return nodes


# Define category colors globally to avoid recreating in functions
CATEGORY_COLORS = {
    "ENTITY": "#3498DB",
    "INFRASTRUCTURE": "#E74C3C",
    "CAPABILITY": "#2ECC71",
    "THREAT": "#F39C12",
    "ACTOR": "#9B59B6",
    "LOCATION": "#1ABC9C",
    "EVENT": "#E67E22",
    "INTELLIGENCE": "#34495E",
}


def extract_node_properties(node):
    """
    Extract properties from a node data dictionary.
    
    Args:
        node: Dictionary containing node data
        
    Returns:
        tuple: Extracted node properties
    """
    # Extract basic node properties
    name = node.get("name", "Unknown")
    symbol = node.get("symbol", "??")
    atomic_number = node.get("atomic_number", 0)
    category = node.get("category", "ENTITY")
    description = node.get("description", "No description available")
    
    # Extract metric properties
    props = node.get("properties", {})
    reliability = props.get("reliability", 0.8)
    confidence = props.get("confidence", 0.8)
    complexity = props.get("complexity", 5)
    maturity = props.get("maturity", 5)
    priority = props.get("priority", 5)
    
    return (
        name, symbol, atomic_number, category, description,
        reliability, confidence, complexity, maturity, priority
    )


def create_property_bar_html(name, value, max_value, color):
    """
    Create HTML for a property bar visualization.
    
    Args:
        name: Property name
        value: Property value
        max_value: Maximum possible value (for percentage calculation)
        color: Bar color
        
    Returns:
        str: HTML for the property bar
    """
    # Calculate percentage for bar width
    percentage = (value / max_value) * 100 if max_value else 0
    
    # Format display value
    if max_value == 1:  # Percentage properties like reliability
        display_value = f"{int(value * 100)}%"
    else:  # Integer scale properties like complexity
        display_value = f"{value}/{max_value}"
    
    return f"""
    <div class="property-row">
        <div class="property-name">{name}:</div>
        <div class="property-value">{display_value}</div>
    </div>
    <div class="property-bar">
        <div class="property-fill" style="width: {percentage}%; background-color: {color};"></div>
    </div>
    """


def render_node_card(node):
    """
    Render an HTML card for a node.
    
    Args:
        node: Dictionary containing node data
        
    Returns:
        str: HTML representation of the node card
    """
    # Extract all node properties
    (
        name, symbol, atomic_number, category, description,
        reliability, confidence, complexity, maturity, priority
    ) = extract_node_properties(node)
    
    # Get color based on category
    color = CATEGORY_COLORS.get(category, "#1f77b4")
    
    # Create the HTML card header
    card_html = f"""
    <div class="node-card" style="background-color: {color};">
        <div class="node-header">
            <div class="node-title">{atomic_number}.0 {category}</div>
            <div>TTL: 6.5</div>
        </div>
        
        <div class="node-symbol">{symbol}</div>
        <div class="node-name">{name}</div>
        
        <div class="node-description">{description[:100]}...</div>
        
        <div class="node-properties">
    """
    
    # Add property bars
    card_html += create_property_bar_html(
        "Reliability", reliability, 1, "#2ECC71"
    )
    card_html += create_property_bar_html(
        "Confidence", confidence, 1, "#3498DB"
    )
    card_html += create_property_bar_html(
        "Complexity", complexity, 10, "#E74C3C"
    )
    card_html += create_property_bar_html(
        "Maturity", maturity, 10, "#F39C12"
    )
    
    # Close HTML tags
    card_html += """
        </div>
    </div>
    """
    
    return card_html


def setup_page_header():
    """
    Set up the page header and title section
    """
    st.markdown("# CTAS Node Card Viewer")
    st.markdown("Explore and compare node cards from the CTAS Periodic Table")


def initialize_session_state():
    """
    Initialize the session state with nodes if not already loaded
    """
    if "nodes" not in st.session_state:
        st.session_state.nodes = load_sample_nodes()
        st.session_state.selected_nodes = []


def create_filter_controls():
    """
    Create and render the filter controls
    
    Returns:
        tuple: (selected_category, search_query)
    """
    st.markdown("### Filter Nodes")
    col1, col2 = st.columns(2)
    
    # Category filter
    with col1:
        categories = list(
            set(node.get("category") for node in st.session_state.nodes)
        )
        selected_category = st.selectbox("Category", ["All"] + categories)
    
    # Search filter
    with col2:
        search_query = st.text_input("Search by name")
        
    return selected_category, search_query


def filter_nodes(nodes, category=None, search_query=None):
    """
    Filter nodes based on category and search query
    
    Args:
        nodes: List of node dictionaries
        category: Optional category to filter by
        search_query: Optional search string for name filtering
        
    Returns:
        List of filtered nodes
    """
    filtered = nodes
    
    # Filter by category if specified
    if category and category != "All":
        filtered = [n for n in filtered if n.get("category") == category]
    
    # Filter by search query if specified
    if search_query:
        search_lower = search_query.lower()
        filtered = [
            n for n in filtered if search_lower in n.get("name", "").lower()
        ]
        
    return filtered


def display_node_details(node):
    """
    Display detailed properties of a node in an expander
    
    Args:
        node: Node dictionary
    """
    with st.expander("Node Details"):
        st.markdown(
            f"""
        **Name:** {node.get('name', 'Unknown')}  
        **Symbol:** {node.get('symbol', '??')}  
        **Category:** {node.get('category', 'Unknown')}  
        **Description:** {node.get('description', 'No description available')}
        """
        )
        
        # Show properties
        st.markdown("**Properties:**")
        props = node.get("properties", {})
        for prop_name, prop_value in props.items():
            if isinstance(prop_value, float):
                st.markdown(
                    f"* **{prop_name.capitalize()}:** {prop_value:.2f}"
                )
            else:
                st.markdown(
                    f"* **{prop_name.capitalize()}:** {prop_value}"
                )


def display_node_grid(filtered_nodes):
    """
    Display a grid of node cards with selection checkboxes
    
    Args:
        filtered_nodes: List of node dictionaries to display
    """
    if not filtered_nodes:
        st.warning("No nodes match the current filters.")
        return
        
    # Display selection instructions
    st.markdown("Select nodes to compare them in the 'Compare Nodes' tab")
    
    # Create 3-column grid
    cols = st.columns(3)
    
    # Populate grid with node cards
    for i, node in enumerate(filtered_nodes):
        col_idx = i % 3
        
        with cols[col_idx]:
            # Add checkbox for comparison
            node_id = node.get("id")
            is_selected = st.checkbox(
                f"Compare {node.get('symbol')}",
                key=f"compare_{node_id}",
            )
            
            # Update selected nodes tracking
            if (
                is_selected
                and node_id not in st.session_state.selected_nodes
            ):
                st.session_state.selected_nodes.append(node_id)
            elif (
                not is_selected
                and node_id in st.session_state.selected_nodes
            ):
                st.session_state.selected_nodes.remove(node_id)
            
            # Render node card and details
            st.markdown(render_node_card(node), unsafe_allow_html=True)
            display_node_details(node)


def render_browse_tab():
    """
    Render the "Browse Nodes" tab content
    """
    # Create filter controls
    selected_category, search_query = create_filter_controls()
    
    # Apply filters
    filtered_nodes = filter_nodes(
        st.session_state.nodes, selected_category, search_query
    )
    
    # Display nodes in a grid
    st.markdown("### Node Cards")
    display_node_grid(filtered_nodes)


def create_comparison_dataframe(nodes):
    """
    Create a dataframe for node property comparison
    
    Args:
        nodes: List of node dictionaries
        
    Returns:
        DataFrame containing node properties
    """
    comp_data = []
    for node in nodes:
        props = node.get("properties", {})
        row = {
            "Node": f"{node.get('symbol')} - {node.get('name')}",
            "Reliability": props.get("reliability", 0) * 100,
            "Confidence": props.get("confidence", 0) * 100,
            "Complexity": props.get("complexity", 0),
            "Maturity": props.get("maturity", 0),
            "Priority": props.get("priority", 0),
        }
        comp_data.append(row)
        
    return pd.DataFrame(comp_data) if comp_data else None


def display_property_charts(df):
    """
    Display comparison charts for each property
    
    Args:
        df: DataFrame with node properties
    """
    properties = [
        "Reliability",
        "Confidence",
        "Complexity",
        "Maturity",
        "Priority",
    ]
    
    for prop in properties:
        st.markdown(f"#### {prop} Comparison")
        chart_data = df[["Node", prop]].set_index("Node")
        st.bar_chart(chart_data)


def render_comparison_tab():
    """
    Render the "Compare Nodes" tab content
    """
    st.markdown("### Compare Selected Nodes")
    
    # Get selected nodes
    selected_node_objects = [
        n
        for n in st.session_state.nodes
        if n.get("id") in st.session_state.selected_nodes
    ]
    
    # Handle cases with insufficient nodes
    if not selected_node_objects:
        st.info("Select nodes in the 'Browse Nodes' tab to compare them here.")
        return
    elif len(selected_node_objects) == 1:
        st.info("Select at least one more node to enable comparison.")
        return
    
    # Display selected nodes side by side
    st.markdown(f"#### Comparing {len(selected_node_objects)} Nodes")
    
    # Create columns for each selected node (limit to 4)
    display_nodes = selected_node_objects[:4]
    cols = st.columns(len(display_nodes))
    
    for i, node in enumerate(display_nodes):
        with cols[i]:
            st.markdown(f"**{node.get('name')} ({node.get('symbol')})**")
            st.markdown(render_node_card(node), unsafe_allow_html=True)
    
    # Show property comparison charts
    st.markdown("### Property Comparison")
    
    # Create and display comparison data
    df = create_comparison_dataframe(selected_node_objects)
    if df is not None:
        display_property_charts(df)
        
        # Show a summary table
        st.markdown("### Summary Table")
        st.dataframe(df.set_index("Node"))


def main():
    """
    Main entry point for the Node Card Viewer application
    """
    # Set up page header and initialize data
    setup_page_header()
    initialize_session_state()
    
    # Create tabs
    tab1, tab2 = st.tabs(["Browse Nodes", "Compare Nodes"])
    
    # Render content for each tab
    with tab1:
        render_browse_tab()
    
    with tab2:
        render_comparison_tab()


if __name__ == "__main__":
    main()