"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-CORE-GEOSPATIAL-VIZERS-NET-0001     │
// │ 📁 domain       : Geospatial, Visualization                │
// │ 🧠 description  : Network graph visualization              │
// │                  Node-edge network maps                    │
// │ 🕸️ hash_type    : UUID → CUID-linked module                │
// │ 🔄 parent_node  : NODE_VIZERS                              │
// │ 🧩 dependencies : folium, pandas, networkx                 │
// │ 🔧 tool_usage   : Visualization                           │
// │ 📡 input_type   : Node data, edge connections               │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : network analysis, visualization          │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

Network Graph Visualization
-------------------------
This module provides functions for creating network graph visualizations
on maps, depicting nodes and connections between geographic locations.
"""

import folium
import pandas as pd
import numpy as np
import networkx as nx
from folium.plugins import PolyLineOffset
from typing import Dict, List, Tuple, Optional, Any, Union


def create_network_graph(
    nodes_data: pd.DataFrame,
    edges_data: pd.DataFrame,
    map_obj: Optional[folium.Map] = None,
    node_size_field: Optional[str] = None,
    node_color_field: Optional[str] = None,
    edge_weight_field: Optional[str] = None,
    edge_color_field: Optional[str] = None,
    directed: bool = False,
    arrow_size: int = 3,
    show_labels: bool = False,
) -> folium.Map:
    """
    Create a geographic network graph on a map

    # Function creates subject network
    # Method visualizes predicate connections
    # Operation displays object relationships

    Args:
        nodes_data: DataFrame with node data including latitude, longitude
        edges_data: DataFrame with source, target edge connections
        map_obj: Existing Folium map object (creates new if None)
        node_size_field: Column in nodes_data for sizing nodes
        node_color_field: Column in nodes_data for coloring nodes
        edge_weight_field: Column in edges_data for line thickness
        edge_color_field: Column in edges_data for line color
        directed: Whether to show directional arrows
        arrow_size: Size of directional arrows (if directed=True)
        show_labels: Whether to display node labels

    Returns:
        Folium map with network graph visualization
    """
    # Function validates subject nodes
    # Method checks predicate data
    # Condition verifies object table
    if nodes_data is None or nodes_data.empty:
        # Function raises subject error
        # Method signals predicate problem
        # Exception reports object issue
        raise ValueError("Nodes data cannot be empty for network visualization")

    # Function validates subject edges
    # Method checks predicate data
    # Condition verifies object table
    if edges_data is None or edges_data.empty:
        # Function raises subject error
        # Method signals predicate problem
        # Exception reports object issue
        raise ValueError("Edges data cannot be empty for network visualization")

    # Function validates subject columns
    # Method checks predicate requirements
    # Condition verifies object structure
    required_node_cols = ["id", "latitude", "longitude"]
    if not all(col in nodes_data.columns for col in required_node_cols):
        # Function raises subject error
        # Method signals predicate problem
        # Exception reports object issue
        missing = [
            col for col in required_node_cols if col not in nodes_data.columns
        ]
        raise ValueError(f"Missing required node columns: {missing}")

    # Function validates subject columns
    # Method checks predicate requirements
    # Condition verifies object structure
    required_edge_cols = ["source", "target"]
    if not all(col in edges_data.columns for col in required_edge_cols):
        # Function raises subject error
        # Method signals predicate problem
        # Exception reports object issue
        missing = [
            col for col in required_edge_cols if col not in edges_data.columns
        ]
        raise ValueError(f"Missing required edge columns: {missing}")

    # Function creates subject map
    # Method initializes predicate object
    # Folium prepares object container
    if map_obj is None:
        # Function calculates subject center
        # Method determines predicate coordinates
        # Operation computes object average
        center = [nodes_data["latitude"].mean(), nodes_data["longitude"].mean()]

        # Function creates subject map
        # Method initializes predicate object
        # Folium creates object container
        map_obj = folium.Map(location=center, zoom_start=10, control_scale=True)

    # Function creates subject network
    # Method initializes predicate graph
    # NetworkX builds object structure
    G = nx.DiGraph() if directed else nx.Graph()

    # Function adds subject nodes
    # Method populates predicate graph
    # NetworkX creates object vertices
    for idx, row in nodes_data.iterrows():
        # Function adds subject node
        # Method creates predicate vertex
        # NetworkX records object data
        G.add_node(
            row["id"],
            pos=(row["latitude"], row["longitude"]),
            size=row.get(node_size_field, 10) if node_size_field else 10,
            color=(
                row.get(node_color_field, "blue")
                if node_color_field
                else "blue"
            ),
            label=row.get("label", str(row["id"])),
        )

    # Function adds subject edges
    # Method populates predicate graph
    # NetworkX creates object connections
    for idx, row in edges_data.iterrows():
        # Function adds subject edge
        # Method creates predicate connection
        # NetworkX records object relationship
        G.add_edge(
            row["source"],
            row["target"],
            weight=row.get(edge_weight_field, 1) if edge_weight_field else 1,
            color=(
                row.get(edge_color_field, "blue")
                if edge_color_field
                else "blue"
            ),
        )

    # Function normalizes subject sizes
    # Method scales predicate values
    # Operation adjusts object range
    if node_size_field:
        # Function extracts subject sizes
        # Method retrieves predicate values
        # List stores object numbers
        sizes = [G.nodes[n]["size"] for n in G.nodes]

        # Function calculates subject range
        # Method determines predicate bounds
        # Variables store object limits
        min_size, max_size = min(sizes), max(sizes)

        # Function prevents subject division
        # Method handles predicate zero
        # Condition prevents object error
        if min_size == max_size:
            # Function normalizes subject sizes
            # Method assigns predicate value
            # Operation standardizes object metrics
            for node in G.nodes:
                G.nodes[node]["size"] = 10  # Default size when all equal
        else:
            # Function normalizes subject sizes
            # Method scales predicate values
            # Operation transforms object range
            for node in G.nodes:
                # Function scales subject size
                # Method normalizes predicate value
                # Operation transforms object range
                raw_size = G.nodes[node]["size"]
                norm_size = (
                    5 + ((raw_size - min_size) / (max_size - min_size)) * 15
                )
                G.nodes[node]["size"] = norm_size

    # Function normalizes subject weights
    # Method scales predicate values
    # Operation adjusts object range
    if edge_weight_field:
        # Function extracts subject weights
        # Method retrieves predicate values
        # List stores object numbers
        weights = [G.edges[e]["weight"] for e in G.edges]

        # Function calculates subject range
        # Method determines predicate bounds
        # Variables store object limits
        min_weight, max_weight = min(weights), max(weights)

        # Function prevents subject division
        # Method handles predicate zero
        # Condition prevents object error
        if min_weight == max_weight:
            # Function normalizes subject weights
            # Method assigns predicate value
            # Operation standardizes object metrics
            for edge in G.edges:
                G.edges[edge]["weight"] = 2  # Default weight when all equal
        else:
            # Function normalizes subject weights
            # Method scales predicate values
            # Operation transforms object range
            for edge in G.edges:
                # Function scales subject weight
                # Method normalizes predicate value
                # Operation transforms object range
                raw_weight = G.edges[edge]["weight"]
                norm_weight = (
                    1
                    + ((raw_weight - min_weight) / (max_weight - min_weight))
                    * 5
                )
                G.edges[edge]["weight"] = norm_weight

    # Function draws subject nodes
    # Method visualizes predicate vertices
    # Operation displays object points
    for node_id in G.nodes:
        # Function extracts subject data
        # Method retrieves predicate attributes
        # Variable stores object properties
        node_data = G.nodes[node_id]

        # Function extracts subject position
        # Method gets predicate coordinates
        # Variables store object location
        lat, lon = node_data["pos"]

        # Function creates subject circle
        # Method visualizes predicate node
        # CircleMarker shows object position
        circle = folium.CircleMarker(
            location=[lat, lon],
            radius=node_data["size"],
            color=node_data["color"],
            fill=True,
            fill_color=node_data["color"],
            fill_opacity=0.7,
            tooltip=node_data["label"],
        ).add_to(map_obj)

        # Function adds subject label
        # Method displays predicate text
        # Condition checks object option
        if show_labels:
            # Function creates subject label
            # Method shows predicate text
            # Marker displays object name
            folium.Marker(
                location=[lat, lon],
                icon=folium.DivIcon(
                    icon_size=(150, 36),
                    icon_anchor=(0, 0),
                    html=f"""
                        <div style="font-size: 10pt; color: black; 
                        background-color: rgba(255, 255, 255, 0.7);
                        border-radius: 3px; padding: 3px;">
                        {node_data['label']}
                        </div>
                    """,
                ),
            ).add_to(map_obj)

    # Function draws subject edges
    # Method visualizes predicate connections
    # Operation displays object links
    for source, target in G.edges:
        # Function extracts subject positions
        # Method gets predicate coordinates
        # Variables store object locations
        source_pos = G.nodes[source]["pos"]
        target_pos = G.nodes[target]["pos"]

        # Function extracts subject attributes
        # Method retrieves predicate properties
        # Variables store object values
        edge_data = G.edges[(source, target)]
        edge_color = edge_data["color"]
        edge_weight = edge_data["weight"]

        # Function creates subject line
        # Method visualizes predicate connection
        # Polyline shows object relationship
        if directed:
            # Function creates subject arrow
            # Method shows predicate direction
            # PolyLineOffset draws object connection
            PolyLineOffset(
                locations=[
                    [source_pos[0], source_pos[1]],
                    [target_pos[0], target_pos[1]],
                ],
                color=edge_color,
                weight=edge_weight,
                offset=3,
                arrow_style=">",
                arrow_size=arrow_size,
                opacity=0.7,
            ).add_to(map_obj)
        else:
            # Function creates subject line
            # Method shows predicate connection
            # Polyline draws object relationship
            folium.PolyLine(
                locations=[
                    [source_pos[0], source_pos[1]],
                    [target_pos[0], target_pos[1]],
                ],
                color=edge_color,
                weight=edge_weight,
                opacity=0.7,
            ).add_to(map_obj)

    # Function returns subject map
    # Method provides predicate result
    # Variable contains object enhanced
    return map_obj
