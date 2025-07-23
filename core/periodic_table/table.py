"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-PERIODIC-TABLE-LAYOUT-0001          │
// │ 📁 domain       : Classification, Visualization             │
// │ 🧠 description  : Table layout and display for the          │
// │                  CTAS Periodic Table of Nodes               │
// │ 🕸️ hash_type    : UUID → CUID-linked module                 │
// │ 🔄 parent_node  : NODE_CLASSIFICATION                      │
// │ 🧩 dependencies : plotly, numpy, pandas                    │
// │ 🔧 tool_usage   : Visualization, Navigation                │
// │ 📡 input_type   : Element data                             │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : visualization, navigation                 │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

Table Implementation
----------------
This module provides the layout and visualization for the CTAS Periodic Table
of Nodes, including interactive display and navigation capabilities.
"""

import uuid
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Any, Tuple, Union, Callable
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import logging
from datetime import datetime

from core.periodic_table.element import Element, ElementProperty
from core.periodic_table.group import Group, Period, Category, CATEGORY_COLORS
from core.periodic_table.registry import PeriodicTableRegistry
from core.periodic_table.relationships import Relationship, RelationshipType

# Function configures subject logger
# Method initializes predicate output
# Operation sets object format
logger = logging.getLogger("periodic_table.table")
logger.setLevel(logging.INFO)

# Default color scale for visualization
DEFAULT_COLOR_SCALE = [
    "#1f77b4",  # Blue
    "#ff7f0e",  # Orange
    "#2ca02c",  # Green
    "#d62728",  # Red
    "#9467bd",  # Purple
    "#8c564b",  # Brown
    "#e377c2",  # Pink
    "#7f7f7f",  # Gray
    "#bcbd22",  # Olive
    "#17becf",  # Cyan
]


# Symbol for elements in the table
class ElementSymbol:
    """
    Class for generating and formatting element symbols.

    # Class encapsulates subject symbols
    # Object formats predicate display
    # Component generates object representations
    """

    @staticmethod
    def format_symbol(
        symbol: str, properties: Dict[ElementProperty, Any] = None
    ) -> str:
        """
        Format an element symbol with properties.

        # Function formats subject symbol
        # Method styles predicate text
        # Operation returns object representation

        Args:
            symbol: Element symbol
            properties: Element properties

        Returns:
            Formatted symbol HTML
        """
        if not properties:
            return f"<b>{symbol}</b>"

        # Get properties
        atomic_number = properties.get(ElementProperty.ATOMIC_NUMBER, "")
        reliability = properties.get(ElementProperty.RELIABILITY, 1.0)

        # Format reliability as percentage
        reliability_pct = int(reliability * 100)

        # Create formatted symbol
        html = f"""
        <div style="text-align: center;">
            <div style="font-size: 0.9em; color: #666;">{atomic_number}</div>
            <div style="font-size: 1.2em; font-weight: bold;">{symbol}</div>
            <div style="font-size: 0.8em; color: {'green' if reliability_pct >= 70 else 'orange' if reliability_pct >= 40 else 'red'};">
                {reliability_pct}%
            </div>
        </div>
        """

        return html

    @staticmethod
    def get_node_card(element: Element) -> str:
        """
        Generate a node card for an element.

        # Function generates subject card
        # Method formats predicate display
        # Operation returns object HTML

        Args:
            element: Element to create card for

        Returns:
            HTML for node card
        """
        # Get properties
        name = element.name
        symbol = element.symbol
        atomic_number = element.atomic_number
        ttl = element.get_property(ElementProperty.TTL, "N/A")
        reliability = element.get_property(ElementProperty.RELIABILITY, 1.0)
        confidence = element.get_property(ElementProperty.CONFIDENCE, 1.0)
        accessibility = element.get_property(ElementProperty.ACCESSIBILITY, 1.0)
        maturity = element.get_property(ElementProperty.MATURITY, 5)
        complexity = element.get_property(ElementProperty.COMPLEXITY, 5)
        description = element.get_property(ElementProperty.DESCRIPTION, "")

        # Get color based on category
        category_id = element.category_id
        color = "#1f77b4"  # Default blue
        border_color = "#ff0000"  # Red border

        # Format reliability as integer percentage
        reliability_pct = int(reliability * 100)
        confidence_pct = int(confidence * 100)

        # Create node card HTML
        html = f"""
        <div style="border: 3px solid {border_color}; border-radius: 10px; padding: 10px; 
                    background-color: {color}; color: white; max-width: 300px;">
            
            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                <div style="background-color: rgba(255,255,255,0.2); padding: 2px 5px; border-radius: 5px;">
                    {atomic_number}.0 {name} (TTL:{ttl})
                </div>
            </div>
            
            <div style="display: flex; justify-content: space-between; margin: 10px 0;">
                <div style="background-color: rgba(0,100,255,0.5); width: 50px; height: 50px; 
                            border-radius: 8px; display: flex; align-items: center; justify-content: center; 
                            font-weight: bold; font-size: 1.2em;">
                    θ<br>{-3}
                </div>
                
                <div style="background-color: rgba(255,255,255,0.9); color: black; padding: 5px; 
                            text-align: center; border-radius: 5px; flex-grow: 1; margin: 0 5px;">
                    <div>Symbols</div>
                    <div style="font-weight: bold;">α αΩ♥♠Δ</div>
                </div>
                
                <div style="background-color: rgba(0,200,0,0.5); width: 50px; height: 50px; 
                            border-radius: 8px; display: flex; align-items: center; justify-content: center; 
                            font-weight: bold; font-size: 1.2em;">
                    Δ<br>{6}
                </div>
            </div>
            
            <div style="background-color: rgba(200,0,200,0.7); padding: 8px; text-align: center; 
                        margin: 10px 0; border-radius: 5px; font-weight: bold; animation: pulse 2s infinite;">
                {name}<br>with color and<br>border incl<br>animation
            </div>
            
            <div style="background-color: rgba(255,255,255,0.2); padding: 8px; text-align: left; 
                        margin: 10px 0; border-radius: 5px; font-size: 0.9em;">
                Each element is a link to<br>additional info
            </div>
            
            <div style="font-size: 0.8em; margin-top: 10px;">
                <div style="display: flex; justify-content: space-between;">
                    <span>Reliability: {reliability_pct}%</span>
                    <span>Confidence: {confidence_pct}%</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span>Complexity: {complexity}/10</span>
                    <span>Maturity: {maturity}/10</span>
                </div>
            </div>
        </div>
        
        <style>
        .node-card:hover {
          transform: scale(1.05);
          transition: transform 0.3s ease;
        }
        </style>
        """

        return html


class PeriodicTable:
    """
    Class for displaying and interacting with the CTAS Periodic Table.

    # Class represents subject table
    # Object displays predicate elements
    # Component visualizes object relationships
    """

    def __init__(self, registry: PeriodicTableRegistry):
        """
        Initialize periodic table with registry.

        # Function initializes subject table
        # Method configures predicate display
        # Operation prepares object visualization

        Args:
            registry: Element registry
        """
        self.registry = registry
        self.selected_element = None

        # Cache for visualization data
        self._groups_cache = None
        self._periods_cache = None
        self._categories_cache = None
        self._elements_cache = None

        # Load data
        self._load_data()

    def _load_data(self):
        """
        Load data from registry.

        # Function loads subject data
        # Method retrieves predicate elements
        # Operation populates object cache
        """
        # Load groups, periods, and categories
        self._groups_cache = self.registry.get_all_groups()
        self._periods_cache = self.registry.get_all_periods()
        self._categories_cache = self.registry.get_all_categories()

        # Load elements
        self._elements_cache = self.registry.get_all_elements()

        # Load relationships
        self.registry.load_relationships_from_db()

    def refresh(self):
        """
        Refresh data from registry.

        # Function refreshes subject data
        # Method updates predicate cache
        # Operation reloads object state
        """
        self._load_data()

    def get_table_data(self) -> pd.DataFrame:
        """
        Get table data as pandas DataFrame.

        # Function gets subject data
        # Method formats predicate table
        # Operation returns object dataframe

        Returns:
            DataFrame with table data
        """
        # Refresh data if needed
        if not self._elements_cache:
            self.refresh()

        # Create elements data
        elements_data = []
        for element in self._elements_cache:
            # Get group and period numbers
            group_id = element.group_id
            period_id = element.period_id
            category_id = element.category_id

            # Find group and period objects
            group = next(
                (g for g in self._groups_cache if g.id == group_id), None
            )
            period = next(
                (p for p in self._periods_cache if p.id == period_id), None
            )
            category = next(
                (c for c in self._categories_cache if c.id == category_id), None
            )

            # Get coordinates
            group_num = group.number if group else 0
            period_num = period.number if period else 0
            category_name = category.name if category else "Unknown"
            category_color = category.color if category else "#CCCCCC"

            # Get properties
            reliability = element.get_property(ElementProperty.RELIABILITY, 1.0)
            confidence = element.get_property(ElementProperty.CONFIDENCE, 1.0)
            maturity = element.get_property(ElementProperty.MATURITY, 5)
            complexity = element.get_property(ElementProperty.COMPLEXITY, 5)

            # Add element data
            elements_data.append(
                {
                    "id": str(element.id),
                    "symbol": element.symbol,
                    "name": element.name,
                    "atomic_number": element.atomic_number,
                    "group": group_num,
                    "period": period_num,
                    "category": category_name,
                    "color": category_color,
                    "reliability": reliability,
                    "confidence": confidence,
                    "maturity": maturity,
                    "complexity": complexity,
                }
            )

        # Create DataFrame
        df = pd.DataFrame(elements_data)

        return df

    def create_plotly_table(
        self,
        width: int = 1200,
        height: int = 800,
        color_by: str = "category",
        on_element_click: Optional[Callable] = None,
    ) -> go.Figure:
        """
        Create a Plotly figure for interactive periodic table.

        # Function creates subject figure
        # Method generates predicate visualization
        # Operation returns object plot

        Args:
            width: Figure width
            height: Figure height
            color_by: Property to color elements by
            on_element_click: Callback for element click

        Returns:
            Plotly figure
        """
        # Get table data
        df = self.get_table_data()

        # Create figure
        fig = go.Figure()

        # Add elements as scatter markers
        if color_by == "category":
            # Color by category
            categories = df["category"].unique()
            color_map = {
                cat: CATEGORY_COLORS.get(
                    cat, DEFAULT_COLOR_SCALE[i % len(DEFAULT_COLOR_SCALE)]
                )
                for i, cat in enumerate(categories)
            }

            # Create scatter plot for each category
            for category in categories:
                cat_df = df[df["category"] == category]

                fig.add_trace(
                    go.Scatter(
                        x=cat_df["group"],
                        y=cat_df["period"],
                        mode="markers+text",
                        marker=dict(
                            size=50,
                            color=color_map[category],
                            line=dict(width=2, color="rgba(0,0,0,0.2)"),
                        ),
                        text=cat_df["symbol"],
                        textposition="middle center",
                        textfont=dict(size=14, color="white"),
                        customdata=cat_df[
                            [
                                "id",
                                "name",
                                "atomic_number",
                                "reliability",
                                "confidence",
                            ]
                        ].values,
                        name=category,
                        hovertemplate="<b>%{text}</b><br>%{customdata[1]}<br>Atomic #: %{customdata[2]}<br>Reliability: %{customdata[3]:.0%}<br>Confidence: %{customdata[4]:.0%}<extra></extra>",
                    )
                )
        else:
            # Color by numeric property (reliability, confidence, etc.)
            color_property = color_by
            if color_property in df.columns and df[color_property].dtype in [
                np.float64,
                np.int64,
            ]:
                fig.add_trace(
                    go.Scatter(
                        x=df["group"],
                        y=df["period"],
                        mode="markers+text",
                        marker=dict(
                            size=50,
                            color=df[color_property],
                            colorscale="Viridis",
                            showscale=True,
                            colorbar=dict(title=color_property.capitalize()),
                            line=dict(width=2, color="rgba(0,0,0,0.2)"),
                        ),
                        text=df["symbol"],
                        textposition="middle center",
                        textfont=dict(size=14, color="white"),
                        customdata=df[
                            ["id", "name", "atomic_number", color_property]
                        ].values,
                        hovertemplate="<b>%{text}</b><br>%{customdata[1]}<br>Atomic #: %{customdata[2]}<br>"
                        + f"{color_property.capitalize()}: "
                        + "%{customdata[3]:.2f}<extra></extra>",
                    )
                )
            else:
                # Default to category coloring
                fig.add_trace(
                    go.Scatter(
                        x=df["group"],
                        y=df["period"],
                        mode="markers+text",
                        marker=dict(
                            size=50,
                            color=df["color"],
                            line=dict(width=2, color="rgba(0,0,0,0.2)"),
                        ),
                        text=df["symbol"],
                        textposition="middle center",
                        textfont=dict(size=14, color="white"),
                        customdata=df[["id", "name", "atomic_number"]].values,
                        hovertemplate="<b>%{text}</b><br>%{customdata[1]}<br>Atomic #: %{customdata[2]}<extra></extra>",
                    )
                )

        # Configure layout
        fig.update_layout(
            title="CTAS Periodic Table of Nodes",
            width=width,
            height=height,
            xaxis=dict(
                title="Group",
                zeroline=False,
                gridcolor="rgba(0,0,0,0.1)",
                showgrid=True,
                showticklabels=True,
                range=[0, max(df["group"]) + 1],
            ),
            yaxis=dict(
                title="Period",
                zeroline=False,
                gridcolor="rgba(0,0,0,0.1)",
                showgrid=True,
                showticklabels=True,
                range=[max(df["period"]) + 1, 0],  # Invert y-axis
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0.02)",
            hovermode="closest",
            margin=dict(l=50, r=50, t=80, b=50),
            showlegend=True,
            legend_title_text="Categories",
        )

        return fig

    def create_network_graph(
        self,
        element_id: Optional[uuid.UUID] = None,
        max_depth: int = 2,
        width: int = 1200,
        height: int = 800,
    ) -> go.Figure:
        """
        Create a network graph visualization for element relationships.

        # Function creates subject graph
        # Method visualizes predicate relationships
        # Operation returns object network

        Args:
            element_id: Root element ID (None for full graph)
            max_depth: Maximum depth of relationships to show
            width: Figure width
            height: Figure height

        Returns:
            Plotly figure
        """
        if not self._elements_cache:
            self.refresh()

        # If no element specified, use the first one
        if not element_id and self._elements_cache:
            element_id = self._elements_cache[0].id

        # Traverse relationships to build graph
        nodes = {}
        edges = []

        # Function to add element to graph
        def add_element_to_graph(element_id, depth=0):
            if depth > max_depth or element_id in nodes:
                return

            # Get element
            element = self.registry.get_element(element_id)
            if not element:
                return

            # Add node
            category_id = element.category_id
            category = next(
                (c for c in self._categories_cache if c.id == category_id), None
            )
            category_color = category.color if category else "#CCCCCC"

            nodes[str(element_id)] = {
                "id": str(element_id),
                "symbol": element.symbol,
                "name": element.name,
                "color": category_color,
                "size": 20 if depth == 0 else 15 - (depth * 2),
            }

            # Get relationships
            if depth < max_depth:
                relationships = self.registry.get_element_relationships(
                    element_id
                )

                # Process each relationship
                for rel_type, connections in relationships.items():
                    for relationship, connected_id in connections:
                        # Add connected element
                        add_element_to_graph(connected_id, depth + 1)

                        # Add edge
                        edges.append(
                            {
                                "source": str(relationship.source_id),
                                "target": str(relationship.target_id),
                                "type": rel_type.lower().replace("_", " "),
                                "width": 2 if depth == 0 else 1,
                            }
                        )

        # Start with the specified element
        add_element_to_graph(element_id)

        # Build node and edge datasets
        node_df = pd.DataFrame(list(nodes.values()))
        edge_df = pd.DataFrame(edges)

        # Check if we have data
        if node_df.empty or edge_df.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No relationship data available for this element",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=20),
            )
            fig.update_layout(width=width, height=height)
            return fig

        # Create network graph using Force Atlas 2 algorithm
        # (simplified version for demonstration)
        positions = self._force_atlas2_layout(node_df, edge_df)

        # Create figure
        fig = go.Figure()

        # Add edges
        for i, row in edge_df.iterrows():
            source = row["source"]
            target = row["target"]

            if source in positions and target in positions:
                fig.add_trace(
                    go.Scatter(
                        x=[positions[source][0], positions[target][0]],
                        y=[positions[source][1], positions[target][1]],
                        mode="lines",
                        line=dict(width=row["width"], color="rgba(0,0,0,0.3)"),
                        hoverinfo="text",
                        text=row["type"],
                        showlegend=False,
                    )
                )

        # Add nodes
        for i, row in node_df.iterrows():
            node_id = row["id"]

            if node_id in positions:
                fig.add_trace(
                    go.Scatter(
                        x=[positions[node_id][0]],
                        y=[positions[node_id][1]],
                        mode="markers+text",
                        marker=dict(
                            size=row["size"],
                            color=row["color"],
                            line=dict(width=2, color="rgba(0,0,0,0.2)"),
                        ),
                        text=row["symbol"],
                        textposition="middle center",
                        textfont=dict(size=10, color="white"),
                        name=row["name"],
                        hovertemplate="<b>%{text}</b><br>%{meta[0]}<extra></extra>",
                        meta=[[row["name"]]],
                        showlegend=False,
                    )
                )

        # Configure layout
        fig.update_layout(
            title="Element Relationships",
            width=width,
            height=height,
            showlegend=False,
            hovermode="closest",
            margin=dict(l=20, r=20, t=60, b=20),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0.02)",
        )

        return fig

    def _force_atlas2_layout(
        self, nodes: pd.DataFrame, edges: pd.DataFrame, iterations: int = 100
    ) -> Dict[str, Tuple[float, float]]:
        """
        Compute layout positions using Force Atlas 2 algorithm.

        # Function computes subject layout
        # Method calculates predicate positions
        # Operation returns object coordinates

        Args:
            nodes: Nodes DataFrame
            edges: Edges DataFrame
            iterations: Number of iterations

        Returns:
            Dictionary mapping node IDs to (x, y) positions
        """
        # Initialize positions randomly
        positions = {
            row["id"]: (np.random.random(), np.random.random())
            for _, row in nodes.iterrows()
        }

        # Convert to adjacency list
        adjacency = {}
        for _, row in edges.iterrows():
            source = row["source"]
            target = row["target"]

            if source not in adjacency:
                adjacency[source] = []
            if target not in adjacency:
                adjacency[target] = []

            adjacency[source].append(target)
            adjacency[target].append(source)

        # Constants for the algorithm
        k = 1.0  # Optimal distance
        gravity = 1.0  # Gravity strength
        speed = 1.0  # Speed factor

        # Run iterations
        for _ in range(iterations):
            # Calculate forces
            forces = {node_id: [0.0, 0.0] for node_id in positions}

            # Repulsive forces (between all nodes)
            for id1, pos1 in positions.items():
                for id2, pos2 in positions.items():
                    if id1 != id2:
                        dx = pos1[0] - pos2[0]
                        dy = pos1[1] - pos2[1]
                        distance = max(0.01, np.sqrt(dx * dx + dy * dy))

                        # Repulsive force
                        force = k * k / distance

                        # Apply force
                        forces[id1][0] += dx / distance * force
                        forces[id1][1] += dy / distance * force

            # Attractive forces (between connected nodes)
            for node_id, neighbors in adjacency.items():
                pos1 = positions.get(node_id)
                if not pos1:
                    continue

                for neighbor in neighbors:
                    pos2 = positions.get(neighbor)
                    if not pos2:
                        continue

                    dx = pos1[0] - pos2[0]
                    dy = pos1[1] - pos2[1]
                    distance = max(0.01, np.sqrt(dx * dx + dy * dy))

                    # Attractive force
                    force = distance * distance / k

                    # Apply force
                    forces[node_id][0] -= dx / distance * force
                    forces[node_id][1] -= dy / distance * force

            # Apply gravity towards center
            for node_id, pos in positions.items():
                gx = 0.0 - pos[0]
                gy = 0.0 - pos[1]
                distance = max(0.01, np.sqrt(gx * gx + gy * gy))

                # Gravity force
                forces[node_id][0] += gx / distance * gravity
                forces[node_id][1] += gy / distance * gravity

            # Update positions
            for node_id, force in forces.items():
                # Calculate displacement
                dx = np.clip(force[0], -10, 10) * speed
                dy = np.clip(force[1], -10, 10) * speed

                # Update position
                x, y = positions[node_id]
                positions[node_id] = (x + dx, y + dy)

        # Normalize positions to fit in [0, 1] range
        min_x = min(x for x, _ in positions.values())
        max_x = max(x for x, _ in positions.values())
        min_y = min(y for _, y in positions.values())
        max_y = max(y for _, y in positions.values())

        x_range = max(0.01, max_x - min_x)
        y_range = max(0.01, max_y - min_y)

        for node_id, (x, y) in positions.items():
            normalized_x = (x - min_x) / x_range
            normalized_y = (y - min_y) / y_range
            positions[node_id] = (normalized_x, normalized_y)

        return positions

    def get_element_card(self, element_id: uuid.UUID) -> str:
        """
        Get HTML card for an element.

        # Function gets subject card
        # Method generates predicate HTML
        # Operation returns object visualization

        Args:
            element_id: Element ID

        Returns:
            HTML for element card
        """
        element = self.registry.get_element(element_id)
        if not element:
            return "<div>Element not found</div>"

        return ElementSymbol.get_node_card(element)

    def create_element_detail_plot(
        self, element_id: uuid.UUID, width: int = 1200, height: int = 800
    ) -> go.Figure:
        """
        Create a detail plot for an element.

        # Function creates subject plot
        # Method generates predicate visualization
        # Operation returns object figure

        Args:
            element_id: Element ID
            width: Figure width
            height: Figure height

        Returns:
            Plotly figure
        """
        element = self.registry.get_element(element_id)
        if not element:
            fig = go.Figure()
            fig.add_annotation(
                text="Element not found",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=20),
            )
            fig.update_layout(width=width, height=height)
            return fig

        # Get properties for radar chart
        props = {
            "Reliability": element.get_property(
                ElementProperty.RELIABILITY, 0.5
            ),
            "Confidence": element.get_property(ElementProperty.CONFIDENCE, 0.5),
            "Accessibility": element.get_property(
                ElementProperty.ACCESSIBILITY, 0.5
            ),
            "Stability": element.get_property(ElementProperty.STABILITY, 0.5),
            "Maturity": element.get_property(ElementProperty.MATURITY, 5) / 10,
            "Complexity": element.get_property(ElementProperty.COMPLEXITY, 5)
            / 10,
            "Priority": element.get_property(ElementProperty.PRIORITY, 5) / 10,
            "Computation Cost": element.get_property(
                ElementProperty.COMPUTATION_COST, 5
            )
            / 10,
        }

        # Get values for radar chart
        categories = list(props.keys())
        values = [props[cat] for cat in categories]

        # Create figure with subplots
        fig = make_subplots(
            rows=1,
            cols=2,
            specs=[[{"type": "polar"}, {"type": "xy"}]],
            subplot_titles=["Element Properties", "Element Relationships"],
        )

        # Add radar chart
        fig.add_trace(
            go.Scatterpolar(
                r=values, theta=categories, fill="toself", name=element.name
            ),
            row=1,
            col=1,
        )

        # Get relationship data
        relationships = self.registry.get_element_relationships(element_id)

        # Count relationship types
        rel_counts = {}
        for rel_type, connections in relationships.items():
            rel_counts[rel_type] = len(connections)

        # Convert to lists for bar chart
        rel_types = list(rel_counts.keys())
        rel_values = [rel_counts[rt] for rt in rel_types]

        # Add bar chart for relationships
        fig.add_trace(
            go.Bar(x=rel_types, y=rel_values, marker_color="rgb(26, 118, 255)"),
            row=1,
            col=2,
        )

        # Update layout
        fig.update_layout(
            title=f"Element Details: {element.name} ({element.symbol})",
            width=width,
            height=height,
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=False,
            margin=dict(l=20, r=20, t=80, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0.02)",
        )

        # Update x-axis of bar chart
        fig.update_xaxes(title="Relationship Type", row=1, col=2)
        fig.update_yaxes(title="Count", row=1, col=2)

        return fig
