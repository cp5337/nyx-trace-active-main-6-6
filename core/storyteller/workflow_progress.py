"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-STORYTELLER-WORKFLOW-0001           │
// │ 📁 domain       : Storytelling, Workflow, Visualization     │
// │ 🧠 description  : Interactive workflow progress storyteller │
// │                  for CTAS operations                        │
// │ 🕸️ hash_type    : UUID → CUID-linked visualization          │
// │ 🔄 parent_node  : NODE_INTERFACE                           │
// │ 🧩 dependencies : streamlit, plotly, story_elements         │
// │ 🔧 tool_usage   : Visualization, Tracking, Presentation     │
// │ 📡 input_type   : Workflow events, progress data            │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : narrative visualization, progress tracking │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

CTAS Workflow Progress Storyteller
--------------------------------
This module provides an interactive workflow progress storyteller for
visualizing and narrating the progression of CTAS operations and
investigations. It creates compelling visual narratives of complex
workflows with interactive elements.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
import logging
from typing import List, Dict, Tuple, Optional, Any, Union, Callable

from core.storyteller.story_elements import (
    StoryElement,
    StoryMilestone,
    StoryTimeline,
    StoryElementType,
    ElementStatus,
)

# Function creates subject logger
# Method configures predicate output
# Logger tracks object events
# Variable supports subject debugging
logger = logging.getLogger(__name__)


# Function defines subject class
# Method implements predicate visualization
# Class provides object functionality
# Definition delivers subject implementation
class WorkflowProgressStoryteller:
    """
    Interactive workflow progress storyteller for CTAS operations

    # Class implements subject storyteller
    # Method provides predicate visualization
    # Object narrates workflow progress
    # Definition creates subject implementation

    This class creates interactive visualizations that tell the story
    of a workflow's progression, with timelines, milestones, and narrative
    elements that help users understand complex operational sequences.
    """

    # Function defines subject constructor
    # Method initializes predicate instance
    # Operation creates object state
    # Code sets up subject attributes
    def __init__(
        self,
        title: str = "Workflow Progress",
        description: str = "Interactive visualization of workflow progress",
        data_dir: str = "data/storyteller",
    ):
        """
        Initialize the workflow progress storyteller

        # Function initializes subject storyteller
        # Method creates predicate instance
        # Operation sets object state
        # Code configures subject properties

        Args:
            title: Title for the storyteller
            description: Description text
            data_dir: Directory for storing story data
        """
        # Function sets subject properties
        # Method defines predicate attributes
        # Operation initializes object state
        # Code configures subject instance
        self.title = title
        self.description = description
        self.data_dir = data_dir
        self.current_timeline = None
        self.theme = {
            "milestone": {"color": "#4CAF50", "icon": "✓", "size": 20},
            "event": {"color": "#2196F3", "icon": "•", "size": 15},
            "discovery": {"color": "#9C27B0", "icon": "★", "size": 15},
            "decision": {"color": "#FF9800", "icon": "⬧", "size": 15},
            "obstacle": {"color": "#F44336", "icon": "⚠", "size": 15},
            "insight": {"color": "#00BCD4", "icon": "💡", "size": 15},
            "action": {"color": "#607D8B", "icon": "➤", "size": 15},
            "resource": {"color": "#8BC34A", "icon": "⚙", "size": 15},
            "planned": {"color": "#9E9E9E", "pattern": "dot"},
            "in_progress": {"color": "#42A5F5", "pattern": "solid"},
            "completed": {"color": "#4CAF50", "pattern": "solid"},
            "blocked": {"color": "#F44336", "pattern": "dot"},
            "skipped": {"color": "#9E9E9E", "pattern": "dash"},
            "failed": {"color": "#FF5252", "pattern": "solid"},
        }

        # Create data directory if it doesn't exist
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
            logger.info(f"Created data directory at {self.data_dir}")

    # Function creates subject timeline
    # Method generates predicate framework
    # Operation builds object structure
    # Code initializes subject narrative
    def create_timeline(self, title: str, description: str) -> StoryTimeline:
        """
        Create a new story timeline

        # Function creates subject timeline
        # Method generates predicate structure
        # Operation builds object framework
        # Code initializes subject narrative

        Args:
            title: Title for the timeline
            description: Description of the timeline

        Returns:
            New StoryTimeline instance
        """
        # Function creates subject instance
        # Method generates predicate timeline
        # Operation builds object structure
        # Code initializes subject narrative
        timeline = StoryTimeline(title=title, description=description)
        self.current_timeline = timeline
        return timeline

    # Function loads subject timeline
    # Method retrieves predicate narrative
    # Operation restores object state
    # Code reads subject data
    def load_timeline(self, timeline_id: str) -> Optional[StoryTimeline]:
        """
        Load a timeline from storage

        # Function loads subject timeline
        # Method retrieves predicate data
        # Operation restores object state
        # Code reads subject narrative

        Args:
            timeline_id: ID of the timeline to load

        Returns:
            StoryTimeline instance or None if not found
        """
        filepath = os.path.join(self.data_dir, f"{timeline_id}.json")

        # Function checks subject existence
        # Method verifies predicate file
        # Operation validates object location
        # Code confirms subject availability
        if not os.path.exists(filepath):
            logger.warning(f"Timeline file not found: {filepath}")
            return None

        try:
            # Function loads subject timeline
            # Method reads predicate data
            # Operation restores object state
            # Code deserializes subject information
            timeline = StoryTimeline.load_from_file(filepath)
            self.current_timeline = timeline
            return timeline
        except Exception as e:
            logger.error(f"Error loading timeline: {e}")
            return None

    # Function saves subject timeline
    # Method persists predicate data
    # Operation stores object state
    # Code writes subject information
    def save_timeline(self, timeline: Optional[StoryTimeline] = None) -> bool:
        """
        Save a timeline to storage

        # Function saves subject timeline
        # Method persists predicate data
        # Operation stores object state
        # Code serializes subject narrative

        Args:
            timeline: Timeline to save (uses current_timeline if None)

        Returns:
            True if successful, False otherwise
        """
        # Use provided timeline or current one
        timeline = timeline or self.current_timeline

        # Function validates subject input
        # Method checks predicate parameter
        # Operation verifies object existence
        # Code ensures subject availability
        if timeline is None:
            logger.warning("No timeline to save")
            return False

        # Function generates subject filepath
        # Method determines predicate location
        # Operation sets object destination
        # Code specifies subject storage
        filepath = os.path.join(self.data_dir, f"{timeline.id}.json")

        try:
            # Function saves subject timeline
            # Method writes predicate data
            # Operation persists object state
            # Code serializes subject information
            timeline.save_to_file(filepath)
            logger.info(f"Saved timeline to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving timeline: {e}")
            return False

    # Function lists subject timelines
    # Method retrieves predicate files
    # Operation finds object narratives
    # Code discovers subject resources
    def list_timelines(self) -> List[Dict[str, Any]]:
        """
        Get a list of available timelines

        # Function lists subject timelines
        # Method retrieves predicate narratives
        # Operation finds object resources
        # Code discovers subject files

        Returns:
            List of timeline metadata dictionaries
        """
        # Function checks subject directory
        # Method verifies predicate existence
        # Operation validates object location
        # Code confirms subject availability
        if not os.path.exists(self.data_dir):
            logger.warning(f"Data directory not found: {self.data_dir}")
            return []

        # Function creates subject container
        # Method initializes predicate list
        # Operation prepares object storage
        # Code sets up subject collection
        timelines = []

        # Function scans subject directory
        # Method searches predicate files
        # Operation discovers object resources
        # Code finds subject timelines
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(self.data_dir, filename)

                try:
                    # Function reads subject metadata
                    # Method extracts predicate information
                    # Operation retrieves object properties
                    # Code loads subject summary
                    with open(filepath, "r") as f:
                        data = json.load(f)

                    timelines.append(
                        {
                            "id": data.get("id"),
                            "title": data.get("title", "Untitled"),
                            "description": data.get("description", ""),
                            "created_at": data.get("created_at"),
                            "updated_at": data.get("updated_at"),
                            "element_count": len(data.get("elements", [])),
                            "filepath": filepath,
                        }
                    )
                except Exception as e:
                    logger.error(f"Error reading timeline file {filepath}: {e}")

        # Sort by updated date (most recent first)
        timelines.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return timelines

    # Function gets subject timelines
    # Method retrieves predicate identifiers
    # Operation lists object available
    # Code provides subject options
    def get_available_timelines(self) -> List[str]:
        """
        Get a list of available timeline IDs

        # Function gets subject timelines
        # Method retrieves predicate identifiers
        # Operation lists object available
        # Code provides subject options

        Returns:
            List of timeline IDs
        """
        # Function gets subject metadata
        # Method retrieves predicate info
        # Operation collects object list
        # Code obtains subject details
        timeline_data = self.list_timelines()

        # Function extracts subject ids
        # Method collects predicate identifiers
        # Operation selects object keys
        # Code gathers subject references
        timeline_ids = [timeline["id"] for timeline in timeline_data]

        return timeline_ids

    # Function adds subject element
    # Method appends predicate item
    # Operation extends object timeline
    # Code updates subject narrative
    def add_element(self, element: StoryElement) -> None:
        """
        Add an element to the current timeline

        # Function adds subject element
        # Method appends predicate item
        # Operation extends object timeline
        # Code updates subject narrative

        Args:
            element: StoryElement to add
        """
        # Function validates subject state
        # Method checks predicate timeline
        # Operation verifies object existence
        # Code ensures subject availability
        if self.current_timeline is None:
            logger.warning("No current timeline")
            return

        # Function adds subject element
        # Method extends predicate timeline
        # Operation updates object collection
        # Code modifies subject narrative
        self.current_timeline.add_element(element)

    # Function creates subject milestone
    # Method generates predicate element
    # Operation builds object marker
    # Code constructs subject achievement
    def add_milestone(
        self,
        title: str,
        description: str,
        status: ElementStatus = ElementStatus.COMPLETED,
        **kwargs,
    ) -> StoryMilestone:
        """
        Add a milestone to the current timeline

        # Function adds subject milestone
        # Method creates predicate marker
        # Operation builds object achievement
        # Code extends subject narrative

        Args:
            title: Milestone title
            description: Detailed description
            status: Element status (default: COMPLETED)
            **kwargs: Additional attributes

        Returns:
            Created StoryMilestone instance
        """
        # Function creates subject instance
        # Method generates predicate milestone
        # Operation builds object element
        # Code constructs subject marker
        milestone = StoryMilestone(
            title=title, description=description, status=status, **kwargs
        )

        # Function adds subject milestone
        # Method extends predicate timeline
        # Operation updates object collection
        # Code modifies subject narrative
        self.add_element(milestone)

        return milestone

    # Function visualizes subject timeline
    # Method renders predicate narrative
    # Operation displays object story
    # Code presents subject visualization
    def display_timeline_visualization(
        self,
        timeline: Optional[StoryTimeline] = None,
        height: int = 600,
        show_details: bool = True,
    ) -> None:
        """
        Display an interactive timeline visualization in Streamlit

        # Function displays subject visualization
        # Method renders predicate timeline
        # Operation shows object narrative
        # Code presents subject storytelling

        Args:
            timeline: Timeline to visualize (uses current_timeline if None)
            height: Height of the visualization in pixels
            show_details: Whether to show element details panel
        """
        # Use provided timeline or current one
        timeline = timeline or self.current_timeline

        # Function validates subject input
        # Method checks predicate parameter
        # Operation verifies object existence
        # Code ensures subject availability
        if timeline is None or not timeline.elements:
            st.warning("No timeline or elements to display")
            return

        # Function creates subject layout
        # Method sets predicate structure
        # Operation organizes object interface
        # Code arranges subject visualization
        if show_details:
            col1, col2 = st.columns([3, 1])
        else:
            col1 = st

        with col1:
            # Function creates subject figure
            # Method generates predicate visualization
            # Operation builds object chart
            # Code constructs subject timeline
            fig = self._create_timeline_figure(timeline, height)
            st.plotly_chart(fig, use_container_width=True)

        # Function shows subject details
        # Method displays predicate information
        # Operation presents object data
        # Code provides subject context
        if show_details:
            with col2:
                self._display_element_details(timeline)

    # Function creates subject figure
    # Method generates predicate chart
    # Operation builds object visualization
    # Code constructs subject graphic
    def _create_timeline_figure(
        self, timeline: StoryTimeline, height: int = 600
    ) -> go.Figure:
        """
        Create a Plotly figure for the timeline visualization

        # Function creates subject figure
        # Method generates predicate chart
        # Operation builds object visualization
        # Code constructs subject graphic

        Args:
            timeline: Timeline to visualize
            height: Height of the visualization in pixels

        Returns:
            Plotly Figure object
        """
        # Function prepares subject data
        # Method transforms predicate elements
        # Operation structures object information
        # Code organizes subject content
        elements = timeline.elements

        # Sort elements by timestamp
        elements.sort(key=lambda x: x.timestamp)

        # Function creates subject dataframe
        # Method transforms predicate data
        # Operation structures object information
        # Code prepares subject visualization
        df = pd.DataFrame(
            [
                {
                    "id": e.id,
                    "title": e.title,
                    "description": e.description,
                    "type": e.element_type.value,
                    "status": e.status.value,
                    "timestamp": e.timestamp,
                    "parent_id": e.parent_id,
                    "attributes": e.attributes,
                }
                for e in elements
            ]
        )

        # Function handles subject edge-case
        # Method checks predicate dataframe
        # Operation verifies object content
        # Code ensures subject rendering
        if df.empty:
            # Create empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="No timeline elements to display",
                showarrow=False,
                font=dict(size=14),
            )
            fig.update_layout(height=height)
            return fig

        # Get time range with buffer
        min_time = df["timestamp"].min() - timedelta(hours=1)
        max_time = df["timestamp"].max() + timedelta(hours=1)

        # Function creates subject figure
        # Method initializes predicate chart
        # Operation prepares object visualization
        # Code sets up subject layout
        fig = go.Figure()

        # Vertical timeline axis
        fig.add_shape(
            type="line",
            x0=0.05,
            x1=0.05,
            y0=0,
            y1=1,
            line=dict(color="#888888", width=2),
            xref="paper",
            yref="paper",
        )

        # Function processes subject elements
        # Method visualizes predicate items
        # Operation renders object components
        # Code displays subject narrative
        for i, row in df.iterrows():
            element_type = row["type"]
            status = row["status"]

            # Get styling from theme
            type_style = self.theme.get(element_type, self.theme["event"])
            status_style = self.theme.get(status, self.theme["planned"])

            # Normalize time to 0-1 range for y-axis
            time_range = (max_time - min_time).total_seconds()
            if time_range == 0:  # Handle single point
                y_pos = 0.5
            else:
                y_pos = (
                    row["timestamp"] - min_time
                ).total_seconds() / time_range

            # Add marker
            fig.add_trace(
                go.Scatter(
                    x=[0.05],
                    y=[y_pos],
                    mode="markers+text",
                    text=[type_style["icon"]],
                    textposition="middle center",
                    textfont=dict(color="white", size=type_style["size"] * 0.7),
                    marker=dict(
                        symbol="circle",
                        size=type_style["size"],
                        color=status_style["color"],
                        line=dict(color=type_style["color"], width=2),
                    ),
                    customdata=[row["id"]],
                    hoverinfo="text",
                    hovertext=f"{row['title']} ({row['type']}, {row['status']})<br>{row['timestamp'].strftime('%Y-%m-%d %H:%M')}",
                    name=row["type"],
                    showlegend=False,
                )
            )

            # Add text label
            fig.add_trace(
                go.Scatter(
                    x=[0.12],
                    y=[y_pos],
                    mode="text",
                    text=[row["title"]],
                    textposition="middle left",
                    textfont=dict(color="#333333", size=12),
                    customdata=[row["id"]],
                    hoverinfo="text",
                    hovertext=row["description"],
                    showlegend=False,
                )
            )

            # Add timestamp
            fig.add_trace(
                go.Scatter(
                    x=[0.02],
                    y=[y_pos],
                    mode="text",
                    text=[row["timestamp"].strftime("%H:%M")],
                    textposition="middle right",
                    textfont=dict(color="#666666", size=10),
                    showlegend=False,
                )
            )

            # Connect elements with lines if they have parent relationships
            if row["parent_id"] and row["parent_id"] in df["id"].values:
                parent_row = df[df["id"] == row["parent_id"]].iloc[0]
                parent_time = parent_row["timestamp"]
                parent_y_pos = (
                    ((parent_time - min_time).total_seconds() / time_range)
                    if time_range > 0
                    else 0.5
                )

                fig.add_shape(
                    type="line",
                    x0=0.05,
                    x1=0.08,
                    y0=parent_y_pos,
                    y1=y_pos,
                    line=dict(
                        color="#AAAAAA", width=1, dash=status_style["pattern"]
                    ),
                    xref="paper",
                    yref="paper",
                )

        # Function styles subject figure
        # Method enhances predicate appearance
        # Operation improves object aesthetics
        # Code refines subject visualization
        fig.update_layout(
            height=height,
            margin=dict(l=50, r=50, t=50, b=50),
            paper_bgcolor="rgba(245,245,245,1)",
            plot_bgcolor="rgba(245,245,245,1)",
            title=dict(text=timeline.title, x=0.5, xanchor="center"),
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=[-0.1, 1.1],
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=[-0.05, 1.05],
            ),
            showlegend=False,
            annotations=[
                dict(
                    x=0.5,
                    y=1.05,
                    xref="paper",
                    yref="paper",
                    text=timeline.description,
                    showarrow=False,
                    font=dict(size=12),
                    align="center",
                )
            ],
        )

        # Add date labels for days
        days = {}
        for ts in df["timestamp"]:
            day_str = ts.strftime("%Y-%m-%d")
            days[day_str] = ts

        # Add day markers
        for i, (day_str, ts) in enumerate(days.items()):
            y_pos = (
                ((ts - min_time).total_seconds() / time_range)
                if time_range > 0
                else 0.5
            )

            fig.add_annotation(
                x=-0.01,
                y=y_pos,
                xref="paper",
                yref="paper",
                text=day_str,
                showarrow=False,
                font=dict(size=10),
                align="right",
                bgcolor="rgba(255,255,255,0.7)",
                bordercolor="#DDDDDD",
                borderwidth=1,
                borderpad=2,
            )

            # Day separator line
            fig.add_shape(
                type="line",
                x0=0,
                x1=1,
                y0=y_pos,
                y1=y_pos,
                line=dict(color="#DDDDDD", width=1, dash="dot"),
                xref="paper",
                yref="paper",
            )

        # Add legend for element types and statuses
        y_pos = 1.05
        x_pos = 0.7
        for element_type, style in self.theme.items():
            if element_type in [t.value for t in StoryElementType]:
                fig.add_annotation(
                    x=x_pos,
                    y=y_pos,
                    xref="paper",
                    yref="paper",
                    text=f"{style['icon']} {element_type.capitalize()}",
                    showarrow=False,
                    font=dict(size=10, color="#333333"),
                    align="left",
                    bgcolor="rgba(255,255,255,0.7)",
                    bordercolor=style["color"],
                    borderwidth=1,
                    borderpad=2,
                )
                y_pos -= 0.03

                if y_pos < 0.8:
                    y_pos = 1.05
                    x_pos += 0.15

        # Make interactive - click events
        fig.update_layout(clickmode="event+select")

        return fig

    # Function displays subject details
    # Method shows predicate information
    # Operation presents object data
    # Code provides subject context
    def _display_element_details(self, timeline: StoryTimeline) -> None:
        """
        Display details panel for timeline elements

        # Function displays subject details
        # Method shows predicate information
        # Operation presents object context
        # Code provides subject explanation

        Args:
            timeline: Timeline to display details for
        """
        # Function creates subject header
        # Method displays predicate title
        # Operation shows object heading
        # Code presents subject section
        st.subheader("Element Details")

        # Function retrieves subject selection
        # Method gets predicate choice
        # Operation obtains object selection
        # Code determines subject focus
        selected_id = st.session_state.get("selected_timeline_element")

        # Function creates subject filters
        # Method provides predicate controls
        # Operation enables object selection
        # Code enhances subject interaction

        # Type filter
        element_types = [t.value for t in StoryElementType]
        selected_type = st.selectbox(
            "Filter by type",
            ["All Types"] + element_types,
            key="timeline_type_filter",
        )

        # Status filter
        status_types = [s.value for s in ElementStatus]
        selected_status = st.selectbox(
            "Filter by status",
            ["All Statuses"] + status_types,
            key="timeline_status_filter",
        )

        # Function applies subject filters
        # Method filters predicate elements
        # Operation narrows object collection
        # Code reduces subject scope
        filtered_elements = timeline.elements

        if selected_type != "All Types":
            filtered_elements = [
                e
                for e in filtered_elements
                if e.element_type.value == selected_type
            ]

        if selected_status != "All Statuses":
            filtered_elements = [
                e
                for e in filtered_elements
                if e.status.value == selected_status
            ]

        # Sort by timestamp (newest first)
        filtered_elements.sort(key=lambda x: x.timestamp, reverse=True)

        # Function shows subject count
        # Method displays predicate statistic
        # Operation presents object metric
        # Code shows subject information
        st.write(
            f"Showing {len(filtered_elements)} of {len(timeline.elements)} elements"
        )

        # Function creates subject container
        # Method prepares predicate display
        # Operation sets object area
        # Code establishes subject section
        details_container = st.container()

        # Function displays subject elements
        # Method shows predicate details
        # Operation presents object information
        # Code reveals subject data
        for element in filtered_elements:
            with details_container:
                # Get styling
                type_style = self.theme.get(
                    element.element_type.value, self.theme["event"]
                )
                status_style = self.theme.get(
                    element.status.value, self.theme["planned"]
                )

                # Create expandable element card
                with st.expander(
                    f"{type_style['icon']} {element.title}",
                    expanded=(element.id == selected_id),
                ):
                    st.markdown(
                        f"**Type:** {element.element_type.value.capitalize()}"
                    )
                    st.markdown(
                        f"**Status:** {element.status.value.capitalize()}"
                    )
                    st.markdown(
                        f"**Time:** {element.timestamp.strftime('%Y-%m-%d %H:%M')}"
                    )
                    st.markdown(f"**Description:**")
                    st.markdown(element.description)

                    # Display attributes if any
                    if element.attributes:
                        st.markdown("**Attributes:**")
                        for key, value in element.attributes.items():
                            st.markdown(f"- **{key}:** {value}")

                    # Display linked entities if any
                    if element.linked_entities:
                        st.markdown("**Linked Entities:**")
                        for entity in element.linked_entities:
                            st.markdown(f"- {entity}")

                    # Display linked artifacts if any
                    if element.linked_artifacts:
                        st.markdown("**Linked Artifacts:**")
                        for artifact in element.linked_artifacts:
                            st.markdown(f"- {artifact}")

    # Function visualizes subject summary
    # Method renders predicate metrics
    # Operation displays object statistics
    # Code presents subject overview
    def display_timeline_summary(
        self, timeline: Optional[StoryTimeline] = None
    ) -> None:
        """
        Display summary statistics for a timeline

        # Function displays subject summary
        # Method shows predicate statistics
        # Operation presents object metrics
        # Code reveals subject overview

        Args:
            timeline: Timeline to summarize (uses current_timeline if None)
        """
        # Use provided timeline or current one
        timeline = timeline or self.current_timeline

        # Function validates subject input
        # Method checks predicate parameter
        # Operation verifies object existence
        # Code ensures subject availability
        if timeline is None:
            st.warning("No timeline to summarize")
            return

        # Function creates subject layout
        # Method organizes predicate structure
        # Operation arranges object interface
        # Code sets subject presentation
        st.subheader("Timeline Summary")

        # Basic statistics
        element_count = len(timeline.elements)
        created_date = timeline.created_at.strftime("%Y-%m-%d")
        updated_date = timeline.updated_at.strftime("%Y-%m-%d")
        date_range = ""

        if timeline.elements:
            min_date = min(e.timestamp for e in timeline.elements).strftime(
                "%Y-%m-%d"
            )
            max_date = max(e.timestamp for e in timeline.elements).strftime(
                "%Y-%m-%d"
            )
            date_range = f"{min_date} to {max_date}"

        # Function creates subject columns
        # Method organizes predicate layout
        # Operation arranges object display
        # Code structures subject presentation
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Elements", element_count)

        with col2:
            st.metric("Timeline Created", created_date)

        with col3:
            st.metric("Last Updated", updated_date)

        if date_range:
            st.metric("Date Range", date_range)

        # Calculate element type distribution
        type_counts = {}
        for element in timeline.elements:
            element_type = element.element_type.value
            type_counts[element_type] = type_counts.get(element_type, 0) + 1

        # Calculate status distribution
        status_counts = {}
        for element in timeline.elements:
            status = element.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

        # Function creates subject layout
        # Method organizes predicate display
        # Operation arranges object charts
        # Code structures subject visuals
        col1, col2 = st.columns(2)

        # Function creates subject chart
        # Method generates predicate visualization
        # Operation builds object figure
        # Code constructs subject graph
        with col1:
            if type_counts:
                st.subheader("Element Types")

                fig = px.pie(
                    values=list(type_counts.values()),
                    names=list(type_counts.keys()),
                    color=list(type_counts.keys()),
                    color_discrete_map={
                        t: self.theme.get(t, self.theme["event"])["color"]
                        for t in type_counts.keys()
                    },
                    hole=0.4,
                )

                fig.update_layout(
                    margin=dict(l=10, r=10, t=10, b=10), height=300
                )

                st.plotly_chart(fig, use_container_width=True)

        # Function creates subject chart
        # Method generates predicate visualization
        # Operation builds object figure
        # Code constructs subject graph
        with col2:
            if status_counts:
                st.subheader("Element Status")

                fig = px.bar(
                    x=list(status_counts.keys()),
                    y=list(status_counts.values()),
                    color=list(status_counts.keys()),
                    color_discrete_map={
                        s: self.theme.get(s, self.theme["planned"])["color"]
                        for s in status_counts.keys()
                    },
                )

                fig.update_layout(
                    xaxis_title="Status",
                    yaxis_title="Count",
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=300,
                )

                st.plotly_chart(fig, use_container_width=True)

        # Function displays subject milestones
        # Method shows predicate key events
        # Operation presents object landmarks
        # Code reveals subject progress
        milestones = timeline.get_elements_by_type(StoryElementType.MILESTONE)

        if milestones:
            st.subheader("Key Milestones")

            for milestone in sorted(milestones, key=lambda m: m.timestamp):
                status_style = self.theme.get(
                    milestone.status.value, self.theme["planned"]
                )

                st.markdown(
                    f"<div style='padding: 10px; margin: 5px 0; border-left: 4px solid {status_style['color']};'>"
                    f"<span style='font-weight: bold;'>{milestone.timestamp.strftime('%Y-%m-%d')}:</span> "
                    f"{milestone.title} <span style='color: {status_style['color']}'>({milestone.status.value})</span><br>"
                    f"<span style='font-size: 0.9em;'>{milestone.description}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    # Function creates subject interface
    # Method builds predicate UI
    # Operation generates object controls
    # Code constructs subject editor
    def create_timeline_editor(self) -> None:
        """
        Create an interface for editing timelines

        # Function creates subject editor
        # Method builds predicate interface
        # Operation generates object controls
        # Code constructs subject UI
        """
        # Function creates subject header
        # Method displays predicate title
        # Operation shows object section
        # Code presents subject heading
        st.subheader("Timeline Editor")

        # Function creates subject tabs
        # Method organizes predicate interface
        # Operation structures object sections
        # Code separates subject functions
        tab1, tab2, tab3 = st.tabs(["Create/Edit", "Load", "Export"])

        # Function implements subject creation
        # Method builds predicate interface
        # Operation generates object controls
        # Code constructs subject editor
        with tab1:
            # New timeline form
            if not self.current_timeline:
                st.subheader("Create New Timeline")

                with st.form("new_timeline_form"):
                    title = st.text_input(
                        "Timeline Title", "My Workflow Progress"
                    )
                    description = st.text_area(
                        "Description",
                        "Interactive visualization of workflow progress",
                    )

                    if st.form_submit_button("Create Timeline"):
                        self.create_timeline(title, description)
                        st.success(f"Created new timeline: {title}")
                        st.experimental_rerun()

            # Edit current timeline
            else:
                st.subheader(f"Editing: {self.current_timeline.title}")

                # Timeline properties
                with st.expander("Timeline Properties", expanded=False):
                    new_title = st.text_input(
                        "Title", self.current_timeline.title
                    )
                    new_description = st.text_area(
                        "Description", self.current_timeline.description
                    )

                    if st.button("Update Timeline"):
                        self.current_timeline.title = new_title
                        self.current_timeline.description = new_description
                        self.current_timeline.updated_at = datetime.now()
                        self.save_timeline()
                        st.success("Timeline updated")

                # Add new element
                with st.expander("Add New Element", expanded=True):
                    with st.form("new_element_form"):
                        element_type = st.selectbox(
                            "Element Type", [t.value for t in StoryElementType]
                        )

                        element_title = st.text_input("Title")
                        element_description = st.text_area("Description")

                        status = st.selectbox(
                            "Status", [s.value for s in ElementStatus]
                        )

                        # Optional: parent element
                        if self.current_timeline.elements:
                            parent_options = [("None", None)] + [
                                (e.title, e.id)
                                for e in self.current_timeline.elements
                            ]
                            parent_titles, parent_ids = zip(*parent_options)
                            parent_idx = st.selectbox(
                                "Parent Element",
                                range(len(parent_options)),
                                format_func=lambda x: parent_titles[x],
                            )
                            parent_id = parent_ids[parent_idx]
                        else:
                            parent_id = None

                        # Optional: attributes
                        st.subheader("Attributes (Optional)")
                        attr_key1 = st.text_input("Attribute 1 Key")
                        attr_val1 = st.text_input("Attribute 1 Value")

                        attr_key2 = st.text_input("Attribute 2 Key")
                        attr_val2 = st.text_input("Attribute 2 Value")

                        if st.form_submit_button("Add Element"):
                            # Create attributes dictionary
                            attributes = {}
                            if attr_key1 and attr_val1:
                                attributes[attr_key1] = attr_val1
                            if attr_key2 and attr_val2:
                                attributes[attr_key2] = attr_val2

                            # Create element based on type
                            if element_type == StoryElementType.MILESTONE.value:
                                element = StoryMilestone(
                                    title=element_title,
                                    description=element_description,
                                    status=ElementStatus(status),
                                    parent_id=parent_id,
                                    attributes=attributes,
                                )
                            else:
                                element = StoryElement(
                                    title=element_title,
                                    description=element_description,
                                    element_type=StoryElementType(element_type),
                                    status=ElementStatus(status),
                                    parent_id=parent_id,
                                    attributes=attributes,
                                )

                            # Add element to timeline
                            self.add_element(element)
                            self.save_timeline()
                            st.success(
                                f"Added new {element_type} element: {element_title}"
                            )
                            st.experimental_rerun()

        # Function implements subject loading
        # Method builds predicate interface
        # Operation generates object controls
        # Code constructs subject loader
        with tab2:
            st.subheader("Load Timeline")

            # List available timelines
            timelines = self.list_timelines()

            if not timelines:
                st.info("No saved timelines found")
            else:
                # Display timeline selection
                timeline_options = [(t["title"], t["id"]) for t in timelines]
                timeline_titles, timeline_ids = zip(*timeline_options)

                selected_idx = st.selectbox(
                    "Select a timeline to load",
                    range(len(timeline_options)),
                    format_func=lambda x: timeline_titles[x],
                )

                selected_id = timeline_ids[selected_idx]
                selected_timeline = timelines[selected_idx]

                # Display timeline info
                st.markdown(
                    f"**Description:** {selected_timeline.get('description', '')}"
                )
                st.markdown(
                    f"**Created:** {selected_timeline.get('created_at', '')}"
                )
                st.markdown(
                    f"**Updated:** {selected_timeline.get('updated_at', '')}"
                )
                st.markdown(
                    f"**Elements:** {selected_timeline.get('element_count', 0)}"
                )

                if st.button("Load Selected Timeline"):
                    self.load_timeline(selected_id)
                    st.success(f"Loaded timeline: {selected_timeline['title']}")
                    st.experimental_rerun()

                if st.button("Delete Selected Timeline", type="secondary"):
                    filepath = selected_timeline.get("filepath")
                    if filepath and os.path.exists(filepath):
                        try:
                            os.remove(filepath)
                            st.success(
                                f"Deleted timeline: {selected_timeline['title']}"
                            )

                            # Reset current timeline if it was the deleted one
                            if (
                                self.current_timeline
                                and self.current_timeline.id == selected_id
                            ):
                                self.current_timeline = None

                            st.experimental_rerun()
                        except Exception as e:
                            st.error(f"Error deleting timeline: {e}")

        # Function implements subject export
        # Method builds predicate interface
        # Operation generates object controls
        # Code constructs subject exporter
        with tab3:
            st.subheader("Export Timeline")

            if not self.current_timeline:
                st.warning("No timeline currently loaded")
            else:
                st.info(f"Current timeline: {self.current_timeline.title}")

                # Export to JSON
                timeline_json = json.dumps(
                    self.current_timeline.to_dict(), indent=2
                )
                st.download_button(
                    label="Download JSON",
                    data=timeline_json,
                    file_name=f"{self.current_timeline.id}.json",
                    mime="application/json",
                )

                # Display JSON preview
                with st.expander("JSON Preview"):
                    st.code(timeline_json, language="json")

    # Function creates subject interface
    # Method builds predicate visualization
    # Operation generates object display
    # Code constructs subject dashboard
    def create_storyteller_dashboard(self) -> None:
        """
        Create a complete storyteller dashboard in Streamlit

        # Function creates subject dashboard
        # Method builds predicate interface
        # Operation generates object visualization
        # Code constructs subject presentation
        """
        # Function creates subject header
        # Method displays predicate title
        # Operation shows object section
        # Code presents subject heading
        st.title(self.title)
        st.markdown(self.description)

        # Function creates subject tabs
        # Method organizes predicate interface
        # Operation structures object sections
        # Code separates subject functions
        tab1, tab2, tab3 = st.tabs(
            ["Timeline Visualization", "Timeline Editor", "Analytics"]
        )

        with tab1:
            if self.current_timeline:
                self.display_timeline_visualization()
            else:
                st.info(
                    "No timeline loaded. Create or load a timeline in the Timeline Editor tab."
                )

        with tab2:
            self.create_timeline_editor()

        with tab3:
            if self.current_timeline:
                self.display_timeline_summary()
            else:
                st.info("No timeline loaded. Create or load a timeline first.")
