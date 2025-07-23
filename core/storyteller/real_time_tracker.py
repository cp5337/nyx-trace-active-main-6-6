"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-STORYTELLER-REALTIME-0001           │
// │ 📁 domain       : Storytelling, Tracking, Visualization      │
// │ 🧠 description  : Real-time workflow tracker with            │
// │                  interactive progress visualization          │
// │ 🕸️ hash_type    : UUID → CUID-linked module                 │
// │ 🔄 parent_node  : NODE_INTERFACE                           │
// │ 🧩 dependencies : streamlit, plotly, datetime, story_elements│
// │ 🔧 tool_usage   : Visualization, Real-time Tracking         │
// │ 📡 input_type   : Workflow events, progress updates         │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : real-time monitoring, visualization       │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

Real-Time Workflow Tracker
------------------------
This module provides enhanced real-time tracking capabilities for
the Interactive Workflow Progress Storyteller. It enables live
updates of workflow progress, animated transitions, and rich
interactive visualizations for operational timelines.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from typing import List, Dict, Tuple, Optional, Any, Union

from core.storyteller.story_elements import (
    StoryElement,
    StoryMilestone,
    StoryTimeline,
    StoryElementType,
    ElementStatus,
)

# Set up logging
logger = logging.getLogger(__name__)


class RealTimeWorkflowTracker:
    """
    Real-time tracker for workflow progress visualization

    # Class manages subject tracking
    # Module monitors predicate progress
    # Component visualizes object workflows
    """

    def __init__(
        self,
        timeline: StoryTimeline,
        auto_update: bool = False,
        update_interval: int = 5,
        animation_speed: int = 1000,
    ):
        """
        Initialize the real-time workflow tracker

        # Function initializes subject tracker
        # Method sets predicate properties
        # Constructor configures object state

        Args:
            timeline: The StoryTimeline to track
            auto_update: Whether to automatically update the display
            update_interval: Seconds between automatic updates
            animation_speed: Speed of animations in milliseconds
        """
        self.timeline = timeline
        self.auto_update = auto_update
        self.update_interval = update_interval
        self.animation_speed = animation_speed
        self.last_update = datetime.now()
        self.active_elements = []
        self.color_map = {
            "milestone": "#4CAF50",
            "event": "#2196F3",
            "discovery": "#9C27B0",
            "decision": "#FF9800",
            "obstacle": "#F44336",
            "insight": "#00BCD4",
            "action": "#607D8B",
            "resource": "#8BC34A",
            "planned": "#9E9E9E",
            "in_progress": "#42A5F5",
            "completed": "#4CAF50",
            "blocked": "#F44336",
            "skipped": "#9E9E9E",
            "failed": "#FF5252",
        }

        # Initialize session state for tracking
        if "last_active_elements" not in st.session_state:
            st.session_state.last_active_elements = []
        if "update_counter" not in st.session_state:
            st.session_state.update_counter = 0

    def create_live_timeline_visualization(self) -> None:
        """
        Create an interactive live timeline visualization

        # Function creates subject visualization
        # Method renders predicate timeline
        # Operation displays object progress
        """
        st.subheader("Live Workflow Timeline")

        # Control panel
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

        with col1:
            self.auto_update = st.toggle("Auto-Update", value=self.auto_update)

        with col2:
            if self.auto_update:
                self.update_interval = st.slider(
                    "Update Interval (seconds)",
                    min_value=1,
                    max_value=60,
                    value=self.update_interval,
                )

        with col3:
            self.animation_speed = st.slider(
                "Animation Speed",
                min_value=200,
                max_value=2000,
                value=self.animation_speed,
                step=100,
            )

        with col4:
            if st.button("Force Update"):
                st.session_state.update_counter += 1

        # Update automatically if enabled
        if self.auto_update:
            current_time = datetime.now()
            if (
                current_time - self.last_update
            ).total_seconds() >= self.update_interval:
                self.last_update = current_time
                st.session_state.update_counter += 1

        # Get all elements in the timeline
        elements = self.timeline.elements

        if not elements:
            st.warning(
                "No elements in the timeline. Add elements to visualize progress."
            )
            return

        # Create a DataFrame for visualization
        df = self._create_elements_dataframe(elements)

        # Determine which elements are currently active
        current_time = datetime.now()
        self.active_elements = [
            e.id for e in elements if e.status == ElementStatus.IN_PROGRESS
        ]

        # Create the visualization if we have data
        if len(df) > 0:
            fig = self._create_timeline_visualization(df)

            # Display the visualization
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No elements to display in the timeline yet.")

        # Update session state
        st.session_state.last_active_elements = self.active_elements.copy()

        # Display active tasks
        active_count = len(self.active_elements)
        if active_count > 0:
            st.subheader(f"Active Tasks ({active_count})")
            active_elements = [
                e for e in elements if e.id in self.active_elements
            ]

            for element in active_elements:
                color = self.color_map.get(
                    element.element_type.name.lower(), "#888888"
                )
                self._render_active_element_card(element, color)

    def _create_elements_dataframe(
        self, elements: List[Union[StoryElement, StoryMilestone]]
    ) -> pd.DataFrame:
        """
        Create a DataFrame from timeline elements for visualization

        # Function creates subject dataframe
        # Method organizes predicate data
        # Operation structures object elements

        Args:
            elements: List of timeline elements

        Returns:
            DataFrame with element data
        """
        data = []

        for e in elements:
            element_type = e.element_type.name.lower()
            status = e.status.name.lower()

            data.append(
                {
                    "id": e.id,
                    "title": e.title,
                    "type": element_type,
                    "status": status,
                    "timestamp": e.timestamp,
                    "parent_id": e.parent_id,
                    "description": e.description,
                }
            )

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("date")

        # Add some fields for visualization
        df["color"] = df["type"].apply(
            lambda x: self.color_map.get(x, "#888888")
        )
        df["status_color"] = df["status"].apply(
            lambda x: self.color_map.get(x, "#888888")
        )
        df["active"] = df["id"].apply(lambda x: x in self.active_elements)
        df["was_active"] = df["id"].apply(
            lambda x: x in st.session_state.last_active_elements
        )
        df["highlight"] = df["active"] | ((df["was_active"]) & ~df["active"])

        return df

    def _create_timeline_visualization(self, df: pd.DataFrame) -> go.Figure:
        """
        Create an interactive timeline visualization

        # Function creates subject visualization
        # Method renders predicate timeline
        # Operation generates object figure

        Args:
            df: DataFrame with timeline elements

        Returns:
            Plotly figure object
        """
        # Determine time range
        if len(df) == 0:
            start_date = datetime.now() - timedelta(days=7)
            end_date = datetime.now() + timedelta(days=7)
        else:
            # Add buffer before and after
            # Convert to Python datetime if necessary
            min_date = df["date"].min()
            max_date = df["date"].max()

            if isinstance(min_date, pd.Timestamp):
                min_date = min_date.to_pydatetime()
            if isinstance(max_date, pd.Timestamp):
                max_date = max_date.to_pydatetime()

            start_date = min_date - timedelta(days=1)
            end_date = max_date + timedelta(days=1)

        # Create figure
        fig = go.Figure()

        # Add milestone markers
        milestones = df[df["type"] == "milestone"]
        if not milestones.empty:
            fig.add_trace(
                go.Scatter(
                    x=milestones["date"],
                    y=[0] * len(milestones),
                    mode="markers",
                    marker=dict(
                        size=20,
                        color=milestones["status_color"],
                        symbol="diamond",
                        line=dict(width=2, color="DarkSlateGrey"),
                    ),
                    text=milestones["title"],
                    hoverinfo="text",
                    name="Milestones",
                )
            )

        # Add other elements
        for element_type in df["type"].unique():
            if element_type == "milestone":
                continue

            elements = df[df["type"] == element_type]
            if elements.empty:
                continue

            # Determine y position based on element type
            y_positions = {
                "event": -0.5,
                "discovery": -1.0,
                "decision": -1.5,
                "obstacle": -2.0,
                "insight": -2.5,
                "action": -3.0,
                "resource": -3.5,
            }
            y_pos = y_positions.get(element_type, -4.0)

            # Add scatter trace
            fig.add_trace(
                go.Scatter(
                    x=elements["date"],
                    y=[y_pos] * len(elements),
                    mode="markers",
                    marker=dict(
                        size=12,
                        color=elements["status_color"],
                        symbol="circle",
                        line=dict(width=2, color="DarkSlateGrey"),
                    ),
                    text=elements["title"],
                    hoverinfo="text",
                    name=element_type.capitalize(),
                )
            )

            # Add highlights for active elements
            active_elements = elements[elements["highlight"]]
            if not active_elements.empty:
                fig.add_trace(
                    go.Scatter(
                        x=active_elements["date"],
                        y=[y_pos] * len(active_elements),
                        mode="markers",
                        marker=dict(
                            size=20,
                            color=active_elements["status_color"],
                            opacity=0.7,
                            symbol="circle-open",
                            line=dict(width=3, color="white"),
                        ),
                        text=active_elements["title"],
                        hoverinfo="text",
                        name=f"Active {element_type.capitalize()}",
                    )
                )

        # Add time axis line
        fig.add_shape(
            type="line",
            x0=start_date,
            y0=0,
            x1=end_date,
            y1=0,
            line=dict(
                color="RoyalBlue",
                width=3,
                dash="solid",
            ),
        )

        # Add current time marker
        fig.add_shape(
            type="line",
            x0=datetime.now(),
            y0=-4,
            x1=datetime.now(),
            y1=1,
            line=dict(
                color="Red",
                width=2,
                dash="dash",
            ),
        )

        fig.add_annotation(
            x=datetime.now(),
            y=1,
            text="Current Time",
            showarrow=False,
            yshift=10,
            bgcolor="rgba(255, 0, 0, 0.1)",
        )

        # Update layout
        fig.update_layout(
            title="Real-Time Workflow Progress",
            height=500,
            xaxis=dict(
                title="Timeline", type="date", range=[start_date, end_date]
            ),
            yaxis=dict(
                title="", showticklabels=False, fixedrange=True, range=[-4, 1]
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
            ),
            hovermode="closest",
            margin=dict(l=10, r=10, t=50, b=30),
            plot_bgcolor="rgba(240, 240, 240, 0.7)",
            transition={"duration": self.animation_speed},
        )

        return fig

    def _render_active_element_card(
        self, element: StoryElement, color: str
    ) -> None:
        """
        Render a card for an active element

        # Function renders subject card
        # Method displays predicate element
        # Operation shows object details

        Args:
            element: The StoryElement to display
            color: Color for the card border
        """
        with st.container():
            st.markdown(
                f"""
                <div style="border-left: 4px solid {color}; padding: 10px; margin: 5px 0; background-color: #f5f5f5;">
                <h4 style="margin: 0; color: #333;">{element.title}</h4>
                <p style="margin: 5px 0; color: #666; font-size: 0.9em;">{element.description}</p>
                <div style="display: flex; justify-content: space-between;">
                <span style="font-size: 0.8em; color: #888;">Type: {element.element_type.name}</span>
                <span style="font-size: 0.8em; color: #888;">Status: {element.status.name}</span>
                <span style="font-size: 0.8em; color: #888;">Updated: {element.timestamp.strftime('%Y-%m-%d %H:%M')}</span>
                </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    def create_workflow_activity_feed(self, max_events: int = 10) -> None:
        """
        Create a real-time activity feed for workflow events

        # Function creates subject feed
        # Method displays predicate activity
        # Operation shows object updates

        Args:
            max_events: Maximum number of events to display
        """
        st.subheader("Workflow Activity Feed")

        # Get elements sorted by timestamp (newest first)
        elements = sorted(
            self.timeline.elements, key=lambda e: e.timestamp, reverse=True
        )[:max_events]

        if not elements:
            st.info("No activity to display.")
            return

        for element in elements:
            element_type = element.element_type.name.lower()
            status = element.status.name.lower()
            color = self.color_map.get(element_type, "#888888")

            # Time difference
            time_diff = datetime.now() - element.timestamp
            if time_diff.days > 0:
                time_text = f"{time_diff.days} days ago"
            elif time_diff.seconds // 3600 > 0:
                time_text = f"{time_diff.seconds // 3600} hours ago"
            elif time_diff.seconds // 60 > 0:
                time_text = f"{time_diff.seconds // 60} minutes ago"
            else:
                time_text = f"{time_diff.seconds} seconds ago"

            # Activity icon
            icon_map = {
                "milestone": "✓",
                "event": "•",
                "discovery": "★",
                "decision": "⬧",
                "obstacle": "⚠",
                "insight": "💡",
                "action": "➤",
                "resource": "⚙",
            }
            icon = icon_map.get(element_type, "•")

            with st.container():
                st.markdown(
                    f"""
                    <div style="display: flex; align-items: center; margin-bottom: 10px;">
                    <div style="background-color: {color}; color: white; border-radius: 50%; width: 30px; height: 30px; 
                          display: flex; align-items: center; justify-content: center; margin-right: 10px;">
                        {icon}
                    </div>
                    <div style="flex-grow: 1;">
                        <div style="display: flex; justify-content: space-between;">
                            <b>{element.title}</b>
                            <span style="color: #888; font-size: 0.8em;">{time_text}</span>
                        </div>
                        <div style="color: #666; font-size: 0.9em; margin-top: 3px;">{element.description}</div>
                        <div style="display: flex; margin-top: 3px;">
                            <span style="background-color: {color}; color: white; font-size: 0.7em; padding: 2px 5px; 
                                  border-radius: 10px; margin-right: 5px;">
                                {element_type.upper()}
                            </span>
                            <span style="background-color: {self.color_map.get(status, '#888888')}; color: white; 
                                  font-size: 0.7em; padding: 2px 5px; border-radius: 10px;">
                                {status.upper()}
                            </span>
                        </div>
                    </div>
                    </div>
                    <hr style="margin: 0; border-top: 1px solid #eee;">
                    """,
                    unsafe_allow_html=True,
                )

    def create_workflow_metrics_dashboard(self) -> None:
        """
        Create a metrics dashboard for workflow progress

        # Function creates subject dashboard
        # Method displays predicate metrics
        # Operation shows object statistics
        """
        st.subheader("Workflow Metrics")

        elements = self.timeline.elements

        if not elements:
            st.info("No elements to analyze.")
            return

        # Calculate metrics
        total_elements = len(elements)
        milestones = [
            e for e in elements if e.element_type == StoryElementType.MILESTONE
        ]
        total_milestones = len(milestones)

        completed_elements = len(
            [e for e in elements if e.status == ElementStatus.COMPLETED]
        )
        in_progress_elements = len(
            [e for e in elements if e.status == ElementStatus.IN_PROGRESS]
        )
        planned_elements = len(
            [e for e in elements if e.status == ElementStatus.PLANNED]
        )
        blocked_elements = len(
            [e for e in elements if e.status == ElementStatus.BLOCKED]
        )

        completed_milestones = len(
            [m for m in milestones if m.status == ElementStatus.COMPLETED]
        )

        # Create metrics display
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Overall Progress", f"{completed_elements / total_elements:.1%}"
            )
            st.metric(
                "Milestone Completion",
                f"{completed_milestones}/{total_milestones}",
            )

        with col2:
            st.metric("In Progress", in_progress_elements)
            st.metric("Planned", planned_elements)

        with col3:
            st.metric("Blocked", blocked_elements)

            # Calculate estimated completion
            if completed_elements > 0 and total_milestones > 0:
                first_timestamp = min(
                    [
                        e.timestamp
                        for e in elements
                        if e.status == ElementStatus.COMPLETED
                    ]
                )
                latest_timestamp = max(
                    [
                        e.timestamp
                        for e in elements
                        if e.status == ElementStatus.COMPLETED
                    ]
                )

                if latest_timestamp > first_timestamp:
                    elapsed_days = (latest_timestamp - first_timestamp).days
                    if elapsed_days > 0:
                        completion_rate = completed_elements / elapsed_days
                        remaining_elements = total_elements - completed_elements

                        if completion_rate > 0:
                            days_remaining = (
                                remaining_elements / completion_rate
                            )
                            st.metric(
                                "Est. Days Remaining", f"{days_remaining:.1f}"
                            )

        # Progress by element type
        element_types = {}
        for e in elements:
            element_type = e.element_type.name
            if element_type not in element_types:
                element_types[element_type] = {"total": 0, "completed": 0}

            element_types[element_type]["total"] += 1
            if e.status == ElementStatus.COMPLETED:
                element_types[element_type]["completed"] += 1

        # Create progress bars
        st.subheader("Progress by Element Type")

        for element_type, counts in element_types.items():
            if counts["total"] > 0:
                progress = counts["completed"] / counts["total"]
                st.markdown(
                    f"**{element_type}**: {counts['completed']}/{counts['total']}"
                )
                st.progress(progress)


# CTAS Module Line Count:
# USIM Header: 17 lines
# Comments: 44 lines
# Code: 505 lines
# Total: 566 lines
