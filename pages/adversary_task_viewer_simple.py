# ┌─────────────────────────────────────────────────────────────┐
# │ █████████████████ CTAS USIM HEADER ███████████████████████ │
# ├─────────────────────────────────────────────────────────────┤
# │ 🔖 hash_id      : USIM-ADVERSARY-TASK-VIEWER-0001          │
# │ 📁 domain       : Visualization, Task Viewer                │
# │ 🧠 description  : Task viewer for CTAS                      │
# │                  adversary task model visualization         │
# │ 🕸️ hash_type    : UUID → CUID-linked module                 │
# │ 🔄 parent_node  : NODE_VISUALIZATION                       │
# │ 🧩 dependencies : streamlit, plotly, uuid                  │
# │ 🔧 tool_usage   : Visualization, Analysis                   │
# │ 📡 input_type   : User interface events                     │
# │ 🧪 test_status  : stable                                   │
# │ 🧠 cognitive_fn : visualization, analysis                   │
# │ ⌛ TTL Policy   : 6.5 Persistent                           │
# └─────────────────────────────────────────────────────────────┘

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import os
import logging
import random

from core.periodic_table.task_registry import TaskRegistry

# Configure logger
logger = logging.getLogger("adversary_task_viewer")
logger.setLevel(logging.INFO)

# Error-safe CSS loader
def load_css():
    css_path = os.path.join("public", "style.css")
    try:
        with open(css_path, "r") as f:
            css = f"<style>{f.read()}</style>"
            st.markdown(css, unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("CSS file not found at 'public/style.css'.")

# Error-safe registry loader
def initialize_registry():
    try:
        data_dir = "attached_assets"
        if not os.path.isdir(data_dir):
            st.error("Data directory 'attached_assets' not found.")
            return None

        registry = TaskRegistry(db_path=":memory:")
        loaded = registry.load_tasks_from_directory(data_dir)
        if loaded == 0:
            st.error("No tasks loaded from 'attached_assets'.")
            return None

        registry.load_relationships_from_db()
        return registry
    except Exception as e:
        logger.error(f"Failed to initialize registry: {e}")
        st.error("Registry initialization failed.")
        return None

# Main entry
def main():
    st.set_page_config(page_title="CTAS Adversary Task Viewer", layout="wide")
    load_css()
    st.title("CTAS Periodic Table of Adversary Tasks")

    if "registry" not in st.session_state:
        with st.spinner("Loading task registry..."):
            st.session_state.registry = initialize_registry()

    registry = st.session_state.get("registry")
    if not registry:
        st.stop()

    tasks = registry.get_all_tasks()
    if not tasks:
        st.warning("No tasks to display.")
        return

    # Display tasks in a clean, simple format
    st.subheader("Adversary Tasks in Registry")
    
    # Basic task display with columns for organization
    cols = st.columns(3)
    for i, task in enumerate(tasks):
        with cols[i % 3]:
            with st.expander(f"{task.hash_id}: {task.task_name}", expanded=False):
                st.write(f"**Category:** {task.get_category()}")
                st.write(f"**Description:** {task.description}")
                
                # Display metrics in a clean way
                metrics_col1, metrics_col2 = st.columns(2)
                with metrics_col1:
                    st.metric("Reliability", f"{int(task.get_reliability() * 100)}%")
                with metrics_col2:
                    st.metric("Confidence", f"{int(task.get_confidence() * 100)}%")
                
                # Display additional properties if available
                if hasattr(task, 'capabilities') and task.capabilities:
                    st.write("**Capabilities:**")
                    st.write(task.capabilities)
                
                if hasattr(task, 'limitations') and task.limitations:
                    st.write("**Limitations:**")
                    st.write(task.limitations)

if __name__ == "__main__":
    main()