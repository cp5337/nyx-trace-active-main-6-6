"""
Enhanced HTML Renderer with Streamlit Native Component Fallbacks
=================================================================

This module provides improved HTML rendering that gracefully falls back
to native Streamlit components when HTML rendering fails.
"""

import streamlit as st
import streamlit.components.v1 as components
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

def render_task_card_native(task_data: Dict[str, Any]) -> None:
    """
    Render task card using native Streamlit components as fallback.
    
    Args:
        task_data: Dictionary containing task information
    """
    # Use native Streamlit components for reliable rendering
    with st.container():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader(f"🧠 {task_data.get('task_name', 'Unknown Task')}")
            st.text(f"ID: {task_data.get('hash_id', 'N/A')}")
            st.text(f"Category: {task_data.get('category', 'Unknown')}")
        
        with col2:
            # Display metrics using native progress bars
            reliability = task_data.get('reliability', 0.5)
            confidence = task_data.get('confidence', 0.5)
            
            st.metric("Reliability", f"{int(reliability * 100)}%")
            st.progress(reliability)
            
            st.metric("Confidence", f"{int(confidence * 100)}%")  
            st.progress(confidence)
        
        # Description in expandable section
        if task_data.get('description'):
            with st.expander("Description"):
                st.write(task_data['description'])

def render_with_fallback(html_content: str, task_data: Optional[Dict[str, Any]] = None) -> bool:
    """
    Render HTML with automatic fallback to native components.
    
    Args:
        html_content: HTML string to render
        task_data: Optional task data for native fallback
        
    Returns:
        bool: Success status
    """
    try:
        # Try HTML rendering first
        components.html(html_content, height=300, scrolling=True)
        logger.info("HTML rendered successfully using components.html")
        return True
    except Exception as e:
        logger.warning(f"HTML rendering failed: {e}. Using native fallback.")
        
        # Fallback to native Streamlit components
        if task_data:
            render_task_card_native(task_data)
            return True
        else:
            # Last resort: display as code
            st.warning("HTML rendering failed. Displaying raw content:")
            st.code(html_content[:500] + "..." if len(html_content) > 500 else html_content)
            return False
