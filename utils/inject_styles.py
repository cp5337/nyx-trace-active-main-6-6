"""
Utility for injecting CSS styles into Streamlit pages
"""

import streamlit as st
import streamlit.components.v1 as components
import os
from pathlib import Path

def load_css_file(css_file_path):
    """
    Load CSS from a file path
    
    Args:
        css_file_path: Path to the CSS file
        
    Returns:
        str: The CSS content as a string
    """
    try:
        with open(css_file_path, 'r') as f:
            return f.read()
    except Exception as e:
        st.error(f"Failed to load CSS file: {e}")
        return ""

def inject_css(css_content=None, css_file_path=None):
    """
    Inject CSS into a Streamlit app using components.html
    
    Args:
        css_content: CSS content as string (optional)
        css_file_path: Path to CSS file (optional)
        
    If both are provided, css_content takes precedence
    """
    if not css_content and css_file_path:
        css_content = load_css_file(css_file_path)
    
    if not css_content:
        return
    
    # Inject the CSS using components.html for reliable rendering
    components.html(
        f"""
        <style>
        {css_content}
        </style>
        """,
        height=0,
        width=0
    )

def inject_default_styles():
    """
    Inject the default application styles from public/style.css
    """
    # Determine the path to style.css
    current_dir = Path(__file__).parent
    root_dir = current_dir.parent
    css_path = root_dir / "public" / "style.css"
    
    if css_path.exists():
        inject_css(css_file_path=str(css_path))
    else:
        st.warning("Style file not found: public/style.css")