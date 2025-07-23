"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-HTML-RENDERER-0001                  │
// │ 📁 domain       : Utilities, Rendering                     │
// │ 🧠 description  : HTML renderer for CTAS components        │
// │                  with Replit-safe display                  │
// │ 🕸️ hash_type    : UUID → CUID-linked module               │
// │ 🔄 parent_node  : NODE_UTILITIES                          │
// │ 🧩 dependencies : streamlit                               │
// │ 🔧 tool_usage   : Rendering, UI                           │
// │ 📡 input_type   : HTML strings                            │
// │ 🧪 test_status  : stable                                  │
// │ 🧠 cognitive_fn : rendering, display                      │
// │ ⌛ TTL Policy   : 6.5 Persistent                          │
// └─────────────────────────────────────────────────────────────┘

HTML Safe Renderer Utility for Replit
-------------------------------------
This module provides utilities for safely rendering HTML content in Streamlit,
especially when running in Replit's environment where some rendering approaches
may not work correctly.
"""

import streamlit as st
import streamlit.components.v1 as components
import logging

# Configure logger
logger = logging.getLogger("html_renderer")
logger.setLevel(logging.INFO)

def render_html(html_content, use_component=True, height=None, container=None):
    """
    Render HTML content safely in Streamlit, with special handling for Replit.
    
    This function uses multiple methods to ensure content renders correctly:
    1. Primary: streamlit.components.v1.html for isolated rendering
    2. Backup: st.markdown with unsafe_allow_html=True
    
    Args:
        html_content: HTML string content to render
        use_component: Whether to use components.html (more reliable in Replit)
        height: Height for components.html rendering (auto if None)
        container: Optional container to render in (st.container)
    
    Returns:
        bool: Success status
    """
    if not html_content:
        logger.warning("Empty HTML content provided")
        return False
    
    target = container if container else st
    
    try:
        if use_component:
            # Calculate appropriate height if not specified
            if height is None:
                # Estimate height based on content length and complexity
                # Roughly 20px per line of content
                num_lines = html_content.count('\n') + 1
                height = max(100, min(800, num_lines * 20))
            
            # Render using components.html for better isolation
            components.html(
                f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <style>
                    body {{ margin: 0; padding: 0; }}
                    </style>
                </head>
                <body>
                    {html_content}
                </body>
                </html>
                """,
                height=height,
                scrolling=True
            )
            logger.info(f"Rendered HTML content using components.html with height={height}")
            return True
        else:
            # Use st.markdown as fallback or when specifically requested
            target.markdown(html_content, unsafe_allow_html=True)
            logger.info("Rendered HTML content using st.markdown")
            return True
    except Exception as e:
        logger.error(f"Failed to render HTML content: {str(e)}")
        # Last resort fallback - basic markdown with warning
        try:
            target.warning("HTML rendering failed. Displaying basic content.")
            target.code(html_content[:500] + "..." if len(html_content) > 500 else html_content)
            return False
        except:
            logger.error("Even fallback rendering failed")
            return False

def render_css(css_content):
    """
    Render CSS content safely in Streamlit with Replit-compatible approach.
    
    Args:
        css_content: CSS string content to inject
    
    Returns:
        bool: Success status
    """
    try:
        # Primary method: Use components.html for reliable CSS injection
        components.html(
            f"""
            <style>
            {css_content}
            </style>
            """,
            height=0,
            width=0
        )
        logger.info("Injected CSS using components.html")
        
        # Backup method: Also inject via markdown for wider compatibility
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
        logger.info("Injected CSS using st.markdown as backup")
        
        return True
    except Exception as e:
        logger.error(f"Failed to inject CSS: {str(e)}")
        return False