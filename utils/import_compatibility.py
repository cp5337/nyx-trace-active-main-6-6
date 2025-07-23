"""
Import Compatibility Module
===========================

This module provides safe imports with fallbacks to prevent import errors.
"""

import sys
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

def safe_import(module_name: str, fallback: Optional[Any] = None) -> Any:
    """
    Safely import a module with fallback handling.
    
    Args:
        module_name: Name of module to import
        fallback: Fallback value if import fails
        
    Returns:
        Imported module or fallback value
    """
    try:
        return __import__(module_name, fromlist=[''])
    except ImportError as e:
        logger.warning(f"Failed to import {module_name}: {e}")
        if fallback is not None:
            return fallback
        raise

def check_dependencies() -> Dict[str, bool]:
    """
    Check if all required dependencies are available.
    
    Returns:
        Dictionary mapping dependency names to availability status
    """
    dependencies = {
        'streamlit': False,
        'plotly': False,
        'pandas': False,
        'numpy': False,
        'sqlalchemy': False
    }
    
    for dep_name in dependencies.keys():
        try:
            __import__(dep_name)
            dependencies[dep_name] = True
            logger.info(f"✅ {dep_name} available")
        except ImportError:
            logger.warning(f"❌ {dep_name} not available")
    
    return dependencies
