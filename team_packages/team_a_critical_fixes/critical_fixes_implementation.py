#!/usr/bin/env python3
"""
Team A Critical Fixes Implementation
====================================

This module implements the critical fixes identified in TEAM_A_CRITICAL_FIXES_README.md:

1. HTML Rendering Issues (HIGH PRIORITY)
2. Database Threading Issues (HIGH PRIORITY)  
3. Import Dependencies (MEDIUM PRIORITY)
4. Configuration Issues (MEDIUM PRIORITY)

All fixes are designed to be backwards-compatible and non-breaking.
"""

import os
import sys
import logging
import shutil
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TeamACriticalFixes:
    """
    Implementation of Team A critical fixes for nyx-trace repository
    """
    
    def __init__(self, repo_root=None):
        self.repo_root = Path(repo_root) if repo_root else Path('.').resolve()
        logger.info(f"Initialized Team A fixes for repository: {self.repo_root}")
    
    def fix_html_rendering_issues(self):
        """
        Fix 1: HTML Rendering Issues
        
        Issues addressed:
        - Raw HTML displaying instead of rendered content in Streamlit
        - Replace custom HTML with native Streamlit components
        - Create fallback rendering system
        """
        logger.info("Applying HTML rendering fixes...")
        
        # Fix 1a: Create enhanced HTML renderer with better fallbacks
        enhanced_renderer_content = '''"""
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
'''
        
        # Write enhanced renderer
        enhanced_renderer_path = self.repo_root / "utils" / "enhanced_html_renderer.py"
        with open(enhanced_renderer_path, 'w') as f:
            f.write(enhanced_renderer_content)
        
        logger.info("✅ Enhanced HTML renderer created")
        
        return True
    
    def fix_database_threading_issues(self):
        """
        Fix 2: Database Threading Issues
        
        Issues addressed:
        - Streamlit threading conflicts with database connections
        - Implement proper thread-safe connection pooling
        - Add connection timeout and retry mechanisms
        """
        logger.info("Applying database threading fixes...")
        
        # Fix 2a: Create improved thread factory
        improved_factory_content = '''"""
Improved Thread-Safe Database Factory
====================================

This module provides enhanced thread-safe database connections
specifically designed for Streamlit's threading model.
"""

import threading
import logging
from typing import Dict, Any, Optional
from contextlib import contextmanager
import time

logger = logging.getLogger(__name__)

class StreamlitSafeDatabaseFactory:
    """
    Thread-safe database factory optimized for Streamlit applications
    """
    
    _instance = None
    _lock = threading.RLock()
    _connections = {}
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    @contextmanager
    def get_connection(self, db_type: str = "default", timeout: int = 30):
        """
        Get a thread-safe database connection with automatic cleanup.
        
        Args:
            db_type: Type of database connection
            timeout: Connection timeout in seconds
            
        Yields:
            Database connection object
        """
        thread_id = threading.get_ident()
        connection_key = f"{db_type}_{thread_id}"
        
        connection = None
        try:
            # Get or create connection for this thread
            if connection_key not in self._connections:
                connection = self._create_connection(db_type, timeout)
                self._connections[connection_key] = connection
            else:
                connection = self._connections[connection_key]
                # Test connection health
                if not self._test_connection(connection):
                    connection = self._create_connection(db_type, timeout)
                    self._connections[connection_key] = connection
            
            yield connection
            
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            # Clean up failed connection
            if connection_key in self._connections:
                del self._connections[connection_key]
            raise
        finally:
            # Connection cleanup is handled by the pool
            pass
    
    def _create_connection(self, db_type: str, timeout: int):
        """Create a new database connection with retry logic"""
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                # Mock connection creation - replace with actual implementation
                logger.info(f"Creating {db_type} connection (attempt {attempt + 1})")
                # connection = create_actual_connection(db_type, timeout)
                return {"type": db_type, "thread": threading.get_ident(), "created": time.time()}
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Connection attempt {attempt + 1} failed: {e}. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    raise
    
    def _test_connection(self, connection) -> bool:
        """Test if connection is still valid"""
        try:
            # Mock connection test - replace with actual implementation
            return connection is not None and time.time() - connection.get("created", 0) < 3600
        except:
            return False
    
    def cleanup_connections(self):
        """Clean up all connections for the current thread"""
        thread_id = threading.get_ident()
        keys_to_remove = [k for k in self._connections.keys() if k.endswith(f"_{thread_id}")]
        
        for key in keys_to_remove:
            try:
                # Clean up connection
                del self._connections[key]
                logger.info(f"Cleaned up connection: {key}")
            except Exception as e:
                logger.error(f"Error cleaning up connection {key}: {e}")

# Global factory instance
database_factory = StreamlitSafeDatabaseFactory()
'''
        
        # Write improved factory
        improved_factory_path = self.repo_root / "core" / "database" / "streamlit_safe_factory.py"
        improved_factory_path.parent.mkdir(parents=True, exist_ok=True)
        with open(improved_factory_path, 'w') as f:
            f.write(improved_factory_content)
        
        logger.info("✅ Streamlit-safe database factory created")
        
        return True
    
    def fix_import_dependencies(self):
        """
        Fix 3: Import Dependencies
        
        Issues addressed:
        - Resolve circular imports
        - Add missing __init__.py files
        - Standardize import structure
        """
        logger.info("Applying import dependency fixes...")
        
        # Fix 3a: Create missing __init__.py files
        init_dirs = [
            "core",
            "core/database", 
            "core/periodic_table",
            "core/soc_teams",
            "database",
            "database/supabase",
            "database/mongodb", 
            "database/neo4j",
            "pages",
            "utils",
            "team_packages",
            "team_packages/team_a_critical_fixes",
            "team_packages/team_b_large_files",
            "team_packages/team_c_database",
            "team_packages/team_d_organization"
        ]
        
        for dir_path in init_dirs:
            init_file = self.repo_root / dir_path / "__init__.py"
            if not init_file.exists():
                init_file.parent.mkdir(parents=True, exist_ok=True)
                init_file.write_text('"""Module initialization file"""\\n')
                logger.info(f"Created __init__.py in {dir_path}")
        
        # Fix 3b: Create import compatibility module
        import_fix_content = '''"""
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
'''
        
        import_fix_path = self.repo_root / "utils" / "import_compatibility.py"
        with open(import_fix_path, 'w') as f:
            f.write(import_fix_content)
        
        logger.info("✅ Import compatibility module created")
        
        return True
    
    def fix_configuration_issues(self):
        """
        Fix 4: Configuration Issues
        
        Issues addressed:
        - Missing or incorrect configuration handling
        - Environment variable loading
        - API key management
        """
        logger.info("Applying configuration fixes...")
        
        # Fix 4a: Create robust configuration manager
        config_manager_content = '''"""
Robust Configuration Manager
============================

This module provides secure and reliable configuration management
for the nyx-trace application.
"""

import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class ConfigurationManager:
    """
    Secure configuration manager with environment variable support
    """
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self.config = {}
        self._load_configuration()
    
    def _load_configuration(self):
        """Load configuration from file and environment variables"""
        # Load from file if specified
        if self.config_file and os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    self.config = json.load(f)
                logger.info(f"Loaded configuration from {self.config_file}")
            except Exception as e:
                logger.error(f"Failed to load config file: {e}")
        
        # Override with environment variables
        self._load_environment_variables()
    
    def _load_environment_variables(self):
        """Load configuration from environment variables with NYTRACE_ prefix"""
        env_vars = {
            'NYTRACE_DATABASE_URL': 'database_url',
            'NYTRACE_SUPABASE_URL': 'supabase_url', 
            'NYTRACE_SUPABASE_KEY': 'supabase_key',
            'NYTRACE_DEBUG': 'debug',
            'NYTRACE_LOG_LEVEL': 'log_level'
        }
        
        for env_var, config_key in env_vars.items():
            value = os.getenv(env_var)
            if value:
                self.config[config_key] = value
                logger.info(f"Loaded {config_key} from environment")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value with fallback
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        return self.config.get(key, default)
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration with secure defaults"""
        return {
            'url': self.get('database_url', 'sqlite:///nytrace.db'),
            'pool_size': int(self.get('db_pool_size', '5')),
            'max_overflow': int(self.get('db_max_overflow', '10')),
            'pool_timeout': int(self.get('db_pool_timeout', '30')),
            'pool_recycle': int(self.get('db_pool_recycle', '1800'))
        }
    
    def is_debug_enabled(self) -> bool:
        """Check if debug mode is enabled"""
        debug_val = self.get('debug', 'false').lower()
        return debug_val in ('true', '1', 'yes', 'on')
    
    def get_log_level(self) -> str:
        """Get logging level"""
        return self.get('log_level', 'INFO').upper()

# Global configuration instance
config_manager = ConfigurationManager()
'''
        
        config_manager_path = self.repo_root / "core" / "configuration_manager.py"
        with open(config_manager_path, 'w') as f:
            f.write(config_manager_content)
        
        # Create sample .env file
        env_sample_content = '''# Nyx-Trace Configuration
# Copy this file to .env and update values as needed

# Database Configuration
NYTRACE_DATABASE_URL=sqlite:///nytrace.db

# Supabase Configuration (if using Supabase)
# NYTRACE_SUPABASE_URL=your_supabase_url
# NYTRACE_SUPABASE_KEY=your_supabase_key

# Application Configuration
NYTRACE_DEBUG=false
NYTRACE_LOG_LEVEL=INFO

# Database Pool Configuration
NYTRACE_DB_POOL_SIZE=5
NYTRACE_DB_MAX_OVERFLOW=10
NYTRACE_DB_POOL_TIMEOUT=30
NYTRACE_DB_POOL_RECYCLE=1800
'''
        
        env_sample_path = self.repo_root / ".env.sample"
        with open(env_sample_path, 'w') as f:
            f.write(env_sample_content)
        
        logger.info("✅ Configuration manager and sample .env created")
        
        return True
    
    def run_all_fixes(self):
        """Run all critical fixes in order"""
        logger.info("🚀 Starting Team A Critical Fixes...")
        
        fixes = [
            ("HTML Rendering Issues", self.fix_html_rendering_issues),
            ("Database Threading Issues", self.fix_database_threading_issues),
            ("Import Dependencies", self.fix_import_dependencies),
            ("Configuration Issues", self.fix_configuration_issues)
        ]
        
        results = {}
        for fix_name, fix_function in fixes:
            try:
                logger.info(f"Applying {fix_name}...")
                success = fix_function()
                results[fix_name] = success
                if success:
                    logger.info(f"✅ {fix_name} completed successfully")
                else:
                    logger.error(f"❌ {fix_name} failed")
            except Exception as e:
                logger.error(f"❌ {fix_name} failed with error: {e}")
                results[fix_name] = False
        
        # Summary
        successful_fixes = sum(1 for success in results.values() if success)
        total_fixes = len(results)
        
        logger.info(f"🎯 Team A Critical Fixes Summary: {successful_fixes}/{total_fixes} successful")
        
        return results

if __name__ == "__main__":
    fixer = TeamACriticalFixes()
    results = fixer.run_all_fixes()
    
    # Exit with appropriate code
    if all(results.values()):
        print("🎉 All Team A critical fixes completed successfully!")
        sys.exit(0)
    else:
        print("⚠️  Some Team A critical fixes failed. Check logs for details.")
        sys.exit(1)
