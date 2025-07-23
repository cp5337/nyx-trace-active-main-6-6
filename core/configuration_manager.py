"""
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
