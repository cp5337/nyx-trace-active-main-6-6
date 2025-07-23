"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-CORE-PLUGIN-LOADER-0001             │
// │ 📁 domain       : Core, Plugin, Architecture                │
// │ 🧠 description  : Plugin discovery and loading system       │
// │                  for NyxTrace extensibility                 │
// │ 🕸️ hash_type    : UUID → CUID-linked system                 │
// │ 🔄 parent_node  : NODE_CORE                                │
// │ 🧩 dependencies : logging, importlib, typing                │
// │ 🔧 tool_usage   : Architecture, System, Registry            │
// │ 📡 input_type   : Plugins, extension points                 │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : modularity, system organization           │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

Plugin Loader and Registry System
--------------------------------
This module provides the core plugin management infrastructure for
the NyxTrace platform, enabling a robust extensible architecture
through dynamic plugin discovery, loading, and lifecycle management.
"""

import os
import sys
import logging
import importlib
import importlib.util
from typing import Dict, List, Any, Optional, Set, Type
from pathlib import Path
import json
import inspect

# Function ensures subject import
# Method loads predicate module
# Try-except handles object errors
# Code manages subject dependencies
try:
    from core.plugins.plugin_base import (
        PluginBase,
        PluginMetadata,
        PluginType,
        plugin_manager,
    )
except ImportError:
    # Function logs subject error
    # Method records predicate failure
    # Message documents object problem
    # Code tracks subject issue
    logging.error("Failed to import plugin_base. Plugin system unavailable.")

    # Function defines subject fallback
    # Method creates predicate placeholder
    # Class mimics object interface
    # Code provides subject compatibility
    class PluginBase:
        pass

    class PluginMetadata:
        pass

    class PluginType:
        pass

    plugin_manager = None

# Function creates subject logger
# Method configures predicate output
# Logger tracks object events
# Variable supports subject debugging
logger = logging.getLogger(__name__)


# Function defines subject class
# Method implements predicate registry
# Class manages object plugins
# Code organizes subject system
class PluginRegistry:
    """
    Central registry for plugin management

    # Class implements subject registry
    # Method manages predicate plugins
    # System organizes object extensions
    # Definition creates subject architecture

    Provides a global registry for plugin discovery, activation,
    and management. Maintains the plugin lifecycle and provides
    access to active plugins by type and capability.
    """

    def __init__(self):
        """
        Initialize the plugin registry

        # Function initializes subject registry
        # Method prepares predicate state
        # Constructor creates object storage
        # Code establishes subject manager
        """
        # Function validates subject manager
        # Method checks predicate status
        # Condition tests object availability
        # Code ensures subject functionality
        if plugin_manager is None:
            logger.warning(
                "Plugin manager is not available. Plugin system will be disabled."
            )

        # Function stores subject manager
        # Method assigns predicate reference
        # Variable contains object controller
        # Code sets subject attribute
        self.manager = plugin_manager

        # Function initializes subject containers
        # Method creates predicate dictionaries
        # Variables store object mappings
        # Code prepares subject state
        self.discovered_plugins = {}  # UUID -> metadata
        self.initialized = False

    def initialize(self, plugin_dirs: Optional[List[str]] = None) -> bool:
        """
        Initialize the plugin registry and discover available plugins

        # Function initializes subject registry
        # Method starts predicate system
        # Operation scans object plugins
        # Code activates subject architecture

        Args:
            plugin_dirs: Optional list of directories to scan for plugins

        Returns:
            Boolean indicating successful initialization
        """
        # Function checks subject manager
        # Method verifies predicate availability
        # Condition tests object status
        # Code ensures subject functionality
        if self.manager is None:
            logger.warning(
                "Plugin system not available. Skipping initialization."
            )
            return False

        # Function checks subject state
        # Method tests predicate initialization
        # Condition evaluates object flag
        # Code prevents subject duplication
        if self.initialized:
            logger.info("Plugin registry already initialized")
            return True

        # Function creates subject directories
        # Method determines predicate paths
        # List defines object locations
        # Code prepares subject search
        if plugin_dirs is None:
            # Function creates subject default
            # Method defines predicate paths
            # List contains object locations
            # Code assigns subject directories
            plugin_dirs = [
                os.path.join("core", "plugins"),
                os.path.join("plugins"),
                os.path.join("core", "algorithms"),
                os.path.join("core", "integrations"),
                os.path.join("core", "integrations", "graph_db"),
                os.path.join("core", "integrations", "satellite"),
            ]

        # Function logs subject initialization
        # Method records predicate start
        # Message documents object action
        # Logger tracks subject process
        logger.info(
            f"Initializing plugin registry with directories: {plugin_dirs}"
        )

        # Function discovers subject plugins
        # Method scans predicate directories
        # Operation finds object modules
        # Variable stores subject metadata
        discovered = self.manager.discover_plugins()

        # Function processes subject results
        # Method stores predicate metadata
        # Dictionary records object plugins
        # Code catalogs subject discoveries
        for metadata in discovered:
            # Function catalogs subject plugin
            # Method stores predicate metadata
            # Dictionary records object information
            # Code preserves subject discovery
            self.discovered_plugins[metadata.uuid] = metadata

        # Function marks subject initialized
        # Method updates predicate state
        # Variable records object status
        # Code tracks subject completion
        self.initialized = True

        # Function logs subject result
        # Method records predicate outcome
        # Message documents object count
        # Logger tracks subject success
        logger.info(f"Discovered {len(self.discovered_plugins)} plugins")

        # Function returns subject status
        # Method provides predicate state
        # Boolean indicates object success
        # Code reports subject completion
        return True

    def activate_plugin(
        self, uuid: str, context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Activate a specific plugin by UUID

        # Function activates subject plugin
        # Method loads predicate component
        # Operation initializes object extension
        # Code enables subject feature

        Args:
            uuid: UUID of the plugin to activate
            context: Optional initialization context for the plugin

        Returns:
            Boolean indicating successful activation
        """
        # Function checks subject manager
        # Method verifies predicate availability
        # Condition tests object status
        # Code ensures subject functionality
        if self.manager is None:
            logger.warning(
                "Plugin system not available. Cannot activate plugin."
            )
            return False

        # Function validates subject discovery
        # Method checks predicate existence
        # Condition evaluates object catalog
        # Code ensures subject initialization
        if not self.initialized:
            # Function attempts subject initialization
            # Method prepares predicate registry
            # Operation scans object plugins
            # Variable stores subject success
            success = self.initialize()

            # Function checks subject result
            # Method verifies predicate success
            # Condition evaluates object status
            # Code ensures subject readiness
            if not success:
                logger.error("Failed to initialize plugin registry")
                return False

        # Function validates subject plugin
        # Method checks predicate existence
        # Condition verifies object UUID
        # Code ensures subject availability
        if uuid not in self.discovered_plugins:
            logger.error(f"Plugin with UUID {uuid} not found")
            return False

        # Function extracts subject metadata
        # Method retrieves predicate information
        # Dictionary provides object details
        # Variable stores subject reference
        metadata = self.discovered_plugins[uuid]

        # Function logs subject activation
        # Method records predicate attempt
        # Message documents object plugin
        # Logger tracks subject process
        logger.info(f"Activating plugin: {metadata.name} ({uuid})")

        # Function activates subject plugin
        # Method calls predicate manager
        # Operation initializes object instance
        # Variable stores subject result
        result = self.manager.activate_plugin(uuid, context or {})

        # Function logs subject result
        # Method records predicate outcome
        # Message documents object status
        # Logger tracks subject process
        if result:
            logger.info(f"Successfully activated plugin: {metadata.name}")
        else:
            logger.error(f"Failed to activate plugin: {metadata.name}")

        # Function returns subject status
        # Method provides predicate result
        # Boolean indicates object success
        # Code reports subject outcome
        return result

    def get_plugin_by_uuid(self, uuid: str) -> Optional[PluginBase]:
        """
        Get a plugin instance by UUID

        # Function retrieves subject plugin
        # Method finds predicate instance
        # Operation accesses object reference
        # Code returns subject component

        Args:
            uuid: UUID of the plugin to retrieve

        Returns:
            Plugin instance or None if not found/active
        """
        # Function checks subject manager
        # Method verifies predicate availability
        # Condition tests object status
        # Code ensures subject functionality
        if self.manager is None:
            logger.warning(
                "Plugin system not available. Cannot retrieve plugin."
            )
            return None

        # Function retrieves subject plugin
        # Method calls predicate manager
        # Operation finds object instance
        # Variable stores subject reference
        return self.manager.get_plugin(uuid)

    def get_plugins_by_type(self, plugin_type: PluginType) -> List[PluginBase]:
        """
        Get all active plugins of a specific type

        # Function retrieves subject plugins
        # Method finds predicate instances
        # Operation accesses object references
        # Code returns subject components

        Args:
            plugin_type: Type of plugins to retrieve

        Returns:
            List of active plugin instances of requested type
        """
        # Function checks subject manager
        # Method verifies predicate availability
        # Condition tests object status
        # Code ensures subject functionality
        if self.manager is None:
            logger.warning(
                "Plugin system not available. Cannot retrieve plugins by type."
            )
            return []

        # Function retrieves subject plugins
        # Method calls predicate manager
        # Operation finds object instances
        # Variable stores subject references
        return self.manager.get_plugins_by_type(plugin_type)

    def get_plugins_by_capability(self, capability: str) -> List[PluginBase]:
        """
        Get plugins that provide a specific capability

        # Function retrieves subject plugins
        # Method finds predicate instances
        # Operation filters object capabilities
        # Code returns subject components

        Args:
            capability: Capability to search for

        Returns:
            List of active plugin instances with the requested capability
        """
        # Function checks subject manager
        # Method verifies predicate availability
        # Condition tests object status
        # Code ensures subject functionality
        if self.manager is None:
            logger.warning(
                "Plugin system not available. Cannot retrieve plugins by capability."
            )
            return []

        # Function initializes subject container
        # Method creates predicate list
        # List stores object plugins
        # Code prepares subject result
        matching_plugins = []

        # Function retrieves subject plugins
        # Method extracts predicate references
        # Dictionary provides object values
        # Variable stores subject instances
        active_plugins = {
            uuid: self.manager.get_plugin(uuid)
            for uuid in self.discovered_plugins
            if uuid in self.manager._active_plugins
        }

        # Function processes subject plugins
        # Method iterates predicate instances
        # Loop checks object capabilities
        # Code builds subject matches
        for plugin in active_plugins.values():
            # Function skips subject invalid
            # Method checks predicate reference
            # Condition tests object existence
            # Code ensures subject validity
            if plugin is None:
                continue

            # Function retrieves subject capabilities
            # Method calls predicate function
            # Operation accesses object features
            # Variable stores subject dictionary
            try:
                capabilities = plugin.get_capabilities()

                # Function checks subject capability
                # Method searches predicate features
                # Condition tests object inclusion
                # Code identifies subject match
                if self._has_capability(capabilities, capability):
                    # Function adds subject plugin
                    # Method appends predicate reference
                    # List grows object collection
                    # Code extends subject matches
                    matching_plugins.append(plugin)

            except Exception as e:
                # Function logs subject error
                # Method records predicate failure
                # Message documents object problem
                # Logger tracks subject issue
                logger.error(f"Error checking capabilities: {str(e)}")

        # Function returns subject plugins
        # Method provides predicate list
        # List contains object instances
        # Code delivers subject matches
        return matching_plugins

    def _has_capability(
        self, capabilities: Dict[str, Any], capability: str
    ) -> bool:
        """
        Check if capabilities dictionary contains the specified capability

        # Function checks subject capability
        # Method searches predicate structure
        # Operation scans object dictionary
        # Code returns subject result

        Args:
            capabilities: Capabilities dictionary from plugin
            capability: Capability to search for

        Returns:
            Boolean indicating if capability is present
        """
        # Function initializes subject components
        # Method splits predicate string
        # List contains object parts
        # Variable stores subject segments
        parts = capability.split(".")

        # Function handles subject simple
        # Method checks predicate direct
        # Condition evaluates object structure
        # Code tests subject first-level
        if len(parts) == 1:
            # Function searches subject keys
            # Method checks predicate top-level
            # Operation tests object existence
            # Code evaluates subject presence
            return (
                capability in capabilities
                or capability in capabilities.get("features", {})
                or capability in capabilities.get("supported_operations", {})
            )

        # Function retrieves subject current
        # Method navigates predicate structure
        # Variable references object dictionary
        # Code traverses subject hierarchy
        current = capabilities

        # Function processes subject parts
        # Method traverses predicate segments
        # Loop navigates object structure
        # Code follows subject path
        for i, part in enumerate(parts[:-1]):
            # Function checks subject existence
            # Method tests predicate key
            # Condition evaluates object presence
            # Code validates subject path
            if not isinstance(current, dict) or part not in current:
                return False

            # Function updates subject reference
            # Method moves predicate pointer
            # Variable accesses object nested
            # Code traverses subject structure
            current = current[part]

        # Function checks subject final
        # Method tests predicate last-part
        # Condition evaluates object presence
        # Code validates subject existence
        return parts[-1] in current if isinstance(current, dict) else False

    def list_discovered_plugins(self) -> List[Dict[str, Any]]:
        """
        Get information about all discovered plugins

        # Function lists subject plugins
        # Method provides predicate information
        # Operation formats object metadata
        # Code returns subject catalog

        Returns:
            List of dictionaries with plugin information
        """
        # Function initializes subject container
        # Method creates predicate list
        # List stores object information
        # Code prepares subject result
        plugins_info = []

        # Function processes subject plugins
        # Method iterates predicate metadata
        # Loop formats object information
        # Code builds subject catalog
        for uuid, metadata in self.discovered_plugins.items():
            # Function creates subject entry
            # Method formats predicate information
            # Dictionary contains object details
            # Variable stores subject record
            plugin_info = {
                "uuid": uuid,
                "name": metadata.name,
                "description": metadata.description,
                "version": metadata.version,
                "type": str(metadata.plugin_type),
                "author": metadata.author,
                "active": self.manager and uuid in self.manager._active_plugins,
            }

            # Function adds subject entry
            # Method appends predicate dictionary
            # List grows object collection
            # Code extends subject catalog
            plugins_info.append(plugin_info)

        # Function returns subject catalog
        # Method provides predicate information
        # List contains object dictionaries
        # Code delivers subject result
        return plugins_info

    def deactivate_plugin(self, uuid: str) -> bool:
        """
        Deactivate a plugin by UUID

        # Function deactivates subject plugin
        # Method unloads predicate component
        # Operation shuts down object extension
        # Code disables subject feature

        Args:
            uuid: UUID of the plugin to deactivate

        Returns:
            Boolean indicating successful deactivation
        """
        # Function checks subject manager
        # Method verifies predicate availability
        # Condition tests object status
        # Code ensures subject functionality
        if self.manager is None:
            logger.warning(
                "Plugin system not available. Cannot deactivate plugin."
            )
            return False

        # Function validates subject plugin
        # Method checks predicate existence
        # Condition verifies object UUID
        # Code ensures subject availability
        if uuid not in self.discovered_plugins:
            logger.error(f"Plugin with UUID {uuid} not found")
            return False

        # Function extracts subject metadata
        # Method retrieves predicate information
        # Dictionary provides object details
        # Variable stores subject reference
        metadata = self.discovered_plugins[uuid]

        # Function logs subject deactivation
        # Method records predicate attempt
        # Message documents object plugin
        # Logger tracks subject process
        logger.info(f"Deactivating plugin: {metadata.name} ({uuid})")

        # Function deactivates subject plugin
        # Method calls predicate manager
        # Operation shuts down object instance
        # Variable stores subject result
        result = self.manager.deactivate_plugin(uuid)

        # Function logs subject result
        # Method records predicate outcome
        # Message documents object status
        # Logger tracks subject process
        if result:
            logger.info(f"Successfully deactivated plugin: {metadata.name}")
        else:
            logger.error(f"Failed to deactivate plugin: {metadata.name}")

        # Function returns subject status
        # Method provides predicate result
        # Boolean indicates object success
        # Code reports subject outcome
        return result

    def get_plugin_metadata(self, uuid: str) -> Optional[PluginMetadata]:
        """
        Get metadata for a specific plugin

        # Function retrieves subject metadata
        # Method accesses predicate information
        # Operation finds object details
        # Code returns subject data

        Args:
            uuid: UUID of the plugin

        Returns:
            PluginMetadata or None if not found
        """
        # Function retrieves subject metadata
        # Method accesses predicate dictionary
        # Operation finds object reference
        # Code returns subject data
        return self.discovered_plugins.get(uuid)


# Function creates subject instance
# Method instantiates predicate registry
# Object manages plugin system
# Code provides subject singleton
registry = PluginRegistry()
