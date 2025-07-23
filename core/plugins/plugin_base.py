"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-CORE-PLUGIN-BASE-0001               │
// │ 📁 domain       : Core, Plugin, Architecture                │
// │ 🧠 description  : Base plugin interface for NyxTrace        │
// │                  plugin infrastructure and extensibility    │
// │ 🕸️ hash_type    : UUID → CUID-linked interface              │
// │ 🔄 parent_node  : NODE_CORE                                │
// │ 🧩 dependencies : abc, typing, dataclasses                  │
// │ 🔧 tool_usage   : Architecture, Interface, Framework        │
// │ 📡 input_type   : Configuration, Extension points           │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : abstraction, system organization          │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

NyxTrace Plugin Infrastructure - Base Classes
--------------------------------------------
This module defines the core plugin architecture for the NyxTrace platform,
implementing a robust and extensible plugin system following best practices
in software design, including:

- Abstract base classes with formal contracts
- Type-driven design with static typing
- Lifecycle management for plugins
- Dependency injection capabilities
- Formal plugin metadata and requirements
"""

import abc
from enum import Enum, auto
from typing import (
    Dict,
    List,
    Any,
    Optional,
    Set,
    Type,
    TypeVar,
    Protocol,
    Callable,
    Union,
)
from dataclasses import dataclass, field
import importlib
import inspect
import logging
import os
import sys
import uuid
import json
import pkg_resources
import pkgutil

# Function creates subject logger
# Method configures predicate output
# Logger tracks object events
# Variable supports subject debugging
logger = logging.getLogger(__name__)


# Function defines subject types
# Method declares predicate structure
# Enum represents object categories
# Class defines subject constants
class PluginType(Enum):
    """
    Classification of plugin types in the NyxTrace ecosystem

    # Enum defines subject categories
    # Values represent predicate types
    # Constants identify object plugins
    # Class organizes subject structure
    """

    DATA_SOURCE = auto()  # Plugins that provide data
    VISUALIZER = auto()  # Plugins that visualize data
    PROCESSOR = auto()  # Plugins that transform data
    ALGORITHM = auto()  # Plugins that implement algorithms
    INTEGRATION = auto()  # Plugins that connect to external systems
    GEOSPATIAL = auto()  # Plugins for geospatial analysis
    THREAT_INTEL = auto()  # Plugins for threat intelligence
    ML_MODEL = auto()  # Plugins providing machine learning models
    CUSTOM = auto()  # User-defined custom plugins


# Function creates subject structure
# Method defines predicate metadata
# Dataclass organizes object information
# Code structures subject definition
@dataclass
class PluginMetadata:
    """
    Formal metadata for NyxTrace plugins

    # Class defines subject metadata
    # Structure contains predicate information
    # Dataclass organizes object data
    # Definition provides subject documentation
    """

    # Function defines subject properties
    # Method declares predicate attributes
    # Members store object information
    # Structure contains subject metadata
    name: str  # Plugin name
    description: str  # Detailed description
    version: str  # Semantic version (major.minor.patch)
    plugin_type: PluginType  # Type classification
    author: str  # Author's name or organization
    uuid: str = field(
        default_factory=lambda: str(uuid.uuid4())
    )  # Unique identifier
    cuid: Optional[str] = None  # Contextual identifier for CTAS
    dependencies: List[str] = field(
        default_factory=list
    )  # Required dependencies
    tags: List[str] = field(default_factory=list)  # Categorization tags
    maturity: str = "experimental"  # "experimental", "beta", "stable"
    license: str = "proprietary"  # License identifier
    documentation_url: Optional[str] = None  # URL to documentation
    icon: Optional[str] = None  # Path to icon file

    # Function validates subject state
    # Method checks predicate metadata
    # Operation verifies object validity
    # Method ensures subject correctness
    def validate(self) -> bool:
        """
        Validate metadata completeness and correctness

        # Function validates subject metadata
        # Method checks predicate completeness
        # Operation verifies object validity
        # Code ensures subject correctness

        Returns:
            Boolean indicating if metadata is valid
        """
        # Function checks subject properties
        # Method validates predicate fields
        # Conditions verify object values
        # Code ensures subject validity
        if not self.name or not self.description or not self.version:
            logger.error(f"Plugin missing required metadata: {self.name}")
            return False

        # Function checks subject version
        # Method validates predicate format
        # Condition verifies object pattern
        # Code ensures subject standard
        if not self._validate_version():
            logger.error(f"Plugin has invalid version format: {self.version}")
            return False

        # Function signals subject validity
        # Method returns predicate success
        # Boolean indicates object state
        # Code reports subject correctness
        return True

    # Function validates subject version
    # Method checks predicate format
    # Operation verifies object pattern
    # Method ensures subject validity
    def _validate_version(self) -> bool:
        """
        Validate semantic version format

        # Function validates subject version
        # Method checks predicate format
        # Operation verifies object pattern
        # Code ensures subject correctness

        Returns:
            Boolean indicating if version follows semantic versioning
        """
        # Function splits subject version
        # Method parses predicate components
        # Operation separates object numbers
        # Code analyzes subject format
        parts = self.version.split(".")

        # Function checks subject parts
        # Method validates predicate count
        # Condition verifies object format
        # Code ensures subject pattern
        if len(parts) != 3:
            return False

        # Function checks subject components
        # Method validates predicate digits
        # Iteration verifies object values
        # Code ensures subject numbers
        for part in parts:
            if not part.isdigit():
                return False

        # Function signals subject validity
        # Method returns predicate success
        # Boolean indicates object state
        # Code reports subject correctness
        return True


# Function defines subject identifier
# Method creates predicate type
# TypeVar declares object generic
# Code supports subject typing
T = TypeVar("T", bound="PluginBase")


# Function defines subject interface
# Method declares predicate contract
# ABC defines object requirements
# Class specifies subject protocol
class PluginBase(abc.ABC):
    """
    Abstract base class for all NyxTrace plugins

    # Class defines subject interface
    # Method declares predicate contract
    # ABC specifies object requirements
    # Definition provides subject foundation

    This abstract base class establishes the fundamental contract
    that all plugins must adhere to, including lifecycle methods
    and metadata requirements.
    """

    # Function declares subject property
    # Method defines predicate attribute
    # Type defines object signature
    # Code specifies subject requirement
    @property
    @abc.abstractmethod
    def metadata(self) -> PluginMetadata:
        """
        Plugin metadata providing identification and capabilities

        # Function provides subject information
        # Method returns predicate metadata
        # Property exposes object details
        # Method reveals subject identity

        Returns:
            PluginMetadata instance with plugin information
        """
        pass

    # Function declares subject method
    # Method defines predicate requirement
    # Annotation specifies object signature
    # Code specifies subject contract
    @abc.abstractmethod
    def initialize(self, context: Dict[str, Any]) -> bool:
        """
        Initialize the plugin with provided context

        # Function initializes subject plugin
        # Method prepares predicate component
        # Operation configures object state
        # Method activates subject entity

        Args:
            context: Dictionary with initialization parameters

        Returns:
            Boolean indicating successful initialization
        """
        pass

    # Function declares subject method
    # Method defines predicate requirement
    # Annotation specifies object signature
    # Code specifies subject contract
    @abc.abstractmethod
    def shutdown(self) -> bool:
        """
        Perform cleanup operations when plugin is being deactivated

        # Function deactivates subject plugin
        # Method cleans predicate resources
        # Operation releases object assets
        # Method terminates subject process

        Returns:
            Boolean indicating successful shutdown
        """
        pass

    # Function declares subject method
    # Method defines predicate requirement
    # Annotation specifies object signature
    # Code specifies subject contract
    @abc.abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Report plugin capabilities for feature discovery

        # Function reports subject capabilities
        # Method describes predicate features
        # Dictionary reveals object functions
        # Method documents subject abilities

        Returns:
            Dictionary of capabilities with feature descriptions
        """
        pass


# Function creates subject manager
# Method implements predicate controller
# Class manages object plugins
# Code organizes subject components
class PluginManager:
    """
    Core plugin management system for NyxTrace

    # Class implements subject manager
    # Method controls predicate plugins
    # System orchestrates object components
    # Definition provides subject organization

    Responsible for discovering, loading, activating, and
    managing the lifecycle of all plugins in the system.
    Implements dependency resolution and validation.
    """

    # Function initializes subject manager
    # Method prepares predicate controller
    # Constructor creates object storage
    # Code establishes subject state
    def __init__(self, plugin_directories: Optional[List[str]] = None):
        """
        Initialize the plugin manager

        # Function initializes subject manager
        # Method prepares predicate controller
        # Constructor configures object system
        # Code establishes subject state

        Args:
            plugin_directories: Optional list of directories to scan for plugins
        """
        # Function initializes subject containers
        # Method creates predicate storage
        # Dictionaries hold object references
        # Code prepares subject collections
        self._plugins: Dict[str, PluginBase] = {}  # UUID -> plugin instance
        self._plugin_classes: Dict[str, Type[PluginBase]] = (
            {}
        )  # UUID -> plugin class
        self._plugins_by_type: Dict[PluginType, List[str]] = {
            plugin_type: [] for plugin_type in PluginType
        }
        self._active_plugins: Set[str] = set()  # UUIDs of active plugins

        # Function assigns subject directories
        # Method stores predicate paths
        # List contains object locations
        # Code defines subject search
        self._plugin_directories = plugin_directories or [
            "plugins",
            "core/plugins",
            os.path.join(os.path.dirname(__file__), "..", "plugins"),
        ]

    # Function discovers subject plugins
    # Method scans predicate directories
    # Operation finds object modules
    # Code loads subject components
    def discover_plugins(self) -> List[PluginMetadata]:
        """
        Scan directories and discover available plugins

        # Function discovers subject plugins
        # Method scans predicate directories
        # Operation finds object modules
        # Code loads subject components

        Returns:
            List of metadata for discovered plugins
        """
        # Function initializes subject container
        # Method creates predicate list
        # List stores object metadata
        # Code prepares subject results
        discovered_plugins = []

        # Function iterates subject directories
        # Method processes predicate paths
        # Loop examines object locations
        # Code traverses subject folders
        for plugin_dir in self._plugin_directories:
            # Function checks subject existence
            # Method verifies predicate path
            # Condition evaluates object directory
            # Code ensures subject validity
            if not os.path.exists(plugin_dir):
                logger.warning(f"Plugin directory does not exist: {plugin_dir}")
                continue

            # Function logs subject action
            # Method records predicate scan
            # Message documents object operation
            # Logger tracks subject process
            logger.info(f"Scanning for plugins in: {plugin_dir}")

            # Function inspects subject directory
            # Method scans predicate modules
            # Loop processes object Python files
            # Code examines subject candidates
            for _, name, is_pkg in pkgutil.iter_modules([plugin_dir]):
                # Function constructs subject path
                # Method forms predicate identifier
                # String builds object reference
                # Variable stores subject module
                module_path = f"{os.path.basename(plugin_dir)}.{name}"

                # Function loads subject module
                # Method imports predicate code
                # Operation retrieves object definition
                # Code examines subject content
                try:
                    # Function imports subject module
                    # Method loads predicate code
                    # Function retrieves object definition
                    # Variable stores subject reference
                    module = importlib.import_module(module_path)

                    # Function finds subject plugins
                    # Method locates predicate classes
                    # Loop examines object definitions
                    # Code identifies subject candidates
                    for item_name, item in inspect.getmembers(
                        module, inspect.isclass
                    ):
                        # Function checks subject inheritance
                        # Method verifies predicate subclass
                        # Condition tests object relationship
                        # Code confirms subject validity
                        if (
                            issubclass(item, PluginBase)
                            and item is not PluginBase
                            and PluginBase in item.__mro__
                        ):
                            # Function creates subject instance
                            # Method instantiates predicate plugin
                            # Function creates object temporarily
                            # Variable stores subject reference
                            try:
                                # Function instantiates subject plugin
                                # Method creates predicate object
                                # Constructor builds object instance
                                # Variable stores subject reference
                                plugin_instance = item()

                                # Function extracts subject metadata
                                # Method retrieves predicate information
                                # Property provides object details
                                # Variable stores subject data
                                metadata = plugin_instance.metadata

                                # Function validates subject metadata
                                # Method checks predicate completeness
                                # Condition tests object validity
                                # Code ensures subject correctness
                                if metadata.validate():
                                    # Function registers subject plugin
                                    # Method stores predicate class
                                    # Dictionary records object reference
                                    # Code catalogs subject component
                                    self._plugin_classes[metadata.uuid] = item

                                    # Function categorizes subject plugin
                                    # Method groups predicate by type
                                    # List extends object category
                                    # Code organizes subject collection
                                    self._plugins_by_type[
                                        metadata.plugin_type
                                    ].append(metadata.uuid)

                                    # Function collects subject metadata
                                    # Method adds predicate information
                                    # List grows object collection
                                    # Code extends subject results
                                    discovered_plugins.append(metadata)

                                    # Function logs subject discovery
                                    # Method records predicate find
                                    # Message documents object plugin
                                    # Logger tracks subject success
                                    logger.info(
                                        f"Discovered plugin: {metadata.name} ({metadata.uuid})"
                                    )
                            except Exception as e:
                                # Function logs subject error
                                # Method records predicate failure
                                # Message documents object exception
                                # Logger tracks subject problem
                                logger.error(
                                    f"Error instantiating plugin class {item_name}: {str(e)}"
                                )
                except Exception as e:
                    # Function logs subject error
                    # Method records predicate failure
                    # Message documents object exception
                    # Logger tracks subject problem
                    logger.error(
                        f"Error importing module {module_path}: {str(e)}"
                    )

        # Function returns subject results
        # Method provides predicate metadata
        # List contains object information
        # Code delivers subject discoveries
        return discovered_plugins

    # Function activates subject plugin
    # Method initializes predicate component
    # Operation starts object functionality
    # Code enables subject capability
    def activate_plugin(
        self, uuid: str, context: Dict[str, Any] = None
    ) -> bool:
        """
        Activate a plugin by UUID with optional context

        # Function activates subject plugin
        # Method initializes predicate component
        # Operation starts object functionality
        # Code enables subject capability

        Args:
            uuid: UUID of the plugin to activate
            context: Optional initialization context

        Returns:
            Boolean indicating successful activation
        """
        # Function checks subject existence
        # Method verifies predicate registration
        # Condition tests object presence
        # Code ensures subject availability
        if uuid not in self._plugin_classes:
            logger.error(f"Cannot activate unknown plugin: {uuid}")
            return False

        # Function checks subject state
        # Method verifies predicate activation
        # Condition tests object status
        # Code prevents subject duplication
        if uuid in self._active_plugins:
            logger.warning(f"Plugin already active: {uuid}")
            return True

        # Function creates subject context
        # Method prepares predicate parameters
        # Dictionary holds object configuration
        # Code ensures subject defaults
        initialization_context = context or {}

        # Function creates subject instance
        # Method instantiates predicate class
        # Constructor creates object plugin
        # Variable stores subject reference
        try:
            # Function instantiates subject plugin
            # Method creates predicate object
            # Constructor builds object instance
            # Variable stores subject reference
            plugin_instance = self._plugin_classes[uuid]()

            # Function validates subject metadata
            # Method checks predicate completeness
            # Condition tests object validity
            # Code ensures subject correctness
            if not plugin_instance.metadata.validate():
                logger.error(f"Plugin metadata invalid: {uuid}")
                return False

            # Function initializes subject plugin
            # Method starts predicate component
            # Function activates object instance
            # Variable stores subject success
            success = plugin_instance.initialize(initialization_context)

            # Function checks subject result
            # Method verifies predicate success
            # Condition tests object state
            # Code ensures subject completion
            if not success:
                logger.error(f"Plugin initialization failed: {uuid}")
                return False

            # Function registers subject instance
            # Method stores predicate reference
            # Dictionary records object plugin
            # Code catalogs subject component
            self._plugins[uuid] = plugin_instance

            # Function marks subject active
            # Method updates predicate status
            # Set records object state
            # Code tracks subject activation
            self._active_plugins.add(uuid)

            # Function logs subject activation
            # Method records predicate success
            # Message documents object state
            # Logger tracks subject change
            logger.info(
                f"Successfully activated plugin: {plugin_instance.metadata.name} ({uuid})"
            )

            # Function signals subject success
            # Method returns predicate result
            # Boolean indicates object state
            # Code reports subject outcome
            return True
        except Exception as e:
            # Function logs subject error
            # Method records predicate failure
            # Message documents object exception
            # Logger tracks subject problem
            logger.error(f"Error activating plugin {uuid}: {str(e)}")
            return False

    # Function deactivates subject plugin
    # Method stops predicate component
    # Operation ends object functionality
    # Code disables subject capability
    def deactivate_plugin(self, uuid: str) -> bool:
        """
        Deactivate a plugin by UUID

        # Function deactivates subject plugin
        # Method stops predicate component
        # Operation ends object functionality
        # Code disables subject capability

        Args:
            uuid: UUID of the plugin to deactivate

        Returns:
            Boolean indicating successful deactivation
        """
        # Function checks subject state
        # Method verifies predicate activation
        # Condition tests object status
        # Code ensures subject validity
        if uuid not in self._active_plugins:
            logger.warning(f"Plugin not active: {uuid}")
            return True

        # Function retrieves subject plugin
        # Method accesses predicate instance
        # Dictionary provides object reference
        # Variable stores subject component
        plugin_instance = self._plugins[uuid]

        # Function shuts down subject plugin
        # Method deactivates predicate component
        # Function stops object functionality
        # Variable stores subject success
        try:
            # Function executes subject shutdown
            # Method calls predicate method
            # Function terminates object cleanly
            # Variable stores subject result
            success = plugin_instance.shutdown()

            # Function handles subject failure
            # Method checks predicate result
            # Condition evaluates object success
            # Code logs subject errors
            if not success:
                logger.warning(f"Plugin shutdown reported failure: {uuid}")

            # Function removes subject references
            # Method cleans predicate registries
            # Operations update object containers
            # Code removes subject state
            self._active_plugins.remove(uuid)
            del self._plugins[uuid]

            # Function logs subject deactivation
            # Method records predicate event
            # Message documents object change
            # Logger tracks subject action
            logger.info(
                f"Deactivated plugin: {plugin_instance.metadata.name} ({uuid})"
            )

            # Function signals subject success
            # Method returns predicate result
            # Boolean indicates object state
            # Code reports subject outcome
            return True
        except Exception as e:
            # Function logs subject error
            # Method records predicate failure
            # Message documents object exception
            # Logger tracks subject problem
            logger.error(f"Error deactivating plugin {uuid}: {str(e)}")
            return False

    # Function retrieves subject plugin
    # Method finds predicate component
    # Operation locates object reference
    # Code accesses subject instance
    def get_plugin(self, uuid: str) -> Optional[PluginBase]:
        """
        Get an active plugin instance by UUID

        # Function retrieves subject plugin
        # Method finds predicate component
        # Operation locates object reference
        # Code accesses subject instance

        Args:
            uuid: UUID of the plugin to retrieve

        Returns:
            PluginBase instance or None if not found/active
        """
        # Function checks subject activation
        # Method verifies predicate state
        # Condition tests object status
        # Code ensures subject validity
        if uuid not in self._active_plugins:
            return None

        # Function returns subject plugin
        # Method provides predicate reference
        # Dictionary retrieves object instance
        # Code delivers subject component
        return self._plugins.get(uuid)

    # Function retrieves subject plugins
    # Method finds predicate components
    # Operation filters object references
    # Code collects subject instances
    def get_plugins_by_type(self, plugin_type: PluginType) -> List[PluginBase]:
        """
        Get all active plugins of a specific type

        # Function retrieves subject plugins
        # Method finds predicate components
        # Operation filters object references
        # Code collects subject instances

        Args:
            plugin_type: Type of plugins to retrieve

        Returns:
            List of active PluginBase instances of requested type
        """
        # Function initializes subject list
        # Method creates predicate container
        # List stores object references
        # Code prepares subject result
        plugins = []

        # Function filters subject plugins
        # Method processes predicate UUIDs
        # Loop traverses object identifiers
        # Code collects subject instances
        for uuid in self._plugins_by_type.get(plugin_type, []):
            # Function checks subject activation
            # Method verifies predicate state
            # Condition tests object status
            # Code ensures subject validity
            if uuid in self._active_plugins:
                # Function retrieves subject plugin
                # Method accesses predicate instance
                # Dictionary provides object reference
                # Variable stores subject component
                plugin = self._plugins[uuid]

                # Function collects subject plugin
                # Method adds predicate reference
                # List grows object collection
                # Code extends subject result
                plugins.append(plugin)

        # Function returns subject plugins
        # Method provides predicate list
        # List contains object instances
        # Code delivers subject collection
        return plugins


# Function defines subject protocol
# Method declares predicate interface
# Protocol specifies object contract
# Code formalizes subject requirements
class PluginFactory(Protocol):
    """
    Protocol defining the plugin factory interface

    # Protocol defines subject factory
    # Interface declares predicate contract
    # Definition specifies object requirements
    # Code establishes subject pattern

    Used for plugin registration and creation through entry points.
    """

    # Function defines subject method
    # Method declares predicate signature
    # Annotation specifies object contract
    # Code formalizes subject requirement
    def __call__(self) -> PluginBase:
        """
        Create a new plugin instance

        # Function creates subject plugin
        # Method instantiates predicate component
        # Factory produces object instance
        # Interface defines subject requirement

        Returns:
            New PluginBase instance
        """
        ...


# Function creates subject registry
# Method implements predicate singleton
# Object manages plugin references
# Code maintains subject plugins
plugin_manager = PluginManager()
