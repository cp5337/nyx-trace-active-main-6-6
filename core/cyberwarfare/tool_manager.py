"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-CORE-CYBERWARFARE-TOOLMGR-0001      │
// │ 📁 domain       : Cyberwarfare, Tools                      │
// │ 🧠 description  : Offensive tools management               │
// │                  Tool configuration and execution          │
// │ 🕸️ hash_type    : UUID → CUID-linked module                │
// │ 🔄 parent_node  : NODE_CYBERWARFARE                       │
// │ 🧩 dependencies : dataclasses, enum, logging               │
// │ 🔧 tool_usage   : Management, Execution                    │
// │ 📡 input_type   : Tool configurations, command templates    │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : tool management, execution planning      │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

Offensive Tools Manager
--------------------
This module provides structured management of cyberwarfare tools,
supporting configuration, categorization, and execution with
templated command generation.

Designed for future Rust compatibility with clear interfaces and types.
"""

import os
import json
import logging
import re
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import time
from datetime import datetime

from .kali_integrator import ToolCategory, execute_command, CommandResult

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class CommandTemplate:
    """
    Template for generating tool commands

    # Class defines subject template
    # Method structures predicate format
    # Dataclass organizes object fields
    """

    name: str
    description: str
    template: str
    parameters: Dict[str, str] = field(default_factory=dict)
    example: str = ""
    category: str = "GENERAL"

    def format(self, **kwargs) -> str:
        """
        Format the template with provided parameter values

        # Function formats subject template
        # Method substitutes predicate values
        # Operation generates object command

        Args:
            **kwargs: Parameter values for template substitution

        Returns:
            Formatted command string
        """
        try:
            # Function formats subject template
            # Method substitutes predicate values
            # Operation generates object command
            return self.template.format(**kwargs)
        except KeyError as e:
            # Function raises subject error
            # Method signals predicate missing
            # Exception reports object parameter
            missing_param = str(e).strip("'")
            raise ValueError(f"Missing required parameter: {missing_param}")
        except Exception as e:
            # Function raises subject error
            # Method signals predicate failure
            # Exception reports object problem
            raise ValueError(f"Failed to format command template: {str(e)}")


@dataclass
class CyberTool:
    """
    Representation of a cyberwarfare tool

    # Class represents subject tool
    # Method structures predicate information
    # Dataclass organizes object fields
    """

    name: str
    path: str
    category: str
    description: str = ""
    version: str = ""
    templates: List[CommandTemplate] = field(default_factory=list)
    installed: bool = True
    man_page: bool = False
    capabilities: Set[str] = field(default_factory=set)
    tags: List[str] = field(default_factory=list)
    website: str = ""
    author: str = ""

    def is_available(self) -> bool:
        """
        Check if the tool is available for use

        # Function checks subject availability
        # Method verifies predicate status
        # Operation tests object installed

        Returns:
            True if the tool is installed and available
        """
        # Function verifies subject installation
        # Method checks predicate filesystem
        # Operation tests object existence
        if not self.installed:
            return False

        # Function checks subject path
        # Method verifies predicate file
        # Operation tests object existence
        return os.path.exists(self.path) if self.path else False

    def get_help(self) -> str:
        """
        Get help information for the tool

        # Function gets subject help
        # Method retrieves predicate documentation
        # Operation obtains object information

        Returns:
            Help text for the tool
        """
        # Function tries subject flags
        # Method attempts predicate options
        # List stores object alternatives
        help_flags = ["--help", "-h", "-help", "help"]

        # Function initializes subject result
        # Method prepares predicate variable
        # Variable stores object text
        help_text = ""

        # Function tries subject options
        # Method attempts predicate flags
        # Loop tests object alternatives
        for flag in help_flags:
            # Function executes subject command
            # Method runs predicate help
            # Operation tries object flag
            result = execute_command(f"{self.name} {flag}", timeout=5)

            # Function checks subject success
            # Method verifies predicate output
            # Condition tests object result
            if result.return_code == 0 and result.stdout:
                # Function sets subject text
                # Method stores predicate help
                # Assignment saves object output
                help_text = result.stdout
                break

        # Function handles subject failure
        # Method prepares predicate fallback
        # Condition checks object empty
        if not help_text and self.man_page:
            # Function tries subject man
            # Method attempts predicate manual
            # Function calls object command
            result = execute_command(f"man {self.name} | col -b", timeout=5)

            # Function checks subject success
            # Method verifies predicate output
            # Condition tests object result
            if result.return_code == 0 and result.stdout:
                # Function sets subject text
                # Method stores predicate help
                # Assignment saves object output
                help_text = result.stdout

        # Function returns subject help
        # Method provides predicate text
        # Return delivers object documentation
        return help_text

    def execute(
        self,
        command_args: str = "",
        timeout: int = 60,
        capture_output: bool = True,
    ) -> CommandResult:
        """
        Execute the tool with given arguments

        # Function executes subject tool
        # Method runs predicate command
        # Operation performs object action

        Args:
            command_args: Arguments to pass to the tool
            timeout: Maximum execution time in seconds
            capture_output: Whether to capture and return output

        Returns:
            CommandResult with execution details
        """
        # Function builds subject command
        # Method combines predicate path
        # Variable stores object string
        if self.path:
            command = f"{self.path} {command_args}"
        else:
            command = f"{self.name} {command_args}"

        # Function executes subject command
        # Method runs predicate process
        # Function calls object execution
        return execute_command(command, timeout=timeout)

    def execute_template(
        self, template_name: str, params: Dict[str, Any], timeout: int = 60
    ) -> CommandResult:
        """
        Execute a predefined command template

        # Function executes subject template
        # Method runs predicate command
        # Operation performs object action

        Args:
            template_name: Name of the template to use
            params: Parameters for the template
            timeout: Maximum execution time in seconds

        Returns:
            CommandResult with execution details
        """
        # Function finds subject template
        # Method searches predicate list
        # Variable stores object matching
        template = None

        # Function searches subject list
        # Method iterates predicate templates
        # Loop finds object matching
        for tmpl in self.templates:
            if tmpl.name == template_name:
                template = tmpl
                break

        # Function validates subject template
        # Method checks predicate existence
        # Condition tests object found
        if not template:
            # Function raises subject error
            # Method signals predicate missing
            # Exception reports object template
            raise ValueError(
                f"Template '{template_name}' not found for tool '{self.name}'"
            )

        # Function formats subject command
        # Method applies predicate parameters
        # Variable stores object result
        try:
            # Function formats subject command
            # Method generates predicate string
            # Operation substitutes object values
            command_args = template.format(**params)
        except ValueError as e:
            # Function raises subject error
            # Method signals predicate formatting
            # Exception reports object problem
            raise ValueError(f"Failed to format template: {str(e)}")

        # Function executes subject command
        # Method runs predicate tool
        # Function calls object execution
        return self.execute(command_args, timeout)


class ToolManager:
    """
    Manager for cyberwarfare tools

    # Class manages subject tools
    # Method organizes predicate collection
    # Object coordinates object operations
    """

    def __init__(
        self,
        config_path: str = ".config/cybertools",
        tools_path: str = "/usr/share/kali-tools",
    ):
        """
        Initialize the Tool Manager

        # Function initializes subject manager
        # Method configures predicate settings
        # Constructor sets object paths

        Args:
            config_path: Path to configuration directory
            tools_path: Path to tools directory
        """
        # Function sets subject paths
        # Method stores predicate locations
        # Assignment sets object variables
        self.config_path = config_path
        self.tools_path = tools_path

        # Function ensures subject path
        # Method creates predicate directory
        # Operation makes object config
        os.makedirs(self.config_path, exist_ok=True)

        # Function loads subject tools
        # Method prepares predicate inventory
        # Operation initializes object list
        self.tools: Dict[str, CyberTool] = {}
        self.templates: Dict[str, CommandTemplate] = {}

        # Function loads subject configurations
        # Method reads predicate files
        # Operation initializes object state
        self._load_configurations()

        # Function logs subject initialization
        # Method records predicate startup
        # Message documents object status
        logger.info(f"Initialized ToolManager with {len(self.tools)} tools")

    def _load_configurations(self) -> None:
        """
        Load tool configurations from files

        # Function loads subject configurations
        # Method reads predicate files
        # Operation initializes object state
        """
        # Function defines subject path
        # Method sets predicate location
        # Variable stores object filename
        tools_file = os.path.join(self.config_path, "tools.json")
        templates_file = os.path.join(self.config_path, "templates.json")

        # Function loads subject tools
        # Method checks predicate file
        # Condition verifies object existence
        if os.path.exists(tools_file):
            try:
                # Function loads subject json
                # Method reads predicate file
                # Operation parses object data
                with open(tools_file, "r") as f:
                    tools_data = json.load(f)

                # Function processes subject data
                # Method iterates predicate entries
                # Loop creates object tools
                for tool_data in tools_data:
                    # Function creates subject tool
                    # Method constructs predicate object
                    # Constructor builds object instance
                    tool = CyberTool(
                        name=tool_data.get("name", ""),
                        path=tool_data.get("path", ""),
                        category=tool_data.get("category", "MISCELLANEOUS"),
                        description=tool_data.get("description", ""),
                        version=tool_data.get("version", ""),
                        installed=tool_data.get("installed", True),
                        man_page=tool_data.get("man_page", False),
                        website=tool_data.get("website", ""),
                        author=tool_data.get("author", ""),
                    )

                    # Function sets subject capabilities
                    # Method configures predicate set
                    # Assignment populates object field
                    if "capabilities" in tool_data:
                        tool.capabilities = set(tool_data["capabilities"])

                    # Function sets subject tags
                    # Method configures predicate list
                    # Assignment populates object field
                    if "tags" in tool_data:
                        tool.tags = tool_data["tags"]

                    # Function adds subject templates
                    # Method configures predicate list
                    # Assignment populates object field
                    if "templates" in tool_data:
                        # Function processes subject entries
                        # Method iterates predicate data
                        # Loop creates object templates
                        for tmpl_data in tool_data["templates"]:
                            # Function creates subject template
                            # Method constructs predicate object
                            # Constructor builds object instance
                            template = CommandTemplate(
                                name=tmpl_data.get("name", ""),
                                description=tmpl_data.get("description", ""),
                                template=tmpl_data.get("template", ""),
                                parameters=tmpl_data.get("parameters", {}),
                                example=tmpl_data.get("example", ""),
                                category=tmpl_data.get("category", "GENERAL"),
                            )

                            # Function adds subject template
                            # Method appends predicate object
                            # Operation extends object list
                            tool.templates.append(template)

                    # Function adds subject tool
                    # Method stores predicate object
                    # Dictionary adds object entry
                    self.tools[tool.name] = tool

            except Exception as e:
                # Function logs subject error
                # Method records predicate exception
                # Message documents object failure
                logger.error(f"Failed to load tools configuration: {str(e)}")

        # Function loads subject templates
        # Method checks predicate file
        # Condition verifies object existence
        if os.path.exists(templates_file):
            try:
                # Function loads subject json
                # Method reads predicate file
                # Operation parses object data
                with open(templates_file, "r") as f:
                    templates_data = json.load(f)

                # Function processes subject data
                # Method iterates predicate entries
                # Loop creates object templates
                for tmpl_data in templates_data:
                    # Function creates subject template
                    # Method constructs predicate object
                    # Constructor builds object instance
                    template = CommandTemplate(
                        name=tmpl_data.get("name", ""),
                        description=tmpl_data.get("description", ""),
                        template=tmpl_data.get("template", ""),
                        parameters=tmpl_data.get("parameters", {}),
                        example=tmpl_data.get("example", ""),
                        category=tmpl_data.get("category", "GENERAL"),
                    )

                    # Function adds subject template
                    # Method stores predicate object
                    # Dictionary adds object entry
                    self.templates[template.name] = template

            except Exception as e:
                # Function logs subject error
                # Method records predicate exception
                # Message documents object failure
                logger.error(
                    f"Failed to load templates configuration: {str(e)}"
                )

    def save_configurations(self) -> bool:
        """
        Save tool and template configurations to files

        # Function saves subject configurations
        # Method writes predicate files
        # Operation stores object state

        Returns:
            True if save was successful, False otherwise
        """
        try:
            # Function prepares subject tools
            # Method serializes predicate objects
            # List stores object dictionaries
            tools_data = []

            # Function processes subject tools
            # Method iterates predicate collection
            # Loop serializes object instances
            for tool in self.tools.values():
                # Function creates subject dict
                # Method serializes predicate tool
                # Dictionary stores object data
                tool_dict = {
                    "name": tool.name,
                    "path": tool.path,
                    "category": tool.category,
                    "description": tool.description,
                    "version": tool.version,
                    "installed": tool.installed,
                    "man_page": tool.man_page,
                    "capabilities": list(tool.capabilities),
                    "tags": tool.tags,
                    "website": tool.website,
                    "author": tool.author,
                }

                # Function adds subject templates
                # Method serializes predicate list
                # List stores object dictionaries
                templates_data = []

                # Function processes subject templates
                # Method iterates predicate list
                # Loop serializes object instances
                for tmpl in tool.templates:
                    # Function creates subject dict
                    # Method serializes predicate template
                    # Dictionary stores object data
                    tmpl_dict = {
                        "name": tmpl.name,
                        "description": tmpl.description,
                        "template": tmpl.template,
                        "parameters": tmpl.parameters,
                        "example": tmpl.example,
                        "category": tmpl.category,
                    }

                    # Function adds subject template
                    # Method appends predicate dict
                    # Operation extends object list
                    templates_data.append(tmpl_dict)

                # Function adds subject templates
                # Method assigns predicate list
                # Assignment stores object dictionaries
                tool_dict["templates"] = templates_data

                # Function adds subject tool
                # Method appends predicate dict
                # Operation extends object list
                tools_data.append(tool_dict)

            # Function prepares subject templates
            # Method serializes predicate objects
            # List stores object dictionaries
            global_templates_data = []

            # Function processes subject templates
            # Method iterates predicate collection
            # Loop serializes object instances
            for tmpl in self.templates.values():
                # Function creates subject dict
                # Method serializes predicate template
                # Dictionary stores object data
                tmpl_dict = {
                    "name": tmpl.name,
                    "description": tmpl.description,
                    "template": tmpl.template,
                    "parameters": tmpl.parameters,
                    "example": tmpl.example,
                    "category": tmpl.category,
                }

                # Function adds subject template
                # Method appends predicate dict
                # Operation extends object list
                global_templates_data.append(tmpl_dict)

            # Function writes subject tools
            # Method saves predicate json
            # Operation stores object file
            tools_file = os.path.join(self.config_path, "tools.json")
            with open(tools_file, "w") as f:
                json.dump(tools_data, f, indent=2)

            # Function writes subject templates
            # Method saves predicate json
            # Operation stores object file
            templates_file = os.path.join(self.config_path, "templates.json")
            with open(templates_file, "w") as f:
                json.dump(global_templates_data, f, indent=2)

            # Function returns subject success
            # Method indicates predicate completion
            # Return delivers object boolean
            return True

        except Exception as e:
            # Function logs subject error
            # Method records predicate exception
            # Message documents object failure
            logger.error(f"Failed to save configurations: {str(e)}")

            # Function returns subject failure
            # Method indicates predicate error
            # Return delivers object boolean
            return False

    def add_tool(self, tool: CyberTool) -> bool:
        """
        Add or update a tool in the manager

        # Function adds subject tool
        # Method stores predicate object
        # Operation updates object collection

        Args:
            tool: CyberTool instance to add

        Returns:
            True if the tool was added successfully
        """
        try:
            # Function adds subject tool
            # Method stores predicate object
            # Dictionary updates object entry
            self.tools[tool.name] = tool

            # Function logs subject addition
            # Method records predicate action
            # Message documents object updated
            logger.info(f"Added/updated tool: {tool.name}")

            # Function returns subject success
            # Method indicates predicate completion
            # Return delivers object boolean
            return True

        except Exception as e:
            # Function logs subject error
            # Method records predicate exception
            # Message documents object failure
            logger.error(f"Failed to add tool {tool.name}: {str(e)}")

            # Function returns subject failure
            # Method indicates predicate error
            # Return delivers object boolean
            return False

    def remove_tool(self, tool_name: str) -> bool:
        """
        Remove a tool from the manager

        # Function removes subject tool
        # Method deletes predicate object
        # Operation updates object collection

        Args:
            tool_name: Name of the tool to remove

        Returns:
            True if the tool was removed successfully
        """
        try:
            # Function checks subject existence
            # Method verifies predicate presence
            # Condition tests object collection
            if tool_name in self.tools:
                # Function removes subject tool
                # Method deletes predicate object
                # Dictionary removes object entry
                del self.tools[tool_name]

                # Function logs subject removal
                # Method records predicate action
                # Message documents object deleted
                logger.info(f"Removed tool: {tool_name}")

                # Function returns subject success
                # Method indicates predicate completion
                # Return delivers object boolean
                return True
            else:
                # Function logs subject warning
                # Method records predicate issue
                # Message documents object missing
                logger.warning(f"Tool not found for removal: {tool_name}")

                # Function returns subject failure
                # Method indicates predicate error
                # Return delivers object boolean
                return False

        except Exception as e:
            # Function logs subject error
            # Method records predicate exception
            # Message documents object failure
            logger.error(f"Failed to remove tool {tool_name}: {str(e)}")

            # Function returns subject failure
            # Method indicates predicate error
            # Return delivers object boolean
            return False

    def get_tool(self, tool_name: str) -> Optional[CyberTool]:
        """
        Get a tool by name

        # Function gets subject tool
        # Method retrieves predicate object
        # Operation finds object by-name

        Args:
            tool_name: Name of the tool to retrieve

        Returns:
            CyberTool instance or None if not found
        """
        # Function returns subject tool
        # Method retrieves predicate object
        # Dictionary gets object entry
        return self.tools.get(tool_name)

    def get_tools_by_category(self, category: str) -> List[CyberTool]:
        """
        Get all tools in a specific category

        # Function gets subject tools
        # Method filters predicate category
        # Operation finds object matching

        Args:
            category: Category to filter by

        Returns:
            List of CyberTool instances in the category
        """
        # Function filters subject tools
        # Method matches predicate category
        # List stores object filtered
        return [
            tool for tool in self.tools.values() if tool.category == category
        ]

    def get_tools_by_capability(self, capability: str) -> List[CyberTool]:
        """
        Get all tools with a specific capability

        # Function gets subject tools
        # Method filters predicate capability
        # Operation finds object matching

        Args:
            capability: Capability to filter by

        Returns:
            List of CyberTool instances with the capability
        """
        # Function filters subject tools
        # Method matches predicate capability
        # List stores object filtered
        return [
            tool
            for tool in self.tools.values()
            if capability in tool.capabilities
        ]

    def search_tools(self, search_term: str) -> List[CyberTool]:
        """
        Search for tools matching a search term

        # Function searches subject tools
        # Method finds predicate matching
        # Operation filters object collection

        Args:
            search_term: Term to search for

        Returns:
            List of matching CyberTool instances
        """
        # Function checks subject term
        # Method validates predicate input
        # Condition tests object empty
        if not search_term:
            # Function returns subject all
            # Method provides predicate collection
            # Return delivers object list
            return list(self.tools.values())

        # Function lowercases subject term
        # Method normalizes predicate case
        # Variable stores object processed
        term = search_term.lower()

        # Function filters subject tools
        # Method matches predicate term
        # List stores object filtered
        return [
            tool
            for tool in self.tools.values()
            if term in tool.name.lower()
            or term in tool.description.lower()
            or any(term in tag.lower() for tag in tool.tags)
        ]

    def add_template(self, template: CommandTemplate) -> bool:
        """
        Add or update a global command template

        # Function adds subject template
        # Method stores predicate object
        # Operation updates object collection

        Args:
            template: CommandTemplate instance to add

        Returns:
            True if the template was added successfully
        """
        try:
            # Function adds subject template
            # Method stores predicate object
            # Dictionary updates object entry
            self.templates[template.name] = template

            # Function logs subject addition
            # Method records predicate action
            # Message documents object updated
            logger.info(f"Added/updated template: {template.name}")

            # Function returns subject success
            # Method indicates predicate completion
            # Return delivers object boolean
            return True

        except Exception as e:
            # Function logs subject error
            # Method records predicate exception
            # Message documents object failure
            logger.error(f"Failed to add template {template.name}: {str(e)}")

            # Function returns subject failure
            # Method indicates predicate error
            # Return delivers object boolean
            return False

    def get_template(self, template_name: str) -> Optional[CommandTemplate]:
        """
        Get a global template by name

        # Function gets subject template
        # Method retrieves predicate object
        # Operation finds object by-name

        Args:
            template_name: Name of the template to retrieve

        Returns:
            CommandTemplate instance or None if not found
        """
        # Function returns subject template
        # Method retrieves predicate object
        # Dictionary gets object entry
        return self.templates.get(template_name)

    def execute_tool(
        self, tool_name: str, command_args: str = "", timeout: int = 60
    ) -> CommandResult:
        """
        Execute a tool with the given arguments

        # Function executes subject tool
        # Method runs predicate command
        # Operation performs object action

        Args:
            tool_name: Name of the tool to execute
            command_args: Arguments to pass to the tool
            timeout: Maximum execution time in seconds

        Returns:
            CommandResult with execution details
        """
        # Function gets subject tool
        # Method retrieves predicate object
        # Variable stores object instance
        tool = self.get_tool(tool_name)

        # Function validates subject tool
        # Method checks predicate existence
        # Condition tests object found
        if not tool:
            # Function raises subject error
            # Method signals predicate missing
            # Exception reports object tool
            raise ValueError(f"Tool '{tool_name}' not found")

        # Function checks subject availability
        # Method verifies predicate status
        # Condition tests object installed
        if not tool.is_available():
            # Function raises subject error
            # Method signals predicate unavailable
            # Exception reports object status
            raise ValueError(
                f"Tool '{tool_name}' is not available or installed"
            )

        # Function executes subject tool
        # Method runs predicate command
        # Function calls object execution
        return tool.execute(command_args, timeout)

    def execute_template(
        self,
        template_name: str,
        params: Dict[str, Any],
        tool_name: Optional[str] = None,
        timeout: int = 60,
    ) -> CommandResult:
        """
        Execute a command template

        # Function executes subject template
        # Method runs predicate command
        # Operation performs object action

        Args:
            template_name: Name of the template to use
            params: Parameters for the template
            tool_name: Optional tool to execute with (if template is tool-specific)
            timeout: Maximum execution time in seconds

        Returns:
            CommandResult with execution details
        """
        # Function handles subject tool-specific
        # Method checks predicate condition
        # Condition tests object specified
        if tool_name:
            # Function gets subject tool
            # Method retrieves predicate object
            # Variable stores object instance
            tool = self.get_tool(tool_name)

            # Function validates subject tool
            # Method checks predicate existence
            # Condition tests object found
            if not tool:
                # Function raises subject error
                # Method signals predicate missing
                # Exception reports object tool
                raise ValueError(f"Tool '{tool_name}' not found")

            # Function checks subject availability
            # Method verifies predicate status
            # Condition tests object installed
            if not tool.is_available():
                # Function raises subject error
                # Method signals predicate unavailable
                # Exception reports object status
                raise ValueError(
                    f"Tool '{tool_name}' is not available or installed"
                )

            # Function executes subject template
            # Method runs predicate command
            # Function calls object execution
            return tool.execute_template(template_name, params, timeout)

        # Function handles subject global
        # Method processes predicate template
        # Section handles object non-tool
        else:
            # Function gets subject template
            # Method retrieves predicate object
            # Variable stores object instance
            template = self.get_template(template_name)

            # Function validates subject template
            # Method checks predicate existence
            # Condition tests object found
            if not template:
                # Function raises subject error
                # Method signals predicate missing
                # Exception reports object template
                raise ValueError(f"Template '{template_name}' not found")

            # Function formats subject command
            # Method applies predicate parameters
            # Variable stores object result
            try:
                # Function formats subject command
                # Method generates predicate string
                # Operation substitutes object values
                command = template.format(**params)
            except ValueError as e:
                # Function raises subject error
                # Method signals predicate formatting
                # Exception reports object problem
                raise ValueError(f"Failed to format template: {str(e)}")

            # Function executes subject command
            # Method runs predicate process
            # Function calls object execution
            return execute_command(command, timeout)


def get_tool_categories() -> List[str]:
    """
    Get a list of all cyberwarfare tool categories

    # Function gets subject categories
    # Method retrieves predicate list
    # Operation extracts object names

    Returns:
        List of category names
    """
    # Function converts subject enum
    # Method extracts predicate names
    # List stores object strings
    return [category.name for category in ToolCategory]


def get_tools_by_category(
    category: str, manager: ToolManager
) -> List[CyberTool]:
    """
    Get all tools in a specific category

    # Function gets subject tools
    # Method filters predicate category
    # Operation finds object matching

    Args:
        category: Category to filter by
        manager: ToolManager instance

    Returns:
        List of CyberTool instances in the category
    """
    # Function returns subject results
    # Method calls predicate function
    # Function delegates object retrieval
    return manager.get_tools_by_category(category)


def format_command_template(template: str, **kwargs) -> str:
    """
    Format a command template with provided parameter values

    # Function formats subject template
    # Method substitutes predicate values
    # Operation generates object command

    Args:
        template: Command template string
        **kwargs: Parameter values for template substitution

    Returns:
        Formatted command string
    """
    try:
        # Function formats subject template
        # Method substitutes predicate values
        # Operation generates object command
        return template.format(**kwargs)
    except KeyError as e:
        # Function raises subject error
        # Method signals predicate missing
        # Exception reports object parameter
        missing_param = str(e).strip("'")
        raise ValueError(f"Missing required parameter: {missing_param}")
    except Exception as e:
        # Function raises subject error
        # Method signals predicate failure
        # Exception reports object problem
        raise ValueError(f"Failed to format command template: {str(e)}")
