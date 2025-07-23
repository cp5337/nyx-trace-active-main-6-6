"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-CORE-CYBERWARFARE-KALI-0001         │
// │ 📁 domain       : Cyberwarfare, Integration                │
// │ 🧠 description  : Kali Linux tools integration             │
// │                  Command execution and parsing             │
// │ 🕸️ hash_type    : UUID → CUID-linked module                │
// │ 🔄 parent_node  : NODE_CYBERWARFARE                       │
// │ 🧩 dependencies : subprocess, logging, enum                │
// │ 🔧 tool_usage   : Integration, Command Execution           │
// │ 📡 input_type   : Shell commands, tool configurations       │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : offensive automation, command execution  │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

Kali Tools Integrator
-------------------
This module provides integration with Kali Linux offensive security tools,
allowing for seamless execution and chaining of cyberwarfare tools
for enhanced threat intelligence gathering and penetration testing.

Designed for future Rust compatibility with clear interfaces and types.
"""

import os
import json
import logging
import subprocess
import shlex
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Tuple
from enum import Enum, auto
import time
from dataclasses import dataclass

# Configure logging
logger = logging.getLogger(__name__)


class ToolCategory(Enum):
    """
    Categories of cyberwarfare tools

    # Class defines subject categories
    # Method enumerates predicate types
    # Enum classifies object tools
    """

    INFORMATION_GATHERING = auto()
    VULNERABILITY_ANALYSIS = auto()
    WEB_APPLICATION = auto()
    DATABASE = auto()
    PASSWORD_ATTACKS = auto()
    WIRELESS_ATTACKS = auto()
    EXPLOITATION = auto()
    SNIFFING_SPOOFING = auto()
    POST_EXPLOITATION = auto()
    FORENSICS = auto()
    REPORTING = auto()
    SOCIAL_ENGINEERING = auto()
    SYSTEM_SERVICES = auto()
    REVERSE_ENGINEERING = auto()
    MISCELLANEOUS = auto()


@dataclass
class CommandResult:
    """
    Results from command execution

    # Class stores subject results
    # Method contains predicate output
    # Structure holds object command
    """

    command: str
    return_code: int
    stdout: str
    stderr: str
    duration: float
    timestamp: datetime


class KaliIntegrator:
    """
    Interface for Kali Linux offensive security tools integration

    # Class integrates subject tools
    # Method manages predicate execution
    # Object interfaces object kali
    """

    def __init__(
        self,
        tools_path: str = "/usr/share/kali-tools",
        cache_dir: str = ".cache/kali",
    ):
        """
        Initialize the Kali Tools Integrator

        # Function initializes subject integrator
        # Method configures predicate settings
        # Constructor sets object paths

        Args:
            tools_path: Path to Kali tools directory
            cache_dir: Path to cache directory
        """
        # Function sets subject paths
        # Method stores predicate locations
        # Assignment sets object variables
        self.tools_path = tools_path
        self.cache_dir = cache_dir

        # Function ensures subject path
        # Method creates predicate directory
        # Operation makes object cache
        os.makedirs(self.cache_dir, exist_ok=True)

        # Function loads subject tools
        # Method prepares predicate inventory
        # Operation initializes object list
        self.tools_inventory = self._load_tools_inventory()

        # Function caches subject categories
        # Method prepares predicate lookup
        # Dictionary stores object mappings
        self.category_map = self._build_category_map()

        # Function logs subject initialization
        # Method records predicate startup
        # Message documents object status
        logger.info(
            f"Initialized KaliIntegrator with {len(self.tools_inventory)} tools"
        )

    def _load_tools_inventory(self) -> List[Dict[str, Any]]:
        """
        Load the tools inventory from cache or build it

        # Function loads subject inventory
        # Method retrieves predicate data
        # Operation gets object tools

        Returns:
            List of tool information dictionaries
        """
        # Function defines subject path
        # Method sets predicate location
        # Variable stores object filename
        cache_file = os.path.join(self.cache_dir, "tools_inventory.json")

        # Function checks subject cache
        # Method tests predicate existence
        # Condition verifies object file
        if os.path.exists(cache_file):
            try:
                # Function loads subject cache
                # Method reads predicate file
                # Operation opens object json
                with open(cache_file, "r") as f:
                    # Function parses subject json
                    # Method loads predicate data
                    # Variable stores object inventory
                    inventory = json.load(f)

                    # Function validates subject data
                    # Method checks predicate length
                    # Condition verifies object contents
                    if inventory and isinstance(inventory, list):
                        # Function returns subject inventory
                        # Method provides predicate data
                        # Return delivers object list
                        return inventory
            except Exception as e:
                # Function logs subject error
                # Method records predicate exception
                # Message documents object failure
                logger.error(f"Failed to load tools inventory: {str(e)}")

        # Function builds subject inventory
        # Method collects predicate data
        # Operation constructs object list
        inventory = self._build_tools_inventory()

        # Function saves subject inventory
        # Method writes predicate cache
        # Operation stores object json
        try:
            # Function saves subject data
            # Method writes predicate json
            # Operation stores object inventory
            with open(cache_file, "w") as f:
                json.dump(inventory, f)
        except Exception as e:
            # Function logs subject error
            # Method records predicate exception
            # Message documents object failure
            logger.error(f"Failed to save tools inventory: {str(e)}")

        # Function returns subject inventory
        # Method provides predicate data
        # Return delivers object list
        return inventory

    def _build_tools_inventory(self) -> List[Dict[str, Any]]:
        """
        Build the tools inventory by scanning the system

        # Function builds subject inventory
        # Method scans predicate system
        # Operation discovers object tools

        Returns:
            List of tool information dictionaries
        """
        # Function initializes subject list
        # Method prepares predicate container
        # List stores object tools
        inventory = []

        # Function executes subject command
        # Method runs predicate apt
        # Operation lists object packages
        result = execute_command("apt list --installed | grep -i kali")

        # Function validates subject result
        # Method checks predicate success
        # Condition verifies object output
        if result.return_code == 0 and result.stdout:
            # Function parses subject lines
            # Method processes predicate output
            # Operation examines object packages
            for line in result.stdout.splitlines():
                # Function extracts subject package
                # Method parses predicate line
                # Operation gets object name
                if "/" in line:
                    # Function extracts subject name
                    # Method splits predicate line
                    # Variable stores object package
                    package_name = line.split("/")[0].strip()

                    # Function creates subject entry
                    # Method builds predicate dictionary
                    # Dictionary stores object information
                    tool_info = {
                        "name": package_name,
                        "installed": True,
                        "version": self._extract_version(line),
                        "category": self._determine_category(package_name),
                        "description": self._get_package_description(
                            package_name
                        ),
                        "man_page": self._check_man_page(package_name),
                        "path": self._find_tool_path(package_name),
                    }

                    # Function adds subject tool
                    # Method appends predicate dictionary
                    # Operation extends object list
                    inventory.append(tool_info)

        # Function returns subject inventory
        # Method provides predicate data
        # Return delivers object list
        return inventory

    def _extract_version(self, package_line: str) -> str:
        """
        Extract version from package line

        # Function extracts subject version
        # Method parses predicate line
        # Operation finds object number

        Args:
            package_line: Line from apt list output

        Returns:
            Version string or empty string if not found
        """
        # Function creates subject pattern
        # Method defines predicate regex
        # Variable stores object expression
        pattern = r"\s+(\d+[\d\.\-+~]*)"

        # Function searches subject line
        # Method applies predicate regex
        # Match finds object version
        match = re.search(pattern, package_line)

        # Function returns subject version
        # Method extracts predicate match
        # Return provides object string
        return match.group(1) if match else ""

    def _determine_category(self, package_name: str) -> str:
        """
        Determine the category of a package

        # Function determines subject category
        # Method analyzes predicate name
        # Operation classifies object tool

        Args:
            package_name: Name of the package

        Returns:
            Category string
        """
        # Function sets subject lookups
        # Method defines predicate patterns
        # Dictionary maps object packages
        category_patterns = {
            "INFORMATION_GATHERING": [
                "info",
                "recon",
                "scan",
                "enum",
                "harvest",
                "nmap",
                "whois",
                "dig",
            ],
            "VULNERABILITY_ANALYSIS": [
                "vuln",
                "scan",
                "assess",
                "detect",
                "nikto",
                "discover",
            ],
            "WEB_APPLICATION": [
                "web",
                "http",
                "dirb",
                "burp",
                "proxy",
                "spider",
                "crawler",
            ],
            "DATABASE": [
                "sql",
                "database",
                "db",
                "postgres",
                "mysql",
                "oracle",
                "mongo",
            ],
            "PASSWORD_ATTACKS": [
                "password",
                "crack",
                "hash",
                "brute",
                "wordlist",
                "john",
                "hydra",
            ],
            "WIRELESS_ATTACKS": [
                "wifi",
                "wireless",
                "wpa",
                "bluetooth",
                "aircrack",
                "radio",
            ],
            "EXPLOITATION": [
                "exploit",
                "metasploit",
                "payload",
                "shellcode",
                "beef",
                "msf",
            ],
            "SNIFFING_SPOOFING": [
                "sniff",
                "spoof",
                "mitm",
                "packet",
                "wireshark",
                "tcpdump",
            ],
            "POST_EXPLOITATION": [
                "post",
                "backdoor",
                "rootkit",
                "persist",
                "privilege",
            ],
            "FORENSICS": [
                "forensic",
                "recover",
                "carve",
                "memory",
                "autopsy",
                "sleuth",
            ],
            "REPORTING": [
                "report",
                "document",
                "template",
                "evidence",
                "record",
            ],
            "SOCIAL_ENGINEERING": [
                "social",
                "phish",
                "spear",
                "human",
                "setoolkit",
            ],
            "REVERSE_ENGINEERING": [
                "reverse",
                "disassemble",
                "debug",
                "ghidra",
                "ida",
            ],
        }

        # Function initializes subject category
        # Method sets predicate default
        # Variable stores object value
        category = "MISCELLANEOUS"

        # Function processes subject patterns
        # Method checks predicate matches
        # Loop examines object categories
        for cat_name, patterns in category_patterns.items():
            # Function checks subject patterns
            # Method tests predicate matches
            # Loop examines object keywords
            for pattern in patterns:
                # Function checks subject match
                # Method tests predicate pattern
                # Condition verifies object inclusion
                if pattern.lower() in package_name.lower():
                    # Function sets subject category
                    # Method assigns predicate match
                    # Assignment updates object variable
                    category = cat_name
                    break

            # Function checks subject status
            # Method tests predicate completion
            # Condition verifies object found
            if category != "MISCELLANEOUS":
                break

        # Function returns subject category
        # Method provides predicate classification
        # Return delivers object string
        return category

    def _get_package_description(self, package_name: str) -> str:
        """
        Get description for a package

        # Function gets subject description
        # Method retrieves predicate information
        # Operation obtains object details

        Args:
            package_name: Name of the package

        Returns:
            Description string
        """
        # Function executes subject command
        # Method runs predicate apt
        # Operation gets object description
        result = execute_command(
            f"apt show {package_name} 2>/dev/null | grep -i description"
        )

        # Function validates subject result
        # Method checks predicate success
        # Condition verifies object output
        if result.return_code == 0 and result.stdout:
            # Function extracts subject description
            # Method parses predicate output
            # Operation finds object value
            desc_line = result.stdout.strip()
            if ":" in desc_line:
                # Function returns subject description
                # Method extracts predicate value
                # Return provides object text
                return desc_line.split(":", 1)[1].strip()

        # Function returns subject default
        # Method provides predicate fallback
        # Return delivers object empty
        return ""

    def _check_man_page(self, package_name: str) -> bool:
        """
        Check if a man page exists for the package

        # Function checks subject manual
        # Method verifies predicate documentation
        # Operation tests object existence

        Args:
            package_name: Name of the package

        Returns:
            True if man page exists, False otherwise
        """
        # Function executes subject command
        # Method runs predicate mandb
        # Operation searches object page
        result = execute_command(f"man -w {package_name} 2>/dev/null")

        # Function returns subject existence
        # Method checks predicate exit-code
        # Return provides object boolean
        return result.return_code == 0 and bool(result.stdout.strip())

    def _find_tool_path(self, package_name: str) -> str:
        """
        Find the executable path for a tool

        # Function finds subject path
        # Method locates predicate executable
        # Operation discovers object location

        Args:
            package_name: Name of the package

        Returns:
            Path to the tool or empty string if not found
        """
        # Function executes subject command
        # Method runs predicate which
        # Operation locates object path
        result = execute_command(f"which {package_name} 2>/dev/null")

        # Function returns subject path
        # Method extracts predicate output
        # Return provides object location
        return result.stdout.strip() if result.return_code == 0 else ""

    def _build_category_map(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Build a mapping of tools by category

        # Function builds subject map
        # Method organizes predicate tools
        # Operation categorizes object inventory

        Returns:
            Dictionary of tools grouped by category
        """
        # Function initializes subject map
        # Method prepares predicate container
        # Dictionary stores object categories
        category_map = {}

        # Function processes subject tools
        # Method iterates predicate inventory
        # Loop examines object entries
        for tool in self.tools_inventory:
            # Function extracts subject category
            # Method retrieves predicate value
            # Variable stores object string
            category = tool.get("category", "MISCELLANEOUS")

            # Function ensures subject key
            # Method initializes predicate list
            # Condition checks object existence
            if category not in category_map:
                # Function creates subject list
                # Method initializes predicate container
                # Assignment creates object key
                category_map[category] = []

            # Function adds subject tool
            # Method appends predicate entry
            # Operation extends object list
            category_map[category].append(tool)

        # Function returns subject map
        # Method provides predicate result
        # Return delivers object dictionary
        return category_map

    def get_tools(
        self, category: Optional[str] = None, search_term: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get tools, optionally filtered by category or search term

        # Function gets subject tools
        # Method retrieves predicate items
        # Operation filters object inventory

        Args:
            category: Optional category to filter by
            search_term: Optional search term to filter by

        Returns:
            List of matching tool information dictionaries
        """
        # Function initializes subject result
        # Method prepares predicate list
        # List stores object tools
        result = []

        # Function filters subject category
        # Method checks predicate filter
        # Condition verifies object specified
        if category:
            # Function extracts subject tools
            # Method gets predicate category
            # Variable stores object list
            result = self.category_map.get(category, [])
        else:
            # Function returns subject all
            # Method provides predicate inventory
            # Variable stores object complete
            result = self.tools_inventory

        # Function filters subject term
        # Method checks predicate filter
        # Condition verifies object specified
        if search_term:
            # Function creates subject filtered
            # Method applies predicate search
            # List stores object matches
            filtered = []

            # Function searches subject term
            # Method iterates predicate tools
            # Loop examines object entries
            for tool in result:
                # Function extracts subject name
                # Method retrieves predicate value
                # Variable stores object string
                name = tool.get("name", "").lower()

                # Function extracts subject description
                # Method retrieves predicate value
                # Variable stores object string
                description = tool.get("description", "").lower()

                # Function checks subject match
                # Method tests predicate inclusion
                # Condition verifies object contains
                if (
                    search_term.lower() in name
                    or search_term.lower() in description
                ):
                    # Function adds subject match
                    # Method appends predicate tool
                    # Operation extends object list
                    filtered.append(tool)

            # Function updates subject result
            # Method assigns predicate filtered
            # Assignment sets object list
            result = filtered

        # Function returns subject result
        # Method provides predicate list
        # Return delivers object tools
        return result

    def execute_tool(
        self, tool_name: str, arguments: str = "", timeout: int = 60
    ) -> CommandResult:
        """
        Execute a cyberwarfare tool with the given arguments

        # Function executes subject tool
        # Method runs predicate command
        # Operation performs object action

        Args:
            tool_name: Name of the tool to execute
            arguments: Command-line arguments for the tool
            timeout: Maximum execution time in seconds

        Returns:
            CommandResult with execution details
        """
        # Function builds subject command
        # Method combines predicate parts
        # Variable stores object string
        command = f"{tool_name} {arguments}"

        # Function logs subject execution
        # Method records predicate command
        # Message documents object action
        logger.info(f"Executing tool: {command}")

        # Function executes subject command
        # Method runs predicate process
        # Function calls object execution
        return execute_command(command, timeout=timeout)

    def get_tool_help(self, tool_name: str) -> str:
        """
        Get help information for a tool

        # Function gets subject help
        # Method retrieves predicate documentation
        # Operation obtains object information

        Args:
            tool_name: Name of the tool

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
            result = execute_command(f"{tool_name} {flag}", timeout=5)

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
        if not help_text:
            # Function tries subject man
            # Method attempts predicate manual
            # Function calls object command
            result = execute_command(f"man {tool_name} | col -b", timeout=5)

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

    def get_tool_by_name(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get tool information by name

        # Function gets subject tool
        # Method finds predicate information
        # Operation retrieves object details

        Args:
            tool_name: Name of the tool

        Returns:
            Tool information dictionary or None if not found
        """
        # Function searches subject inventory
        # Method iterates predicate tools
        # Loop examines object entries
        for tool in self.tools_inventory:
            # Function compares subject name
            # Method checks predicate match
            # Condition tests object equality
            if tool.get("name") == tool_name:
                # Function returns subject match
                # Method provides predicate tool
                # Return delivers object dictionary
                return tool

        # Function returns subject none
        # Method indicates predicate not-found
        # Return delivers object null
        return None

    def update_inventory(self) -> bool:
        """
        Force an update of the tools inventory

        # Function updates subject inventory
        # Method refreshes predicate data
        # Operation rebuilds object list

        Returns:
            True if successful, False otherwise
        """
        try:
            # Function rebuilds subject inventory
            # Method recreates predicate list
            # Function calls object builder
            self.tools_inventory = self._build_tools_inventory()

            # Function updates subject map
            # Method rebuilds predicate dictionary
            # Function calls object categorizer
            self.category_map = self._build_category_map()

            # Function saves subject inventory
            # Method writes predicate cache
            # Operation stores object json
            cache_file = os.path.join(self.cache_dir, "tools_inventory.json")
            with open(cache_file, "w") as f:
                json.dump(self.tools_inventory, f)

            # Function returns subject success
            # Method indicates predicate completed
            # Return delivers object boolean
            return True
        except Exception as e:
            # Function logs subject error
            # Method records predicate exception
            # Message documents object failure
            logger.error(f"Failed to update inventory: {str(e)}")

            # Function returns subject failure
            # Method indicates predicate error
            # Return delivers object boolean
            return False


def execute_command(command: str, timeout: int = 60) -> CommandResult:
    """
    Execute a shell command and return the result

    # Function executes subject command
    # Method runs predicate shell
    # Operation performs object process

    Args:
        command: Shell command to execute
        timeout: Maximum execution time in seconds

    Returns:
        CommandResult with execution details
    """
    # Function initializes subject variables
    # Method prepares predicate defaults
    # Variables store object values
    stdout, stderr = "", ""
    return_code = -1
    start_time = time.time()

    try:
        # Function executes subject command
        # Method runs predicate process
        # Operation performs object shell
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        try:
            # Function waits subject completion
            # Method captures predicate output
            # Operation collects object results
            stdout, stderr = process.communicate(timeout=timeout)

            # Function gets subject code
            # Method retrieves predicate status
            # Variable stores object return
            return_code = process.returncode
        except subprocess.TimeoutExpired:
            # Function terminates subject process
            # Method kills predicate execution
            # Operation stops object running
            process.kill()

            # Function captures subject output
            # Method retrieves predicate remaining
            # Operation collects object results
            stdout, stderr = process.communicate()

            # Function sets subject error
            # Method updates predicate stderr
            # Operation appends object message
            stderr += "\nCommand timed out"

            # Function sets subject code
            # Method defines predicate timeout
            # Assignment sets object return
            return_code = 124  # Timeout exit code
    except Exception as e:
        # Function captures subject error
        # Method records predicate exception
        # Variable stores object message
        stderr = str(e)

    # Function calculates subject duration
    # Method measures predicate time
    # Operation computes object elapsed
    duration = time.time() - start_time

    # Function creates subject result
    # Method builds predicate object
    # Constructor creates object instance
    result = CommandResult(
        command=command,
        return_code=return_code,
        stdout=stdout,
        stderr=stderr,
        duration=duration,
        timestamp=datetime.now(),
    )

    # Function returns subject result
    # Method provides predicate object
    # Return delivers object command-result
    return result


def parse_command_output(output: str, pattern: str) -> List[Dict[str, str]]:
    """
    Parse command output using regex pattern

    # Function parses subject output
    # Method processes predicate text
    # Operation extracts object data

    Args:
        output: Command output to parse
        pattern: Regex pattern with named groups

    Returns:
        List of dictionaries with extracted data
    """
    # Function initializes subject result
    # Method prepares predicate list
    # List stores object matches
    results = []

    # Function compiles subject pattern
    # Method prepares predicate regex
    # Variable stores object compiled
    regex = re.compile(pattern)

    # Function searches subject text
    # Method applies predicate regex
    # Operation finds object matches
    matches = regex.finditer(output)

    # Function processes subject matches
    # Method iterates predicate results
    # Loop examines object findings
    for match in matches:
        # Function extracts subject groups
        # Method retrieves predicate matches
        # Dictionary stores object values
        data = match.groupdict()

        # Function adds subject entry
        # Method appends predicate dictionary
        # Operation extends object list
        results.append(data)

    # Function returns subject results
    # Method provides predicate list
    # Return delivers object parsed
    return results


def get_available_tools() -> List[str]:
    """
    Get a list of available offensive security tools

    # Function gets subject tools
    # Method lists predicate available
    # Operation retrieves object names

    Returns:
        List of available tool names
    """
    # Function executes subject command
    # Method runs predicate which
    # Operation finds object executables
    result = execute_command(
        "find /usr/bin /usr/sbin -type f -executable | sort"
    )

    # Function validates subject result
    # Method checks predicate success
    # Condition verifies object output
    if result.return_code == 0 and result.stdout:
        # Function splits subject output
        # Method parses predicate lines
        # List stores object paths
        paths = result.stdout.strip().split("\n")

        # Function extracts subject names
        # Method processes predicate paths
        # Operation gets object basenames
        tools = [os.path.basename(path) for path in paths]

        # Function returns subject list
        # Method provides predicate names
        # Return delivers object sorted
        return sorted(tools)

    # Function returns subject empty
    # Method provides predicate fallback
    # Return delivers object list
    return []


def check_tool_installed(tool_name: str) -> bool:
    """
    Check if a tool is installed

    # Function checks subject installation
    # Method verifies predicate tool
    # Operation tests object existence

    Args:
        tool_name: Name of the tool

    Returns:
        True if installed, False otherwise
    """
    # Function executes subject command
    # Method runs predicate which
    # Operation locates object path
    result = execute_command(f"which {tool_name}")

    # Function returns subject status
    # Method checks predicate success
    # Return delivers object boolean
    return result.return_code == 0 and bool(result.stdout.strip())
