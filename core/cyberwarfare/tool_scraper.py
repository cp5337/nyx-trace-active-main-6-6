"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-CORE-CYBERWARFARE-SCRAPER-0001      │
// │ 📁 domain       : Cyberwarfare, Web Scraping               │
// │ 🧠 description  : Kali tools web scraper                   │
// │                  URL and information extraction            │
// │ 🕸️ hash_type    : UUID → CUID-linked module                │
// │ 🔄 parent_node  : NODE_CYBERWARFARE                       │
// │ 🧩 dependencies : requests, bs4, dataclasses               │
// │ 🔧 tool_usage   : Web Scraping, Data Collection           │
// │ 📡 input_type   : URLs, HTML content                        │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : web scraping, information extraction     │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

Kali Tools Scraper
----------------
This module provides web scraping capabilities for extracting
information about Kali Linux tools from official websites and
documentation sources.

Designed for future Rust compatibility with clear interfaces and types.
"""

import os
import json
import logging
import time
import re
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, field
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ToolInfo:
    """
    Information about a cyberwarfare tool

    # Class stores subject information
    # Method contains predicate data
    # Structure holds object details
    """

    name: str
    category: str
    description: str = ""
    url: str = ""
    homepage: str = ""
    repository: str = ""
    documentation: str = ""
    author: str = ""
    version: str = ""
    license: str = ""
    dependencies: List[str] = field(default_factory=list)
    features: List[str] = field(default_factory=list)
    related_tools: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    last_updated: str = ""


@dataclass
class ScraperResult:
    """
    Results from a scraper operation

    # Class stores subject results
    # Method contains predicate output
    # Structure holds object scraping
    """

    tools: List[ToolInfo]
    errors: List[str]
    total_processed: int
    success_count: int
    failure_count: int
    duration_seconds: float


class ToolScraper:
    """
    Web scraper for Kali Linux tools information

    # Class scrapes subject information
    # Method extracts predicate data
    # Object retrieves object tool-details
    """

    def __init__(
        self,
        cache_dir: str = ".cache/tool_scraper",
        user_agent: str = "NyxTrace/1.0 (CTAS Intelligence Platform)",
    ):
        """
        Initialize the Tool Scraper

        # Function initializes subject scraper
        # Method configures predicate settings
        # Constructor sets object parameters

        Args:
            cache_dir: Directory for caching scraped data
            user_agent: User agent string for HTTP requests
        """
        # Function sets subject variables
        # Method stores predicate settings
        # Assignment initializes object state
        self.cache_dir = cache_dir
        self.user_agent = user_agent
        self.base_urls = {
            "kali_tools": "https://www.kali.org/tools/",
            "exploit_db": "https://www.exploit-db.com/",
            "official_docs": "https://www.kali.org/docs/",
        }

        # Function ensures subject path
        # Method creates predicate directory
        # Operation makes object cache
        os.makedirs(self.cache_dir, exist_ok=True)

        # Function initializes subject session
        # Method creates predicate connection
        # Variable stores object requests
        self.session = requests.Session()

        # Function configures subject headers
        # Method sets predicate user-agent
        # Assignment defines object request
        self.session.headers.update(
            {
                "User-Agent": self.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml",
                "Accept-Language": "en-US,en;q=0.9",
            }
        )

        # Function logs subject initialization
        # Method records predicate startup
        # Message documents object ready
        logger.info("Initialized ToolScraper")

    def _get_cached_data(self, cache_key: str) -> Optional[Any]:
        """
        Retrieve data from cache if available

        # Function gets subject cache
        # Method retrieves predicate data
        # Operation loads object file

        Args:
            cache_key: Identifier for the cached data

        Returns:
            Cached data or None if not available
        """
        # Function builds subject path
        # Method creates predicate filename
        # Variable stores object location
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")

        # Function checks subject existence
        # Method verifies predicate file
        # Condition tests object present
        if os.path.exists(cache_file):
            try:
                # Function calculates subject age
                # Method determines predicate time
                # Operation computes object days
                file_age_days = (time.time() - os.path.getmtime(cache_file)) / (
                    60 * 60 * 24
                )

                # Function validates subject freshness
                # Method checks predicate age
                # Condition verifies object recent
                if file_age_days < 7:  # Cache valid for 7 days
                    # Function loads subject cache
                    # Method reads predicate file
                    # Operation deserializes object json
                    with open(cache_file, "r") as f:
                        return json.load(f)
            except Exception as e:
                # Function logs subject error
                # Method records predicate exception
                # Message documents object failure
                logger.warning(f"Failed to read cache {cache_key}: {str(e)}")

        # Function returns subject none
        # Method indicates predicate missing
        # Return signifies object unavailable
        return None

    def _save_cached_data(self, cache_key: str, data: Any) -> bool:
        """
        Save data to cache

        # Function saves subject cache
        # Method writes predicate data
        # Operation stores object file

        Args:
            cache_key: Identifier for the cached data
            data: Data to cache

        Returns:
            True if successfully cached, False otherwise
        """
        try:
            # Function builds subject path
            # Method creates predicate filename
            # Variable stores object location
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")

            # Function writes subject data
            # Method saves predicate json
            # Operation serializes object content
            with open(cache_file, "w") as f:
                json.dump(data, f)

            # Function returns subject success
            # Method indicates predicate saved
            # Return reports object state
            return True
        except Exception as e:
            # Function logs subject error
            # Method records predicate exception
            # Message documents object failure
            logger.error(f"Failed to cache {cache_key}: {str(e)}")

            # Function returns subject failure
            # Method indicates predicate error
            # Return reports object state
            return False

    def _fetch_html(
        self, url: str, allow_redirects: bool = True
    ) -> Optional[str]:
        """
        Fetch HTML content from a URL

        # Function fetches subject html
        # Method retrieves predicate content
        # Operation downloads object page

        Args:
            url: URL to fetch content from
            allow_redirects: Whether to follow redirects

        Returns:
            HTML content as string or None if failed
        """
        try:
            # Function logs subject request
            # Method records predicate url
            # Message documents object action
            logger.debug(f"Fetching URL: {url}")

            # Function sends subject request
            # Method executes predicate http
            # Operation retrieves object content
            response = self.session.get(
                url, allow_redirects=allow_redirects, timeout=10
            )

            # Function validates subject status
            # Method checks predicate code
            # Condition verifies object success
            if response.status_code == 200:
                # Function returns subject content
                # Method provides predicate html
                # Return delivers object text
                return response.text
            else:
                # Function logs subject error
                # Method records predicate failure
                # Message documents object status
                logger.warning(
                    f"Failed to fetch {url}: HTTP {response.status_code}"
                )

                # Function returns subject none
                # Method indicates predicate failure
                # Return signifies object error
                return None

        except Exception as e:
            # Function logs subject error
            # Method records predicate exception
            # Message documents object failure
            logger.error(f"Error fetching {url}: {str(e)}")

            # Function returns subject none
            # Method indicates predicate failure
            # Return signifies object error
            return None

    def get_tool_list(
        self, force_refresh: bool = False
    ) -> List[Dict[str, str]]:
        """
        Get a list of all Kali tools with basic information

        # Function gets subject tools
        # Method retrieves predicate list
        # Operation extracts object basic-info

        Args:
            force_refresh: Whether to bypass cache and fetch fresh data

        Returns:
            List of dictionaries with basic tool information
        """
        # Function defines subject key
        # Method identifies predicate cache
        # Variable stores object identifier
        cache_key = "kali_tools_list"

        # Function checks subject cache
        # Method verifies predicate freshness
        # Condition tests object bypass
        if not force_refresh:
            # Function gets subject cached
            # Method retrieves predicate data
            # Variable stores object json
            cached_data = self._get_cached_data(cache_key)

            # Function validates subject cache
            # Method checks predicate existence
            # Condition verifies object available
            if cached_data is not None:
                # Function returns subject data
                # Method provides predicate cached
                # Return delivers object list
                return cached_data

        # Function initializes subject list
        # Method prepares predicate container
        # List stores object tools
        tools_list = []

        # Function builds subject url
        # Method creates predicate address
        # Variable stores object location
        url = self.base_urls["kali_tools"]

        # Function fetches subject html
        # Method retrieves predicate content
        # Variable stores object page
        html_content = self._fetch_html(url)

        # Function validates subject html
        # Method checks predicate content
        # Condition verifies object retrieval
        if not html_content:
            # Function logs subject error
            # Method records predicate failure
            # Message documents object missing
            logger.error(f"Failed to fetch tools list from {url}")

            # Function returns subject empty
            # Method provides predicate default
            # Return delivers object list
            return []

        # Function parses subject html
        # Method processes predicate content
        # Variable stores object dom
        soup = BeautifulSoup(html_content, "html.parser")

        # Function finds subject elements
        # Method locates predicate tools
        # List stores object containers
        tool_elements = soup.select(".tool-single")

        # Function processes subject elements
        # Method iterates predicate containers
        # Loop extracts object information
        for element in tool_elements:
            try:
                # Function finds subject link
                # Method locates predicate anchor
                # Variable stores object tag
                link_element = element.select_one("a")

                # Function validates subject link
                # Method checks predicate existence
                # Condition verifies object found
                if not link_element:
                    continue

                # Function extracts subject url
                # Method retrieves predicate href
                # Variable stores object address
                tool_url = link_element.get("href", "")

                # Function validates subject url
                # Method checks predicate value
                # Condition verifies object present
                if not tool_url:
                    continue

                # Function normalizes subject url
                # Method ensures predicate absolute
                # Function formats object address
                tool_url = urljoin(url, tool_url)

                # Function extracts subject name
                # Method parses predicate url
                # Operation derives object identifier
                tool_name = os.path.basename(tool_url.rstrip("/"))

                # Function finds subject description
                # Method locates predicate element
                # Variable stores object text
                description_element = element.select_one(".tool-description")
                tool_description = (
                    description_element.get_text(strip=True)
                    if description_element
                    else ""
                )

                # Function finds subject icon
                # Method locates predicate image
                # Variable stores object element
                icon_element = element.select_one(".tool-logo img")
                tool_icon = icon_element.get("src", "") if icon_element else ""

                # Function normalizes subject icon
                # Method ensures predicate absolute
                # Function formats object url
                if tool_icon:
                    tool_icon = urljoin(url, tool_icon)

                # Function creates subject entry
                # Method builds predicate dictionary
                # Dictionary stores object information
                tool_info = {
                    "name": tool_name,
                    "description": tool_description,
                    "url": tool_url,
                    "icon": tool_icon,
                }

                # Function adds subject tool
                # Method appends predicate dictionary
                # Operation extends object list
                tools_list.append(tool_info)

            except Exception as e:
                # Function logs subject error
                # Method records predicate exception
                # Message documents object failure
                logger.error(f"Error parsing tool element: {str(e)}")

        # Function validates subject list
        # Method checks predicate quantity
        # Condition verifies object populated
        if tools_list:
            # Function caches subject data
            # Method saves predicate list
            # Operation stores object json
            self._save_cached_data(cache_key, tools_list)

        # Function returns subject list
        # Method provides predicate tools
        # Return delivers object information
        return tools_list

    def get_tool_details(
        self, tool_name: str, force_refresh: bool = False
    ) -> Optional[ToolInfo]:
        """
        Get detailed information about a specific tool

        # Function gets subject details
        # Method retrieves predicate information
        # Operation extracts object tool-data

        Args:
            tool_name: Name of the tool to get details for
            force_refresh: Whether to bypass cache and fetch fresh data

        Returns:
            ToolInfo object or None if failed
        """
        # Function defines subject key
        # Method identifies predicate cache
        # Variable stores object identifier
        cache_key = f"tool_details_{tool_name}"

        # Function checks subject cache
        # Method verifies predicate freshness
        # Condition tests object bypass
        if not force_refresh:
            # Function gets subject cached
            # Method retrieves predicate data
            # Variable stores object json
            cached_data = self._get_cached_data(cache_key)

            # Function validates subject cache
            # Method checks predicate existence
            # Condition verifies object available
            if cached_data is not None:
                # Function converts subject dict
                # Method deserializes predicate json
                # Constructor creates object instance
                return ToolInfo(**cached_data)

        # Function builds subject url
        # Method creates predicate address
        # Variable stores object location
        url = f"{self.base_urls['kali_tools']}{tool_name}/"

        # Function fetches subject html
        # Method retrieves predicate content
        # Variable stores object page
        html_content = self._fetch_html(url)

        # Function validates subject html
        # Method checks predicate content
        # Condition verifies object retrieval
        if not html_content:
            # Function logs subject error
            # Method records predicate failure
            # Message documents object missing
            logger.error(f"Failed to fetch tool details from {url}")

            # Function returns subject none
            # Method indicates predicate failure
            # Return signifies object error
            return None

        # Function parses subject html
        # Method processes predicate content
        # Variable stores object dom
        soup = BeautifulSoup(html_content, "html.parser")

        try:
            # Function extracts subject fields
            # Method parses predicate html
            # Dictionary stores object data
            tool_data = {}

            # Function extracts subject title
            # Method finds predicate element
            # Variable stores object text
            title_element = soup.select_one(".hero h1")
            tool_data["name"] = (
                title_element.get_text(strip=True)
                if title_element
                else tool_name
            )

            # Function extracts subject url
            # Method assigns predicate value
            # Dictionary stores object address
            tool_data["url"] = url

            # Function extracts subject description
            # Method finds predicate element
            # Variable stores object text
            description_element = soup.select_one(".hero p.lead")
            tool_data["description"] = (
                description_element.get_text(strip=True)
                if description_element
                else ""
            )

            # Function extracts subject details
            # Method parses predicate section
            # Operation processes object content
            details_section = soup.select_one(".card-body")
            if details_section:
                # Function finds subject homepage
                # Method locates predicate link
                # Variable stores object element
                homepage_link = details_section.select_one('a[href*="//"]')
                tool_data["homepage"] = (
                    homepage_link.get("href", "") if homepage_link else ""
                )

                # Function finds subject elements
                # Method locates predicate paragraphs
                # List stores object text-containers
                paragraphs = details_section.select("p")

                # Function initializes subject text
                # Method prepares predicate variable
                # Variable stores object content
                content_text = ""

                # Function combines subject paragraphs
                # Method joins predicate text
                # Operation builds object content
                for p in paragraphs:
                    content_text += p.get_text(strip=True) + " "

                # Function extracts subject author
                # Method finds predicate mention
                # Operation searches object text
                author_match = re.search(
                    r"(?:author|by|developed by)[:\s]+([^\.]+)",
                    content_text,
                    re.IGNORECASE,
                )
                if author_match:
                    tool_data["author"] = author_match.group(1).strip()

            # Function extracts subject category
            # Method finds predicate section
            # Variable stores object element
            category_element = soup.select_one(".badge")
            tool_data["category"] = (
                category_element.get_text(strip=True)
                if category_element
                else "Uncategorized"
            )

            # Function builds subject examples
            # Method finds predicate section
            # List stores object commands
            example_elements = soup.select("pre code")
            tool_data["examples"] = [
                e.get_text(strip=True) for e in example_elements
            ]

            # Function extracts subject package
            # Method finds predicate section
            # Variable stores object element
            package_section = soup.select_one(".package-meta")
            if package_section:
                # Function finds subject version
                # Method locates predicate span
                # Variable stores object text
                version_element = package_section.select_one(".badge-version")
                tool_data["version"] = (
                    version_element.get_text(strip=True)
                    if version_element
                    else ""
                )

            # Function finds subject repository
            # Method locates predicate link
            # Variable stores object element
            repo_link = soup.select_one('a[href*="github.com"]')
            tool_data["repository"] = (
                repo_link.get("href", "") if repo_link else ""
            )

            # Function finds subject documentation
            # Method locates predicate link
            # Variable stores object element
            docs_link = soup.select_one('a[href*="docs"]')
            tool_data["documentation"] = (
                docs_link.get("href", "") if docs_link else ""
            )

            # Function extracts subject features
            # Method finds predicate list
            # List stores object capabilities
            features_list = []
            features_section = soup.select_one(".features-list")
            if features_section:
                # Function finds subject items
                # Method locates predicate elements
                # List stores object features
                feature_items = features_section.select("li")

                # Function builds subject list
                # Method collects predicate text
                # Operation extracts object items
                features_list = [f.get_text(strip=True) for f in feature_items]

            # Function assigns subject features
            # Method stores predicate list
            # Dictionary updates object field
            tool_data["features"] = features_list

            # Function finds subject related
            # Method locates predicate section
            # List stores object tools
            related_elements = soup.select(".card-tool a")
            tool_data["related_tools"] = [
                re.sub(r"/$", "", os.path.basename(e.get("href", "")))
                for e in related_elements
                if e.get("href") and "/tools/" in e.get("href", "")
            ]

            # Function extracts subject tags
            # Method finds predicate elements
            # List stores object keywords
            tag_elements = soup.select(".tag")
            tool_data["tags"] = [t.get_text(strip=True) for t in tag_elements]

            # Function creates subject info
            # Method constructs predicate object
            # Constructor builds object instance
            tool_info = ToolInfo(**tool_data)

            # Function caches subject data
            # Method saves predicate dictionary
            # Operation stores object json
            self._save_cached_data(cache_key, tool_data)

            # Function returns subject info
            # Method provides predicate object
            # Return delivers object instance
            return tool_info

        except Exception as e:
            # Function logs subject error
            # Method records predicate exception
            # Message documents object failure
            logger.error(
                f"Error parsing tool details for {tool_name}: {str(e)}"
            )

            # Function returns subject none
            # Method indicates predicate failure
            # Return signifies object error
            return None

    def scrape_all_tools(
        self, max_tools: Optional[int] = None
    ) -> ScraperResult:
        """
        Scrape detailed information for all Kali tools

        # Function scrapes subject tools
        # Method retrieves predicate details
        # Operation extracts object information

        Args:
            max_tools: Maximum number of tools to scrape (None for all)

        Returns:
            ScraperResult with outcome and tools data
        """
        # Function initializes subject variables
        # Method prepares predicate tracking
        # Variables store object state
        start_time = time.time()
        tools_info = []
        errors = []
        total_processed = 0
        success_count = 0
        failure_count = 0

        # Function gets subject list
        # Method retrieves predicate tools
        # Variable stores object names
        tools_list = self.get_tool_list()

        # Function validates subject list
        # Method checks predicate empty
        # Condition verifies object populated
        if not tools_list:
            # Function logs subject error
            # Method records predicate failure
            # Message documents object empty
            logger.error("Failed to get tools list for scraping")

            # Function creates subject result
            # Method builds predicate object
            # Constructor creates object instance
            return ScraperResult(
                tools=[],
                errors=["Failed to get tools list"],
                total_processed=0,
                success_count=0,
                failure_count=1,
                duration_seconds=time.time() - start_time,
            )

        # Function limits subject scope
        # Method applies predicate maximum
        # Condition checks object specified
        if max_tools is not None:
            # Function limits subject list
            # Method truncates predicate array
            # Operation slices object length
            tools_list = tools_list[:max_tools]

        # Function processes subject tools
        # Method iterates predicate list
        # Loop retrieves object details
        for tool_info in tools_list:
            try:
                # Function increments subject counter
                # Method updates predicate tracking
                # Operation counts object processed
                total_processed += 1

                # Function extracts subject name
                # Method retrieves predicate value
                # Variable stores object identifier
                tool_name = tool_info["name"]

                # Function logs subject processing
                # Method records predicate action
                # Message documents object current
                logger.info(
                    f"Scraping details for tool {total_processed}/{len(tools_list)}: {tool_name}"
                )

                # Function gets subject details
                # Method retrieves predicate information
                # Variable stores object tool-info
                tool_details = self.get_tool_details(tool_name)

                # Function checks subject success
                # Method verifies predicate result
                # Condition tests object retrieved
                if tool_details:
                    # Function adds subject tool
                    # Method appends predicate object
                    # Operation extends object list
                    tools_info.append(tool_details)

                    # Function increments subject counter
                    # Method updates predicate tracking
                    # Operation counts object success
                    success_count += 1
                else:
                    # Function adds subject error
                    # Method records predicate failure
                    # Operation logs object message
                    errors.append(f"Failed to scrape details for {tool_name}")

                    # Function increments subject counter
                    # Method updates predicate tracking
                    # Operation counts object failure
                    failure_count += 1

                # Function adds subject delay
                # Method prevents predicate rate-limiting
                # Operation pauses object execution
                time.sleep(0.5)  # Polite delay to prevent rate limiting

            except Exception as e:
                # Function logs subject error
                # Method records predicate exception
                # Message documents object failure
                logger.error(
                    f"Error scraping tool {tool_info.get('name', 'unknown')}: {str(e)}"
                )

                # Function adds subject error
                # Method records predicate message
                # Operation logs object exception
                errors.append(
                    f"Error with {tool_info.get('name', 'unknown')}: {str(e)}"
                )

                # Function increments subject counter
                # Method updates predicate tracking
                # Operation counts object failure
                failure_count += 1

        # Function calculates subject duration
        # Method measures predicate time
        # Operation computes object seconds
        duration = time.time() - start_time

        # Function creates subject result
        # Method builds predicate object
        # Constructor creates object instance
        result = ScraperResult(
            tools=tools_info,
            errors=errors,
            total_processed=total_processed,
            success_count=success_count,
            failure_count=failure_count,
            duration_seconds=duration,
        )

        # Function returns subject result
        # Method provides predicate object
        # Return delivers object instance
        return result

    def export_tools_data(
        self, output_file: str, scraped_data: Optional[ScraperResult] = None
    ) -> bool:
        """
        Export scraped tools data to a JSON file

        # Function exports subject data
        # Method saves predicate tools
        # Operation writes object file

        Args:
            output_file: Path to output JSON file
            scraped_data: Optional ScraperResult to export (scrapes all if None)

        Returns:
            True if export successful, False otherwise
        """
        try:
            # Function handles subject missing
            # Method checks predicate data
            # Condition tests object provided
            if scraped_data is None:
                # Function scrapes subject data
                # Method retrieves predicate tools
                # Function calls object scraper
                scraped_data = self.scrape_all_tools()

            # Function validates subject data
            # Method checks predicate result
            # Condition tests object success
            if not scraped_data or not scraped_data.tools:
                # Function logs subject error
                # Method records predicate problem
                # Message documents object empty
                logger.error("No data available to export")

                # Function returns subject failure
                # Method signals predicate error
                # Return indicates object status
                return False

            # Function prepares subject data
            # Method serializes predicate tools
            # List stores object dictionaries
            tools_data = []

            # Function processes subject tools
            # Method iterates predicate list
            # Loop converts object instances
            for tool in scraped_data.tools:
                # Function serializes subject tool
                # Method converts predicate object
                # Variable stores object dictionary
                tool_dict = {
                    key: getattr(tool, key) for key in tool.__annotations__
                }

                # Function adds subject entry
                # Method appends predicate dictionary
                # Operation extends object list
                tools_data.append(tool_dict)

            # Function creates subject directory
            # Method ensures predicate path
            # Operation checks object parent
            os.makedirs(
                os.path.dirname(os.path.abspath(output_file)), exist_ok=True
            )

            # Function writes subject file
            # Method saves predicate json
            # Operation stores object data
            with open(output_file, "w") as f:
                json.dump(tools_data, f, indent=2)

            # Function logs subject success
            # Method records predicate action
            # Message documents object result
            logger.info(f"Exported {len(tools_data)} tools to {output_file}")

            # Function returns subject success
            # Method signals predicate completion
            # Return indicates object status
            return True

        except Exception as e:
            # Function logs subject error
            # Method records predicate exception
            # Message documents object failure
            logger.error(f"Error exporting tools data: {str(e)}")

            # Function returns subject failure
            # Method signals predicate error
            # Return indicates object status
            return False
