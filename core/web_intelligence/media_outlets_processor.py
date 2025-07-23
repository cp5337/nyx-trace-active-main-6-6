"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-MEDIA-OUTLETS-PROCESSOR-0001        │
// │ 📁 domain       : Media Analysis, OSINT                     │
// │ 🧠 description  : Media outlets processor for tracking and   │
// │                  monitoring large sets of news sources       │
// │ 🕸️ hash_type    : UUID → CUID-linked processor              │
// │ 🔄 parent_node  : NODE_PROCESSOR                           │
// │ 🧩 dependencies : pandas, numpy, requests, bs4             │
// │ 🔧 tool_usage   : Analysis, Collection, Intelligence        │
// │ 📡 input_type   : Spreadsheets, URLs, keywords              │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : data collection, pattern analysis         │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

Advanced Media Outlets Processor
-------------------------------
This module provides comprehensive capabilities for processing, tracking,
and monitoring large sets of media outlets. It supports importing outlet data
from spreadsheets, automatic discovery of related outlets, and continuous
keyword-based monitoring for content generation.
"""

import os
import re
import json
import time
import logging
import hashlib
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Set
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import trafilatura
from urllib.parse import urlparse, urljoin
import concurrent.futures
import traceback
from dataclasses import dataclass, field
import sqlite3

# Function creates subject logger
# Method initializes predicate output
# Operation configures object format
logger = logging.getLogger("ctas_media_outlets_processor")
logger.setLevel(logging.INFO)


@dataclass
class MediaOutlet:
    """
    Data class representing a media outlet

    # Class represents subject outlet
    # Structure stores predicate information
    # Container holds object properties
    """

    outlet_id: str
    name: str
    domain: str
    url: str
    category: Optional[str] = None
    country: Optional[str] = None
    language: Optional[str] = None
    rss_feeds: List[str] = field(default_factory=list)
    article_selectors: Dict[str, str] = field(default_factory=dict)
    social_media: Dict[str, str] = field(default_factory=dict)
    reliability_score: Optional[float] = None
    bias_rating: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    active: bool = True
    last_checked: Optional[datetime] = None
    discovered_date: datetime = field(default_factory=datetime.now)


@dataclass
class MonitoringKeyword:
    """
    Data class for media monitoring keyword

    # Class represents subject keyword
    # Structure stores predicate monitoring
    # Container holds object configuration
    """

    keyword: str
    category: Optional[str] = None
    priority: int = 1  # 1-5 scale, 5 being highest priority
    active: bool = True
    last_matched: Optional[datetime] = None
    match_count: int = 0
    created_date: datetime = field(default_factory=datetime.now)


@dataclass
class ContentMatch:
    """
    Data class for keyword matches in content

    # Class represents subject match
    # Structure stores predicate occurrence
    # Container holds object details
    """

    content_id: str
    keyword: str
    outlet_id: str
    outlet_name: str
    url: str
    title: Optional[str] = None
    match_context: Optional[str] = None
    match_date: datetime = field(default_factory=datetime.now)
    processed: bool = False
    exported: bool = False


class MediaOutletsProcessor:
    """
    Advanced processor for media outlets monitoring and content extraction

    # Class processes subject outlets
    # Processor analyzes predicate media
    # Component monitors object sources
    """

    def __init__(
        self,
        database_path: str = "data/web_intelligence/media_outlets.db",
        outlets_dir: str = "data/web_intelligence/outlets",
        content_cache_dir: str = "data/web_intelligence/content_cache",
    ):
        """
        Initialize the media outlets processor

        # Function initializes subject processor
        # Method configures predicate settings
        # Operation sets object parameters

        Args:
            database_path: Path to SQLite database
            outlets_dir: Directory for outlet data files
            content_cache_dir: Directory for cached content
        """
        self.database_path = database_path
        self.outlets_dir = outlets_dir
        self.content_cache_dir = content_cache_dir

        # Create directory structure
        os.makedirs(os.path.dirname(database_path), exist_ok=True)
        os.makedirs(outlets_dir, exist_ok=True)
        os.makedirs(content_cache_dir, exist_ok=True)

        # Initialize database
        self._initialize_database()

        # Set up requests session with appropriate headers
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
            }
        )

        logger.info(
            f"Initialized MediaOutletsProcessor with database at {database_path}"
        )

    def _initialize_database(self) -> None:
        """
        Initialize SQLite database for media outlets

        # Function initializes subject database
        # Method creates predicate tables
        # Operation sets object schema
        """
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()

            # Create media outlets table
            cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS media_outlets (
                outlet_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                domain TEXT NOT NULL,
                url TEXT NOT NULL,
                category TEXT,
                country TEXT,
                language TEXT,
                reliability_score REAL,
                bias_rating TEXT,
                active INTEGER DEFAULT 1,
                last_checked TIMESTAMP,
                discovered_date TIMESTAMP
            )
            """
            )

            # Create RSS feeds table
            cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS outlet_rss_feeds (
                feed_id INTEGER PRIMARY KEY AUTOINCREMENT,
                outlet_id TEXT,
                feed_url TEXT,
                FOREIGN KEY (outlet_id) REFERENCES media_outlets(outlet_id),
                UNIQUE(outlet_id, feed_url)
            )
            """
            )

            # Create article selectors table
            cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS outlet_selectors (
                selector_id INTEGER PRIMARY KEY AUTOINCREMENT,
                outlet_id TEXT,
                selector_type TEXT,
                selector_value TEXT,
                FOREIGN KEY (outlet_id) REFERENCES media_outlets(outlet_id)
            )
            """
            )

            # Create social media table
            cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS outlet_social_media (
                social_id INTEGER PRIMARY KEY AUTOINCREMENT,
                outlet_id TEXT,
                platform TEXT,
                url TEXT,
                FOREIGN KEY (outlet_id) REFERENCES media_outlets(outlet_id)
            )
            """
            )

            # Create keywords table
            cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS outlet_keywords (
                keyword_id INTEGER PRIMARY KEY AUTOINCREMENT,
                outlet_id TEXT,
                keyword TEXT,
                FOREIGN KEY (outlet_id) REFERENCES media_outlets(outlet_id)
            )
            """
            )

            # Create monitoring keywords table
            cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS monitoring_keywords (
                keyword_id INTEGER PRIMARY KEY AUTOINCREMENT,
                keyword TEXT UNIQUE,
                category TEXT,
                priority INTEGER DEFAULT 1,
                active INTEGER DEFAULT 1,
                last_matched TIMESTAMP,
                match_count INTEGER DEFAULT 0,
                created_date TIMESTAMP
            )
            """
            )

            # Create content matches table
            cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS content_matches (
                match_id INTEGER PRIMARY KEY AUTOINCREMENT,
                content_id TEXT,
                keyword TEXT,
                outlet_id TEXT,
                outlet_name TEXT,
                url TEXT,
                title TEXT,
                match_context TEXT,
                match_date TIMESTAMP,
                processed INTEGER DEFAULT 0,
                exported INTEGER DEFAULT 0,
                FOREIGN KEY (outlet_id) REFERENCES media_outlets(outlet_id),
                FOREIGN KEY (keyword) REFERENCES monitoring_keywords(keyword)
            )
            """
            )

            conn.commit()
            conn.close()

            logger.info("Media outlets database initialized")

        except Exception as e:
            logger.error(f"Error initializing database: {e}")

    def import_outlets_from_excel(self, excel_path: str) -> int:
        """
        Import media outlets from Excel spreadsheet

        # Function imports subject outlets
        # Method reads predicate spreadsheet
        # Operation processes object data

        Args:
            excel_path: Path to Excel file

        Returns:
            Number of outlets imported
        """
        try:
            # Read Excel file
            df = pd.read_excel(excel_path)

            # Check if required columns exist
            required_columns = ["name", "domain", "url"]
            missing_columns = [
                col for col in required_columns if col not in df.columns
            ]

            if missing_columns:
                raise ValueError(
                    f"Missing required columns: {', '.join(missing_columns)}"
                )

            # Process each row
            imported_count = 0

            for _, row in df.iterrows():
                try:
                    # Extract basic info
                    name = row["name"]
                    domain = row["domain"]
                    url = row["url"]

                    # Generate outlet ID
                    outlet_id = hashlib.md5(
                        f"{domain}:{name}".encode()
                    ).hexdigest()

                    # Create outlet object
                    outlet = MediaOutlet(
                        outlet_id=outlet_id, name=name, domain=domain, url=url
                    )

                    # Add optional fields if they exist
                    optional_fields = {
                        "category": "category",
                        "country": "country",
                        "language": "language",
                        "reliability_score": "reliability_score",
                        "bias_rating": "bias_rating",
                    }

                    for attr, col in optional_fields.items():
                        if col in df.columns and not pd.isna(row[col]):
                            setattr(outlet, attr, row[col])

                    # Check for RSS feeds column
                    if "rss_feeds" in df.columns and not pd.isna(
                        row["rss_feeds"]
                    ):
                        # Split RSS feeds if they're in a comma-separated list
                        feeds = row["rss_feeds"].split(",")
                        outlet.rss_feeds = [
                            feed.strip() for feed in feeds if feed.strip()
                        ]

                    # Check for keywords column
                    if "keywords" in df.columns and not pd.isna(
                        row["keywords"]
                    ):
                        # Split keywords if they're in a comma-separated list
                        keywords = row["keywords"].split(",")
                        outlet.keywords = [
                            keyword.strip()
                            for keyword in keywords
                            if keyword.strip()
                        ]

                    # Check for article selectors column
                    if "article_selectors" in df.columns and not pd.isna(
                        row["article_selectors"]
                    ):
                        # Parse article selectors (expected format: type1:selector1,type2:selector2)
                        selector_pairs = row["article_selectors"].split(",")
                        for pair in selector_pairs:
                            if ":" in pair:
                                sel_type, sel_value = pair.split(":", 1)
                                outlet.article_selectors[sel_type.strip()] = (
                                    sel_value.strip()
                                )

                    # Check for social media column
                    if "social_media" in df.columns and not pd.isna(
                        row["social_media"]
                    ):
                        # Parse social media links (expected format: platform1:url1,platform2:url2)
                        social_pairs = row["social_media"].split(",")
                        for pair in social_pairs:
                            if ":" in pair:
                                platform, social_url = pair.split(":", 1)
                                outlet.social_media[platform.strip()] = (
                                    social_url.strip()
                                )

                    # Save outlet to database
                    self.save_outlet(outlet)
                    imported_count += 1

                except Exception as e:
                    logger.warning(
                        f"Error importing outlet {row.get('name', 'unknown')}: {e}"
                    )

            logger.info(f"Imported {imported_count} outlets from {excel_path}")
            return imported_count

        except Exception as e:
            logger.error(f"Error importing outlets from Excel: {e}")
            return 0

    def save_outlet(self, outlet: MediaOutlet) -> bool:
        """
        Save media outlet to database

        # Function saves subject outlet
        # Method stores predicate data
        # Operation writes object database

        Args:
            outlet: MediaOutlet object to save

        Returns:
            True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()

            # Insert or update outlet in media_outlets table
            cursor.execute(
                """
            INSERT OR REPLACE INTO media_outlets (
                outlet_id, name, domain, url, category, country, language,
                reliability_score, bias_rating, active, last_checked, discovered_date
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    outlet.outlet_id,
                    outlet.name,
                    outlet.domain,
                    outlet.url,
                    outlet.category,
                    outlet.country,
                    outlet.language,
                    outlet.reliability_score,
                    outlet.bias_rating,
                    1 if outlet.active else 0,
                    (
                        outlet.last_checked.isoformat()
                        if outlet.last_checked
                        else None
                    ),
                    outlet.discovered_date.isoformat(),
                ),
            )

            # Clear existing related data for this outlet
            cursor.execute(
                "DELETE FROM outlet_rss_feeds WHERE outlet_id = ?",
                (outlet.outlet_id,),
            )
            cursor.execute(
                "DELETE FROM outlet_selectors WHERE outlet_id = ?",
                (outlet.outlet_id,),
            )
            cursor.execute(
                "DELETE FROM outlet_social_media WHERE outlet_id = ?",
                (outlet.outlet_id,),
            )
            cursor.execute(
                "DELETE FROM outlet_keywords WHERE outlet_id = ?",
                (outlet.outlet_id,),
            )

            # Insert RSS feeds
            for feed_url in outlet.rss_feeds:
                cursor.execute(
                    """
                INSERT INTO outlet_rss_feeds (outlet_id, feed_url) VALUES (?, ?)
                """,
                    (outlet.outlet_id, feed_url),
                )

            # Insert article selectors
            for (
                selector_type,
                selector_value,
            ) in outlet.article_selectors.items():
                cursor.execute(
                    """
                INSERT INTO outlet_selectors (outlet_id, selector_type, selector_value) VALUES (?, ?, ?)
                """,
                    (outlet.outlet_id, selector_type, selector_value),
                )

            # Insert social media links
            for platform, social_url in outlet.social_media.items():
                cursor.execute(
                    """
                INSERT INTO outlet_social_media (outlet_id, platform, url) VALUES (?, ?, ?)
                """,
                    (outlet.outlet_id, platform, social_url),
                )

            # Insert keywords
            for keyword in outlet.keywords:
                cursor.execute(
                    """
                INSERT INTO outlet_keywords (outlet_id, keyword) VALUES (?, ?)
                """,
                    (outlet.outlet_id, keyword),
                )

            conn.commit()
            conn.close()

            return True

        except Exception as e:
            logger.error(f"Error saving outlet {outlet.name}: {e}")
            return False

    def get_outlet(self, outlet_id: str) -> Optional[MediaOutlet]:
        """
        Get media outlet by ID

        # Function gets subject outlet
        # Method retrieves predicate data
        # Operation loads object record

        Args:
            outlet_id: The outlet ID

        Returns:
            MediaOutlet object if found, None otherwise
        """
        try:
            conn = sqlite3.connect(self.database_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Get outlet basic info
            cursor.execute(
                "SELECT * FROM media_outlets WHERE outlet_id = ?", (outlet_id,)
            )
            row = cursor.fetchone()

            if not row:
                return None

            # Create outlet object
            outlet = MediaOutlet(
                outlet_id=row["outlet_id"],
                name=row["name"],
                domain=row["domain"],
                url=row["url"],
                category=row["category"],
                country=row["country"],
                language=row["language"],
                reliability_score=row["reliability_score"],
                bias_rating=row["bias_rating"],
                active=bool(row["active"]),
            )

            # Set dates if they exist
            if row["last_checked"]:
                outlet.last_checked = datetime.fromisoformat(
                    row["last_checked"]
                )
            if row["discovered_date"]:
                outlet.discovered_date = datetime.fromisoformat(
                    row["discovered_date"]
                )

            # Get RSS feeds
            cursor.execute(
                "SELECT feed_url FROM outlet_rss_feeds WHERE outlet_id = ?",
                (outlet_id,),
            )
            outlet.rss_feeds = [row[0] for row in cursor.fetchall()]

            # Get article selectors
            cursor.execute(
                "SELECT selector_type, selector_value FROM outlet_selectors WHERE outlet_id = ?",
                (outlet_id,),
            )
            outlet.article_selectors = {
                row[0]: row[1] for row in cursor.fetchall()
            }

            # Get social media links
            cursor.execute(
                "SELECT platform, url FROM outlet_social_media WHERE outlet_id = ?",
                (outlet_id,),
            )
            outlet.social_media = {row[0]: row[1] for row in cursor.fetchall()}

            # Get keywords
            cursor.execute(
                "SELECT keyword FROM outlet_keywords WHERE outlet_id = ?",
                (outlet_id,),
            )
            outlet.keywords = [row[0] for row in cursor.fetchall()]

            conn.close()

            return outlet

        except Exception as e:
            logger.error(f"Error getting outlet {outlet_id}: {e}")
            return None

    def get_all_outlets(self, active_only: bool = True) -> List[MediaOutlet]:
        """
        Get all media outlets

        # Function gets subject outlets
        # Method retrieves predicate records
        # Operation loads object collection

        Args:
            active_only: If True, return only active outlets

        Returns:
            List of MediaOutlet objects
        """
        outlets = []

        try:
            conn = sqlite3.connect(self.database_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Get all outlets (active only if specified)
            if active_only:
                cursor.execute(
                    "SELECT outlet_id FROM media_outlets WHERE active = 1"
                )
            else:
                cursor.execute("SELECT outlet_id FROM media_outlets")

            rows = cursor.fetchall()

            conn.close()

            # Get each outlet by ID
            for row in rows:
                outlet = self.get_outlet(row["outlet_id"])
                if outlet:
                    outlets.append(outlet)

        except Exception as e:
            logger.error(f"Error getting all outlets: {e}")

        return outlets

    def search_outlets(self, query: str) -> List[MediaOutlet]:
        """
        Search for media outlets

        # Function searches subject outlets
        # Method finds predicate matches
        # Operation retrieves object results

        Args:
            query: Search query

        Returns:
            List of matching MediaOutlet objects
        """
        outlets = []

        try:
            conn = sqlite3.connect(self.database_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Search for outlets matching query
            cursor.execute(
                """
            SELECT outlet_id FROM media_outlets 
            WHERE name LIKE ? OR domain LIKE ? OR url LIKE ? OR category LIKE ? 
            OR country LIKE ? OR language LIKE ? OR bias_rating LIKE ?
            """,
                (
                    f"%{query}%",
                    f"%{query}%",
                    f"%{query}%",
                    f"%{query}%",
                    f"%{query}%",
                    f"%{query}%",
                    f"%{query}%",
                ),
            )

            rows = cursor.fetchall()

            conn.close()

            # Get each outlet by ID
            for row in rows:
                outlet = self.get_outlet(row["outlet_id"])
                if outlet:
                    outlets.append(outlet)

        except Exception as e:
            logger.error(f"Error searching outlets: {e}")

        return outlets

    def add_monitoring_keyword(
        self, keyword: str, category: Optional[str] = None, priority: int = 1
    ) -> bool:
        """
        Add a keyword for monitoring

        # Function adds subject keyword
        # Method creates predicate monitoring
        # Operation configures object tracking

        Args:
            keyword: Keyword to monitor
            category: Category for organization
            priority: Priority level (1-5)

        Returns:
            True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()

            # Insert or update monitoring keyword
            cursor.execute(
                """
            INSERT OR REPLACE INTO monitoring_keywords (
                keyword, category, priority, active, created_date
            ) VALUES (?, ?, ?, ?, ?)
            """,
                (
                    keyword,
                    category,
                    min(
                        max(1, priority), 5
                    ),  # Ensure priority is between 1 and 5
                    1,  # Active by default
                    datetime.now().isoformat(),
                ),
            )

            conn.commit()
            conn.close()

            logger.info(f"Added monitoring keyword: {keyword}")
            return True

        except Exception as e:
            logger.error(f"Error adding monitoring keyword: {e}")
            return False

    def get_monitoring_keywords(
        self, active_only: bool = True, category: Optional[str] = None
    ) -> List[MonitoringKeyword]:
        """
        Get monitoring keywords

        # Function gets subject keywords
        # Method retrieves predicate monitoring
        # Operation loads object configuration

        Args:
            active_only: If True, return only active keywords
            category: Filter by category

        Returns:
            List of MonitoringKeyword objects
        """
        keywords = []

        try:
            conn = sqlite3.connect(self.database_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Build query based on filters
            query = "SELECT * FROM monitoring_keywords"
            params = []

            conditions = []
            if active_only:
                conditions.append("active = 1")
            if category:
                conditions.append("category = ?")
                params.append(category)

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            # Execute query
            cursor.execute(query, params)
            rows = cursor.fetchall()

            # Create keyword objects
            for row in rows:
                keyword = MonitoringKeyword(
                    keyword=row["keyword"],
                    category=row["category"],
                    priority=row["priority"],
                    active=bool(row["active"]),
                    match_count=row["match_count"],
                )

                # Set dates if they exist
                if row["last_matched"]:
                    keyword.last_matched = datetime.fromisoformat(
                        row["last_matched"]
                    )
                if row["created_date"]:
                    keyword.created_date = datetime.fromisoformat(
                        row["created_date"]
                    )

                keywords.append(keyword)

            conn.close()

        except Exception as e:
            logger.error(f"Error getting monitoring keywords: {e}")

        return keywords

    def extract_content_from_url(
        self, url: str, max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Extract content from a URL

        # Function extracts subject content
        # Method retrieves predicate article
        # Operation processes object text

        Args:
            url: URL to extract content from
            max_retries: Maximum number of retry attempts

        Returns:
            Dictionary with extracted content
        """
        # Generate content ID from URL
        content_id = hashlib.md5(url.encode()).hexdigest()

        # Check cache first
        cache_path = os.path.join(self.content_cache_dir, f"{content_id}.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading cached content: {e}")

        # Extract domain to identify the outlet
        domain = self._extract_domain(url)

        # Try to get outlet for this domain
        outlets = self.search_outlets(domain)
        outlet = outlets[0] if outlets else None

        # Set up selectors if outlet is known
        selectors = {}
        if outlet and outlet.article_selectors:
            selectors = outlet.article_selectors

        # Initialize content dictionary
        content = {
            "content_id": content_id,
            "url": url,
            "domain": domain,
            "outlet_name": outlet.name if outlet else domain,
            "extraction_date": datetime.now().isoformat(),
            "title": None,
            "text": None,
            "html": None,
            "images": [],
            "links": [],
        }

        # Extract content
        retries = 0
        while retries < max_retries:
            try:
                # Download content
                response = self.session.get(url, timeout=20)
                response.raise_for_status()

                # Get HTML content
                html_content = response.text
                content["html"] = html_content

                # Use trafilatura for extraction
                extracted = trafilatura.extract(
                    html_content,
                    url=url,
                    include_comments=False,
                    include_tables=True,
                    output_format="json",
                )

                if extracted:
                    # Parse JSON output
                    data = json.loads(extracted)

                    # Set extracted fields
                    content["title"] = data.get("title")
                    content["text"] = data.get("text")

                    # Use BeautifulSoup for additional extraction
                    soup = BeautifulSoup(html_content, "html.parser")

                    # Extract images
                    for img in soup.find_all("img"):
                        if img.get("src"):
                            img_url = img["src"]
                            if not img_url.startswith(("http://", "https://")):
                                img_url = urljoin(url, img_url)
                            if img_url not in content["images"]:
                                content["images"].append(img_url)

                    # Extract links
                    for a_tag in soup.find_all("a", href=True):
                        href = a_tag["href"]
                        if not href.startswith(("http://", "https://")):
                            href = urljoin(url, href)
                        if href != url and href not in content["links"]:
                            content["links"].append(href)

                    # Cache the content
                    self._save_to_cache(content)

                    return content
                else:
                    # Fallback to BeautifulSoup if trafilatura fails
                    soup = BeautifulSoup(html_content, "html.parser")

                    # Extract title
                    title_tag = soup.find("title")
                    if title_tag:
                        content["title"] = title_tag.text.strip()

                    # Extract content using known selectors if available
                    if "content" in selectors and selectors["content"]:
                        content_elements = soup.select(selectors["content"])
                        if content_elements:
                            content["text"] = "\n\n".join(
                                [
                                    el.get_text().strip()
                                    for el in content_elements
                                ]
                            )

                    # Fallback: Extract paragraphs if no selector exists
                    if not content["text"]:
                        paragraphs = []
                        for p in soup.find_all("p"):
                            text = p.get_text().strip()
                            if (
                                text and len(text) > 20
                            ):  # Filter out short paragraphs
                                paragraphs.append(text)

                        if paragraphs:
                            content["text"] = "\n\n".join(paragraphs)

                    # Extract images
                    for img in soup.find_all("img"):
                        if img.get("src"):
                            img_url = img["src"]
                            if not img_url.startswith(("http://", "https://")):
                                img_url = urljoin(url, img_url)
                            if img_url not in content["images"]:
                                content["images"].append(img_url)

                    # Extract links
                    for a_tag in soup.find_all("a", href=True):
                        href = a_tag["href"]
                        if not href.startswith(("http://", "https://")):
                            href = urljoin(url, href)
                        if href != url and href not in content["links"]:
                            content["links"].append(href)

                    # Cache the content if we have title and text
                    if content["title"] and content["text"]:
                        self._save_to_cache(content)
                        return content

                    # If we don't have enough content, retry
                    retries += 1
                    logger.warning(
                        f"Insufficient content extracted on attempt {retries}"
                    )
                    time.sleep(2 * retries)  # Exponential backoff

            except requests.RequestException as e:
                retries += 1
                logger.warning(f"Request error on attempt {retries}: {e}")
                time.sleep(2 * retries)  # Exponential backoff

            except Exception as e:
                logger.error(f"Error extracting content: {e}")
                traceback.print_exc()
                return content

        logger.warning(
            f"Failed to extract sufficient content after {max_retries} retries"
        )
        return content

    def _extract_domain(self, url: str) -> str:
        """
        Extract domain from URL

        # Function extracts subject domain
        # Method parses predicate URL
        # Operation processes object structure

        Args:
            url: URL to extract domain from

        Returns:
            Domain name
        """
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc.lower()

            # Remove www. if present
            if domain.startswith("www."):
                domain = domain[4:]

            return domain

        except Exception as e:
            logger.error(f"Error extracting domain from {url}: {e}")
            return ""

    def _save_to_cache(self, content: Dict[str, Any]) -> None:
        """
        Save content to cache

        # Function saves subject content
        # Method caches predicate data
        # Operation stores object file

        Args:
            content: Content dictionary to cache
        """
        try:
            cache_path = os.path.join(
                self.content_cache_dir, f"{content['content_id']}.json"
            )

            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(content, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"Error saving content to cache: {e}")

    def scan_content_for_keywords(
        self, content: Dict[str, Any]
    ) -> List[ContentMatch]:
        """
        Scan content for monitoring keywords

        # Function scans subject content
        # Method searches predicate keywords
        # Operation finds object matches

        Args:
            content: Content dictionary

        Returns:
            List of ContentMatch objects for matched keywords
        """
        matches = []

        try:
            # Get active monitoring keywords
            keywords = self.get_monitoring_keywords(active_only=True)

            # Get content text to scan
            text = ""
            if content.get("title"):
                text += content["title"] + "\n\n"
            if content.get("text"):
                text += content["text"]

            # Ensure we have text to scan
            if not text:
                return matches

            text_lower = text.lower()

            # Scan for each keyword
            for keyword_obj in keywords:
                keyword = keyword_obj.keyword.lower()

                # Check if keyword is in text
                if keyword in text_lower:
                    # Find positions of keyword in text
                    positions = []
                    start_pos = 0
                    while True:
                        pos = text_lower.find(keyword, start_pos)
                        if pos == -1:
                            break
                        positions.append(pos)
                        start_pos = pos + len(keyword)

                    # For each position, get context (100 chars before and after)
                    for pos in positions:
                        context_start = max(0, pos - 100)
                        context_end = min(len(text), pos + len(keyword) + 100)
                        context = text[context_start:context_end]

                        # Create match object
                        match = ContentMatch(
                            content_id=content["content_id"],
                            keyword=keyword_obj.keyword,
                            outlet_id="",  # Will be set later if outlet is known
                            outlet_name=content.get(
                                "outlet_name", content.get("domain", "")
                            ),
                            url=content["url"],
                            title=content.get("title"),
                            match_context=context,
                        )

                        # Check if we know the outlet
                        domain = content.get("domain", "")
                        outlets = self.search_outlets(domain)
                        if outlets:
                            match.outlet_id = outlets[0].outlet_id
                            match.outlet_name = outlets[0].name

                        matches.append(match)

                        # Only save one context per keyword to avoid duplication
                        break

                    # Update keyword match statistics
                    self._update_keyword_match_stats(keyword_obj.keyword)

            # Save matches to database
            for match in matches:
                self._save_content_match(match)

        except Exception as e:
            logger.error(f"Error scanning content for keywords: {e}")

        return matches

    def _update_keyword_match_stats(self, keyword: str) -> None:
        """
        Update match statistics for a keyword

        # Function updates subject statistics
        # Method increments predicate counter
        # Operation tracks object matches

        Args:
            keyword: Keyword that was matched
        """
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()

            # Update match count and last matched date
            cursor.execute(
                """
            UPDATE monitoring_keywords
            SET match_count = match_count + 1, last_matched = ?
            WHERE keyword = ?
            """,
                (datetime.now().isoformat(), keyword),
            )

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Error updating keyword match stats: {e}")

    def _save_content_match(self, match: ContentMatch) -> bool:
        """
        Save content match to database

        # Function saves subject match
        # Method stores predicate occurrence
        # Operation records object hit

        Args:
            match: ContentMatch object

        Returns:
            True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()

            # Insert content match
            cursor.execute(
                """
            INSERT INTO content_matches (
                content_id, keyword, outlet_id, outlet_name, url, title,
                match_context, match_date, processed, exported
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    match.content_id,
                    match.keyword,
                    match.outlet_id,
                    match.outlet_name,
                    match.url,
                    match.title,
                    match.match_context,
                    match.match_date.isoformat(),
                    1 if match.processed else 0,
                    1 if match.exported else 0,
                ),
            )

            conn.commit()
            conn.close()

            return True

        except Exception as e:
            logger.error(f"Error saving content match: {e}")
            return False

    def get_content_matches(
        self,
        processed: Optional[bool] = None,
        exported: Optional[bool] = None,
        start_date: Optional[datetime] = None,
        keyword: Optional[str] = None,
        outlet_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[ContentMatch]:
        """
        Get content matches with filtering options

        # Function gets subject matches
        # Method retrieves predicate occurrences
        # Operation loads object records

        Args:
            processed: Filter by processed state
            exported: Filter by exported state
            start_date: Filter by start date
            keyword: Filter by keyword
            outlet_id: Filter by outlet ID
            limit: Maximum number of matches to return

        Returns:
            List of ContentMatch objects
        """
        matches = []

        try:
            conn = sqlite3.connect(self.database_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Build query
            query = "SELECT * FROM content_matches"
            params = []

            conditions = []
            if processed is not None:
                conditions.append("processed = ?")
                params.append(1 if processed else 0)
            if exported is not None:
                conditions.append("exported = ?")
                params.append(1 if exported else 0)
            if start_date:
                conditions.append("match_date >= ?")
                params.append(start_date.isoformat())
            if keyword:
                conditions.append("keyword = ?")
                params.append(keyword)
            if outlet_id:
                conditions.append("outlet_id = ?")
                params.append(outlet_id)

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            query += " ORDER BY match_date DESC LIMIT ?"
            params.append(limit)

            # Execute query
            cursor.execute(query, params)
            rows = cursor.fetchall()

            # Create match objects
            for row in rows:
                match = ContentMatch(
                    content_id=row["content_id"],
                    keyword=row["keyword"],
                    outlet_id=row["outlet_id"],
                    outlet_name=row["outlet_name"],
                    url=row["url"],
                    title=row["title"],
                    match_context=row["match_context"],
                    processed=bool(row["processed"]),
                    exported=bool(row["exported"]),
                )

                # Set match date
                if row["match_date"]:
                    match.match_date = datetime.fromisoformat(row["match_date"])

                matches.append(match)

            conn.close()

        except Exception as e:
            logger.error(f"Error getting content matches: {e}")

        return matches

    def discover_related_outlets(
        self, outlet: MediaOutlet, max_links: int = 20
    ) -> List[MediaOutlet]:
        """
        Discover outlets related to a given outlet

        # Function discovers subject outlets
        # Method finds predicate relationships
        # Operation identifies object connections

        Args:
            outlet: MediaOutlet to find related outlets for
            max_links: Maximum number of links to check

        Returns:
            List of discovered related MediaOutlet objects
        """
        discovered_outlets = []

        try:
            # Extract content from outlet URL
            content = self.extract_content_from_url(outlet.url)

            # Get links from content
            links = content.get("links", [])

            # Process each link, up to the maximum
            processed_count = 0
            for link in links:
                if processed_count >= max_links:
                    break

                # Extract domain from link
                domain = self._extract_domain(link)

                # Skip if domain is the same as the original outlet
                if domain == outlet.domain:
                    continue

                # Skip if we already have this outlet
                existing_outlets = self.search_outlets(domain)
                if existing_outlets:
                    continue

                try:
                    # Process the link to see if it's a valid outlet
                    link_content = self.extract_content_from_url(link)

                    # Skip if no title or text was extracted
                    if not link_content.get("title") or not link_content.get(
                        "text"
                    ):
                        continue

                    # Generate outlet ID
                    outlet_id = hashlib.md5(
                        f"{domain}:{link_content.get('title', domain)}".encode()
                    ).hexdigest()

                    # Create new outlet
                    new_outlet = MediaOutlet(
                        outlet_id=outlet_id,
                        name=link_content.get("title", domain),
                        domain=domain,
                        url=link,
                        category=outlet.category,  # Inherit category from original outlet
                        country=outlet.country,  # Inherit country from original outlet
                        discovered_date=datetime.now(),
                    )

                    # Save the outlet to database
                    if self.save_outlet(new_outlet):
                        discovered_outlets.append(new_outlet)

                    processed_count += 1

                except Exception as e:
                    logger.warning(f"Error processing link {link}: {e}")

            logger.info(
                f"Discovered {len(discovered_outlets)} related outlets for {outlet.name}"
            )

        except Exception as e:
            logger.error(f"Error discovering related outlets: {e}")

        return discovered_outlets

    def batch_monitor_outlets(
        self,
        outlets: Optional[List[MediaOutlet]] = None,
        max_outlets: int = 10,
        keywords: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Batch monitor outlets for keyword matches

        # Function monitors subject outlets
        # Method processes predicate content
        # Operation finds object matches

        Args:
            outlets: List of outlets to monitor (or None for all active outlets)
            max_outlets: Maximum number of outlets to process in one batch
            keywords: List of keywords to monitor (or None for all active keywords)

        Returns:
            Dictionary with monitoring results
        """
        results = {
            "outlets_processed": 0,
            "urls_checked": 0,
            "content_extracted": 0,
            "matches_found": 0,
            "errors": 0,
        }

        try:
            # Get outlets to monitor
            if outlets is None:
                outlets = self.get_all_outlets(active_only=True)

            # Limit number of outlets to process
            outlets_to_process = outlets[:max_outlets]

            # Get monitoring keywords
            if keywords is None:
                keyword_objects = self.get_monitoring_keywords(active_only=True)
                keywords = [k.keyword for k in keyword_objects]

            # Process each outlet
            for outlet in outlets_to_process:
                try:
                    # Extract content from outlet URL
                    content = self.extract_content_from_url(outlet.url)

                    if content.get("title") and content.get("text"):
                        results["content_extracted"] += 1

                        # Scan content for keywords
                        matches = self.scan_content_for_keywords(content)
                        results["matches_found"] += len(matches)

                    # Update outlet's last checked timestamp
                    self._update_outlet_last_checked(outlet.outlet_id)

                    results["urls_checked"] += 1

                except Exception as e:
                    logger.error(f"Error monitoring outlet {outlet.name}: {e}")
                    results["errors"] += 1

                # Mark outlet as processed
                results["outlets_processed"] += 1

            logger.info(f"Batch monitoring completed: {results}")

        except Exception as e:
            logger.error(f"Error in batch monitoring: {e}")
            results["errors"] += 1

        return results

    def _update_outlet_last_checked(self, outlet_id: str) -> None:
        """
        Update last checked timestamp for an outlet

        # Function updates subject timestamp
        # Method modifies predicate record
        # Operation updates object attribute

        Args:
            outlet_id: Outlet ID to update
        """
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()

            # Update last checked timestamp
            cursor.execute(
                """
            UPDATE media_outlets
            SET last_checked = ?
            WHERE outlet_id = ?
            """,
                (datetime.now().isoformat(), outlet_id),
            )

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Error updating outlet last checked: {e}")

    def export_content_matches(
        self,
        format: str = "json",
        output_dir: str = "exports",
        start_date: Optional[datetime] = None,
        mark_as_exported: bool = True,
    ) -> str:
        """
        Export content matches to a file

        # Function exports subject matches
        # Method generates predicate file
        # Operation creates object export

        Args:
            format: Export format ('json' or 'csv')
            output_dir: Directory to save export file
            start_date: Only export matches after this date
            mark_as_exported: Whether to mark exported matches as exported

        Returns:
            Path to the exported file
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            # Get matches to export
            matches = self.get_content_matches(
                exported=False, start_date=start_date, limit=1000
            )

            if not matches:
                logger.warning("No matches to export")
                return ""

            # Generate file name with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            if format.lower() == "json":
                # Export to JSON
                file_path = os.path.join(
                    output_dir, f"content_matches_{timestamp}.json"
                )

                # Convert matches to dictionaries
                match_dicts = []
                for match in matches:
                    match_dict = {
                        "content_id": match.content_id,
                        "keyword": match.keyword,
                        "outlet_id": match.outlet_id,
                        "outlet_name": match.outlet_name,
                        "url": match.url,
                        "title": match.title,
                        "match_context": match.match_context,
                        "match_date": match.match_date.isoformat(),
                    }
                    match_dicts.append(match_dict)

                # Write to file
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(match_dicts, f, ensure_ascii=False, indent=2)

            elif format.lower() == "csv":
                # Export to CSV
                file_path = os.path.join(
                    output_dir, f"content_matches_{timestamp}.csv"
                )

                # Convert matches to dictionaries
                match_dicts = []
                for match in matches:
                    match_dict = {
                        "content_id": match.content_id,
                        "keyword": match.keyword,
                        "outlet_id": match.outlet_id,
                        "outlet_name": match.outlet_name,
                        "url": match.url,
                        "title": match.title,
                        "match_context": match.match_context,
                        "match_date": match.match_date.isoformat(),
                    }
                    match_dicts.append(match_dict)

                # Create DataFrame and export to CSV
                df = pd.DataFrame(match_dicts)
                df.to_csv(file_path, index=False, encoding="utf-8")

            else:
                logger.error(f"Unsupported export format: {format}")
                return ""

            # Mark matches as exported if requested
            if mark_as_exported:
                self._mark_matches_as_exported([m.content_id for m in matches])

            logger.info(f"Exported {len(matches)} matches to {file_path}")
            return file_path

        except Exception as e:
            logger.error(f"Error exporting content matches: {e}")
            return ""

    def _mark_matches_as_exported(self, content_ids: List[str]) -> None:
        """
        Mark content matches as exported

        # Function marks subject matches
        # Method updates predicate status
        # Operation modifies object attribute

        Args:
            content_ids: List of content IDs to mark as exported
        """
        if not content_ids:
            return

        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()

            # Build query with placeholders for each ID
            placeholders = ",".join(["?" for _ in content_ids])
            query = f"UPDATE content_matches SET exported = 1 WHERE content_id IN ({placeholders})"

            # Execute query
            cursor.execute(query, content_ids)

            conn.commit()
            conn.close()

            logger.info(f"Marked {len(content_ids)} matches as exported")

        except Exception as e:
            logger.error(f"Error marking matches as exported: {e}")


# Function reads subject example
# Method demonstrates predicate usage
# Operation shows object capabilities
def usage_example():
    """
    Example usage of MediaOutletsProcessor

    # Function shows subject example
    # Method demonstrates predicate usage
    # Operation illustrates object capabilities
    """
    processor = MediaOutletsProcessor()

    # Import outlets from Excel
    # processor.import_outlets_from_excel('path/to/outlets.xlsx')

    # Add monitoring keywords
    processor.add_monitoring_keyword(
        "climate change", category="Environment", priority=3
    )
    processor.add_monitoring_keyword(
        "election", category="Politics", priority=4
    )
    processor.add_monitoring_keyword("pandemic", category="Health", priority=5)

    # Get all outlets
    outlets = processor.get_all_outlets()
    print(f"Found {len(outlets)} outlets")

    # Monitor a few outlets
    results = processor.batch_monitor_outlets(outlets[:5])
    print(f"Monitoring results: {results}")

    # Get content matches
    matches = processor.get_content_matches(limit=10)
    print(f"Found {len(matches)} content matches")

    # Export matches
    export_path = processor.export_content_matches(format="json")
    print(f"Exported matches to: {export_path}")


if __name__ == "__main__":
    usage_example()
