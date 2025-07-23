"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-DARKWEB-ANALYZER-0001               │
// │ 📁 domain       : Darkweb Analysis, OSINT                  │
// │ 🧠 description  : Professional darkweb analysis engine     │
// │                  for intelligence gathering and monitoring  │
// │ 🕸️ hash_type    : UUID → CUID-linked processor              │
// │ 🔄 parent_node  : NODE_PROCESSOR                           │
// │ 🧩 dependencies : requests, stem, numpy, pandas            │
// │ 🔧 tool_usage   : Intelligence, Analysis, Monitoring        │
// │ 📡 input_type   : Onion URLs, search terms, keywords        │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : data extraction, pattern analysis         │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

Professional Darkweb Intelligence Module
--------------------------------------
Advanced darkweb analysis engine for secure access, monitoring,
and intelligence extraction from darkweb sources with comprehensive
security measures and analysis capabilities.
"""

import os
import re
import json
import time
import logging
import hashlib
import requests
import pandas as pd
import numpy as np
import socket
from typing import Dict, List, Tuple, Optional, Union, Any, Set
from datetime import datetime
import traceback
from dataclasses import dataclass, field
import sqlite3
from urllib.parse import urlparse

# Function creates subject logger
# Method initializes predicate output
# Operation configures object format
logger = logging.getLogger("ctas_darkweb_analyzer")
logger.setLevel(logging.INFO)


@dataclass
class DarkwebContent:
    """
    Data class for darkweb content

    # Class represents subject content
    # Structure stores predicate data
    # Container holds object properties
    """

    url: str
    title: Optional[str] = None
    content: Optional[str] = None
    html: Optional[str] = None
    extracted_date: datetime = field(default_factory=datetime.now)
    last_modified: Optional[datetime] = None
    site_type: Optional[str] = None
    language: Optional[str] = None
    marketplace_data: Optional[Dict[str, Any]] = None
    forum_data: Optional[Dict[str, Any]] = None
    content_hash: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    categories: List[str] = field(default_factory=list)
    mentions: Dict[str, List[str]] = field(default_factory=dict)
    entities: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class DarkwebMonitoringTarget:
    """
    Data class for darkweb monitoring target

    # Class represents subject target
    # Structure stores predicate monitoring
    # Container holds object configuration
    """

    name: str
    keywords: List[str] = field(default_factory=list)
    urls: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    monitoring_frequency: int = 24  # hours
    last_checked: Optional[datetime] = None
    is_active: bool = True
    threshold_score: float = 0.5
    notification_enabled: bool = True


class DarkwebIntelligenceEngine:
    """
    Professional darkweb intelligence gathering and analysis engine

    # Class analyzes subject darkweb
    # Engine processes predicate intelligence
    # Component gathers object data
    """

    def __init__(
        self,
        cache_dir: str = "data/darkweb_cache",
        db_path: str = "data/darkweb/darkweb_intelligence.db",
        proxy_url: Optional[str] = None,
    ):
        """
        Initialize the darkweb intelligence engine

        # Function initializes subject engine
        # Method configures predicate settings
        # Operation sets object parameters

        Args:
            cache_dir: Directory to cache extracted content
            db_path: Path to darkweb intelligence database
            proxy_url: Proxy URL for TOR access (e.g., 'socks5h://127.0.0.1:9050')
        """
        self.cache_dir = cache_dir
        self.db_path = db_path
        self.proxy_url = proxy_url

        # Create directory structure
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # Initialize database
        self._initialize_database()

        # Set up proxy configuration for requests
        self.session = requests.Session()
        if proxy_url:
            self.session.proxies = {"http": proxy_url, "https": proxy_url}

        logger.info(
            f"Initialized DarkwebIntelligenceEngine with proxy: {proxy_url}"
        )

        # Map of site types to their common indicators
        self.site_type_indicators = {
            "marketplace": [
                "market",
                "shop",
                "store",
                "buy",
                "sell",
                "vendor",
                "product",
                "bitcoin",
                "btc",
                "monero",
                "xmr",
                "escrow",
                "pgp",
            ],
            "forum": [
                "forum",
                "board",
                "thread",
                "post",
                "topic",
                "message",
                "member",
                "user",
                "comment",
                "discussion",
            ],
            "blog": [
                "blog",
                "article",
                "news",
                "post",
                "author",
                "date",
                "published",
                "journalist",
            ],
            "service": [
                "service",
                "hosting",
                "mail",
                "email",
                "anonymity",
                "privacy",
                "secure",
                "communication",
            ],
        }

    def _initialize_database(self) -> None:
        """
        Initialize SQLite database for darkweb intelligence

        # Function initializes subject database
        # Method creates predicate tables
        # Operation sets object schema
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Create tables if they don't exist

            # Sites table
            cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS darkweb_sites (
                url TEXT PRIMARY KEY,
                title TEXT,
                site_type TEXT,
                first_discovered TIMESTAMP,
                last_checked TIMESTAMP,
                status TEXT,
                categorization TEXT,
                reliability_score REAL,
                notes TEXT
            )
            """
            )

            # Content table
            cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS darkweb_content (
                content_id TEXT PRIMARY KEY,
                url TEXT,
                title TEXT,
                extraction_date TIMESTAMP,
                content_hash TEXT,
                language TEXT,
                content_type TEXT,
                cache_path TEXT,
                FOREIGN KEY (url) REFERENCES darkweb_sites(url)
            )
            """
            )

            # Monitoring targets table
            cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS monitoring_targets (
                target_id TEXT PRIMARY KEY,
                name TEXT,
                created_date TIMESTAMP,
                is_active INTEGER,
                monitoring_frequency INTEGER,
                last_checked TIMESTAMP,
                threshold_score REAL,
                notification_enabled INTEGER
            )
            """
            )

            # Monitoring keywords table
            cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS monitoring_keywords (
                keyword_id INTEGER PRIMARY KEY AUTOINCREMENT,
                target_id TEXT,
                keyword TEXT,
                FOREIGN KEY (target_id) REFERENCES monitoring_targets(target_id)
            )
            """
            )

            # Monitoring URLs table
            cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS monitoring_urls (
                url_id INTEGER PRIMARY KEY AUTOINCREMENT,
                target_id TEXT,
                url TEXT,
                FOREIGN KEY (target_id) REFERENCES monitoring_targets(target_id)
            )
            """
            )

            # Alerts table
            cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS darkweb_alerts (
                alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
                target_id TEXT,
                content_id TEXT,
                alert_date TIMESTAMP,
                score REAL,
                processed INTEGER DEFAULT 0,
                notes TEXT,
                FOREIGN KEY (target_id) REFERENCES monitoring_targets(target_id),
                FOREIGN KEY (content_id) REFERENCES darkweb_content(content_id)
            )
            """
            )

            conn.commit()
            conn.close()

            logger.info("Darkweb intelligence database initialized")

        except Exception as e:
            logger.error(f"Error initializing database: {e}")

    def is_onion_url(self, url: str) -> bool:
        """
        Check if a URL is a valid onion address

        # Function checks subject URL
        # Method validates predicate address
        # Operation verifies object format

        Args:
            url: URL to check

        Returns:
            True if URL is a valid onion address, False otherwise
        """
        try:
            parsed = urlparse(url)
            host = parsed.netloc.lower()

            # Check for .onion TLD
            if host.endswith(".onion"):
                # Check structure for v3 onion addresses
                # V3 addresses are 56 characters (including .onion)
                onion_part = (
                    host.split(".")[-2]
                    if len(host.split(".")) > 1
                    else host.split(".")[0]
                )
                return (
                    len(onion_part) == 56 - 6 or len(onion_part) == 16
                )  # v3 or v2

            return False

        except Exception:
            return False

    def extract_content(
        self, url: str, force_refresh: bool = False
    ) -> Optional[DarkwebContent]:
        """
        Extract content from a darkweb URL

        # Function extracts subject content
        # Method retrieves predicate data
        # Operation processes object page

        Args:
            url: Darkweb URL to extract content from
            force_refresh: Whether to force refresh cached content

        Returns:
            DarkwebContent object containing extracted content or None if extraction fails
        """
        # Validate URL
        if not self.is_onion_url(url):
            logger.warning(f"Not a valid onion URL: {url}")
            return None

        # Generate content ID from URL
        content_id = hashlib.sha256(url.encode()).hexdigest()

        # Check cache first if not forcing refresh
        if not force_refresh:
            cache_path = os.path.join(self.cache_dir, f"{content_id}.json")
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                        # Convert date strings back to datetime objects
                        if "extracted_date" in data:
                            data["extracted_date"] = datetime.fromisoformat(
                                data["extracted_date"]
                            )
                        if "last_modified" in data and data["last_modified"]:
                            data["last_modified"] = datetime.fromisoformat(
                                data["last_modified"]
                            )

                        return DarkwebContent(**data)

                except Exception as e:
                    logger.warning(f"Error loading cached content: {e}")

        # Check if TOR proxy is configured
        if not self.proxy_url:
            logger.error("TOR proxy not configured. Cannot access .onion URLs.")
            return None

        try:
            # Set up request headers
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; rv:91.0) Gecko/20100101 Firefox/91.0",
                "Accept": "text/html,application/xhtml+xml,application/xml",
                "Accept-Language": "en-US,en;q=0.5",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
            }

            # Make request through TOR proxy
            response = self.session.get(url, headers=headers, timeout=60)

            # Check if request was successful
            if response.status_code != 200:
                logger.warning(
                    f"Failed to retrieve content: HTTP {response.status_code}"
                )
                return None

            # Extract HTML content
            html_content = response.text

            # Create content object
            content = DarkwebContent(
                url=url,
                html=html_content,
                content_hash=hashlib.md5(html_content.encode()).hexdigest(),
                extracted_date=datetime.now(),
            )

            # Extract title
            title_match = re.search(
                r"<title>(.*?)</title>", html_content, re.IGNORECASE | re.DOTALL
            )
            if title_match:
                content.title = title_match.group(1).strip()

            # Extract main content text
            # This is a simplified approach - in production, use a proper HTML parser
            body_text = re.sub(
                r"<[^>]+>", " ", html_content
            )  # Remove HTML tags
            body_text = re.sub(
                r"\s+", " ", body_text
            ).strip()  # Normalize whitespace
            content.content = body_text

            # Determine site type
            content.site_type = self._determine_site_type(
                html_content, content.title or ""
            )

            # Extract marketplace data if applicable
            if content.site_type == "marketplace":
                content.marketplace_data = self._extract_marketplace_data(
                    html_content
                )

            # Extract forum data if applicable
            elif content.site_type == "forum":
                content.forum_data = self._extract_forum_data(html_content)

            # Save to cache
            self._save_content_to_cache(content)

            # Save to database
            self._save_to_database(content)

            return content

        except requests.RequestException as e:
            logger.error(f"Request error for {url}: {e}")
            return None

        except Exception as e:
            logger.error(f"Error extracting content from {url}: {e}")
            traceback.print_exc()
            return None

    def _determine_site_type(self, html_content: str, title: str) -> str:
        """
        Determine the type of darkweb site

        # Function determines subject type
        # Method classifies predicate site
        # Operation categorizes object content

        Args:
            html_content: HTML content of the site
            title: Site title

        Returns:
            Site type ('marketplace', 'forum', 'blog', 'service', or 'unknown')
        """
        html_lower = html_content.lower()
        title_lower = title.lower()

        # Count indicators for each site type
        type_scores = {}

        for site_type, indicators in self.site_type_indicators.items():
            score = 0
            for indicator in indicators:
                if indicator in title_lower:
                    score += 2  # Title matches are more significant

                # Count occurrences in HTML
                score += html_lower.count(indicator) * 0.1

            type_scores[site_type] = score

        # Get the site type with the highest score
        if type_scores:
            max_type = max(type_scores.items(), key=lambda x: x[1])
            if max_type[1] > 0:
                return max_type[0]

        return "unknown"

    def _extract_marketplace_data(self, html_content: str) -> Dict[str, Any]:
        """
        Extract marketplace-specific data

        # Function extracts subject marketplace
        # Method retrieves predicate product
        # Operation obtains object listings

        Args:
            html_content: HTML content of the marketplace

        Returns:
            Dictionary containing marketplace data
        """
        marketplace_data = {
            "products": [],
            "vendors": [],
            "categories": [],
            "payment_methods": [],
        }

        # Extract cryptocurrency mentions
        crypto_patterns = {
            "bitcoin": r"\b(bitcoin|btc)\b",
            "monero": r"\b(monero|xmr)\b",
            "ethereum": r"\b(ethereum|eth)\b",
            "litecoin": r"\b(litecoin|ltc)\b",
        }

        payment_methods = []
        for crypto, pattern in crypto_patterns.items():
            if re.search(pattern, html_content, re.IGNORECASE):
                payment_methods.append(crypto)

        marketplace_data["payment_methods"] = payment_methods

        # This is a placeholder for more sophisticated extraction
        # In production, use specific parsers for known marketplace structures
        logger.info("Marketplace data extraction not fully implemented")

        return marketplace_data

    def _extract_forum_data(self, html_content: str) -> Dict[str, Any]:
        """
        Extract forum-specific data

        # Function extracts subject forum
        # Method retrieves predicate discussions
        # Operation obtains object threads

        Args:
            html_content: HTML content of the forum

        Returns:
            Dictionary containing forum data
        """
        forum_data = {"topics": [], "users": [], "posts": [], "categories": []}

        # This is a placeholder for more sophisticated extraction
        # In production, use specific parsers for known forum structures
        logger.info("Forum data extraction not fully implemented")

        return forum_data

    def _save_content_to_cache(self, content: DarkwebContent) -> None:
        """
        Save darkweb content to cache

        # Function saves subject content
        # Method stores predicate data
        # Operation writes object cache

        Args:
            content: DarkwebContent object to save
        """
        try:
            # Convert to serializable dict
            content_dict = content.__dict__.copy()

            # Convert datetime objects to ISO format strings
            if content.extracted_date:
                content_dict["extracted_date"] = (
                    content.extracted_date.isoformat()
                )
            if content.last_modified:
                content_dict["last_modified"] = (
                    content.last_modified.isoformat()
                )

            # Save to cache
            content_id = hashlib.sha256(content.url.encode()).hexdigest()
            cache_path = os.path.join(self.cache_dir, f"{content_id}.json")

            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(content_dict, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"Error saving content to cache: {e}")

    def _save_to_database(self, content: DarkwebContent) -> None:
        """
        Save darkweb content to database

        # Function saves subject content
        # Method stores predicate data
        # Operation writes object database

        Args:
            content: DarkwebContent object to save
        """
        try:
            content_id = hashlib.sha256(content.url.encode()).hexdigest()

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Check if site exists in database
            cursor.execute(
                "SELECT url FROM darkweb_sites WHERE url = ?", (content.url,)
            )
            site_exists = cursor.fetchone()

            # Current timestamp
            now = datetime.now().isoformat()

            # Insert or update site information
            if site_exists:
                cursor.execute(
                    """
                UPDATE darkweb_sites
                SET title = ?, site_type = ?, last_checked = ?, status = ?
                WHERE url = ?
                """,
                    (
                        content.title,
                        content.site_type,
                        now,
                        "active",
                        content.url,
                    ),
                )
            else:
                cursor.execute(
                    """
                INSERT INTO darkweb_sites
                (url, title, site_type, first_discovered, last_checked, status)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        content.url,
                        content.title,
                        content.site_type,
                        now,
                        now,
                        "active",
                    ),
                )

            # Cache path
            cache_path = os.path.join(self.cache_dir, f"{content_id}.json")

            # Insert content information
            cursor.execute(
                """
            INSERT OR REPLACE INTO darkweb_content
            (content_id, url, title, extraction_date, content_hash, language, content_type, cache_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    content_id,
                    content.url,
                    content.title,
                    content.extracted_date.isoformat(),
                    content.content_hash,
                    content.language,
                    content.site_type,
                    cache_path,
                ),
            )

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Error saving to database: {e}")

    def add_monitoring_target(self, target: DarkwebMonitoringTarget) -> bool:
        """
        Add a new darkweb monitoring target

        # Function adds subject target
        # Method creates predicate monitoring
        # Operation sets object configuration

        Args:
            target: DarkwebMonitoringTarget object to add

        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate target ID
            target_id = hashlib.md5(target.name.encode()).hexdigest()

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Insert target information
            cursor.execute(
                """
            INSERT OR REPLACE INTO monitoring_targets
            (target_id, name, created_date, is_active, monitoring_frequency, last_checked, threshold_score, notification_enabled)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    target_id,
                    target.name,
                    datetime.now().isoformat(),
                    1 if target.is_active else 0,
                    target.monitoring_frequency,
                    (
                        target.last_checked.isoformat()
                        if target.last_checked
                        else None
                    ),
                    target.threshold_score,
                    1 if target.notification_enabled else 0,
                ),
            )

            # Insert keywords
            for keyword in target.keywords:
                cursor.execute(
                    """
                INSERT INTO monitoring_keywords (target_id, keyword)
                VALUES (?, ?)
                """,
                    (target_id, keyword),
                )

            # Insert URLs
            for url in target.urls:
                cursor.execute(
                    """
                INSERT INTO monitoring_urls (target_id, url)
                VALUES (?, ?)
                """,
                    (target_id, url),
                )

            conn.commit()
            conn.close()

            logger.info(f"Added monitoring target: {target.name}")
            return True

        except Exception as e:
            logger.error(f"Error adding monitoring target: {e}")
            return False

    def get_monitoring_targets(self) -> List[DarkwebMonitoringTarget]:
        """
        Get all monitoring targets

        # Function gets subject targets
        # Method retrieves predicate monitoring
        # Operation obtains object configurations

        Returns:
            List of DarkwebMonitoringTarget objects
        """
        targets = []

        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Get all targets
            cursor.execute("SELECT * FROM monitoring_targets")
            target_rows = cursor.fetchall()

            for row in target_rows:
                target_id = row["target_id"]

                # Get keywords for this target
                cursor.execute(
                    "SELECT keyword FROM monitoring_keywords WHERE target_id = ?",
                    (target_id,),
                )
                keywords = [r["keyword"] for r in cursor.fetchall()]

                # Get URLs for this target
                cursor.execute(
                    "SELECT url FROM monitoring_urls WHERE target_id = ?",
                    (target_id,),
                )
                urls = [r["url"] for r in cursor.fetchall()]

                # Create target object
                target = DarkwebMonitoringTarget(
                    name=row["name"],
                    keywords=keywords,
                    urls=urls,
                    monitoring_frequency=row["monitoring_frequency"],
                    is_active=bool(row["is_active"]),
                    threshold_score=row["threshold_score"],
                    notification_enabled=bool(row["notification_enabled"]),
                )

                # Add last_checked if it exists
                if row["last_checked"]:
                    target.last_checked = datetime.fromisoformat(
                        row["last_checked"]
                    )

                targets.append(target)

            conn.close()

        except Exception as e:
            logger.error(f"Error getting monitoring targets: {e}")

        return targets

    def scan_for_keywords(
        self, content: DarkwebContent, keywords: List[str]
    ) -> Dict[str, List[int]]:
        """
        Scan content for keywords and return occurrences

        # Function scans subject content
        # Method finds predicate keywords
        # Operation identifies object occurrences

        Args:
            content: DarkwebContent object to scan
            keywords: List of keywords to scan for

        Returns:
            Dictionary of keyword to list of occurrence positions
        """
        results = {}

        if not content.content:
            return results

        text = content.content.lower()

        for keyword in keywords:
            keyword_lower = keyword.lower()
            positions = []

            # Find all occurrences
            start = 0
            while True:
                start = text.find(keyword_lower, start)
                if start == -1:
                    break
                positions.append(start)
                start += len(keyword_lower)

            if positions:
                results[keyword] = positions

        return results

    def monitor_targets(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Monitor all active targets and return alerts

        # Function monitors subject targets
        # Method checks predicate content
        # Operation generates object alerts

        Returns:
            Dictionary of target name to list of alerts
        """
        alerts = {}

        # Get all active monitoring targets
        targets = self.get_monitoring_targets()
        active_targets = [t for t in targets if t.is_active]

        for target in active_targets:
            target_alerts = []

            # Check if it's time to monitor this target
            current_time = datetime.now()
            if (
                target.last_checked
                and (current_time - target.last_checked).total_seconds()
                < target.monitoring_frequency * 3600
            ):
                continue

            logger.info(f"Monitoring target: {target.name}")

            # Check specific URLs for this target
            for url in target.urls:
                # Extract content from URL
                content = self.extract_content(url)
                if not content:
                    continue

                # Scan content for keywords
                keyword_occurrences = self.scan_for_keywords(
                    content, target.keywords
                )

                # If any keywords found, create an alert
                if keyword_occurrences:
                    # Calculate alert score based on keyword occurrences
                    total_occurrences = sum(
                        len(positions)
                        for positions in keyword_occurrences.values()
                    )
                    score = min(
                        1.0, total_occurrences / 10
                    )  # Normalize score, max at 10 occurrences

                    # Only create alert if score exceeds threshold
                    if score >= target.threshold_score:
                        alert = {
                            "url": url,
                            "title": content.title,
                            "score": score,
                            "keywords": keyword_occurrences,
                            "content_id": hashlib.sha256(
                                url.encode()
                            ).hexdigest(),
                            "alert_date": current_time.isoformat(),
                        }
                        target_alerts.append(alert)

                        # Save alert to database
                        self._save_alert(target.name, alert)

            if target_alerts:
                alerts[target.name] = target_alerts

            # Update last checked time
            self._update_target_last_checked(target.name, current_time)

        return alerts

    def _save_alert(self, target_name: str, alert: Dict[str, Any]) -> None:
        """
        Save an alert to the database

        # Function saves subject alert
        # Method stores predicate notification
        # Operation writes object record

        Args:
            target_name: Name of the monitoring target
            alert: Alert information
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get target ID
            cursor.execute(
                "SELECT target_id FROM monitoring_targets WHERE name = ?",
                (target_name,),
            )
            result = cursor.fetchone()
            if not result:
                logger.warning(f"Target not found: {target_name}")
                return

            target_id = result[0]

            # Insert alert
            cursor.execute(
                """
            INSERT INTO darkweb_alerts
            (target_id, content_id, alert_date, score, notes)
            VALUES (?, ?, ?, ?, ?)
            """,
                (
                    target_id,
                    alert["content_id"],
                    alert["alert_date"],
                    alert["score"],
                    json.dumps({"keywords": alert["keywords"]}),
                ),
            )

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Error saving alert: {e}")

    def _update_target_last_checked(
        self, target_name: str, timestamp: datetime
    ) -> None:
        """
        Update the last checked timestamp for a target

        # Function updates subject timestamp
        # Method modifies predicate record
        # Operation changes object value

        Args:
            target_name: Name of the monitoring target
            timestamp: New last checked timestamp
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
            UPDATE monitoring_targets
            SET last_checked = ?
            WHERE name = ?
            """,
                (timestamp.isoformat(), target_name),
            )

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Error updating target last checked time: {e}")

    def get_alerts(
        self,
        target_name: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get alerts from the database with filtering options

        # Function gets subject alerts
        # Method retrieves predicate notifications
        # Operation obtains object records

        Args:
            target_name: Filter by target name
            start_date: Filter by start date
            end_date: Filter by end date

        Returns:
            List of alert records
        """
        alerts = []

        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Build query with filters
            query = """
            SELECT a.*, t.name as target_name, c.url, c.title
            FROM darkweb_alerts a
            JOIN monitoring_targets t ON a.target_id = t.target_id
            JOIN darkweb_content c ON a.content_id = c.content_id
            WHERE 1=1
            """
            params = []

            if target_name:
                query += " AND t.name = ?"
                params.append(target_name)

            if start_date:
                query += " AND a.alert_date >= ?"
                params.append(start_date.isoformat())

            if end_date:
                query += " AND a.alert_date <= ?"
                params.append(end_date.isoformat())

            query += " ORDER BY a.alert_date DESC"

            # Execute query
            cursor.execute(query, params)
            rows = cursor.fetchall()

            for row in rows:
                alert = dict(row)

                # Parse notes JSON
                if alert["notes"]:
                    try:
                        alert["notes"] = json.loads(alert["notes"])
                    except:
                        pass

                # Convert alert date to datetime
                if alert["alert_date"]:
                    alert["alert_date"] = datetime.fromisoformat(
                        alert["alert_date"]
                    )

                alerts.append(alert)

            conn.close()

        except Exception as e:
            logger.error(f"Error getting alerts: {e}")

        return alerts

    def search_darkweb_content(self, search_term: str) -> List[Dict[str, Any]]:
        """
        Search for darkweb content matching a search term

        # Function searches subject content
        # Method finds predicate matches
        # Operation retrieves object results

        Args:
            search_term: Term to search for

        Returns:
            List of content matches
        """
        results = []

        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # This is a simplified search - in production, use a full-text search engine
            cursor.execute(
                """
            SELECT c.*, s.site_type
            FROM darkweb_content c
            JOIN darkweb_sites s ON c.url = s.url
            WHERE c.title LIKE ? OR c.cache_path LIKE ?
            ORDER BY c.extraction_date DESC
            LIMIT 100
            """,
                (f"%{search_term}%", f"%{search_term}%"),
            )

            rows = cursor.fetchall()

            for row in rows:
                result = dict(row)

                # Check if content contains search term
                content_id = result["content_id"]
                cache_path = result["cache_path"]

                if os.path.exists(cache_path):
                    try:
                        with open(cache_path, "r", encoding="utf-8") as f:
                            content_data = json.load(f)

                            # Check if content contains search term
                            if (
                                "content" in content_data
                                and content_data["content"]
                            ):
                                if (
                                    search_term.lower()
                                    in content_data["content"].lower()
                                ):
                                    result["matched_in_content"] = True

                                    # Add snippet with context
                                    content_lower = content_data[
                                        "content"
                                    ].lower()
                                    search_term_lower = search_term.lower()
                                    pos = content_lower.find(search_term_lower)

                                    if pos >= 0:
                                        start = max(0, pos - 100)
                                        end = min(
                                            len(content_data["content"]),
                                            pos + len(search_term) + 100,
                                        )
                                        result["snippet"] = (
                                            "..."
                                            + content_data["content"][start:end]
                                            + "..."
                                        )
                    except:
                        pass

                results.append(result)

            conn.close()

        except Exception as e:
            logger.error(f"Error searching darkweb content: {e}")

        return results


def setup_tor_connection(tor_port: int = 9050) -> Optional[str]:
    """
    Set up connection to TOR network

    # Function sets subject connection
    # Method configures predicate TOR
    # Operation establishes object proxy

    Args:
        tor_port: Port for TOR SOCKS proxy

    Returns:
        Proxy URL if successful, None otherwise
    """
    # Check if TOR is running
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(("127.0.0.1", tor_port))
        sock.close()

        if result == 0:
            # TOR is running
            proxy_url = f"socks5h://127.0.0.1:{tor_port}"

            # Verify connection through TOR
            session = requests.Session()
            session.proxies = {"http": proxy_url, "https": proxy_url}

            # Try to access check.torproject.org
            response = session.get("https://check.torproject.org/", timeout=30)
            if "Congratulations" in response.text:
                logger.info("Successfully connected to TOR network")
                return proxy_url
            else:
                logger.warning("TOR is running but connection check failed")
                return None
        else:
            logger.warning(f"TOR proxy not running on port {tor_port}")
            return None

    except Exception as e:
        logger.error(f"Error setting up TOR connection: {e}")
        return None


if __name__ == "__main__":
    pass  # Add test code here if needed
