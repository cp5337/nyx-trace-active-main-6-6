"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-WEBINT-NEWS-SCRAPER-0001            │
// │ 📁 domain       : Web Intelligence, OSINT                   │
// │ 🧠 description  : Advanced news organization scraper for    │
// │                  extracting and analyzing news content      │
// │ 🕸️ hash_type    : UUID → CUID-linked collector              │
// │ 🔄 parent_node  : NODE_COLLECTOR                           │
// │ 🧩 dependencies : requests, trafilatura, bs4, pandas        │
// │ 🔧 tool_usage   : Collection, Analysis, Intelligence        │
// │ 📡 input_type   : URLs, domains, search terms               │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : content extraction, pattern analysis      │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

Advanced News Organization Scraper
---------------------------------
Professional module for extracting, processing, and analyzing content
from news organizations worldwide. Supports article extraction, metadata
parsing, and entity recognition across multiple languages and formats.
"""

import os
import re
import json
import time
import logging
import hashlib
import requests
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union, Set
from datetime import datetime
from urllib.parse import urlparse, urljoin
import concurrent.futures
import traceback
from dataclasses import dataclass, field
from bs4 import BeautifulSoup
import trafilatura
from trafilatura.settings import use_config

# Function creates subject logger
# Method initializes predicate output
# Operation configures object format
logger = logging.getLogger("ctas_news_scraper")
logger.setLevel(logging.INFO)

# Configure trafilatura
config = use_config()
config.set("DEFAULT", "EXTRACTION_TIMEOUT", "20")


@dataclass
class NewsArticle:
    """
    Data class representing a news article

    # Class represents subject article
    # Structure stores predicate content
    # Container holds object metadata
    """

    url: str
    title: Optional[str] = None
    text: Optional[str] = None
    html: Optional[str] = None
    author: Optional[str] = None
    date_published: Optional[datetime] = None
    date_modified: Optional[datetime] = None
    categories: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    image_urls: List[str] = field(default_factory=list)
    main_image_url: Optional[str] = None
    language: Optional[str] = None
    domain: Optional[str] = None
    summary: Optional[str] = None
    word_count: int = 0
    source_name: Optional[str] = None
    extracted_date: datetime = field(default_factory=datetime.now)
    article_id: Optional[str] = None


@dataclass
class NewsSource:
    """
    Data class representing a news source

    # Class represents subject source
    # Structure stores predicate outlet
    # Container holds object metadata
    """

    domain: str
    name: Optional[str] = None
    url: Optional[str] = None
    rss_feeds: List[str] = field(default_factory=list)
    language: Optional[str] = None
    country: Optional[str] = None
    article_selector: Optional[str] = None  # CSS selector for article links
    categories: List[str] = field(default_factory=list)
    reliability_score: Optional[float] = None  # 0-1 score
    bias_rating: Optional[str] = None  # e.g., "left", "center", "right"
    fact_check_rating: Optional[str] = None
    is_verified: bool = False


class NewsScraper:
    """
    Advanced news organization web scraper for OSINT intelligence gathering

    # Class scrapes subject news
    # Scraper collects predicate articles
    # Engine extracts object content
    """

    def __init__(
        self,
        cache_dir: str = "data/news_cache",
        news_db_path: str = "data/news_sources/news_organizations.csv",
        user_agent: Optional[str] = None,
    ):
        """
        Initialize the news scraper

        # Function initializes subject scraper
        # Method configures predicate settings
        # Operation sets object parameters

        Args:
            cache_dir: Directory to cache scraped articles
            news_db_path: Path to database of news sources
            user_agent: Custom user agent string for requests
        """
        self.cache_dir = cache_dir
        self.news_db_path = news_db_path

        # Set up user agent
        self.user_agent = (
            user_agent
            or "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )

        # Create directory structure if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(os.path.dirname(news_db_path), exist_ok=True)

        # Load news sources database
        self.news_sources = self._load_news_sources()

        logger.info(
            f"Initialized NewsScraper with {len(self.news_sources)} news sources"
        )

    def _load_news_sources(self) -> Dict[str, NewsSource]:
        """
        Load news sources from database

        # Function loads subject sources
        # Method reads predicate database
        # Operation retrieves object outlets

        Returns:
            Dictionary of domain to NewsSource objects
        """
        sources = {}

        # Check if news sources database exists
        if not os.path.exists(self.news_db_path):
            logger.warning(
                f"News sources database not found at {self.news_db_path}"
            )
            return sources

        try:
            # Load from CSV
            df = pd.read_csv(self.news_db_path)

            for _, row in df.iterrows():
                domain = row.get("domain", "")
                if not domain:
                    continue

                # Create NewsSource object
                source = NewsSource(
                    domain=domain,
                    name=row.get("name"),
                    url=row.get("url", f"https://{domain}"),
                    language=row.get("language"),
                    country=row.get("country"),
                    is_verified=bool(row.get("is_verified", False)),
                )

                # Add RSS feeds if available
                rss_feeds = row.get("rss_feeds")
                if rss_feeds and isinstance(rss_feeds, str):
                    source.rss_feeds = [
                        feed.strip() for feed in rss_feeds.split(",")
                    ]

                # Add article selector if available
                selector = row.get("article_selector")
                if selector and isinstance(selector, str):
                    source.article_selector = selector

                # Add categories if available
                categories = row.get("categories")
                if categories and isinstance(categories, str):
                    source.categories = [
                        c.strip() for c in categories.split(",")
                    ]

                # Add reliability score if available
                reliability = row.get("reliability_score")
                if reliability and not pd.isna(reliability):
                    source.reliability_score = float(reliability)

                # Add bias rating if available
                bias = row.get("bias_rating")
                if bias and isinstance(bias, str):
                    source.bias_rating = bias

                # Add fact check rating if available
                fact_check = row.get("fact_check_rating")
                if fact_check and isinstance(fact_check, str):
                    source.fact_check_rating = fact_check

                sources[domain] = source

            logger.info(f"Loaded {len(sources)} news sources from database")

        except Exception as e:
            logger.error(f"Error loading news sources: {e}")

        return sources

    def is_news_source(self, url: str) -> bool:
        """
        Check if a URL belongs to a known news source

        # Function checks subject URL
        # Method verifies predicate domain
        # Operation validates object source

        Args:
            url: URL to check

        Returns:
            True if URL is from a known news source, False otherwise
        """
        domain = self._extract_domain(url)
        return domain in self.news_sources

    def _extract_domain(self, url: str) -> str:
        """
        Extract domain from URL

        # Function extracts subject domain
        # Method parses predicate URL
        # Operation retrieves object hostname

        Args:
            url: URL to extract domain from

        Returns:
            Domain without subdomain
        """
        try:
            parsed = urlparse(url)
            hostname = parsed.netloc.lower()

            # Remove www. if present
            if hostname.startswith("www."):
                hostname = hostname[4:]

            # Get the main domain (last two parts)
            parts = hostname.split(".")
            if len(parts) > 2:
                # Handle country-specific domains (e.g., co.uk)
                country_tlds = {"co.uk", "com.au", "co.nz", "co.jp"}
                if ".".join(parts[-2:]) in country_tlds:
                    return ".".join(parts[-3:])
                return ".".join(parts[-2:])
            return hostname

        except Exception as e:
            logger.error(f"Error extracting domain from {url}: {e}")
            return ""

    def get_news_source(self, url: str) -> Optional[NewsSource]:
        """
        Get the news source for a URL

        # Function gets subject source
        # Method retrieves predicate outlet
        # Operation finds object organization

        Args:
            url: URL to get news source for

        Returns:
            NewsSource object if found, None otherwise
        """
        domain = self._extract_domain(url)
        return self.news_sources.get(domain)

    def extract_article_content(
        self, url: str, max_retries: int = 3
    ) -> Optional[NewsArticle]:
        """
        Extract article content from a URL

        # Function extracts subject content
        # Method retrieves predicate article
        # Operation obtains object text

        Args:
            url: URL of the article
            max_retries: Maximum number of retries for failed requests

        Returns:
            NewsArticle object with extracted content or None if extraction fails
        """
        # Generate article ID from URL
        article_id = hashlib.md5(url.encode()).hexdigest()

        # Check cache first
        cache_path = os.path.join(self.cache_dir, f"{article_id}.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    article = NewsArticle(**data)

                    # Convert date strings back to datetime objects
                    if data.get("date_published"):
                        article.date_published = datetime.fromisoformat(
                            data["date_published"]
                        )
                    if data.get("date_modified"):
                        article.date_modified = datetime.fromisoformat(
                            data["date_modified"]
                        )
                    if data.get("extracted_date"):
                        article.extracted_date = datetime.fromisoformat(
                            data["extracted_date"]
                        )

                    logger.info(f"Loaded article from cache: {url}")
                    return article
            except Exception as e:
                logger.warning(f"Error loading article from cache: {e}")

        # Extract domain and get news source info
        domain = self._extract_domain(url)
        source = self.news_sources.get(domain)
        source_name = source.name if source else domain

        # Initialize article object
        article = NewsArticle(
            url=url,
            domain=domain,
            source_name=source_name,
            article_id=article_id,
        )

        # Set up headers
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
            "Referer": f"https://{domain}",
        }

        # Try to download and extract content
        retries = 0
        while retries < max_retries:
            try:
                # Download content
                response = requests.get(url, headers=headers, timeout=20)
                response.raise_for_status()

                # Get HTML content
                html_content = response.text
                article.html = html_content

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
                    article.title = data.get("title")
                    article.text = data.get("text")
                    article.author = data.get("author")

                    # Parse dates
                    if "date" in data and data["date"]:
                        try:
                            article.date_published = datetime.fromisoformat(
                                data["date"]
                            )
                        except (ValueError, TypeError):
                            pass

                    # Get tags/categories
                    if "categories" in data and data["categories"]:
                        article.categories = [
                            c.strip() for c in data["categories"].split(",")
                        ]

                    # Set language
                    article.language = data.get("language")

                    # Count words
                    if article.text:
                        article.word_count = len(article.text.split())

                    # Extract images using BeautifulSoup if trafilatura didn't get them
                    soup = BeautifulSoup(html_content, "html.parser")

                    # Try to find the main image (usually og:image or similar)
                    og_image = soup.find("meta", property="og:image")
                    if og_image and og_image.get("content"):
                        article.main_image_url = og_image["content"]
                        if (
                            article.main_image_url
                            and not article.main_image_url.startswith(
                                ("http://", "https://")
                            )
                        ):
                            article.main_image_url = urljoin(
                                url, article.main_image_url
                            )
                        article.image_urls.append(article.main_image_url)

                    # Find all images in content
                    for img in soup.find_all("img"):
                        if img.get("src"):
                            img_url = img["src"]
                            if not img_url.startswith(("http://", "https://")):
                                img_url = urljoin(url, img_url)
                            if img_url not in article.image_urls:
                                article.image_urls.append(img_url)

                    # Save to cache
                    self._save_article_to_cache(article)

                    return article
                else:
                    # Fallback to BeautifulSoup if trafilatura fails
                    soup = BeautifulSoup(html_content, "html.parser")

                    # Extract title
                    if not article.title:
                        title_tag = soup.find("title")
                        if title_tag:
                            article.title = title_tag.text.strip()

                    # Extract content - this is a simple fallback, not as good as trafilatura
                    content = []
                    for paragraph in soup.find_all("p"):
                        text = paragraph.text.strip()
                        if (
                            text and len(text) > 20
                        ):  # Filter out short paragraphs
                            content.append(text)

                    if content:
                        article.text = "\n\n".join(content)
                        article.word_count = len(article.text.split())

                        # Save to cache
                        self._save_article_to_cache(article)

                        return article

                    logger.warning(f"Failed to extract content from {url}")
                    return None

            except requests.RequestException as e:
                retries += 1
                logger.warning(
                    f"Request error ({retries}/{max_retries}) for {url}: {e}"
                )
                time.sleep(2 * retries)  # Exponential backoff

            except Exception as e:
                logger.error(f"Error extracting content from {url}: {e}")
                traceback.print_exc()
                return None

        logger.error(
            f"Failed to extract content after {max_retries} retries: {url}"
        )
        return None

    def _save_article_to_cache(self, article: NewsArticle) -> None:
        """
        Save article to cache

        # Function saves subject article
        # Method stores predicate data
        # Operation writes object cache

        Args:
            article: NewsArticle object to save
        """
        try:
            # Convert to serializable dict
            article_dict = article.__dict__.copy()

            # Convert datetime objects to ISO format strings
            if article.date_published:
                article_dict["date_published"] = (
                    article.date_published.isoformat()
                )
            if article.date_modified:
                article_dict["date_modified"] = (
                    article.date_modified.isoformat()
                )
            if article.extracted_date:
                article_dict["extracted_date"] = (
                    article.extracted_date.isoformat()
                )

            # Save to cache
            cache_path = os.path.join(
                self.cache_dir, f"{article.article_id}.json"
            )
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(article_dict, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"Error saving article to cache: {e}")

    def discover_articles(
        self, source_url: str, max_articles: int = 20
    ) -> List[str]:
        """
        Discover article URLs from a news source homepage

        # Function discovers subject articles
        # Method finds predicate URLs
        # Operation retrieves object links

        Args:
            source_url: URL of the news source homepage
            max_articles: Maximum number of articles to discover

        Returns:
            List of discovered article URLs
        """
        discovered_urls = []

        try:
            # Get domain and news source info
            domain = self._extract_domain(source_url)
            source = self.news_sources.get(domain)

            # Set up headers
            headers = {
                "User-Agent": self.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml",
                "Accept-Language": "en-US,en;q=0.9",
                "Connection": "keep-alive",
                "Referer": "https://www.google.com/",
            }

            # Download the page
            response = requests.get(source_url, headers=headers, timeout=20)
            response.raise_for_status()

            # Parse HTML
            soup = BeautifulSoup(response.text, "html.parser")

            # Find all links
            for a_tag in soup.find_all("a", href=True):
                href = a_tag["href"]

                # Normalize URL
                if not href.startswith(("http://", "https://")):
                    href = urljoin(source_url, href)

                # Skip non-article links
                if not self._is_likely_article(href, domain):
                    continue

                # Add to discovered URLs if not already present
                if href not in discovered_urls:
                    discovered_urls.append(href)

                # Stop if we have enough articles
                if len(discovered_urls) >= max_articles:
                    break

            logger.info(
                f"Discovered {len(discovered_urls)} articles from {source_url}"
            )

        except Exception as e:
            logger.error(f"Error discovering articles from {source_url}: {e}")

        return discovered_urls[:max_articles]

    def _is_likely_article(self, url: str, domain: str) -> bool:
        """
        Check if a URL is likely to be an article

        # Function checks subject URL
        # Method verifies predicate article
        # Operation validates object content

        Args:
            url: URL to check
            domain: Domain of the news source

        Returns:
            True if URL is likely an article, False otherwise
        """
        # Must contain the domain
        if domain not in url:
            return False

        # Common article URL patterns
        article_patterns = [
            r"/\d{4}/\d{2}/\d{2}/",  # Date pattern (YYYY/MM/DD)
            r"/article/",
            r"/story/",
            r"/news/",
            r"/politics/",
            r"/sports/",
            r"/business/",
            r"/opinion/",
            r"/technology/",
            r"/science/",
            r"/health/",
            r"/world/",
            r"/national/",
            r"/lifestyle/",
            r"/entertainment/",
        ]

        # Check if URL matches any article pattern
        for pattern in article_patterns:
            if re.search(pattern, url):
                return True

        return False

    def batch_extract_articles(
        self, urls: List[str], max_workers: int = 5
    ) -> Dict[str, NewsArticle]:
        """
        Extract content from multiple article URLs in parallel

        # Function extracts subject batch
        # Method processes predicate articles
        # Operation retrieves object content

        Args:
            urls: List of article URLs to extract
            max_workers: Maximum number of parallel workers

        Returns:
            Dictionary of URL to NewsArticle objects
        """
        results = {}

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers
        ) as executor:
            # Submit all extraction tasks
            future_to_url = {
                executor.submit(self.extract_article_content, url): url
                for url in urls
            }

            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    article = future.result()
                    if article:
                        results[url] = article
                except Exception as e:
                    logger.error(f"Error extracting article {url}: {e}")

        logger.info(
            f"Extracted {len(results)} articles out of {len(urls)} URLs"
        )
        return results

    def analyze_news_source(
        self, url: str, article_count: int = 10
    ) -> Dict[str, Any]:
        """
        Analyze a news source by extracting and analyzing multiple articles

        # Function analyzes subject source
        # Method examines predicate outlet
        # Operation evaluates object organization

        Args:
            url: URL of the news source
            article_count: Number of articles to analyze

        Returns:
            Dictionary containing analysis results
        """
        results = {
            "source_url": url,
            "domain": self._extract_domain(url),
            "articles_analyzed": 0,
            "total_words": 0,
            "avg_article_length": 0,
            "common_categories": [],
            "common_topics": [],
            "article_urls": [],
            "language": None,
        }

        try:
            # Discover articles
            article_urls = self.discover_articles(
                url, max_articles=article_count
            )
            results["article_urls"] = article_urls

            # Extract articles
            articles = self.batch_extract_articles(article_urls)
            results["articles_analyzed"] = len(articles)

            if articles:
                # Calculate total words and average article length
                total_words = sum(
                    article.word_count for article in articles.values()
                )
                results["total_words"] = total_words
                results["avg_article_length"] = (
                    total_words / len(articles) if articles else 0
                )

                # Get common language
                languages = {}
                for article in articles.values():
                    if article.language:
                        languages[article.language] = (
                            languages.get(article.language, 0) + 1
                        )
                if languages:
                    results["language"] = max(
                        languages.items(), key=lambda x: x[1]
                    )[0]

                # Get common categories
                all_categories = []
                for article in articles.values():
                    all_categories.extend(article.categories)

                category_counts = {}
                for category in all_categories:
                    category_counts[category] = (
                        category_counts.get(category, 0) + 1
                    )

                # Sort by count and return top categories
                sorted_categories = sorted(
                    category_counts.items(), key=lambda x: x[1], reverse=True
                )
                results["common_categories"] = [
                    category for category, count in sorted_categories[:10]
                ]

            logger.info(
                f"Analyzed news source {url}: {results['articles_analyzed']} articles"
            )

        except Exception as e:
            logger.error(f"Error analyzing news source {url}: {e}")

        return results

    def search_for_articles(
        self,
        search_term: str,
        news_domains: Optional[List[str]] = None,
        max_results: int = 20,
    ) -> List[str]:
        """
        Search for news articles about a specific topic

        # Function searches subject articles
        # Method finds predicate content
        # Operation discovers object results

        Args:
            search_term: Term to search for
            news_domains: List of domains to search within, or None for all
            max_results: Maximum number of results to return

        Returns:
            List of article URLs matching the search term
        """
        # This is a placeholder for a more sophisticated search implementation
        # In a real-world scenario, this would use a search engine API or custom crawler
        logger.info(
            f"Search functionality not fully implemented: {search_term}"
        )
        return []


def create_news_organization_database(
    output_path: str, sources: List[Dict[str, Any]]
) -> bool:
    """
    Create or update a database of news organizations

    # Function creates subject database
    # Method builds predicate dataset
    # Operation stores object sources

    Args:
        output_path: Path to save the database
        sources: List of dictionaries containing news source information

    Returns:
        True if successful, False otherwise
    """
    try:
        df = pd.DataFrame(sources)

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save to CSV
        df.to_csv(output_path, index=False)
        logger.info(
            f"Created news organization database with {len(sources)} sources at {output_path}"
        )
        return True

    except Exception as e:
        logger.error(f"Error creating news organization database: {e}")
        return False
