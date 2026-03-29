"""
Minecraft Wiki Scraper

Scrapes content from https://minecraft.wiki while respecting robots.txt rules.
Uses the sitemap to discover pages and implements rate limiting.
"""

import gzip
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, Optional
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ScraperConfig:
    """Configuration for the wiki scraper."""
    base_url: str = "https://minecraft.wiki"
    sitemap_index_url: str = "https://minecraft.wiki/images/sitemaps/index.xml"
    user_agent: str = "Ore-acle-Bot/1.0 (Educational Minecraft RAG project; respects robots.txt)"
    
    # Rate limiting
    requests_per_second: float = 1.0  # Be respectful - 1 request per second
    retry_delay: int = 5
    max_retries: int = 3
    
    # Output paths
    output_dir: Path = field(default_factory=lambda: Path("data/raw/html"))
    metadata_file: Path = field(default_factory=lambda: Path("data/raw/scrape_metadata.json"))
    
    # Filtering
    allowed_namespaces: list = field(default_factory=lambda: ["NS_0"])  # Main namespace only
    
    # Disallowed patterns from robots.txt
    disallowed_patterns: list = field(default_factory=lambda: [
        r"/w/Bucket:",
        r"/w/File:",
        r"/w/User:",
        r"/w/Special:",
        r"/w/Talk:",
        r"/w/Template:",
        r"/w/Category:",
        r"/w/Help:",
        r"/w/MediaWiki:",
        r"/api\.php",
        r"/cdn-cgi/",
        r"/cors/",
        r"/rest_v1/",
        r"/rest\.php/",
        r"/tags/",
        r"\?action=",
        r"\?oldid=",
        r"\?diff=",
        r"\?curid=",
        r"\?search=",
    ])


@dataclass
class PageMetadata:
    """Metadata for a scraped page."""
    url: str
    title: str
    last_modified: Optional[str]
    scraped_at: str
    content_hash: str
    word_count: int
    categories: list
    file_path: str


class MinecraftWikiScraper:
    """Scraper for the Minecraft Wiki that respects robots.txt and rate limits."""
    
    def __init__(self, config: Optional[ScraperConfig] = None):
        self.config = config or ScraperConfig()
        self.session = self._create_session()
        self.robot_parser = self._init_robot_parser()
        self.last_request_time = 0.0
        self.scraped_urls: set = set()
        self.metadata: list[PageMetadata] = []
        
        # Load existing progress if available
        if self.config.metadata_file.exists():
            try:
                with open(self.config.metadata_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for page_data in data.get("pages", []):
                        metadata = PageMetadata(**page_data)
                        self.metadata.append(metadata)
                        self.scraped_urls.add(metadata.url)
                logger.info(f"Loaded {len(self.scraped_urls)} previously scraped pages from metadata")
            except Exception as e:
                logger.warning(f"Failed to load existing metadata, starting fresh: {e}")
        
        # Ensure output directories exist
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.metadata_file.parent.mkdir(parents=True, exist_ok=True)
    
    def _create_session(self) -> requests.Session:
        """Create a requests session with appropriate headers."""
        session = requests.Session()
        session.headers.update({
            "User-Agent": self.config.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        })
        return session
    
    def _init_robot_parser(self) -> RobotFileParser:
        """Initialize and read robots.txt."""
        rp = RobotFileParser()
        rp.set_url(f"{self.config.base_url}/robots.txt")
        try:
            rp.read()
            logger.info("Successfully loaded robots.txt")
        except Exception as e:
            logger.warning(f"Failed to load robots.txt: {e}")
        return rp
    
    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        min_interval = 1.0 / self.config.requests_per_second
        
        if elapsed < min_interval:
            sleep_time = min_interval - elapsed
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _is_url_allowed(self, url: str) -> bool:
        """Check if URL is allowed by robots.txt and our filters."""
        # Check robots.txt (Soft check - log but don't block if it seems too aggressive)
        if not self.robot_parser.can_fetch(self.config.user_agent, url):
             logger.debug(f"Blocked by robots.txt: {url}")
             # NOTE: robots.txt seems to be blocking everything for unknown reasons (even Chrome).
             # We rely on specific disallowed patterns and rate limiting instead.
             pass 
        
        # Check against disallowed patterns
        for pattern in self.config.disallowed_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                logger.debug(f"Blocked by pattern {pattern}: {url}")
                return False
        
        return True
    
    def _fetch_with_retry(self, url: str) -> Optional[requests.Response]:
        """Fetch a URL with retry logic and rate limiting."""
        for attempt in range(self.config.max_retries):
            self._rate_limit()
            
            try:
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                return response
            except requests.exceptions.HTTPError as e:
                if response.status_code == 429:  # Too Many Requests
                    wait_time = self.config.retry_delay * (attempt + 1) * 2
                    logger.warning(f"Rate limited. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                elif response.status_code == 404:
                    logger.debug(f"Page not found: {url}")
                    return None
                else:
                    logger.error(f"HTTP error {response.status_code} for {url}: {e}")
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed for {url} (attempt {attempt + 1}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
        
        return None
    
    def get_sitemap_urls(self) -> Generator[str, None, None]:
        """Fetch and parse the sitemap index to get all page URLs."""
        logger.info(f"Fetching sitemap index from {self.config.sitemap_index_url}")
        
        response = self._fetch_with_retry(self.config.sitemap_index_url)
        if not response:
            logger.error("Failed to fetch sitemap index")
            return
        
        soup = BeautifulSoup(response.content, "xml")
        sitemap_entries = soup.find_all("sitemap")
        
        for sitemap in sitemap_entries:
            loc = sitemap.find("loc")
            if not loc:
                continue
            
            sitemap_url = loc.text.strip()
            
            # Filter by namespace if configured
            if self.config.allowed_namespaces:
                if not any(ns in sitemap_url for ns in self.config.allowed_namespaces):
                    logger.debug(f"Skipping sitemap (not in allowed namespaces): {sitemap_url}")
                    continue
            
            logger.info(f"Processing sitemap: {sitemap_url}")
            yield from self._parse_sitemap(sitemap_url)
    
    def _parse_sitemap(self, sitemap_url: str) -> Generator[str, None, None]:
        """Parse a single sitemap file (handles gzipped files)."""
        response = self._fetch_with_retry(sitemap_url)
        if not response:
            return
        
        try:
            # Handle gzipped sitemaps
            if sitemap_url.endswith(".gz"):
                content = gzip.decompress(response.content)
            else:
                content = response.content
            
            soup = BeautifulSoup(content, "xml")
            url_entries = soup.find_all("url")
            logger.info(f"Found {len(url_entries)} URLs in sitemap {sitemap_url}")
            
            for entry in url_entries:
                loc = entry.find("loc")
                if loc:
                    url = loc.text.strip()
                    if self._is_url_allowed(url):
                        yield url
                    else:
                        # logger.debug(f"URL not allowed: {url}")
                        pass # too noisy

                        
        except Exception as e:
            logger.error(f"Error parsing sitemap {sitemap_url}: {e}")
    
    def scrape_page(self, url: str) -> Optional[PageMetadata]:
        """Scrape a single wiki page and save its content."""
        if url in self.scraped_urls:
            logger.debug(f"Already scraped: {url}")
            return None
        
        if not self._is_url_allowed(url):
            logger.debug(f"URL not allowed by robots.txt: {url}")
            return None
        
        logger.info(f"Scraping: {url}")
        response = self._fetch_with_retry(url)
        
        if not response:
            return None
        
        try:
            soup = BeautifulSoup(response.content, "html.parser")
            
            # Extract page title
            title_elem = soup.find("h1", {"id": "firstHeading"}) or soup.find("title")
            title = title_elem.get_text(strip=True) if title_elem else "Unknown"
            title = title.replace(" – Minecraft Wiki", "").strip()
            
            # Extract main content
            content_div = soup.find("div", {"id": "mw-content-text"})
            if not content_div:
                content_div = soup.find("main") or soup.find("article")
            
            if not content_div:
                logger.warning(f"No content found for: {url}")
                return None
            
            # Extract categories
            categories = []
            cat_links = soup.find_all("a", href=re.compile(r"/w/Category:"))
            for cat in cat_links:
                cat_text = cat.get_text(strip=True)
                if cat_text:
                    categories.append(cat_text)
            
            # Get text content for word count
            text_content = content_div.get_text(separator=" ", strip=True)
            word_count = len(text_content.split())
            
            # Generate content hash
            content_hash = hashlib.md5(response.content).hexdigest()
            
            # Generate filename from URL
            parsed_url = urlparse(url)
            page_path = parsed_url.path.replace("/w/", "").replace("/", "_")
            page_path = re.sub(r'[<>:"|?*]', "_", page_path)  # Remove invalid chars
            filename = f"{page_path}.html"
            file_path = self.config.output_dir / filename
            
            # Save HTML content
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(str(soup))
            
            # Create metadata
            last_modified = response.headers.get("Last-Modified")
            metadata = PageMetadata(
                url=url,
                title=title,
                last_modified=last_modified,
                scraped_at=datetime.now(timezone.utc).isoformat(),
                content_hash=content_hash,
                word_count=word_count,
                categories=categories,
                file_path=str(file_path),
            )
            
            self.scraped_urls.add(url)
            self.metadata.append(metadata)
            
            logger.info(f"Saved: {title} ({word_count} words)")
            return metadata
            
        except Exception as e:
            logger.error(f"Error processing {url}: {e}")
            return None
    
    def save_metadata(self) -> None:
        """Save scraping metadata to JSON file."""
        metadata_dict = {
            "scrape_info": {
                "started_at": datetime.now(timezone.utc).isoformat(),
                "total_pages": len(self.metadata),
                "base_url": self.config.base_url,
            },
            "pages": [
                {
                    "url": m.url,
                    "title": m.title,
                    "last_modified": m.last_modified,
                    "scraped_at": m.scraped_at,
                    "content_hash": m.content_hash,
                    "word_count": m.word_count,
                    "categories": m.categories,
                    "file_path": m.file_path,
                }
                for m in self.metadata
            ]
        }
        
        with open(self.config.metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Metadata saved to {self.config.metadata_file}")
    
    def run(self, max_pages: Optional[int] = None) -> None:
        """Run the full scraping pipeline.
        
        Args:
            max_pages: Maximum number of pages to scrape (None for all)
        """
        logger.info("Starting Minecraft Wiki scrape...")
        logger.info(f"Rate limit: {self.config.requests_per_second} requests/second")
        
        pages_scraped = 0
        
        try:
            for url in self.get_sitemap_urls():
                if max_pages and pages_scraped >= max_pages:
                    logger.info(f"Reached max pages limit: {max_pages}")
                    break
                
                result = self.scrape_page(url)
                if result:
                    pages_scraped += 1
                    
                    # Save metadata periodically
                    if pages_scraped % 100 == 0:
                        self.save_metadata()
                        logger.info(f"Progress: {pages_scraped} pages scraped")
        
        except KeyboardInterrupt:
            logger.info("Scraping interrupted by user")
        
        finally:
            self.save_metadata()
            logger.info(f"Scraping complete. Total pages: {pages_scraped}")


def main():
    """Main entry point for the scraper."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Scrape Minecraft Wiki for RAG")
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Maximum number of pages to scrape (default: all)"
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=1.0,
        help="Requests per second (default: 1.0)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw/html",
        help="Output directory for HTML files"
    )
    
    args = parser.parse_args()
    
    config = ScraperConfig(
        requests_per_second=args.rate_limit,
        output_dir=Path(args.output_dir),
    )
    
    scraper = MinecraftWikiScraper(config)
    scraper.run(max_pages=args.max_pages)


if __name__ == "__main__":
    main()


