"""
Minecraft Wiki Image Downloader

Parses local HTML files (scraped by wiki_scraper.py) to find and download images.
Features:
- Deduplication (content-addressed storage via MD5)
- Filtering (skips small icons/UI elements)
- Rate limiting
- Metadata tracking
"""

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Set
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ImageConfig:      #Configuration for the image downloader
    base_url: str = "https://minecraft.wiki"
    
    # Input/Output
    html_dir: Path = field(default_factory=lambda: Path("data/raw/html"))
    images_dir: Path = field(default_factory=lambda: Path("data/raw/images"))
    metadata_file: Path = field(default_factory=lambda: Path("data/processed/image_metadata.json"))
    
    # Download settings
    user_agent: str = "Ore-acle-Bot/1.0 (Educational Minecraft RAG project)"
    requests_per_second: float = 5.0  # Images are CDN-cached, can fetch faster
    retry_delay: int = 5
    max_retries: int = 3
    
    # Filters
    min_width: int = 150   # Skip thumbnails (120x68) and small icons
    min_height: int = 150  # Skip thumbnails (120x68) and small icons
    skip_patterns: list = field(default_factory=lambda: [
        "editor", "sprite", "icon", "placeholder", "button"
    ])


@dataclass
class ImageMetadata:
    """Metadata for a downloaded image."""
    image_hash: str
    original_url: str
    file_path: str
    file_size_bytes: int
    width: int
    height: int
    format: str
    scraped_at: str
    source_pages: list[str]  # List of HTML files this image appeared in


class ImageDownloader:
    """Downloader that parses local HTML and fetches referenced images."""
    
    def __init__(self, config: Optional[ImageConfig] = None):
        self.config = config or ImageConfig()
        self.session = self._create_session()
        self.last_request_time = 0.0
        
        # State tracking
        self.image_metadata: dict[str, ImageMetadata] = {}  # hash -> metadata
        self.url_to_hash: dict[str, str] = {}  # url -> hash
        self.processed_urls: Set[str] = set()
        self.processed_html_files: Set[str] = set()  # Track fully processed HTML files
        self.source_pages_sets: dict[str, Set[str]] = {}  # hash -> set of source pages (O(1) lookup)
        
        # Ensure output directories exist
        self.config.images_dir.mkdir(parents=True, exist_ok=True)
        self.config.metadata_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing metadata if available
        self._load_metadata()

    def _create_session(self) -> requests.Session:
        session = requests.Session()
        session.headers.update({
            "User-Agent": self.config.user_agent,
        })
        return session

    def _load_metadata(self):
        """Load existing metadata to support incremental runs."""
        if self.config.metadata_file.exists():
            try:
                with open(self.config.metadata_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data.get("images", []):
                        meta = ImageMetadata(**item)
                        self.image_metadata[meta.image_hash] = meta
                        self.url_to_hash[meta.original_url] = meta.image_hash
                        self.processed_urls.add(meta.original_url)
                        # Convert source_pages list to set for O(1) lookup
                        self.source_pages_sets[meta.image_hash] = set(meta.source_pages)
                    # Load processed HTML files
                    self.processed_html_files = set(data.get("processed_html_files", []))
                logger.info(f"Loaded metadata for {len(self.image_metadata)} images, {len(self.processed_html_files)} processed HTML files")
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")

    def _rate_limit(self):
        elapsed = time.time() - self.last_request_time
        min_interval = 1.0 / self.config.requests_per_second
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self.last_request_time = time.time()

    def _should_skip_url(self, url: str) -> bool:
        """Heuristic check to skip obvious junk URLs."""
        lower_url = url.lower()
        
        # Check explicit patterns
        for pattern in self.config.skip_patterns:
            if pattern in lower_url:
                return True
        
        # Check likely non-content extensions
        if lower_url.endswith(('.svg', '.gif')): # Optional: skip animations/vectors if problematic
            pass 
            
        return False

    def _convert_to_original_url(self, url: str) -> str:
        """
        Converts a MediaWiki thumbnail URL to the original file URL.
        Example: 
        Input:  .../images/thumb/a/b/Name.png/300px-Name.png
        Output: .../images/a/b/Name.png
        """
        # Regex to capture the part between /thumb/ and the last slash (which is before the width-px part)
        # Matches: .../images/thumb/(part_to_keep)/resolution-filename
        match = re.search(r'/thumb/(.+)/[^/]+$', url)
        if match:
            # Reconstruct URL: remove /thumb/ and the trailing resolution part
            base_part = url.split('/thumb/')[0]
            clean_path = match.group(1)
            return f"{base_part}/{clean_path}"
        return url

    def download_image(self, url: str, source_page: str) -> Optional[str]:
        """Download image, deduplicate, and save. Returns image hash."""
        url = urljoin(self.config.base_url, url)
        
        # Try to get the original high-res version
        url = self._convert_to_original_url(url)
        
        # If we've seen this URL, just link the source page
        if url in self.processed_urls:
            img_hash = self.url_to_hash.get(url)
            if img_hash and img_hash in self.image_metadata:
                if source_page not in self.source_pages_sets.get(img_hash, set()):
                    self.source_pages_sets.setdefault(img_hash, set()).add(source_page)
                logger.debug(f"Cache hit (URL): {url} from {source_page}")
                return img_hash
            return None

        if self._should_skip_url(url):
            logger.debug(f"Skipped URL (pattern match): {url}")
            return None

        # Fetch image
        for attempt in range(self.config.max_retries):
            self._rate_limit()
            try:
                response = self.session.get(url, timeout=15)
                if response.status_code != 200:
                    return None
                
                content = response.content
                break
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    logger.error(f"Failed to download {url}: {e}")
                    return None
                time.sleep(self.config.retry_delay)
        
        # Validate and Filter with Pillow
        try:
            img = Image.open(BytesIO(content))
            
            # Dimension filter
            if img.width < self.config.min_width or img.height < self.config.min_height:
                logger.debug(f"Skipped (too small {img.width}x{img.height}): {url}")
                return None
            
            # Format normalization
            ext = ".png" # Save everything as PNG or keep original? 
            # keep original extension unless it's weird, but convert to RGB if needed
            
        except Exception:
            # Not a valid image
            return None

        # Calculate Hash
        img_hash = hashlib.md5(content).hexdigest()
        
        # Check if we have this CONTENT already (even if URL distinct)
        if img_hash in self.image_metadata:
            # Just add the source using set for O(1) lookup
            if source_page not in self.source_pages_sets.get(img_hash, set()):
                self.source_pages_sets.setdefault(img_hash, set()).add(source_page)
            # Link this new URL to the old hash
            self.url_to_hash[url] = img_hash
            self.processed_urls.add(url)
            logger.debug(f"Cache hit (content hash {img_hash[:8]}): {url} from {source_page}")
            return img_hash

        # Save new image
        filename = f"{img_hash}.png"
        file_path = self.config.images_dir / filename
        
        try:
            # Convert to RGB if necessary (e.g. for JPEG/PNG consistency)
            if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                pass # Keep transparency
            else:
                 if img.mode != 'RGB':
                    img = img.convert('RGB')
            
            # Actually save as valid PNG (fixing previous bug where raw bytes were written)
            img.save(file_path, "PNG", optimize=True)
            
            # Get actual file size of the converted PNG
            final_size = file_path.stat().st_size
                
            # Update metadata
            meta = ImageMetadata(
                image_hash=img_hash,
                original_url=url,
                file_path=str(file_path),
                file_size_bytes=final_size,
                width=img.width,
                height=img.height,
                format="PNG",
                scraped_at=datetime.now(timezone.utc).isoformat(),
                source_pages=[source_page]  # Will be synced from set on save
            )
            
            self.image_metadata[img_hash] = meta
            self.source_pages_sets[img_hash] = {source_page}  # Use set for O(1) lookup
            self.url_to_hash[url] = img_hash
            self.processed_urls.add(url)
            
            logger.info(f"Downloaded: {filename} ({img.width}x{img.height}) from {url}")
            return img_hash
            
        except Exception as e:
            logger.error(f"Error saving image {img_hash}: {e}")
            return None

    def process_html_files(self, limit: Optional[int] = None):
        """Iterate through local HTML files and download content images."""
        if not self.config.html_dir.exists():
            logger.error(f"HTML directory not found: {self.config.html_dir}")
            return

        html_files = list(self.config.html_dir.glob("*.html"))
        logger.info(f"Found {len(html_files)} HTML files to process")
        
        count = 0
        skipped = 0
        
        pbar = tqdm(html_files, desc="Processing HTML", unit="file")
        
        for file_path in pbar:
            if limit and count >= limit:
                break
            
            # Skip already-processed HTML files
            if file_path.name in self.processed_html_files:
                skipped += 1
                continue
                
            pbar.set_description(f"Processing {file_path.name[:20]}...")

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    soup = BeautifulSoup(f, "html.parser")
                
                # Broad selector for content images
                # 1. Standard thumbnails in articles
                # 2. Infobox images
                # 3. Galleries
                # Standard wiki content is in .mw-parser-output
                content = soup.find("div", class_="mw-parser-output")
                if not content:
                    logger.debug(f"No mw-parser-output in {file_path.name}")
                    count += 1
                    continue

                images = content.find_all("img")
                logger.debug(f"[{count+1}/{len(html_files)}] {file_path.name}: found {len(images)} images")
                
                page_name = file_path.name
                
                for img_tag in images:
                    src = img_tag.get("src")
                    if not src:
                        continue
                        
                    # Wiki typically uses local paths /images/...
                    if src.startswith("//"):
                        src = "https:" + src
                    
                    self.download_image(src, page_name)
                
                # Mark this HTML file as fully processed
                self.processed_html_files.add(file_path.name)
                count += 1
                if count % 10 == 0:
                    self.save_metadata()
                    
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
        
        if skipped > 0:
            logger.info(f"Skipped {skipped} already-processed HTML files")
        self.save_metadata()

    def save_metadata(self):
        """Persist metadata to disk."""
        # Sync source_pages sets back to the metadata lists before saving
        for img_hash, meta in self.image_metadata.items():
            if img_hash in self.source_pages_sets:
                meta.source_pages = list(self.source_pages_sets[img_hash])
        
        data = {
            "last_run": datetime.now(timezone.utc).isoformat(),
            "processed_html_files": list(self.processed_html_files),
            "images": [asdict(meta) for meta in self.image_metadata.values()]
        }
        with open(self.config.metadata_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download images from scraped HTML")
    parser.add_argument("--limit", type=int, help="Max HTML files to process")
    parser.add_argument("--rate-limit", type=float, default=5.0)
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    config = ImageConfig(requests_per_second=args.rate_limit)
    downloader = ImageDownloader(config)
    downloader.process_html_files(limit=args.limit)


if __name__ == "__main__":
    main()
