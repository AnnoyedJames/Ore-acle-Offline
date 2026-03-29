"""
Minecraft Wiki Image Downloader (v2)

Parses local HTML files to find and download content images.
Saves directly as WebP using the wiki filename (human-readable, no hashing).

Features:
- Wiki-filename storage (e.g. Water_JE16-a1.webp)
- Direct WebP conversion on download (no PNG intermediate)
- Dimension filtering (skip icons < 150px)
- /thumb/ URL resolution to full-res originals
- Incremental: skips images already on disk
- Periodic metadata saves
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Optional, Set
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from PIL import Image
from tqdm import tqdm

from backend.utils.image_utils import get_original_url, wiki_url_to_filename

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class ImageConfig:
    """Configuration for the image downloader."""

    base_url: str = "https://minecraft.wiki"

    # Input / Output
    html_dir: Path = field(default_factory=lambda: Path("data/raw/html"))
    images_dir: Path = field(default_factory=lambda: Path("data/raw/images"))
    metadata_file: Path = field(
        default_factory=lambda: Path("data/processed/image_metadata.json")
    )

    # Download settings
    user_agent: str = "Ore-acle-Bot/1.0 (Educational Minecraft RAG project)"
    requests_per_second: float = 5.0
    retry_delay: int = 5
    max_retries: int = 3

    # Image processing
    webp_quality: int = 80
    max_dimension: int = 1280  # Downscale longest side

    # Filters
    min_width: int = 150
    min_height: int = 150
    skip_patterns: list = field(
        default_factory=lambda: ["editor", "sprite", "icon", "placeholder", "button"]
    )

    # Persistence
    save_interval: int = 25  # Save metadata every N HTML files


class ImageDownloader:
    """Downloads wiki images from local HTML, saves as WebP with wiki filenames."""

    def __init__(self, config: Optional[ImageConfig] = None):
        self.config = config or ImageConfig()
        self.session = self._create_session()
        self.last_request_time = 0.0

        # State
        self.filename_to_meta: dict[str, dict] = {}  # local_filename -> metadata
        self.url_to_filename: dict[str, str] = {}  # original_url -> local_filename
        self.processed_html: Set[str] = set()

        # Ensure directories
        self.config.images_dir.mkdir(parents=True, exist_ok=True)
        self.config.metadata_file.parent.mkdir(parents=True, exist_ok=True)

        self._load_metadata()

    def _create_session(self) -> requests.Session:
        s = requests.Session()
        s.headers["User-Agent"] = self.config.user_agent
        return s

    # ------------------------------------------------------------------
    # Metadata I/O
    # ------------------------------------------------------------------

    def _load_metadata(self):
        """Load existing metadata for incremental runs."""
        if self.config.metadata_file.exists():
            try:
                with open(self.config.metadata_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for entry in data.get("images", []):
                    fname = entry["local_filename"]
                    self.filename_to_meta[fname] = entry
                    self.url_to_filename[entry["original_url"]] = fname
                self.processed_html = set(data.get("processed_html_files", []))
                logger.info(
                    f"Loaded metadata: {len(self.filename_to_meta)} images, "
                    f"{len(self.processed_html)} processed HTML files."
                )
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")

    def save_metadata(self):
        """Persist metadata to disk."""
        data = {
            "last_run": datetime.now(timezone.utc).isoformat(),
            "processed_html_files": sorted(self.processed_html),
            "images": list(self.filename_to_meta.values()),
        }
        with open(self.config.metadata_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Download logic
    # ------------------------------------------------------------------

    def _rate_limit(self):
        elapsed = time.time() - self.last_request_time
        min_interval = 1.0 / self.config.requests_per_second
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self.last_request_time = time.time()

    def _should_skip_url(self, url: str) -> bool:
        lower = url.lower()
        return any(p in lower for p in self.config.skip_patterns)

    def download_image(self, url: str, source_page: str) -> Optional[str]:
        """
        Download a single image, convert to WebP, save with wiki filename.

        Returns the local filename on success, None on skip/failure.
        """
        url = urljoin(self.config.base_url, url)
        url = get_original_url(url)

        # Determine local filename
        local_filename = wiki_url_to_filename(url)
        if not local_filename:
            return None

        # Already have this URL mapped? Just update source pages.
        if url in self.url_to_filename:
            existing = self.filename_to_meta.get(self.url_to_filename[url])
            if existing and source_page not in existing.get("source_pages", []):
                existing.setdefault("source_pages", []).append(source_page)
            return self.url_to_filename[url]

        # File already on disk from a previous run? Register and skip download.
        dest = self.config.images_dir / local_filename
        if dest.exists() and local_filename in self.filename_to_meta:
            self.url_to_filename[url] = local_filename
            meta = self.filename_to_meta[local_filename]
            if source_page not in meta.get("source_pages", []):
                meta.setdefault("source_pages", []).append(source_page)
            return local_filename

        if self._should_skip_url(url):
            return None

        # Fetch
        for attempt in range(self.config.max_retries):
            self._rate_limit()
            try:
                resp = self.session.get(url, timeout=15)
                if resp.status_code != 200:
                    return None
                break
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    logger.error(f"Failed to download {url}: {e}")
                    return None
                time.sleep(self.config.retry_delay)

        # Validate dimensions
        try:
            img = Image.open(BytesIO(resp.content))
            if img.width < self.config.min_width or img.height < self.config.min_height:
                return None
        except Exception:
            return None

        # Downscale if needed & save as WebP
        try:
            img.thumbnail(
                (self.config.max_dimension, self.config.max_dimension),
                Image.LANCZOS,
            )
            img.save(dest, format="WEBP", quality=self.config.webp_quality)

            w, h = img.size
            file_size = dest.stat().st_size

            meta = {
                "local_filename": local_filename,
                "original_url": url,
                "file_path": str(dest),
                "file_size_bytes": file_size,
                "width": w,
                "height": h,
                "format": "WEBP",
                "scraped_at": datetime.now(timezone.utc).isoformat(),
                "source_pages": [source_page],
            }
            self.filename_to_meta[local_filename] = meta
            self.url_to_filename[url] = local_filename

            logger.debug(f"Saved: {local_filename} ({w}x{h}) from {url}")
            return local_filename

        except Exception as e:
            logger.error(f"Error saving {local_filename}: {e}")
            return None

    # ------------------------------------------------------------------
    # Batch HTML processing
    # ------------------------------------------------------------------

    def process_html_files(self, limit: Optional[int] = None):
        """Iterate through local HTML files and download content images."""
        if not self.config.html_dir.exists():
            logger.error(f"HTML directory not found: {self.config.html_dir}")
            return

        html_files = sorted(self.config.html_dir.glob("*.html"))
        logger.info(f"Found {len(html_files)} HTML files total.")

        to_process = [f for f in html_files if f.name not in self.processed_html]
        logger.info(
            f"{len(to_process)} remaining ({len(self.processed_html)} already done)."
        )

        if limit:
            to_process = to_process[:limit]

        pbar = tqdm(to_process, desc="Downloading images", unit="html")
        files_since_save = 0
        downloaded = 0

        for html_path in pbar:
            pbar.set_postfix(imgs=len(self.filename_to_meta), dl=downloaded)
            try:
                with open(html_path, "r", encoding="utf-8") as f:
                    soup = BeautifulSoup(f, "html.parser")

                content = soup.find("div", class_="mw-parser-output")
                if not content:
                    self.processed_html.add(html_path.name)
                    continue

                for img_tag in content.find_all("img"):
                    src = img_tag.get("src")
                    if not src:
                        continue
                    if src.startswith("//"):
                        src = "https:" + src

                    result = self.download_image(src, html_path.name)
                    if result:
                        downloaded += 1

                self.processed_html.add(html_path.name)
                files_since_save += 1

                if files_since_save >= self.config.save_interval:
                    self.save_metadata()
                    files_since_save = 0

            except Exception as e:
                logger.error(f"Error processing {html_path.name}: {e}")

        self.save_metadata()
        logger.info(
            f"Done. {len(self.filename_to_meta)} unique images, "
            f"{len(self.processed_html)} HTML files processed."
        )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download images from scraped HTML")
    parser.add_argument("--limit", type=int, help="Max HTML files to process")
    parser.add_argument("--rate-limit", type=float, default=5.0)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    config = ImageConfig(requests_per_second=args.rate_limit)
    downloader = ImageDownloader(config)
    downloader.process_html_files(limit=args.limit)


if __name__ == "__main__":
    main()
