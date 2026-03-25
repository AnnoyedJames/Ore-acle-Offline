"""
Interlink Extractor — builds a page-to-page link graph from raw HTML.

Parses internal wiki links (<a href="/w/...">) within the main content area
to capture how pages relate to each other. This metadata is attached to
chunks so the retrieval system can surface related content.

Usage:
    python -m preprocessing.link_extractor
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import unquote

from bs4 import BeautifulSoup
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class LinkExtractorConfig:
    """Configuration for the link extractor."""
    html_dir: Path = field(default_factory=lambda: Path("data/raw/html"))
    scrape_metadata_file: Path = field(
        default_factory=lambda: Path("data/raw/scrape_metadata.json")
    )
    output_file: Path = field(
        default_factory=lambda: Path("data/processed/interlinks.json")
    )
    # Maximum number of outgoing links to store per page (avoid noise)
    max_links_per_page: int = 100


class LinkExtractor:
    """
    Extracts internal wiki links from raw HTML pages.
    
    Builds a directed graph: {page_title: [linked_page_titles]}
    Only captures links within div.mw-parser-output (main content area),
    excludes navbox, see-also, and meta navigation links.
    """

    def __init__(self, config: Optional[LinkExtractorConfig] = None):
        self.config = config or LinkExtractorConfig()
        self.config.output_file.parent.mkdir(parents=True, exist_ok=True)

        # Load scrape metadata for filename → title mapping
        self.scrape_meta = self._load_json(self.config.scrape_metadata_file)
        self.filename_to_title = {}
        self.known_titles = set()
        for page in self.scrape_meta.get("pages", []):
            fname = page["file_path"].replace("\\", "/").split("/")[-1]
            title = page.get("title", "")
            self.filename_to_title[fname] = title
            self.known_titles.add(title)

    def _load_json(self, path: Path) -> dict:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load {path}: {e}")
        return {}

    def _href_to_title(self, href: str) -> Optional[str]:
        """Convert a wiki href like /w/Diamond_Pickaxe to page title 'Diamond Pickaxe'."""
        if not href or not href.startswith("/w/"):
            return None

        # Strip /w/ prefix and any anchor/query
        path = href[3:].split("#")[0].split("?")[0]
        if not path:
            return None

        # URL decode: %27 → ', %C3%A9 → é, etc.
        title = unquote(path).replace("_", " ")

        # Skip special namespaces
        if ":" in title and title.split(":")[0] in (
            "Category", "File", "Template", "Module", "Help",
            "Talk", "User", "MediaWiki", "Special",
        ):
            return None

        return title

    def extract_links_from_file(self, html_path: Path) -> list[str]:
        """Extract internal wiki link targets from a single HTML file."""
        try:
            with open(html_path, "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f.read(), "lxml")
        except Exception as e:
            logger.warning(f"Failed to parse {html_path.name}: {e}")
            return []

        content = soup.find("div", class_="mw-parser-output")
        if not content:
            return []

        # Remove navbox, hatnote navigation, and external link sections
        for noise in content.find_all(["div", "table"], class_=re.compile(
            r"navbox|navigation-not-searchable|noprint"
        )):
            noise.decompose()

        # Collect unique link targets
        seen = set()
        links = []
        for a in content.find_all("a", href=True):
            title = self._href_to_title(a["href"])
            if title and title not in seen:
                seen.add(title)
                links.append(title)
                if len(links) >= self.config.max_links_per_page:
                    break

        return links

    def extract_all(self) -> dict:
        """Extract interlinks from all HTML files."""
        html_files = sorted(self.config.html_dir.glob("*.html"))
        logger.info(f"Extracting interlinks from {len(html_files)} HTML files...")

        graph: dict[str, list[str]] = {}
        reverse_counts: dict[str, int] = {}  # inbound link count per page

        for html_path in tqdm(html_files, desc="Extracting links"):
            source_title = self.filename_to_title.get(html_path.name)
            if not source_title:
                continue

            targets = self.extract_links_from_file(html_path)
            # Only keep links to pages we actually have
            valid_targets = [t for t in targets if t in self.known_titles]

            if valid_targets:
                graph[source_title] = valid_targets
                for t in valid_targets:
                    reverse_counts[t] = reverse_counts.get(t, 0) + 1

        # Summary stats
        total_links = sum(len(v) for v in graph.values())
        most_linked = sorted(reverse_counts.items(), key=lambda x: -x[1])[:20]

        logger.info(f"Pages with outgoing links: {len(graph)}")
        logger.info(f"Total links: {total_links}")
        logger.info(f"Average links per page: {total_links / max(len(graph), 1):.1f}")
        logger.info("Top 20 most linked-to pages:")
        for title, count in most_linked:
            logger.info(f"  {title}: {count} inbound links")

        output = {
            "total_pages_with_links": len(graph),
            "total_links": total_links,
            "graph": graph,
        }

        with open(self.config.output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved interlink graph to {self.config.output_file}")
        return output


if __name__ == "__main__":
    extractor = LinkExtractor()
    extractor.extract_all()
