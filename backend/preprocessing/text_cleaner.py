"""
Minecraft Wiki Text Cleaner

Parses raw HTML files to extract clean, structured text for RAG.
Features:
- Strips navigation, sidebars, and Wiki-specific noise
- Extracts infobox data as structured JSON
- Converts data tables to Markdown format
- Segments text by section headings (H2-H4)
- Preserves semantic structure for downstream chunking
- Incremental processing with periodic saving
"""

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

from bs4 import BeautifulSoup, Tag, NavigableString
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class CleanerConfig:
    """Configuration for the text cleaning pipeline."""
    html_dir: Path = field(default_factory=lambda: Path("data/raw/html"))
    scrape_metadata_file: Path = field(default_factory=lambda: Path("data/raw/scrape_metadata.json"))
    output_file: Path = field(default_factory=lambda: Path("data/processed/metadata.json"))
    
    # Processing settings
    save_interval: int = 100
    min_word_count_flag: int = 50  # Flag pages below this as stubs (but still process)


@dataclass
class Section:
    """Represents a section of text under a specific heading."""
    heading: Optional[str]
    level: int  # 2 for h2, 3 for h3, etc. 0 for preamble
    text: str
    section_type: str = "content"  # "content", "history", "legacy", "future", "navigation"


@dataclass
class ProcessedPage:
    """Final output structure for a cleaned page."""
    url: str
    title: str
    file_path: str
    clean_path: str  # Original HTML path
    word_count: int
    is_stub: bool
    categories: List[str]
    infobox: Dict[str, str]
    tables: List[str]
    sections: List[dict]  # Serialized Section objects
    images: List[dict]  # Rich image metadata with context
    last_processed: str


class TextCleaner:
    """
    Cleans raw HTML into structured text metadata.
    Does NOT chunk text yet (that is a separate step).
    """

    def __init__(self, config: Optional[CleanerConfig] = None):
        self.config = config or CleanerConfig()
        
        # Ensure directories exist
        self.config.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load inputs
        self.scrape_meta = self._load_json(self.config.scrape_metadata_file)
        self.url_map = {p["file_path"].replace("\\", "/").split("/")[-1]: p 
                       for p in self.scrape_meta.get("pages", [])}
        
        # Load existing state for incremental processing
        self.output_data = self._load_json(self.config.output_file)
        # Auto-retry pages with word_count == 0 (likely selector bugs or transient errors)
        self.processed_files = set(
            entry["clean_path"] for entry in self.output_data.get("pages", [])
            if entry.get("word_count", 0) > 0
        )
        # Remove stale 0-word entries so they get re-appended on reprocessing
        if "pages" in self.output_data:
            stale_count = len([p for p in self.output_data["pages"] if p.get("word_count", 0) == 0])
            if stale_count > 0:
                self.output_data["pages"] = [
                    p for p in self.output_data["pages"] if p.get("word_count", 0) > 0
                ]
                logger.info(f"Removed {stale_count} zero-word entries for reprocessing.")

    def _load_json(self, path: Path) -> dict:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load {path}: {e}")
        return {}

    def _save_metadata(self):
        """Persist processed data to disk."""
        try:
            with open(self.config.output_file, "w", encoding="utf-8") as f:
                json.dump(self.output_data, f, indent=2, ensure_ascii=False)
            logger.debug("Metadata saved.")
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")

    def _clean_text(self, text: str) -> str:
        """Standardize whitespace."""
        return re.sub(r'\s+', ' ', text).strip()

    def _extract_infobox(self, soup: Tag) -> Dict[str, str]:
        """Extract key-value pairs from div.infobox."""
        data = {}
        infobox = soup.select_one("div.infobox, div.notaninfobox")
        if not infobox:
            return data
            
        # Handle both nested tables and direct tr elements
        rows = infobox.select("table.infobox-rows tr, tr")
        for row in rows:
            th = row.select_one("th")
            td = row.select_one("td")
            if th and td:
                # Remove edit links and other noise from key/value
                for noise in th.select(".mw-editsection"):
                    noise.decompose()
                for noise in td.select(".mw-editsection"):
                    noise.decompose()
                    
                key = self._clean_text(th.get_text(separator=" "))
                val = self._clean_text(td.get_text(separator=" "))
                if key and val:
                    data[key] = val
        
        # Remove infobox from DOM so it doesn't duplicate in text
        infobox.decompose()
        return data

    def _extract_images(self, content_div: Tag) -> List[dict]:
        """Extract images with full context metadata."""
        images = []
        seen_urls = set()  # Track to avoid duplicates
        
        # Find all img tags in the content
        for img in content_div.find_all("img"):
            src = img.get("src")
            if not src or "/images/" not in src:
                continue
            
            # Clean URL - remove query params and make absolute
            clean_url = src.split("?")[0]
            if clean_url.startswith("/"):
                clean_url = f"https://minecraft.wiki{clean_url}"
            
            # Skip duplicates
            if clean_url in seen_urls:
                continue
            seen_urls.add(clean_url)
            
            # Extract metadata
            alt_text = img.get("alt", "")
            
            # Determine context type by looking at parent elements
            context_type = self._determine_image_context(img)
            
            # Find caption (look for figcaption or nearby text)
            caption = self._find_image_caption(img)
            
            # Find which section this image belongs to
            section = self._find_image_section(img)
            
            # Get surrounding text for context
            surrounding_text = self._get_surrounding_text(img)
            
            images.append({
                "url": clean_url,
                "alt_text": self._clean_text(alt_text),
                "section": section,
                "context_type": context_type,
                "caption": caption,
                "surrounding_text": surrounding_text
            })
        
        return images
    
    def _determine_image_context(self, img: Tag) -> str:
        """Determine where/how the image is used on the page."""
        # Walk up the parent chain to identify context
        for parent in img.parents:
            if not isinstance(parent, Tag):
                continue
            
            classes = parent.get("class", [])
            
            # Check for specific contexts
            if "infobox" in classes:
                return "infobox"
            if "gallery" in classes:
                return "gallery"
            if "thumb" in classes or "thumbinner" in classes:
                return "thumbnail"
            if parent.name == "figure":
                return "figure"
            if "navbox" in classes:
                return "navbox"
            
            # Stop at content div
            if "mw-parser-output" in classes:
                break
        
        return "inline"
    
    def _find_image_caption(self, img: Tag) -> str:
        """Find caption text associated with an image."""
        # Look for figcaption in parent figure
        for parent in img.parents:
            if not isinstance(parent, Tag):
                continue
            if parent.name == "figure":
                figcaption = parent.find("figcaption")
                if figcaption:
                    return self._clean_text(figcaption.get_text())
            # Look for thumbcaption in thumbnails
            if "thumbinner" in parent.get("class", []):
                caption_div = parent.find("div", class_="thumbcaption")
                if caption_div:
                    return self._clean_text(caption_div.get_text())
            # Stop at content boundary
            if "mw-parser-output" in parent.get("class", []):
                break
        return ""
    
    def _find_image_section(self, img: Tag) -> str:
        """Find which section heading this image falls under."""
        # Walk backwards through siblings to find the nearest heading
        current = img
        
        while current:
            # Check previous siblings
            for sibling in current.previous_siblings:
                if not isinstance(sibling, Tag):
                    continue
                
                # Check if this is a heading wrapper
                if sibling.name == "div" and "mw-heading" in sibling.get("class", []):
                    heading = sibling.find(["h2", "h3", "h4", "h5", "h6"])
                    if heading:
                        return self._clean_text(heading.get_text())
                
                # Check if this is a direct heading
                if sibling.name in ["h2", "h3", "h4", "h5", "h6"]:
                    return self._clean_text(sibling.get_text())
            
            # Move up to parent and continue search
            current = current.parent
            if not isinstance(current, Tag):
                break
            
            # Stop at content boundary
            if "mw-parser-output" in current.get("class", []):
                break
        
        return "Introduction"
    
    def _get_surrounding_text(self, img: Tag, max_length: int = 200) -> str:
        """Get nearby text context around an image."""
        # Look for nearest text-containing sibling or parent text
        text_parts = []
        
        # Check the parent paragraph or container
        for parent in img.parents:
            if not isinstance(parent, Tag):
                continue
            
            if parent.name in ["p", "div", "li", "td"]:
                # Get text from this container, but exclude the img itself
                parent_copy = parent.__copy__()
                for child_img in parent_copy.find_all("img"):
                    child_img.decompose()
                text = self._clean_text(parent_copy.get_text())
                if text:
                    text_parts.append(text)
                break
            
            # Stop at section boundary
            if "mw-parser-output" in parent.get("class", []):
                break
        
        # Get text from previous sibling
        prev_sibling = img.find_previous_sibling(["p", "div", "li"])
        if prev_sibling:
            text = self._clean_text(prev_sibling.get_text())
            if text:
                text_parts.insert(0, text)
        
        # Combine and truncate
        full_text = " ".join(text_parts)
        if len(full_text) > max_length:
            full_text = full_text[:max_length] + "..."
        
        return full_text
    
    def _table_to_markdown(self, table: Tag) -> str:
        """Convert a wikitable to a Markdown string."""
        rows = []
        # Header
        headers = [th.get_text(" ", strip=True) for th in table.select("tr th")]
        if headers:
            rows.append("| " + " | ".join(headers) + " |")
            rows.append("| " + " | ".join(["---"] * len(headers)) + " |")
        
        # Body
        for tr in table.select("tr"):
            cells = [td.get_text(" ", strip=True) for td in tr.select("td")]
            if cells:
                # If row has fewer cells than header (colspan/rowspan), pad it
                if headers and len(cells) < len(headers):
                    cells += [""] * (len(headers) - len(cells))
                rows.append("| " + " | ".join(cells) + " |")
        
        return "\n".join(rows)

    def _process_tables(self, soup: Tag) -> List[str]:
        """Convert wikitables to Markdown in-place."""
        tables = []
        for tbl in soup.select("table.wikitable"):
            md = self._table_to_markdown(tbl)
            if md:
                tables.append(md)
                # Replace table with markdown text in the DOM
                tbl.replace_with(NavigableString(f"\n\n{md}\n\n"))
            else:
                tbl.decompose()
        return tables

    def _process_sections_robust(self, content_div: Tag) -> List[dict]:
        """
        Walks DOM nodes linearly to assign text to the preceding heading.
        Builds hierarchical section paths (e.g., "Obtaining > Mining").
        Identifies section type (content, history, future) based on headings.
        """
        sections = []
        current_heading = "Introduction"
        current_level = 1
        current_section_type = "content"
        heading_stack = []  # Stack of (level, name) tuples for building breadcrumbs
        buffer_text = []

        def flush():
            nonlocal buffer_text
            full_text = self._clean_text(" ".join(buffer_text))
            if full_text:
                sections.append(asdict(Section(
                    heading=current_heading,
                    level=current_level,
                    text=full_text,
                    section_type=current_section_type
                )))
            buffer_text = []
        
        def build_breadcrumb(level: int, name: str) -> str:
            """Build hierarchical breadcrumb path for section."""
            # Pop headings from stack that are at the same or deeper level
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            
            # Add current heading to stack
            heading_stack.append((level, name))
            
            # Build breadcrumb from stack
            if len(heading_stack) > 1:
                return " > ".join(h[1] for h in heading_stack)
            return name

        def update_section_type(level: int, name: str):
            """Update section type based on heading."""
            nonlocal current_section_type
            normalized = name.lower()
            
            # Level 2 headings reset or set the type
            if level == 2:
                if any(w in normalized for w in ["history", "legacy", "past version"]):
                    current_section_type = "history"
                elif any(w in normalized for w in ["future", "upcoming", "snapshot", "preview", "experimental"]):
                    current_section_type = "future"
                elif normalized in ["gallery", "references", "see also", "external links"]:
                    current_section_type = "navigation"
                else:
                    current_section_type = "content"
            
            # Allow deeper levels to opt-in to history/future if their heading is explicit
            # but they cannot switch BACK to content from history without a Level 2 reset
            elif level > 2:
                if "history" in normalized or "past" in normalized or "legacy" in normalized:
                    current_section_type = "history"
                elif "future" in normalized or "upcoming" in normalized:
                    current_section_type = "future"
                # If currently "history", we stay "history" even if heading is normal (e.g. "1.19")

        # Scan all direct children of content div
        for child in content_div.children:
            if isinstance(child, NavigableString):
                t = str(child).strip()
                if t: buffer_text.append(t)
                continue
            
            if not isinstance(child, Tag):
                continue
            
            # Check if this is a heading wrapper div
            if child.name == "div" and "mw-heading" in child.get("class", []):
                # Find the actual heading tag inside
                heading_tag = child.find(["h2", "h3", "h4", "h5", "h6"])
                if heading_tag:
                    flush()
                    # Remove [edit] links from heading text
                    for edit in heading_tag.select(".mw-editsection"):
                        edit.decompose()
                    
                    heading_text = self._clean_text(heading_tag.get_text())
                    current_level = int(heading_tag.name[1])
                    current_heading = build_breadcrumb(current_level, heading_text)
                    update_section_type(current_level, heading_text)
                    
            elif child.name in ["h2", "h3", "h4", "h5", "h6"]:
                # Handle direct heading tags (backup for older wiki formats)
                flush()
                # Remove [edit] links from heading text
                for edit in child.select(".mw-editsection"):
                    edit.decompose()
                
                heading_text = self._clean_text(child.get_text())
                current_level = int(child.name[1])
                current_heading = build_breadcrumb(current_level, heading_text)
                update_section_type(current_level, heading_text)
            
            elif child.name in ["p", "ul", "ol", "dl", "blockquote", "div"]:
                # Extract text from block elements, including hatnotes
                text = child.get_text(" ", strip=True)
                if text:
                    buffer_text.append(text)
            
        flush()
        return sections

    def process_single(self, html_path: Path) -> Optional[dict]:
        """Parse one HTML file into structured metadata."""
        try:
            with open(html_path, "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f, "lxml")
            
            # 1. Locate Content
            # Use scoped selector to avoid hitting mw-parser-output inside
            # mw-indicator badges (e.g., featured article stars, CC license icons)
            content = soup.select_one("#mw-content-text > div.mw-parser-output")
            if not content:
                # Fallback to unscoped selector for non-standard page layouts
                content = soup.select_one("div.mw-parser-output")
            if not content:
                logger.warning(f"No .mw-parser-output found in {html_path.name}")
                return None
            
            # 2. Extract Images (before stripping content)
            images = self._extract_images(content)
            
            # 3. Strip Noise (per copilot instructions)
            NOISE_SELECTORS = [
                "div#toc", "span.mw-editsection", "pre.history-json", 
                "pre.chest-json", "div.printfooter", "div#catlinks", 
                "table.navbox", ".navigation-not-searchable", "noscript", 
                "sup.reference", "div.reflist", "ol.references", 
                "ul.gallery", "style", "script", ".noprint", "div.magnify",
                ".mw-editsection-bracket"  # Remove edit brackets too
            ]
            for sel in NOISE_SELECTORS:
                for tag in content.select(sel):
                    tag.decompose()
                    
            # 4. Handle Sprites (keep text, remove image)
            for sprite_file in content.select(".sprite-file"):
                sprite_file.decompose()
            # .sprite-text elements are kept automatically

            # 5. Extract Infobox (and remove from DOM)
            infobox = self._extract_infobox(content)
            
            # 6. Extract Tables (and remove from DOM)
            tables = self._process_tables(content)
            
            # 7. Extract Sections (from remaining clean content)
            sections = self._process_sections_robust(content)
            
            # 8. Metadata & Stats
            # Link back to original scrape metadata if possible
            filename = html_path.name
            scrape_info = self.url_map.get(filename, {})
            
            # Calculate word counts
            section_words = sum(len(s["text"].split()) for s in sections)
            table_words = sum(len(t.split()) for t in tables)
            infobox_words = sum(len(str(v).split()) for v in infobox.values())
            total_words = section_words + table_words + infobox_words
            
            if total_words == 0:
                logger.warning(f"No text extracted from {html_path.name}")
            
            from datetime import datetime
            
            return asdict(ProcessedPage(
                url=scrape_info.get("url", ""),
                title=scrape_info.get("title", html_path.stem),
                file_path=str(html_path),
                clean_path=filename,
                word_count=total_words,
                is_stub=total_words < self.config.min_word_count_flag,
                categories=scrape_info.get("categories", []),
                infobox=infobox,
                tables=tables,
                sections=sections,
                images=images,
                last_processed=datetime.utcnow().isoformat()
            ))

        except Exception as e:
            logger.error(f"Error processing {html_path.name}: {e}", exc_info=True)
            return None

    def process_all(self):
        """Batch process all HTML files."""
        html_files = sorted(list(self.config.html_dir.glob("*.html")))
        logger.info(f"Found {len(html_files)} HTML files.")
        
        # Filter out already processed
        to_process = [p for p in html_files if p.name not in self.processed_files]
        logger.info(f"Processing {len(to_process)} new files ({len(self.processed_files)} skipped).")
        
        if not to_process:
            logger.info("All files already processed.")
            return

        errors = 0
        warnings = 0
        
        # Initialize output structure if new
        if "pages" not in self.output_data:
            self.output_data["pages"] = []
            self.output_data["processing_info"] = {
                "started_at": "",
                "files_total": len(html_files),
                "files_processed": 0,
                "files_skipped": len(self.processed_files),
                "errors": 0
            }
        
        from datetime import datetime
        self.output_data["processing_info"]["started_at"] = datetime.utcnow().isoformat()
        self.output_data["processing_info"]["files_total"] = len(html_files)

        try:
            for i, path in enumerate(tqdm(to_process, desc="Cleaning Text", unit="pg")):
                result = self.process_single(path)
                if result:
                    self.output_data["pages"].append(result)
                    self.processed_files.add(path.name)
                    if result["word_count"] == 0:
                        warnings += 1
                else:
                    errors += 1
                
                # Periodic Save
                if (i + 1) % self.config.save_interval == 0:
                    self.output_data["processing_info"]["files_processed"] = len(self.output_data["pages"])
                    self.output_data["processing_info"]["errors"] = errors
                    self._save_metadata()
                    logger.info(f"Progress: {i+1}/{len(to_process)} files, {errors} errors, {warnings} warnings")
                    
        finally:
            self.output_data["processing_info"]["files_processed"] = len(self.output_data["pages"])
            self.output_data["processing_info"]["errors"] = errors
            self.output_data["processing_info"]["completed_at"] = datetime.utcnow().isoformat()
            self._save_metadata()
            logger.info("="*60)
            logger.info("Final save complete.")
            logger.info(f"Total processed: {len(self.output_data['pages'])} pages")
            logger.info(f"Errors: {errors}, Warnings (empty pages): {warnings}")
            logger.info("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean Minecraft Wiki HTML")
    parser.add_argument("--html-dir", type=Path, default=Path("data/raw/html"))
    parser.add_argument("--output-file", type=Path, default=Path("data/processed/metadata.json"))
    args = parser.parse_args()
    
    config = CleanerConfig(
        html_dir=args.html_dir,
        output_file=args.output_file
    )
    
    cleaner = TextCleaner(config)
    cleaner.process_all()
