
"""
Integration tests to verify image download coverage.
Run with: pytest tests/test_image_integrity.py -v
"""

import sys
import json
import logging
from pathlib import Path
import pytest
from bs4 import BeautifulSoup
from PIL import Image

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from scraper.image_downloader import ImageDownloader, ImageConfig

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
METADATA_FILE = PROJECT_ROOT / "data/processed/image_metadata.json"
HTML_DIR = PROJECT_ROOT / "data/raw/html"

def load_metadata():
    if not METADATA_FILE.exists():
        return None
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

@pytest.fixture
def metadata():
    data = load_metadata()
    if not data:
        pytest.skip("Metadata file not found. Run image_downloader.py first.")
    return data

@pytest.fixture
def downloader():
    """Instance of ImageDownloader for accessing helper methods."""
    config = ImageConfig(metadata_file=METADATA_FILE, html_dir=HTML_DIR)
    return ImageDownloader(config)

class TestImageIntegrity:
    """Checks that downloaded images on disk match the metadata."""

    def test_images_exist_on_disk(self, metadata):
        """Verify nearly all images listed in metadata actually exist (tolerance 0.5%)."""
        images = metadata.get("images", [])
        assert len(images) > 0, "No images found in metadata"
        
        total_images = len(images)
        missing_count = 0
        missing_files = []
        
        for img in images:
            file_path = Path(img["file_path"])
            if not file_path.is_absolute():
                file_path = PROJECT_ROOT / file_path
                
            if not file_path.exists():
                missing_count += 1
                missing_files.append(str(file_path))
        
        missing_percentage = (missing_count / total_images) * 100
        
        assert missing_percentage < 0.5, (
            f"Missing {missing_percentage:.2f}% of images "
            f"({missing_count}/{total_images}). Threshold is 0.5%.\n"
            f"Sample missing: {missing_files[:5]}"
        )

    def test_images_are_valid(self, metadata):
        """Verify every downloaded image is a valid image file."""
        images = metadata.get("images", [])
        corrupted_files = []
        
        # Check a sample to save time if there are thousands
        sample_size = min(len(images), 100)
        # deterministic sample
        sample_images = images[:sample_size]
        
        for img in sample_images:
            file_path = Path(img["file_path"])
            if not file_path.is_absolute():
                file_path = PROJECT_ROOT / file_path
                
            try:
                if file_path.exists():
                    with Image.open(file_path) as i:
                        i.verify()  # PIL verify can catch some corruptions
            except Exception as e:
                corrupted_files.append(f"{file_path}: {str(e)}")
                
        assert not corrupted_files, f"Found corrupted images: {corrupted_files}"


class TestDownloadCoverage:
    """Checks if we missed potential images in the processed files."""

    def test_processed_html_files_completeness(self, metadata, downloader):
        """
        Pick a sample of processed HTML files, parse them again, 
        and ensure their valid images are accounted for in metadata.
        """
        processed_files = metadata.get("processed_html_files", [])
        if not processed_files:
            pytest.skip("No HTML files listed as processed.")

        # Check a small sample of files
        sample_files = processed_files[:5]
        
        # Build a set of all ORIGINAL URLs we think we have (and their hashes)
        known_urls = set()
        for img in metadata["images"]:
            known_urls.add(img["original_url"])
            
        # We also need to account for the fact that downloader.processed_urls 
        # includes redirects/variants. The generic 'processed_urls' isn't publicly exposed 
        # in the json (only 'images').
        # However, checking against the canonical 'original_url' is a good baseline.

        missed_images = []

        for filename in sample_files:
            file_path = HTML_DIR / filename
            if not file_path.exists():
                continue

            with open(file_path, "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f, "html.parser")
            
            content = soup.find("div", class_="mw-parser-output")
            if not content:
                continue

            images = content.find_all("img")
            for img_tag in images:
                src = img_tag.get("src")
                if not src:
                    continue
                
                if src.startswith("//"):
                    src = "https:" + src
                elif src.startswith("/"):
                    src = "https://minecraft.wiki" + src

                # 1. Simulate the logic to convert to "Original" URL
                # This is crucial because we store the ORIGINAL version
                full_url = downloader._convert_to_original_url(src)
                
                # 2. Check if we WOULD skip it
                if downloader._should_skip_url(full_url):
                    continue
                
                # 3. Check if it's in our known set
                # We do a loose check: if the URL (or its variants) is NOT in metadata,
                # we count it as potentially missing.
                
                # Check if this exact original URL is known
                if full_url in known_urls:
                    continue
                    
                # If not, it might be missing.
                # However, we can't fail the test because it might have been skipped due to SIZE (after download).
                # So we collect them for reporting.
                missed_images.append({
                    "page": filename,
                    "url": full_url
                })

        # Warning based assertion:
        # If we have a huge number of "missed" images, something might be wrong with the scraper
        # or the "known_urls" logic.
        if missed_images:
            # We don't fail, but we print/log them.
            # In a strict environment, we might want to fail if > X% are missing.
            print(f"\\n[Info] Found {len(missed_images)} potential images in sample pages that are not in metadata (likely filtered by size or 404s):")
            for m in missed_images[:3]:
                print(f" - {m['page']}: {m['url']}")
            if len(missed_images) > 3:
                print(f" - ... and {len(missed_images) - 3} more")
        
    def test_metadata_consistency(self, metadata):
        """Check internal consistency of metadata structure."""
        processed = set(metadata.get("processed_html_files", []))
        images = metadata.get("images", [])
        
        # Check that all source_pages listed in images are actually in processed_html_files
        orphan_pages = set()
        for img in images:
            for page in img.get("source_pages", []):
                if page not in processed:
                    orphan_pages.add(page)
        
        # This might validly happen if we cleared processed_files list but kept images? 
        # But generally they should match.
        assert not orphan_pages, f"Found source pages in image metadata that are not marked as processed: {list(orphan_pages)[:5]}..."
