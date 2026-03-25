"""
Minecraft Wiki Image Compressor

Converts the raw PNG image library to WebP format with optional downscaling.
Features:
- PNG → WebP conversion (quality 80, transparency preserved)
- Downscaling to max 1280px on longest side (aspect ratio preserved)
- Moves originals to backup directory for safe deletion later
- Updates image_metadata.json with new paths, sizes, and dimensions
- Incremental: skips images already converted to WebP
- Periodic metadata saves to survive interrupts
"""

import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ProcessorConfig:
    """Configuration for the image compression pipeline."""
    images_dir: Path = field(default_factory=lambda: Path("data/raw/images"))
    backup_dir: Path = field(default_factory=lambda: Path("data/raw/images_backup"))
    metadata_file: Path = field(default_factory=lambda: Path("data/processed/image_metadata.json"))

    # Processing settings
    max_dimension: int = 1280       # Max width or height (aspect ratio preserved)
    output_quality: int = 80        # WebP quality (0-100)
    output_format: str = "WEBP"
    save_interval: int = 100        # Save metadata every N images


class ImageProcessor:
    """Compresses the raw image library from PNG to WebP."""

    def __init__(self, config: Optional[ProcessorConfig] = None):
        self.config = config or ProcessorConfig()

        # Ensure directories exist
        self.config.backup_dir.mkdir(parents=True, exist_ok=True)
        self.config.metadata_file.parent.mkdir(parents=True, exist_ok=True)

        # Metadata loaded once; mutated in-place during processing
        self.metadata = self._load_metadata()

        # Build a lookup: hash → index in the images list for O(1) updates
        self._hash_index: dict[str, int] = {}
        for i, entry in enumerate(self.metadata.get("images", [])):
            self._hash_index[entry["image_hash"]] = i

    # ------------------------------------------------------------------
    # Metadata I/O
    # ------------------------------------------------------------------

    def _load_metadata(self) -> dict:
        """Load the image metadata JSON."""
        if self.config.metadata_file.exists():
            try:
                with open(self.config.metadata_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load metadata: {e}")
        return {}

    def _save_metadata(self):
        """Persist the metadata JSON to disk."""
        try:
            with open(self.config.metadata_file, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
            logger.debug("Metadata saved.")
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")

    # ------------------------------------------------------------------
    # Single-image processing
    # ------------------------------------------------------------------

    def _process_single(self, png_path: Path) -> Tuple[int, int]:
        """
        Convert a single PNG to WebP, move original to backup.

        Returns (original_bytes, new_bytes).
        """
        image_hash = png_path.stem
        original_bytes = png_path.stat().st_size

        with Image.open(png_path) as img:
            # Downscale if either dimension exceeds the cap
            img.thumbnail(
                (self.config.max_dimension, self.config.max_dimension),
                Image.LANCZOS,
            )

            new_width, new_height = img.size

            # Save as WebP (supports transparency natively)
            webp_path = self.config.images_dir / f"{image_hash}.webp"
            img.save(
                webp_path,
                format=self.config.output_format,
                quality=self.config.output_quality,
            )

        new_bytes = webp_path.stat().st_size

        # Update metadata entry if it exists
        if image_hash in self._hash_index:
            idx = self._hash_index[image_hash]
            entry = self.metadata["images"][idx]
            entry["file_path"] = str(
                self.config.images_dir / f"{image_hash}.webp"
            )
            entry["format"] = self.config.output_format
            entry["file_size_bytes"] = new_bytes
            entry["width"] = new_width
            entry["height"] = new_height

        # Move original PNG to backup
        shutil.move(str(png_path), str(self.config.backup_dir / png_path.name))

        return original_bytes, new_bytes

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def process_images(self):
        """
        Convert all PNG images to WebP, backup originals, update metadata.
        """
        png_files = sorted(self.config.images_dir.glob("*.png"))

        if not png_files:
            logger.info("No PNG files found — nothing to process.")
            return

        logger.info(f"Found {len(png_files)} PNG images to compress.")

        total_original = 0
        total_new = 0
        processed = 0
        errors = 0

        pbar = tqdm(png_files, desc="Compressing", unit="img")
        try:
            for png_path in pbar:
                try:
                    orig, new = self._process_single(png_path)
                    total_original += orig
                    total_new += new
                    processed += 1

                    # Live progress in the bar
                    saved_so_far = (total_original - total_new) / (1024 ** 3)
                    pbar.set_postfix(saved=f"{saved_so_far:.2f} GB")

                except Exception as e:
                    logger.error(f"Error processing {png_path.name}: {e}")
                    errors += 1

                # Periodic metadata save
                if processed % self.config.save_interval == 0:
                    self._save_metadata()

        finally:
            # Always save on exit (handles Ctrl-C gracefully)
            self._save_metadata()

        # ---- Summary ----
        saved_bytes = total_original - total_new
        orig_gb = total_original / (1024 ** 3)
        new_gb = total_new / (1024 ** 3)
        saved_gb = saved_bytes / (1024 ** 3)
        pct = (saved_bytes / total_original * 100) if total_original else 0

        logger.info("=" * 50)
        logger.info("  Compression Summary")
        logger.info(f"  Images processed : {processed}")
        logger.info(f"  Errors           : {errors}")
        logger.info(f"  Original size    : {orig_gb:.2f} GB")
        logger.info(f"  New size         : {new_gb:.2f} GB")
        logger.info(f"  Space saved      : {saved_gb:.2f} GB ({pct:.1f}%)")
        logger.info("=" * 50)
        logger.info(
            f"Originals backed up to: {self.config.backup_dir.resolve()}"
        )


# ======================================================================
# CLI entry point
# ======================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compress Minecraft Wiki images (PNG → WebP)"
    )
    parser.add_argument(
        "--max-dim", type=int, default=1280,
        help="Maximum pixel dimension (default: 1280)"
    )
    parser.add_argument(
        "--quality", type=int, default=80,
        help="WebP quality 0-100 (default: 80)"
    )
    parser.add_argument(
        "--images-dir", type=Path, default=Path("data/raw/images"),
        help="Directory containing PNG images"
    )
    parser.add_argument(
        "--backup-dir", type=Path, default=Path("data/raw/images_backup"),
        help="Directory for original PNG backups"
    )
    args = parser.parse_args()

    config = ProcessorConfig(
        images_dir=args.images_dir,
        backup_dir=args.backup_dir,
        max_dimension=args.max_dim,
        output_quality=args.quality,
    )

    try:
        processor = ImageProcessor(config)
        processor.process_images()
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user — metadata already saved.")
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
