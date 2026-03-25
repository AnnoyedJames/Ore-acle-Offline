"""
Page Classifier — normalizes raw MediaWiki categories into a clean taxonomy.

Filters out maintenance/meta categories and assigns a top-level page_type
to each page for downstream filtering in hybrid search.

Usage:
    python -m preprocessing.page_classifier
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# Maintenance / meta category patterns to filter out
# ============================================================

MAINTENANCE_EXACT = frozenset({
    "?", "stub", "Stub", "Upcoming", "Verify",
    "Information needed", "Citation needed",
    "Articles to be expanded", "Articles needing rewrite",
    "Article cleanup", "Testing needed",
    "Cleanup without a reason", "Pending split", "Pending merge",
    "Current split", "In development",
    "Check version", "Verify current information",
})

MAINTENANCE_PREFIXES = (
    "Pages with", "Pages using", "Pages needing",
    "Unknown ", "Java Edition upcoming", "Bedrock Edition upcoming",
    "Java Edition specific", "Bedrock Edition specific",
    "Minecraft Education specific",
    "In development Java", "In development Bedrock",
    "Missing discussion", "Minecraft Work in progress",
    "Asset history", "Verify for", "Verify changelog",
    "Resources with invalid",
)

MAINTENANCE_SUFFIXES = (
    "version history", "upcoming tag",
)

# ============================================================
# Page type inference from semantic categories
# ============================================================

# Order matters — first match wins
PAGE_TYPE_RULES: list[tuple[str, set[str]]] = [
    ("mob", {"Hostile mobs", "Passive mobs", "Neutral mobs", "Boss mobs",
             "Monster mobs", "Undead mobs", "Golem mobs", "Humanoid mobs",
             "Aquatic mobs", "Animal mobs", "Tameable mobs", "Flying mobs",
             "End mobs", "Removed mobs", "Joke mobs"}),
    ("block", {"Blocks", "Natural blocks", "Manufactured blocks",
               "Generated structure blocks", "Technical blocks",
               "Non-solid blocks", "Utility blocks", "Redstone"}),
    ("item", {"Items", "Food", "Combat", "Drinks", "Potions",
              "Non-stackable resources", "Stackable resources",
              "Renewable resources"}),
    ("biome", {"Biomes"}),
    ("structure", {"Generated structures", "World features"}),
    ("dimension", {"Dimensions"}),
    ("enchantment", {"Enchantments"}),
    ("effect", {"Effects"}),
    ("mechanic", {"Gameplay", "User interface", "Commands", "Trading",
                  "Achievements", "Advancements"}),
    ("entity", {"Entities", "Stationary entities", "Playable entities",
                "Players", "NPCs", "Characters", "Paintings"}),
    ("version", {"Java Edition versions", "Bedrock Edition versions",
                 "Snapshots", "Pre-releases", "Release candidates",
                 "Version disambiguation pages", "Named updates",
                 "Java Edition", "Bedrock Edition"}),
    ("tutorial", {"Tutorials"}),
    ("franchise", {"Minecraft (franchise)", "Games", "Minecraft Dungeons",
                   "Minecraft Legends", "Minecraft Earth",
                   "Minecraft Education", "LEGO", "Merchandise",
                   "Collaborations"}),
    ("media", {"Music", "Soundtracks", "Books", "Online content",
               "Animated content", "Game trailers", "Resource packs",
               "Textures", "Texture atlases", "DLC promotions"}),
    ("disambiguation", {"Disambiguation pages", "Set index pages"}),
    ("redirect", {"Soft redirects"}),
]


def is_maintenance_category(cat: str) -> bool:
    """Check if a category is a maintenance/meta tag."""
    if cat in MAINTENANCE_EXACT:
        return True
    if any(cat.startswith(p) for p in MAINTENANCE_PREFIXES):
        return True
    if any(cat.endswith(s) for s in MAINTENANCE_SUFFIXES):
        return True
    return False


def filter_categories(raw_categories: list[str]) -> list[str]:
    """Remove maintenance categories and deduplicate."""
    seen = set()
    result = []
    for cat in raw_categories:
        cat = cat.strip()
        if not cat or is_maintenance_category(cat) or cat in seen:
            continue
        seen.add(cat)
        result.append(cat)
    return result


def infer_page_type(semantic_categories: list[str]) -> str:
    """Assign a top-level page_type from semantic categories."""
    cat_set = set(semantic_categories)
    for page_type, trigger_cats in PAGE_TYPE_RULES:
        if cat_set & trigger_cats:
            return page_type
    return "other"


@dataclass
class ClassifierConfig:
    """Configuration for the page classifier."""
    metadata_file: Path = field(
        default_factory=lambda: Path("data/processed/metadata.json")
    )
    output_file: Path = field(
        default_factory=lambda: Path("data/processed/classified_pages.json")
    )


class PageClassifier:
    """
    Reads metadata.json, normalizes categories, assigns page_type.
    Outputs classified_pages.json with per-page type + clean categories.
    """

    def __init__(self, config: Optional[ClassifierConfig] = None):
        self.config = config or ClassifierConfig()
        self.config.output_file.parent.mkdir(parents=True, exist_ok=True)

    def classify_all(self) -> dict:
        """Classify all pages and save results."""
        logger.info(f"Loading metadata from {self.config.metadata_file}")

        with open(self.config.metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        pages = metadata.get("pages", [])
        logger.info(f"Classifying {len(pages)} pages...")

        classifications = {}
        type_counts: dict[str, int] = {}

        for page in pages:
            title = page.get("title", "")
            raw_cats = page.get("categories", [])

            semantic_cats = filter_categories(raw_cats)
            page_type = infer_page_type(semantic_cats)

            classifications[title] = {
                "page_type": page_type,
                "semantic_categories": semantic_cats,
                "raw_categories": raw_cats,
            }

            type_counts[page_type] = type_counts.get(page_type, 0) + 1

        # Summary
        logger.info("Page type distribution:")
        for ptype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            logger.info(f"  {ptype}: {count}")

        output = {
            "total_pages": len(pages),
            "type_distribution": type_counts,
            "pages": classifications,
        }

        with open(self.config.output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved classifications to {self.config.output_file}")
        return output


if __name__ == "__main__":
    classifier = PageClassifier()
    classifier.classify_all()
