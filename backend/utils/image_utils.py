"""
Shared image path utilities for Ore-acle Offline.

Provides a deterministic mapping from wiki image URLs to local filenames.
Used by both the image downloader and text cleaner so they agree on names.
"""

import re
from urllib.parse import unquote, urlparse


def wiki_url_to_filename(url: str) -> str | None:
    """
    Convert a Minecraft Wiki image URL to a local .webp filename.

    Examples:
        "https://minecraft.wiki/images/Water_JE16-a1.png"
            → "Water_JE16-a1.webp"
        "https://minecraft.wiki/images/thumb/Water_spread.png/300px-Water_spread.png"
            → "Water_spread.webp"
        "https://minecraft.wiki/images/Water_BE_%28animated%29.png"
            → "Water_BE_(animated).webp"

    Returns None if the URL doesn't look like a wiki image.
    """
    # Strip query params and fragments
    clean = url.split("?")[0].split("#")[0]

    # URL-decode percent-encoding
    clean = unquote(clean)

    # Handle /thumb/ URLs: extract the original filename (before the resize)
    # e.g. /images/thumb/Water.png/300px-Water.png → Water.png
    thumb_match = re.search(r"/images/thumb/(.+)/[^/]+$", clean)
    if thumb_match:
        original_path = thumb_match.group(1)
    else:
        # Standard URL: /images/Water.png → Water.png
        img_match = re.search(r"/images/(.+)$", clean)
        if not img_match:
            return None
        original_path = img_match.group(1)

    # The filename is the last path component (handles /images/a/b/Name.png)
    filename = original_path.rsplit("/", 1)[-1]

    # Strip original extension, add .webp
    stem = filename.rsplit(".", 1)[0] if "." in filename else filename
    if not stem:
        return None

    # Sanitize stem for local filesystem (remove invalid Windows/Linux chars)
    stem = re.sub(r'[<>:"/\\|?*]', '_', stem)

    return f"{stem}.webp"


def get_original_url(url: str) -> str:
    """
    Convert a thumbnail URL to the original full-resolution wiki URL.

    Input:  https://minecraft.wiki/images/thumb/a/b/Name.png/300px-Name.png
    Output: https://minecraft.wiki/images/a/b/Name.png
    """
    match = re.search(r"/thumb/(.+)/[^/]+$", url)
    if match:
        base = url.split("/thumb/")[0]
        return f"{base}/{match.group(1)}"
    return url
