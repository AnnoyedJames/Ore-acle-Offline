"""
Script to upload keyword index to Supabase ONLY.

Use this if Pinecone upload succeeded but Supabase failed.
Skips vector upload and only populates the keyword index.
"""

import sys
import logging
from pathlib import Path
import json

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from database.uploader import DatabaseUploader

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    uploader = DatabaseUploader()
    chunks_file = uploader.config.chunks_file
    
    if not chunks_file.exists():
        logger.error(f"Chunks file not found: {chunks_file}")
        return

    logger.info(f"Loading chunks from {chunks_file}...")
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    logger.info(f"Loaded {len(chunks)} chunks. Starting Supabase upload...")
    
    # Upload to Supabase only
    count = uploader.upload_to_supabase(chunks)
    logger.info(f"Supabase upload complete: {count} rows inserted/updated")

if __name__ == "__main__":
    main()
