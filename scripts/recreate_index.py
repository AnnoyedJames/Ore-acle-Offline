import os
import time
import logging
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Load env vars
load_dotenv()
load_dotenv(dotenv_path="backend/.env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def recreate_index():
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        logger.error("PINECONE_API_KEY not found in environment")
        return

    pc = Pinecone(api_key=api_key)
    index_name = "ore-acle"
    dimension = 1024  # Multilingual-e5-large
    metric = "cosine"

    # Check if index exists
    existing_indexes = [i.name for i in pc.list_indexes()]
    
    if index_name in existing_indexes:
        logger.info(f"Deleting existing index '{index_name}'...")
        pc.delete_index(index_name)
        
        # Wait for deletion
        while index_name in [i.name for i in pc.list_indexes()]:
            time.sleep(1)
        logger.info("Index deleted.")

    logger.info(f"Creating new index '{index_name}' with dimension {dimension}...")
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric=metric,
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

    # Wait for ready
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)
    
    logger.info(f"Index '{index_name}' is ready.")

if __name__ == "__main__":
    recreate_index()
