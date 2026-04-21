import json
import logging
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
import ijson

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    input_path = Path("data/processed/metadata.json")
    output_path = Path("data/processed/chunks_langchain.json")
    
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2048,
        chunk_overlap=256,
        length_function=len
    )

    chunks = []
    chunk_id_counter = 0

    logger.info(f"Streaming metadata from {input_path} using ijson...")
    
    try:
        with open(input_path, 'rb') as f:
            pages = ijson.items(f, 'pages.item')
            for page_data in tqdm(pages):
                page_url = page_data.get("url", "Unknown")
                page_title = page_data.get("title", "Unknown")
                
                # Reconstruct full page text by concatenating section texts
                sections = page_data.get("sections", [])
                section_texts = []
                for sec in sections:
                    header = sec.get("heading", "")
                    sec_text = sec.get("text", "")
                    if header and header.lower() != "lead":
                        section_texts.append(f"## {header}\n{sec_text}")
                    else:
                        section_texts.append(sec_text)
                        
                text = "\n\n".join(section_texts)
                    
                if not text.strip():
                    continue

                splits = text_splitter.split_text(text)
                
                for i, split in enumerate(splits):
                    chunk = {
                        "chunk_id": f"langchain_{chunk_id_counter}",
                        "page_url": page_url,
                        "page_title": page_title,
                        "text": split,
                        "chunk_index": i
                    }
                    chunks.append(chunk)
                    chunk_id_counter += 1

        logger.info(f"Generated {len(chunks)} chunks.")
        
        logger.info(f"Writing to {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2)
            
        logger.info("Done!")
    except Exception as e:
        logger.error(f"Fatal error during processing: {e}")

if __name__ == "__main__":
    main()
