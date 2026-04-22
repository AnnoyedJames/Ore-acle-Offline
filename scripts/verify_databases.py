import sqlite3
import os
try:
    import chromadb
except ImportError:
    print("Please install chromadb: pip install chromadb")
    exit(1)

def verify_sqlite(db_path, name):
    print(f"\n--- Verifying SQLite FTS ({name}) ---")
    if not os.path.exists(db_path):
        print(f"File not found: {db_path}")
        return

    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        
        # Total Row count
        cur.execute("SELECT COUNT(*) FROM chunks_fts")
        total_rows = cur.fetchone()[0]
        print(f"Total Rows: {total_rows}")
        
        # Crafting Recipe count
        cur.execute("SELECT COUNT(*) FROM chunks_fts WHERE text LIKE '%[Crafting Recipe:%'")
        recipe_count = cur.fetchone()[0]
        print(f"Crafting Recipe instances: {recipe_count}")
        
        conn.close()
    except Exception as e:
        print(f"Error reading {db_path}: {e}")

def verify_chroma(db_path):
    print("\n--- Verifying ChromaDB Collections ---")
    if not os.path.exists(db_path):
        print(f"Directory not found: {db_path}")
        return

    try:
        client = chromadb.PersistentClient(path=db_path)
        collections = client.list_collections()
        if not collections:
            print("No collections found in ChromaDB.")
            return
            
        for col_name in collections:
            try:
                name = getattr(col_name, 'name', col_name)
                col = client.get_collection(name)
                print(f"Collection '{name}' row count: {col.count()}")
            except Exception as e:
                print(f"Error accessing collection {name}: {e}")
                
    except Exception as e:
        print(f"Error reading ChromaDB at {db_path}: {e}")

if __name__ == "__main__":
    verify_sqlite("data/sqlite_fts.db", "Section-Aware Custom Chunker")
    verify_sqlite("data/sqlite_fts_langchain.db", "LangChain Chunker")
    verify_chroma("data/chroma_db")
    print("\nVerification Complete.")
