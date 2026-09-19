"""
Vector Loader
Milestone 2 - Part 1

Embeds chunks using SentenceTransformers (local embedding model)
and loads them into Qdrant (local file-based vector database) for fast semantic search.
"""
import json
import uuid
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")

# Ensure cache directories are created as defined in .env
import os
os.makedirs(os.getenv("HF_HOME", "models/huggingface"), exist_ok=True)
os.makedirs(os.getenv("SENTENCE_TRANSFORMERS_HOME", "models/sentence_transformers"), exist_ok=True)

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast, 384 dimensions
COLLECTION_NAME = "pubmed_chunks"
DATA_IN         = Path(__file__).parent.parent.parent / "data" / "processed" / "chunks.json"
QDRANT_PATH     = Path(__file__).parent.parent.parent / "data" / "qdrant_db"

def init_qdrant() -> QdrantClient:
    """Initialize local Qdrant client and create collection if missing."""
    os.makedirs(QDRANT_PATH, exist_ok=True)
    
    # We use path mode so data is persisted to disk locally without Docker
    client = QdrantClient(path=str(QDRANT_PATH))
    
    # Check if collection exists
    if not client.collection_exists(COLLECTION_NAME):
        print(f"[Qdrant] Creating collection '{COLLECTION_NAME}' (384 dims, Cosine)...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
    else:
        print(f"[Qdrant] Collection '{COLLECTION_NAME}' already exists.")
        
    return client

def generate_uuid(chunk_id: str) -> str:
    """Qdrant requires UUIDs or Integers. We hash chunk_id to a stable UUID."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))

def main():
    print("=" * 60)
    print("Vector Loader (SentenceTransformers -> Qdrant)")
    print("=" * 60)
    
    if not DATA_IN.exists():
        print(f"Error: {DATA_IN} not found. Run chunker first.")
        return

    # 1. Load Data
    with open(DATA_IN, encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks.")

    # 2. Init Model
    print(f"\nLoading embedding model '{EMBEDDING_MODEL}'...")
    print("  (This will download the model on the first run. Don't worry, it caches on D: drive)")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # 3. Init Vector DB
    client = init_qdrant()

    # 4. Process and Upload (Batching for performance)
    BATCH_SIZE = 64
    total = len(chunks)
    
    print("\nStarting vector embedding and insertion...")
    for i in range(0, total, BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        
        # We embed the "text" field
        texts = [c["text"] for c in batch]
        embeddings = model.encode(texts, show_progress_bar=False)
        
        # Create PointStructs for Qdrant
        points = []
        for c, emb in zip(batch, embeddings):
            # Store the chunk data in the payload so we can retrieve it later
            payload = {
                "chunk_id": c["chunk_id"],
                "pmid": c["pmid"],
                "title": c["title"],
                "text": c["text"]
            }
            point = PointStruct(
                id=generate_uuid(c["chunk_id"]),
                vector=emb.tolist(),
                payload=payload
            )
            points.append(point)
            
        # Insert into Qdrant
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        
        print(f"  Inserted {min(i+BATCH_SIZE, total)} / {total} chunks...")

    print("\nVector ingestion complete! You can now run semantic search.")

if __name__ == "__main__":
    main()
