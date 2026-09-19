import json
from pathlib import Path

with open("data/processed/chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Total chunks: {len(chunks)}\n")
for i in range(3):
    c = chunks[i]
    print(f"--- Chunk {i} ---")
    print(f"ID: {c['chunk_id']}")
    print(f"Text:\n{c['text']}\n")
