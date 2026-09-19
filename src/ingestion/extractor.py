"""
LLM Extractor for GraphRAG
Milestone 1 - Step 1.3

Uses Gemini to extract Entities and Relationships from text chunks.
We use Pydantic models to enforce Structured Output from Gemini,
ensuring we get clean, parseable JSON every time.
"""
import json
import os
import time
from pathlib import Path
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# We use the NEW google-genai SDK
from google import genai
from google.genai import types

# Load env vars
load_dotenv(Path(__file__).parent.parent.parent / ".env")

# ─── 1. Define Structured Output Schema (Pydantic) ─────────────────────────
class Entity(BaseModel):
    name: str = Field(description="Name of the entity (e.g., 'Atorvastatin', 'CYP3A4', 'Colorectal cancer')")
    type: str = Field(description="Category of entity. Allowed types: Drug, Enzyme, Disease, Protein, Chemical, Other")

class Relation(BaseModel):
    source: str = Field(description="Name of the source entity")
    target: str = Field(description="Name of the target entity")
    type: str = Field(description="Type of relationship. E.g., INHIBITS, METABOLIZES, TREATS, INCREASES_RISK, INTERACTS_WITH")

class ExtractionResult(BaseModel):
    entities: list[Entity]
    relations: list[Relation]


# ─── 2. Setup LLM Client ───────────────────────────────────────────────────
client = genai.Client()  # Automatically picks up GEMINI_API_KEY from env
MODEL_ID = "gemini-3.6-flash"


def extract_graph_elements(chunk_text: str) -> ExtractionResult | None:
    """Pass chunk to Gemini and ask for entities and relations."""
    
    prompt = f"""
You are a medical data extraction expert.
Analyze the following text chunk and extract all relevant biomedical entities and their relationships.

Text:
\"\"\"{chunk_text}\"\"\"

Focus on:
- Drugs and medications
- Enzymes (like CYP3A4, P-gp)
- Diseases and conditions
- How they interact (pharmacokinetics, inhibition, treatment)
"""

    try:
        # We enforce the JSON schema via response_schema
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ExtractionResult,
                temperature=0.1, # Low temperature for more deterministic extraction
            ),
        )
        
        # Parse the JSON string back into our Pydantic model for validation
        data_dict = json.loads(response.text)
        return ExtractionResult(**data_dict)
        
    except Exception as e:
        print(f"[LLM Error] Failed to extract: {e}")
        return None


# ─── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    DATA_IN  = Path(__file__).parent.parent.parent / "data" / "processed" / "chunks.json"
    DATA_OUT = Path(__file__).parent.parent.parent / "data" / "processed" / "graph_elements.json"
    
    # Load all chunks
    with open(DATA_IN, encoding="utf-8") as f:
        chunks = json.load(f)
        
    print(f"Loaded {len(chunks)} chunks.")
    
    # LIMIT FOR DEVELOPMENT:
    # Processing 3400+ chunks takes time and hits rate limits.
    # For this milestone test, we will only process the first 3 chunks.
    LIMIT = 3
    print(f"Processing first {LIMIT} chunks to build a mini-graph...")
    
    results = []
    
    for i in range(min(LIMIT, len(chunks))):
        c = chunks[i]
        print(f"\n[{i+1}/{LIMIT}] Extracting from chunk: {c['chunk_id']} ...")
        
        extracted = extract_graph_elements(c['text'])
        
        if extracted:
            print(f"  -> Found {len(extracted.entities)} entities, {len(extracted.relations)} relations.")
            # Store everything together
            results.append({
                "chunk_id": c["chunk_id"],
                "pmid": c["pmid"],
                "text": c["text"],
                "extraction": extracted.model_dump()
            })
        
        # Sleep slightly to respect rate limits if we were doing a big batch
        time.sleep(2)
        
    # Save to disk
    with open(DATA_OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    print(f"\nSaved extraction results to {DATA_OUT.name}")
    
    # Preview
    if results:
        print("\n--- Extraction Preview (First Chunk) ---")
        ex = results[0]["extraction"]
        print("Entities:")
        for ent in ex["entities"]:
            print(f"  - [{ent['type']}] {ent['name']}")
        print("Relations:")
        for rel in ex["relations"]:
            print(f"  - ({rel['source']}) --[{rel['type']}]--> ({rel['target']})")
