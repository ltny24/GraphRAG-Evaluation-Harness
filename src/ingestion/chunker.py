"""
Document Chunker
Milestone 1 - Step 1.2

Splits abstracts into overlapping chunks for downstream NER/RE processing.
Each chunk keeps metadata (source PMID, title) for citation/traceback later.
"""
import io
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

# Fix Unicode output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

DATA_RAW       = Path(__file__).parent.parent.parent / "data" / "raw"
DATA_PROCESSED = Path(__file__).parent.parent.parent / "data" / "processed"


@dataclass
class Chunk:
    """
    A text chunk ready for NER/RE extraction.

    Fields:
        chunk_id   : unique ID = "{pmid}_chunk{index}"
        pmid       : source PubMed article ID (for citation)
        title      : source article title (for citation)
        text       : the actual text content
        char_start : character offset in original abstract (for debugging)
        char_end   : character offset end
    """
    chunk_id:   str
    pmid:       str
    title:      str
    text:       str
    char_start: int
    char_end:   int


def chunk_text(
    text:         str,
    chunk_size:   int = 512,   # characters (not tokens — simpler, no tokenizer needed)
    chunk_overlap: int = 64,   # characters of overlap between consecutive chunks
) -> list[tuple[str, int, int]]:
    """
    Split text into overlapping chunks.

    Returns list of (chunk_text, start, end) tuples.

    Why character-based instead of token-based?
    → Simpler — no need to load a tokenizer just for chunking.
    → 1 token ≈ 4 chars in English, so 512 chars ≈ 128 tokens — well within LLM limits.
    → For this dataset (short abstracts), the difference is negligible.
    """
    chunks = []
    start  = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        # Try to break at a sentence boundary (". ") to avoid mid-sentence cuts
        if end < text_len:
            # Look backwards from 'end' for the last ". " within the chunk
            last_period = text.rfind(". ", start, end)
            if last_period != -1 and last_period > start + chunk_size // 2:
                # Found a good sentence break in the second half of the chunk
                end = last_period + 2  # include the ". "

        chunk_text_content = text[start:end].strip()
        if chunk_text_content:  # skip empty chunks
            chunks.append((chunk_text_content, start, end))

        # Move forward by (chunk_size - overlap) to create the overlap
        next_start = end - chunk_overlap
        if next_start <= start:   # safety guard: always advance forward
            next_start = start + 1
        start = next_start

    return chunks


def chunk_abstracts(
    abstracts:     list[dict],
    chunk_size:    int = 512,
    chunk_overlap: int = 64,
) -> list[Chunk]:
    """
    Process all abstracts into Chunk objects.

    Design decision: We chunk title + abstract together.
    Why? The title often contains the key drug names and relationship type.
    Prepending title to each chunk ensures the LLM always has context
    about what the abstract is about, even mid-text chunks.
    """
    all_chunks = []

    for article in abstracts:
        pmid     = article["pmid"]
        title    = article["title"]
        abstract = article["abstract"]

        # Prepend title for context — critical for mid-abstract chunks
        full_text = f"Title: {title}\n\nAbstract: {abstract}"

        text_chunks = chunk_text(full_text, chunk_size, chunk_overlap)

        for idx, (text, start, end) in enumerate(text_chunks):
            chunk = Chunk(
                chunk_id   = f"{pmid}_chunk{idx}",
                pmid       = pmid,
                title      = title,
                text       = text,
                char_start = start,
                char_end   = end,
            )
            all_chunks.append(chunk)

    return all_chunks


def save_chunks(chunks: list[Chunk], filename: str = "chunks.json") -> Path:
    """Save chunks to data/processed/ for reuse in NER step."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    output_path = DATA_PROCESSED / filename

    data = [asdict(c) for c in chunks]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return output_path


def load_abstracts(path: Path) -> list[dict]:
    """Load abstracts from JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ─── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent.parent / ".env")

    CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", 512))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 64))

    print("=" * 60)
    print("Document Chunker")
    print(f"chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP}")
    print("=" * 60)

    # Load abstracts
    abstracts_path = DATA_RAW / "pubmed_ddi.json"
    abstracts = load_abstracts(abstracts_path)
    print(f"\nLoaded {len(abstracts)} abstracts from {abstracts_path.name}")

    # Chunk
    chunks = chunk_abstracts(abstracts, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"Generated {len(chunks)} chunks total")

    # Stats
    avg_len = sum(len(c.text) for c in chunks) / len(chunks)
    min_len = min(len(c.text) for c in chunks)
    max_len = max(len(c.text) for c in chunks)
    print(f"\nChunk length stats (chars):")
    print(f"  avg = {avg_len:.0f}")
    print(f"  min = {min_len}")
    print(f"  max = {max_len}")

    # Save
    output_path = save_chunks(chunks)
    print(f"\nSaved {len(chunks)} chunks to: data/processed/chunks.json")

    # Preview first 2 chunks
    print("\n--- Preview: first 2 chunks ---")
    for c in chunks[:2]:
        print(f"\n[{c.chunk_id}]")
        print(f"  PMID  : {c.pmid}")
        print(f"  Title : {c.title[:60]}...")
        print(f"  Text  : {c.text[:200]}...")
        print(f"  Length: {len(c.text)} chars")
