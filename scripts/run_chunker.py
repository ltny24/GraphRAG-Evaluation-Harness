"""Runner: chạy chunker và lưu output ra file để tránh encoding issues"""
import json
from pathlib import Path

DATA_RAW = Path("data/raw/pubmed_ddi.json")
DATA_OUT = Path("data/processed/chunks.json")
LOG_OUT  = Path("scripts/chunker_result.txt")

DATA_OUT.parent.mkdir(parents=True, exist_ok=True)

lines = []

# Load abstracts
with open(DATA_RAW, encoding="utf-8") as f:
    abstracts = json.load(f)
lines.append(f"Loaded {len(abstracts)} abstracts")

# Chunk
chunks = []
CHUNK_SIZE = 512
OVERLAP    = 64

for article in abstracts:
    pmid      = article["pmid"]
    title     = article["title"]
    abstract  = article["abstract"]
    full_text = "Title: " + title + "\n\nAbstract: " + abstract
    text_len  = len(full_text)

    start = 0
    idx   = 0
    while start < text_len:
        end = min(start + CHUNK_SIZE, text_len)
        if end < text_len:
            last_p = full_text.rfind(". ", start, end)
            if last_p != -1 and last_p > start + CHUNK_SIZE // 2:
                end = last_p + 2
        text = full_text[start:end].strip()
        if text:
            chunks.append({
                "chunk_id":   pmid + "_chunk" + str(idx),
                "pmid":       pmid,
                "title":      title,
                "text":       text,
                "char_start": start,
                "char_end":   end,
            })
            idx += 1
        next_start = end - OVERLAP
        if next_start <= start:
            next_start = start + 1
        start = next_start

# Save chunks
with open(DATA_OUT, "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

# Stats
lengths = [len(c["text"]) for c in chunks]
lines.append("Total chunks : " + str(len(chunks)))
lines.append("Avg length   : " + str(round(sum(lengths)/len(lengths))) + " chars")
lines.append("Min / Max    : " + str(min(lengths)) + " / " + str(max(lengths)))
lines.append("Saved to     : " + str(DATA_OUT))
lines.append("")
lines.append("--- Sample chunk 0 ---")
c0 = chunks[0]
lines.append("ID    : " + c0["chunk_id"])
lines.append("Text  : " + c0["text"][:300])
lines.append("")
lines.append("--- Sample chunk 1 ---")
if len(chunks) > 1:
    c1 = chunks[1]
    lines.append("ID    : " + c1["chunk_id"])
    lines.append("Text  : " + c1["text"][:300])

# Write log to file
result = "\n".join(lines)
with open(LOG_OUT, "w", encoding="utf-8") as f:
    f.write(result)
