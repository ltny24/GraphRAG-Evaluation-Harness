"""
PubMed Abstract Fetcher
Milestone 1 - Step 1.1

Fetches abstracts about Drug-Drug Interactions from PubMed
using the free NCBI E-utilities API (no API key needed for small requests).

API docs: https://www.ncbi.nlm.nih.gov/books/NBK25499/
"""
import json
import time
import urllib.request
import urllib.parse
import sys
import io
from pathlib import Path
from dataclasses import dataclass, asdict

# Fix Unicode output on Windows terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


# ─── Config ────────────────────────────────────────────────────────────────
BASE_URL   = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
SEARCH_URL = f"{BASE_URL}/esearch.fcgi"
FETCH_URL  = f"{BASE_URL}/efetch.fcgi"

DATA_DIR   = Path(__file__).parent.parent.parent / "data" / "raw"


@dataclass
class Abstract:
    """Represents one PubMed article abstract."""
    pmid: str           # PubMed ID (unique)
    title: str          # Article title
    abstract: str       # Full abstract text
    authors: list[str]  # Author list
    year: str           # Publication year
    journal: str        # Journal name


# ─── Step A: Search for PMIDs ──────────────────────────────────────────────
def search_pubmed(query: str, max_results: int = 50) -> list[str]:
    """
    Search PubMed and return a list of PMIDs matching the query.
    
    Why search first then fetch?
    PubMed E-utilities separates search (esearch) from retrieval (efetch)
    so we can batch-fetch without hitting rate limits.
    """
    params = {
        "db":       "pubmed",
        "term":     query,
        "retmax":   max_results,
        "retmode":  "json",
        "sort":     "relevance",
    }
    url = SEARCH_URL + "?" + urllib.parse.urlencode(params)
    print(f"[Search] Query: {query!r}")
    print(f"[Search] URL: {url}")

    with urllib.request.urlopen(url) as resp:
        data = json.loads(resp.read())

    pmids = data["esearchresult"]["idlist"]
    total = data["esearchresult"]["count"]
    print(f"[Search] Found {total} total results, fetching {len(pmids)} PMIDs")
    return pmids


# ─── Step B: Fetch full abstracts by PMID ──────────────────────────────────
def fetch_abstracts(pmids: list[str]) -> list[Abstract]:
    """
    Fetch full abstract data for a list of PMIDs.
    
    We use rettype=abstract&retmode=text to get plain text,
    then parse it into structured Abstract objects.
    
    Rate limit: NCBI allows 3 requests/sec without API key.
    We add a 0.4s delay between batches to be safe.
    """
    abstracts = []
    batch_size = 10  # fetch 10 at a time

    for i in range(0, len(pmids), batch_size):
        batch = pmids[i : i + batch_size]
        print(f"[Fetch] Batch {i//batch_size + 1}: PMIDs {batch[0]}...{batch[-1]}")

        params = {
            "db":      "pubmed",
            "id":      ",".join(batch),
            "rettype": "xml",
            "retmode": "xml",
        }
        url = FETCH_URL + "?" + urllib.parse.urlencode(params)

        with urllib.request.urlopen(url) as resp:
            xml_data = resp.read().decode("utf-8")

        batch_abstracts = _parse_pubmed_xml(xml_data)
        abstracts.extend(batch_abstracts)
        print(f"         Parsed {len(batch_abstracts)} abstracts from batch")

        time.sleep(0.4)  # respect NCBI rate limit

    return abstracts


# ─── Step C: Parse XML response ────────────────────────────────────────────
def _parse_pubmed_xml(xml_text: str) -> list[Abstract]:
    """
    Parse PubMed XML to extract structured fields.
    
    We use Python's built-in xml.etree.ElementTree — no extra dependencies.
    XPath-like queries to find: PMID, ArticleTitle, AbstractText, etc.
    """
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml_text)
    results = []

    for article in root.findall(".//PubmedArticle"):
        try:
            # PMID
            pmid_el = article.find(".//PMID")
            pmid = pmid_el.text if pmid_el is not None else "unknown"

            # Title
            title_el = article.find(".//ArticleTitle")
            title = "".join(title_el.itertext()) if title_el is not None else ""

            # Abstract (can have multiple structured sections)
            abstract_parts = []
            for ab_el in article.findall(".//AbstractText"):
                label = ab_el.get("Label", "")
                text  = "".join(ab_el.itertext())
                if label:
                    abstract_parts.append(f"{label}: {text}")
                else:
                    abstract_parts.append(text)
            abstract_text = " ".join(abstract_parts)

            # Skip articles with no abstract (just titles/editorials)
            if not abstract_text.strip():
                continue

            # Authors
            authors = []
            for author in article.findall(".//Author"):
                last  = author.findtext("LastName", "")
                first = author.findtext("ForeName", "")
                if last:
                    authors.append(f"{last} {first}".strip())

            # Year
            year_el = article.find(".//PubDate/Year")
            if year_el is None:
                year_el = article.find(".//PubDate/MedlineDate")
            year = year_el.text[:4] if year_el is not None else "unknown"

            # Journal
            journal_el = article.find(".//Journal/Title")
            journal = journal_el.text if journal_el is not None else "unknown"

            results.append(Abstract(
                pmid=pmid,
                title=title,
                abstract=abstract_text,
                authors=authors,
                year=year,
                journal=journal,
            ))
        except Exception as e:
            print(f"  [WARN] Failed to parse article: {e}")
            continue

    return results


# ─── Step D: Save to disk ──────────────────────────────────────────────────
def save_abstracts(abstracts: list[Abstract], filename: str = "pubmed_ddi.json") -> Path:
    """
    Save abstracts as JSON for reuse.
    
    Why JSON not CSV?
    Authors is a list → nested structure → JSON handles it naturally.
    Also easier to load back as dataclasses later.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_DIR / filename

    data = [asdict(a) for a in abstracts]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n[Save] Saved {len(abstracts)} abstracts to: {output_path}")
    return output_path


# ─── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Drug-Drug Interaction abstracts — rich in Entity-Relation pairs
    QUERY = (
        'drug interaction[Title/Abstract] AND '
        '(pharmacokinetic OR inhibit OR induce)[Title/Abstract] AND '
        'clinical[Title/Abstract]'
    )

    print("=" * 60)
    print("PubMed Fetcher — Drug-Drug Interaction Abstracts")
    print("=" * 60)

    pmids     = search_pubmed(QUERY, max_results=50)
    abstracts = fetch_abstracts(pmids)
    path      = save_abstracts(abstracts)

    print(f"\nTotal abstracts saved: {len(abstracts)}")
    print(f"\nSample abstract:")
    if abstracts:
        sample = abstracts[0]
        print(f"  PMID    : {sample.pmid}")
        print(f"  Title   : {sample.title[:80]}...")
        print(f"  Abstract: {sample.abstract[:200]}...")
        print(f"  Year    : {sample.year}")
        print(f"  Journal : {sample.journal}")
