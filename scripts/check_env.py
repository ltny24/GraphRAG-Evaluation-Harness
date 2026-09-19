"""
Environment & Connection Health Check
Milestone 0 - Verify all services are reachable
"""
import os
import sys
import io
from pathlib import Path

# Force UTF-8 output on Windows to avoid cp1252 UnicodeEncodeError
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Load .env from project root
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from rich.console import Console
from rich.table import Table
from rich import print as rprint

console = Console(force_terminal=True, highlight=False)

results = []

OK  = "[green]  OK [/green]"
ERR = "[red] FAIL[/red]"
WARN = "[yellow] SKIP[/yellow]"

# ─────────────────────────────────────────────
# 1. Check environment variables
# ─────────────────────────────────────────────
def check_env_vars():
    console.rule("[bold blue]1. Environment Variables")
    required = [
        "GEMINI_API_KEY",
        "NEO4J_URI",
        "NEO4J_USERNAME",
        "NEO4J_PASSWORD",
    ]
    all_ok = True
    for var in required:
        val = os.getenv(var)
        if val:
            masked = val[:8] + "..." + val[-4:] if len(val) > 12 else "***"
            rprint(f"  [{OK}] {var} = {masked}")
        else:
            rprint(f"  [{ERR}] {var} = NOT SET")
            all_ok = False
    results.append(("Env Vars", "[green]OK[/green]" if all_ok else "[red]MISSING[/red]", "green" if all_ok else "red"))
    return all_ok

# ─────────────────────────────────────────────
# 2. Check Neo4j AuraDB connection
# ─────────────────────────────────────────────
def check_neo4j():
    console.rule("[bold blue]2. Neo4j AuraDB Connection")
    try:
        import certifi
        from neo4j import GraphDatabase, TrustCustomCAs

        uri      = os.getenv("NEO4J_URI")
        user     = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")
        database = os.getenv("NEO4J_DATABASE", "neo4j")

        # neo4j driver v6 + AuraDB: must use neo4j:// (not neo4j+s://)
        # with explicit encrypted=True and TrustCustomCAs for SSL
        bolt_uri = uri.replace("neo4j+s://", "neo4j://")
        driver = GraphDatabase.driver(
            bolt_uri,
            auth=(user, password),
            encrypted=True,
            trusted_certificates=TrustCustomCAs(certifi.where()),
        )
        driver.verify_connectivity()

        with driver.session(database=database) as session:
            result = session.run("RETURN 'GraphRAG Connected!' AS msg, datetime() AS ts")
            record = result.single()
            rprint(f"  [{OK}] Connected!  Server msg: [bold]{record['msg']}[/bold]")
            rprint(f"         Timestamp : {record['ts']}")

        driver.close()
        results.append(("Neo4j AuraDB", "[green]Connected[/green]", "green"))
        return True

    except Exception as e:
        rprint(f"  [{ERR}] Connection failed: {e}")
        results.append(("Neo4j AuraDB", f"[red]{type(e).__name__}[/red]", "red"))
        return False

# ─────────────────────────────────────────────
# 3. Check Gemini API
# ─────────────────────────────────────────────
def check_gemini():
    console.rule("[bold blue]3. Gemini API")
    try:
        from google import genai
        api_key = os.getenv("GEMINI_API_KEY")
        client  = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents="Reply with exactly: 'GraphRAG Harness ready.' -- nothing else."
        )
        reply = response.text.strip()
        rprint(f"  [{OK}] Gemini responded: [bold]{reply}[/bold]")
        results.append(("Gemini API", "[green]OK[/green]", "green"))
        return True

    except Exception as e:
        rprint(f"  [{ERR}] API call failed: {e}")
        results.append(("Gemini API", f"[red]{type(e).__name__}[/red]", "red"))
        return False

# ─────────────────────────────────────────────
# 4. Check Sentence Transformers (Embeddings)
# ─────────────────────────────────────────────
def check_embeddings():
    console.rule("[bold blue]4. Sentence Transformers (Local Embeddings)")
    try:
        import importlib.util
        if importlib.util.find_spec("sentence_transformers") is None:
            rprint(f"  [{WARN}] sentence-transformers not installed yet.")
            rprint("         Run: pip install sentence-transformers  (needs ~2 GB for torch)")
            results.append(("Sentence Transformers", "[yellow]Not installed (skipped)[/yellow]", "yellow"))
            return False

        from sentence_transformers import SentenceTransformer
        rprint("         Loading all-MiniLM-L6-v2 (downloads ~90 MB on first run)...")
        model     = SentenceTransformer("all-MiniLM-L6-v2")
        embedding = model.encode("Drug-drug interactions in clinical pharmacology")
        rprint(f"  [{OK}] Model loaded. Vector dim: [bold]{len(embedding)}[/bold]")
        results.append(("Sentence Transformers", f"[green]dim={len(embedding)}[/green]", "green"))
        return True

    except Exception as e:
        rprint(f"  [{ERR}] Failed: {e}")
        results.append(("Sentence Transformers", f"[red]{type(e).__name__}[/red]", "red"))
        return False

# ─────────────────────────────────────────────
# 5. Check Qdrant (in-memory mode)
# ─────────────────────────────────────────────
def check_qdrant():
    console.rule("[bold blue]5. Qdrant Vector DB (in-memory mode)")
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        client = QdrantClient(":memory:")
        client.create_collection(
            collection_name="test_graphrag",
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        names = [c.name for c in client.get_collections().collections]
        rprint(f"  [{OK}] Qdrant in-memory OK. Collections: {names}")
        rprint("         NOTE: Production will connect to localhost:6333")
        results.append(("Qdrant (in-memory)", "[green]OK[/green]", "green"))
        return True

    except Exception as e:
        rprint(f"  [{ERR}] Failed: {e}")
        results.append(("Qdrant", f"[red]{type(e).__name__}[/red]", "red"))
        return False

# ─────────────────────────────────────────────
# Summary Table
# ─────────────────────────────────────────────
def print_summary():
    console.rule("[bold yellow]MILESTONE 0 -- SUMMARY")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Component",  style="cyan",  width=25)
    table.add_column("Status",     width=30)

    all_pass = True
    for name, status, color in results:
        table.add_row(name, status)
        if color == "red":
            all_pass = False

    console.print(table)

    if all_pass:
        rprint("\n[bold green]All systems GO! Ready for Milestone 1.[/bold green]")
    else:
        rprint("\n[bold red]Some checks failed. Fix issues above before proceeding.[/bold red]")

    return all_pass

# ─────────────────────────────────────────────
if __name__ == "__main__":
    console.print("\n[bold]GraphRAG & Evaluation Harness -- Environment Check[/bold]\n")
    check_env_vars()
    check_neo4j()
    check_gemini()
    check_embeddings()
    check_qdrant()
    ok = print_summary()
    sys.exit(0 if ok else 1)
