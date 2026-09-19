"""
Neo4j Graph Loader
Milestone 1 - Step 1.4

Reads extracted entities and relations from JSON and loads them into Neo4j AuraDB.
Uses Cypher MERGE commands to avoid duplicates (UPSERT behavior).
"""
import json
import os
import certifi
from pathlib import Path
from dotenv import load_dotenv
from neo4j import GraphDatabase, TrustCustomCAs

# Load env vars
load_dotenv(Path(__file__).parent.parent.parent / ".env")

class GraphLoader:
    def __init__(self):
        uri      = os.getenv("NEO4J_URI")
        user     = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")
        self.database = os.getenv("NEO4J_DATABASE", "neo4j")

        # Driver v6 connection fix for AuraDB on Windows
        bolt_uri = uri.replace("neo4j+s://", "neo4j://")
        self.driver = GraphDatabase.driver(
            bolt_uri,
            auth=(user, password),
            encrypted=True,
            trusted_certificates=TrustCustomCAs(certifi.where()),
        )
        self.driver.verify_connectivity()
        print("[Neo4j] Connected successfully.")

    def close(self):
        self.driver.close()

    def clear_database(self):
        """WARNING: Deletes all nodes and relationships. Useful for testing."""
        print("[Neo4j] Clearing database...")
        with self.driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")

    def insert_elements(self, data: list[dict]):
        """Insert entities and relationships into Neo4j."""
        with self.driver.session(database=self.database) as session:
            for item in data:
                extraction = item.get("extraction")
                if not extraction:
                    continue

                # 1. Insert Entities
                for ent in extraction.get("entities", []):
                    # Sanitize label (remove spaces, special chars) to prevent Cypher injection
                    safe_label = "".join(c for c in ent["type"] if c.isalnum())
                    if not safe_label:
                        safe_label = "Entity"
                    
                    # MERGE creates the node if it doesn't exist, or matches it if it does
                    query = f"""
                    MERGE (n:{safe_label} {{name: $name}})
                    ON CREATE SET n.created_from = $pmid
                    """
                    session.run(query, name=ent["name"], pmid=item["pmid"])

                # 2. Insert Relationships
                for rel in extraction.get("relations", []):
                    # Sanitize relation type (uppercase, alphanumeric + underscore)
                    safe_rel_type = "".join(c.upper() for c in rel["type"] if c.isalnum() or c == "_")
                    if not safe_rel_type:
                        safe_rel_type = "RELATED_TO"

                    # Match source and target nodes by name, then create relationship
                    query = f"""
                    MATCH (source {{name: $source_name}})
                    MATCH (target {{name: $target_name}})
                    MERGE (source)-[r:{safe_rel_type}]->(target)
                    """
                    session.run(query, 
                                source_name=rel["source"], 
                                target_name=rel["target"])

        print("[Neo4j] Finished inserting elements.")


# ─── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    DATA_IN = Path(__file__).parent.parent.parent / "data" / "processed" / "graph_elements.json"
    
    if not DATA_IN.exists():
        print(f"Error: {DATA_IN} not found. Run extractor.py first.")
        exit(1)
        
    with open(DATA_IN, encoding="utf-8") as f:
        graph_data = json.load(f)
        
    print(f"Loaded {len(graph_data)} chunks with extractions.")
    
    # Initialize Loader
    loader = GraphLoader()
    
    # Clear old data for a fresh start (optional, but good for Milestone 1)
    loader.clear_database()
    
    # Insert new data
    print("Inserting nodes and relationships...")
    loader.insert_elements(graph_data)
    
    loader.close()
    print("Done! Open Neo4j Browser (https://console.neo4j.io) to view the graph.")
    print("Run this query in Neo4j Browser to see everything: MATCH (n) RETURN n")
