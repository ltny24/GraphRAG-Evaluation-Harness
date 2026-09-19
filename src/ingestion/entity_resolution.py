"""
Entity Resolution
Milestone 1 - Step 1.5

Finds nodes that are conceptually the same (e.g., 'Atorvastatin' vs 'atorvastatin')
and merges them into a single node using APOC (which is available in Neo4j AuraDB).
"""
import os
import certifi
from pathlib import Path
from dotenv import load_dotenv
from neo4j import GraphDatabase, TrustCustomCAs

load_dotenv(Path(__file__).parent.parent.parent / ".env")

def run_entity_resolution():
    uri      = os.getenv("NEO4J_URI")
    user     = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    database = os.getenv("NEO4J_DATABASE", "neo4j")

    bolt_uri = uri.replace("neo4j+s://", "neo4j://")
    driver = GraphDatabase.driver(
        bolt_uri,
        auth=(user, password),
        encrypted=True,
        trusted_certificates=TrustCustomCAs(certifi.where()),
    )

    print("=" * 60)
    print("Entity Resolution (Graph Consolidation)")
    print("=" * 60)
    
    with driver.session(database=database) as session:
        # Step 1: Find duplicates based on case-insensitive names
        print("[1/2] Finding duplicate entities (case-insensitive)...")
        find_query = """
        MATCH (n)
        WITH toLower(n.name) AS canonical_name, collect(n) AS nodes
        WHERE size(nodes) > 1
        RETURN canonical_name, size(nodes) as count, [node IN nodes | node.name] AS variations
        """
        duplicates = session.run(find_query).data()
        
        if not duplicates:
            print("  -> No duplicates found! Graph is already clean.")
        else:
            print(f"  -> Found {len(duplicates)} duplicate groups:")
            for d in duplicates:
                print(f"     - '{d['canonical_name']}' has {d['count']} nodes: {d['variations']}")
        
        # Step 2: Merge them using APOC
        # APOC is built-in for Neo4j AuraDB.
        # properties: "combine" will keep all unique properties
        # mergeRels: true will merge duplicate relationships
        print("\n[2/2] Merging nodes and preserving relationships...")
        merge_query = """
        MATCH (n)
        WITH toLower(n.name) AS canonical_name, collect(n) AS nodes
        WHERE size(nodes) > 1
        CALL apoc.refactor.mergeNodes(nodes, {
            properties: "combine",
            mergeRels: true
        })
        YIELD node
        RETURN node.name AS merged_name
        """
        
        try:
            merged = session.run(merge_query).data()
            print(f"  -> Successfully merged {len(merged)} entity groups.")
        except Exception as e:
            print(f"  [Error] Could not merge using APOC: {e}")
            print("  If APOC is missing, you may need a manual Cypher approach.")

    driver.close()
    print("\nResolution complete. Your Knowledge Graph is now consolidated!")

if __name__ == "__main__":
    run_entity_resolution()
