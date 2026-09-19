"""
Milestone 3 - Evaluation Harness
Step 3.1: Dataset Generator (Multi-hop QA generation)

Queries Neo4j for subgraphs and asks Gemini to generate tricky questions.
Saves the questions to a JSON dataset for A/B testing.
"""
import os
import json
import time
import certifi
from pathlib import Path
from dotenv import load_dotenv

from google import genai
from pydantic import BaseModel
from neo4j import GraphDatabase, TrustCustomCAs

load_dotenv(Path(__file__).parent.parent.parent / ".env")

class QAPair(BaseModel):
    question: str
    ground_truth: str

class DatasetGenerator:
    def __init__(self):
        print("Initializing Dataset Generator...")
        self.llm_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model_id = "gemini-3.6-flash"
        
        uri = os.getenv("NEO4J_URI").replace("neo4j+s://", "neo4j://")
        self.neo4j_driver = GraphDatabase.driver(
            uri,
            auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
            encrypted=True,
            trusted_certificates=TrustCustomCAs(certifi.where())
        )
        self.database = os.getenv("NEO4J_DATABASE", "neo4j")
        
        self.output_dir = Path(__file__).parent.parent.parent / "data" / "eval"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print("Initialization complete!\n")

    def get_graph_paths(self) -> list[dict]:
        """Fetch some interesting paths from Neo4j to build questions."""
        paths = []
        with self.neo4j_driver.session(database=self.database) as session:
            # 1-hop facts
            res_1 = session.run("MATCH (a)-[r]->(b) RETURN a.name AS src, type(r) AS rel, b.name AS tgt LIMIT 3")
            for record in res_1:
                paths.append({
                    "type": "1-hop",
                    "context": f"{record['src']} {record['rel']} {record['tgt']}",
                    "edges": [f"({record['src']})-[{record['rel']}]->({record['tgt']})"]
                })
            
            # 2-hop facts (multi-hop reasoning)
            res_2 = session.run("MATCH (a)-[r1]->(b)-[r2]->(c) RETURN a.name AS src, type(r1) AS r1, b.name AS mid, type(r2) AS r2, c.name AS tgt LIMIT 3")
            for record in res_2:
                paths.append({
                    "type": "2-hop",
                    "context": f"{record['src']} {record['r1']} {record['mid']} AND {record['mid']} {record['r2']} {record['tgt']}",
                    "edges": [
                        f"({record['src']})-[{record['r1']}]->({record['mid']})",
                        f"({record['mid']})-[{record['r2']}]->({record['tgt']})"
                    ]
                })
        return paths

    def generate_qa_dataset(self):
        paths = self.get_graph_paths()
        dataset = []
        
        print(f"Generating questions for {len(paths)} paths...")
        for i, path in enumerate(paths):
            print(f"[{i+1}/{len(paths)}] Generating for {path['type']} context: {path['context']}")
            
            prompt = f"""
            You are a medical evaluator. I will give you a fact (context).
            Generate ONE complex medical question that requires knowing this exact fact to answer.
            Also provide the exact, concise ground truth answer based on the fact.
            
            Fact: {path['context']}
            """
            
            try:
                response = self.llm_client.models.generate_content(
                    model=self.model_id,
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": QAPair
                    }
                )
                qa_data = response.parsed
                
                dataset.append({
                    "id": f"q_{i}",
                    "type": path["type"],
                    "question": qa_data.question,
                    "ground_truth": qa_data.ground_truth,
                    "required_edges": path["edges"]
                })
                # Sleep to avoid rate limits
                time.sleep(4)
            except Exception as e:
                print(f"Error generating question: {e}")
                
        output_file = self.output_dir / "qa_dataset.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=4, ensure_ascii=False)
            
        print(f"\nDataset generation complete! Saved {len(dataset)} questions to {output_file}")
        
    def close(self):
        self.neo4j_driver.close()

if __name__ == "__main__":
    generator = DatasetGenerator()
    generator.generate_qa_dataset()
    generator.close()
