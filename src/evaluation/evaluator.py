"""
Milestone 3 - Evaluation Harness
Step 3.2: Evaluator (A/B Testing & LLM-as-a-Judge)

Runs the QA dataset through 3 modes: Vector-Only, Graph-Only, Hybrid.
Uses Gemini to judge Correctness and Faithfulness.
"""
import os
import json
import time
import certifi
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

from google import genai
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from neo4j import GraphDatabase, TrustCustomCAs

load_dotenv(Path(__file__).parent.parent.parent / ".env")

class JudgeScore(BaseModel):
    correctness_score: int      # 0 to 100
    faithfulness_score: int     # 0 to 100

class Evaluator:
    def __init__(self):
        print("Initializing Evaluator...")
        self.llm_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model_id = "gemini-3.6-flash"
        
        qdrant_path = Path(__file__).parent.parent.parent / "data" / "qdrant_db"
        self.qdrant = QdrantClient(path=str(qdrant_path))
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        uri = os.getenv("NEO4J_URI").replace("neo4j+s://", "neo4j://")
        self.neo4j_driver = GraphDatabase.driver(
            uri,
            auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
            encrypted=True,
            trusted_certificates=TrustCustomCAs(certifi.where())
        )
        self.database = os.getenv("NEO4J_DATABASE", "neo4j")
        
        self.data_dir = Path(__file__).parent.parent.parent / "data" / "eval"
        
    def vector_search(self, query: str, top_k: int = 3) -> list[str]:
        query_vector = self.embed_model.encode(query).tolist()
        results = self.qdrant.query_points(collection_name="pubmed_chunks", query=query_vector, limit=top_k)
        return [hit.payload["text"] for hit in results.points]

    def graph_search(self, query: str) -> list[str]:
        extract_prompt = f"Extract 1 to 3 main biomedical keywords (drugs, diseases) from: '{query}'. Return ONLY comma-separated list."
        try:
            response = self.llm_client.models.generate_content(model=self.model_id, contents=extract_prompt)
            keywords = [k.strip() for k in response.text.split(",") if k.strip()]
        except:
            keywords = []
            
        graph_context = []
        with self.neo4j_driver.session(database=self.database) as session:
            for kw in keywords:
                res = session.run(
                    "MATCH (n)-[r]-(m) WHERE toLower(n.name) CONTAINS toLower($keyword) RETURN n.name AS s, type(r) AS r, m.name AS t LIMIT 3",
                    keyword=kw
                )
                for record in res:
                    graph_context.append(f"({record['s']})-[{record['r']}]->({record['t']})")
        return list(set(graph_context))
        
    def generate_answer(self, query: str, context: list[str]) -> str:
        prompt = f"""
        Answer the question based ONLY on this context. If not enough info, say "I don't know".
        Question: {query}
        Context: {chr(10).join(context)}
        """
        try:
            res = self.llm_client.models.generate_content(model=self.model_id, contents=prompt)
            return res.text
        except:
            return "Error calling LLM."

    def judge_answer(self, query: str, ground_truth: str, answer: str) -> dict:
        if "I don't know" in answer or "Error" in answer:
            return {"correctness_score": 0, "faithfulness_score": 0}
            
        prompt = f"""
        You are an expert judge. Score the AI's Answer from 0 to 100 on two metrics:
        1. Correctness: Does the Answer match the Ground Truth?
        2. Faithfulness: Is the Answer free from hallucinated details outside the Ground Truth?
        
        Question: {query}
        Ground Truth: {ground_truth}
        AI Answer: {answer}
        """
        try:
            res = self.llm_client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config={"response_mime_type": "application/json", "response_schema": JudgeScore}
            )
            return res.parsed.model_dump()
        except:
            return {"correctness_score": 0, "faithfulness_score": 0}

    def run_evaluation(self):
        dataset_path = self.data_dir / "qa_dataset.json"
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
            
        # We only evaluate the first 2 questions to avoid API Rate Limits (503)
        dataset = dataset[:2]
            
        results = []
        print(f"Evaluating {len(dataset)} questions across 3 systems...")
        
        for i, item in enumerate(dataset):
            print(f"\n[{i+1}/{len(dataset)}] Q: {item['question']}")
            
            # 1. Retrieve Contexts
            v_ctx = self.vector_search(item['question'])
            g_ctx = self.graph_search(item['question'])
            h_ctx = v_ctx + g_ctx
            
            # Calculate Graph Recall (Did we find the required edges?)
            edges_found = sum([1 for e in item['required_edges'] if any(e in g for g in g_ctx)])
            graph_recall = (edges_found / len(item['required_edges'])) * 100 if item['required_edges'] else 100
            
            time.sleep(3) # Anti-rate limit
            
            # 2. Generate Answers
            v_ans = self.generate_answer(item['question'], v_ctx)
            time.sleep(3)
            g_ans = self.generate_answer(item['question'], g_ctx)
            time.sleep(3)
            h_ans = self.generate_answer(item['question'], h_ctx)
            time.sleep(3)
            
            # 3. Judge Answers
            v_score = self.judge_answer(item['question'], item['ground_truth'], v_ans)
            time.sleep(3)
            g_score = self.judge_answer(item['question'], item['ground_truth'], g_ans)
            time.sleep(3)
            h_score = self.judge_answer(item['question'], item['ground_truth'], h_ans)
            time.sleep(3)
            
            # Append Results
            for system, ans, score in [
                ("Vector", v_ans, v_score), 
                ("Graph", g_ans, g_score), 
                ("Hybrid", h_ans, h_score)
            ]:
                results.append({
                    "question_id": item['id'],
                    "question_type": item['type'],
                    "system": system,
                    "graph_recall": graph_recall if system in ["Graph", "Hybrid"] else 0,
                    "correctness": score['correctness_score'],
                    "faithfulness": score['faithfulness_score']
                })
                print(f"  - {system} Score: {score['correctness_score']}% Correct")
                
        # Save Results
        df = pd.DataFrame(results)
        df.to_csv(self.data_dir / "results.csv", index=False)
        print("\nEvaluation Complete! Saved to results.csv")

if __name__ == "__main__":
    evaluator = Evaluator()
    evaluator.run_evaluation()
