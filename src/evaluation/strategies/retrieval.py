"""
Retrieval Strategies for Evaluation
GraphRAG v2.1

Three strategies sharing the same interface so the harness can
swap them freely for fair A/B comparison:

1. NaiveRAG  — Vector search only (baseline)
2. GraphRAG  — Graph search only
3. HybridRAG — Vector + Graph combined

Each strategy implements:
  retrieve(query) → list[str]   (context chunks)
  generate(query, context) → tuple[str, dict]  (answer, metadata)
"""
import os
import time
import certifi
from abc import ABC, abstractmethod
from pathlib import Path
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from neo4j import GraphDatabase, TrustCustomCAs

from src.infrastructure.llm_client import LLMClient

load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")


class BaseStrategy(ABC):
    """Common interface for all retrieval strategies."""

    def __init__(self, llm: LLMClient):
        self.llm = llm
        self.generation_prompt_template = """You are a medical AI assistant. Answer the user's question based ONLY on the provided context.
If the context does not contain enough information to answer, say "I don't have enough information to answer this question."

Question: {question}

Context:
{context}

Answer:"""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        """Retrieve relevant context for the query."""
        pass

    def generate(self, query: str, context: list[str]) -> tuple[str, dict]:
        """Generate answer from query + context. Returns (answer, metadata)."""
        context_str = "\n---\n".join(context) if context else "(No context available)"
        prompt = self.generation_prompt_template.format(
            question=query, context=context_str
        )
        try:
            answer, metadata = self.llm.generate(prompt, temperature=0.1)
            return answer, metadata
        except Exception as e:
            return f"Error: {e}", {"input_tokens": 0, "output_tokens": 0, "latency_ms": 0}


class NaiveRAG(BaseStrategy):
    """Baseline: Vector search only. No graph, no fancy retrieval."""

    def __init__(self, llm: LLMClient):
        super().__init__(llm)
        qdrant_path = Path(__file__).parent.parent.parent.parent / "data" / "qdrant_db"
        self.qdrant = QdrantClient(path=str(qdrant_path))
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        query_vector = self.embed_model.encode(query).tolist()
        results = self.qdrant.query_points(
            collection_name="pubmed_chunks",
            query=query_vector,
            limit=top_k
        )
        return [hit.payload["text"] for hit in results.points]


class GraphRAG(BaseStrategy):
    """Graph-only: Extract keywords → search Neo4j → use triplets as context."""

    def __init__(self, llm: LLMClient):
        super().__init__(llm)
        uri = os.getenv("NEO4J_URI", "").replace("neo4j+s://", "neo4j://")
        self.neo4j_driver = GraphDatabase.driver(
            uri,
            auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
            encrypted=True,
            trusted_certificates=TrustCustomCAs(certifi.where())
        )
        self.database = os.getenv("NEO4J_DATABASE", "neo4j")

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        # Step 1: Extract keywords using LLM
        extract_prompt = (
            f"Extract 1 to 3 main biomedical keywords (drugs, diseases, enzymes) "
            f"from this question: '{query}'. Return ONLY a comma-separated list."
        )
        try:
            text, _ = self.llm.generate(extract_prompt, temperature=0.0)
            keywords = [k.strip() for k in text.split(",") if k.strip()]
        except Exception:
            keywords = []

        if not keywords:
            return []

        # Step 2: Query Neo4j for matching triplets
        graph_context = []
        with self.neo4j_driver.session(database=self.database) as session:
            for kw in keywords:
                results = session.run(
                    "MATCH (n)-[r]-(m) WHERE toLower(n.name) CONTAINS toLower($kw) "
                    "RETURN n.name AS src, type(r) AS rel, m.name AS tgt LIMIT 5",
                    kw=kw
                )
                for record in results:
                    triplet = f"({record['src']}) --[{record['rel']}]--> ({record['tgt']})"
                    graph_context.append(triplet)

        return list(set(graph_context))

    def close(self):
        self.neo4j_driver.close()


class HybridRAG(BaseStrategy):
    """Combined Vector + Graph retrieval."""

    def __init__(self, llm: LLMClient, naive: NaiveRAG = None, graph: GraphRAG = None):
        super().__init__(llm)
        self._naive = naive if naive else NaiveRAG(llm)
        self._graph = graph if graph else GraphRAG(llm)

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        vector_ctx = self._naive.retrieve(query, top_k)
        graph_ctx = self._graph.retrieve(query, top_k)
        # Combine: vector chunks first, then graph triplets
        return vector_ctx + graph_ctx

    def close(self):
        self._graph.close()


def get_strategies(names: list[str], llm: LLMClient) -> dict[str, BaseStrategy]:
    """Factory function to get initialized strategies."""
    instances = {}
    
    # Initialize basic strategies first if needed
    naive_instance = None
    graph_instance = None
    
    if "naive" in names or "hybrid" in names:
        naive_instance = NaiveRAG(llm)
        if "naive" in names:
            instances["naive"] = naive_instance
            
    if "graph" in names or "hybrid" in names:
        graph_instance = GraphRAG(llm)
        if "graph" in names:
            instances["graph"] = graph_instance
            
    if "hybrid" in names:
        instances["hybrid"] = HybridRAG(llm, naive=naive_instance, graph=graph_instance)
        
    return instances
