from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import sys
from pathlib import Path
import uvicorn

# Add the src folder to path so we can import retrieval
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.retrieval.hybrid_search import HybridRetriever

app = FastAPI(title="GraphRAG AI API")

# Initialize retriever (this will take a few seconds on startup)
retriever = HybridRetriever()

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str
    vector_context: list[str]
    graph_context: list[str]

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        result = retriever.answer_question(request.query)
        return ChatResponse(
            answer=result.get("answer", "Error generating answer."),
            vector_context=result.get("vector_context", []),
            graph_context=result.get("graph_context", [])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Serve the static frontend files
frontend_dir = Path(__file__).parent.parent / "frontend"
frontend_dir.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

if __name__ == "__main__":
    print(f"Starting GraphRAG server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
