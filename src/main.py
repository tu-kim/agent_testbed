"""
Main Application Server for RAG Pipeline

Provides REST API endpoints for querying the RAG system,
managing the index, and retrieving system statistics.
"""

import os
import sys
import time
import logging
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uvicorn

# Add project root to sys.path to allow absolute imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import components using absolute paths
from src.retrieval.faiss_retriever import create_retriever
from src.retrieval.dataset_loader import create_dataset_manager
from src.rag_pipeline.ircot import IRCoTRAGPipeline, IRCoTConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Agent Testbed RAG Server")

# Global state
state = {
    "retriever": None,
    "pipeline": None,
    "config": {
        "llm_endpoint": os.getenv("LLM_ENDPOINT", "http://localhost:8000"),
        "cache_dir": os.getenv("CACHE_DIR", "./cache/faiss_index"),
        "algorithm": os.getenv("FAISS_ALGORITHM", "hnsw"),
        "model_name": os.getenv("MODEL_NAME", "meta-llama/Llama-2-7b-chat-hf"),
        "tiered": os.getenv("TIERED_STORAGE", "false").lower() == "true",
        "ram_capacity": int(os.getenv("RAM_CAPACITY", "100000")),
        "ssd_dir": os.getenv("SSD_DIR", "./cache/ssd_index")
    }
}

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    max_iterations: int = 3

class QueryResponse(BaseModel):
    question: str
    answer: str
    context: List[str]
    num_iterations: int
    num_retrieved_docs: int
    total_time_ms: float
    retrieval_time_ms: float
    llm_time_ms: float
    reasoning_steps: List[Dict[str, Any]]

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    logger.info("Initializing RAG components...")
    
    # Initialize retriever with tiered storage support
    state["retriever"] = create_retriever(
        algorithm=state["config"]["algorithm"],
        cache_dir=state["config"]["cache_dir"],
        tiered=state["config"]["tiered"],
        ram_capacity=state["config"]["ram_capacity"],
        ssd_dir=state["config"]["ssd_dir"]
    )
    
    if state["config"]["tiered"]:
        logger.info(f"Tiered storage enabled: RAM={state['config']['ram_capacity']}, SSD={state['config']['ssd_dir']}")
    
    # Load index if exists
    if state["retriever"].load():
        logger.info(f"Loaded existing index from {state['config']['cache_dir']}")
    else:
        logger.warning(f"No index found at {state['config']['cache_dir']}. Please run indexing first.")
    
    # Initialize IRCoT pipeline
    ircot_config = IRCoTConfig(
        llm_endpoint=state["config"]["llm_endpoint"],
        model_name=state["config"]["model_name"]
    )
    state["pipeline"] = IRCoTRAGPipeline(
        retriever=state["retriever"],
        config=ircot_config
    )
    
    logger.info("RAG components initialized.")

@app.get("/")
async def root():
    return {
        "service": "Agent Testbed RAG Server",
        "status": "online",
        "config": state["config"]
    }

@app.get("/health")
async def health():
    if state["retriever"] is None or state["pipeline"] is None:
        raise HTTPException(status_code=503, detail="Components not initialized")
    return {"status": "healthy"}

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Process a RAG query using IRCoT."""
    if state["pipeline"] is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    start_time = time.time()
    
    try:
        # Run IRCoT pipeline
        result = await state["pipeline"].arun(
            question=request.question,
            max_iterations=request.max_iterations,
            retrieval_per_step=request.top_k
        )
        
        total_time = (time.time() - start_time) * 1000
        
        return QueryResponse(
            question=request.question,
            answer=result["answer"],
            context=result["context"],
            num_iterations=result["num_iterations"],
            num_retrieved_docs=len(result["context"]),
            total_time_ms=total_time,
            retrieval_time_ms=result["profile"]["retrieval_time_ms"],
            llm_time_ms=result["profile"]["llm_time_ms"],
            reasoning_steps=result["steps"]
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    stats = {
        "config": state["config"],
        "index": state["retriever"].get_stats() if state["retriever"] else "Not initialized"
    }
    return stats

def run_server(host: str = "0.0.0.0", port: int = 8080, reload: bool = False):
    """Run the FastAPI server."""
    uvicorn.run("src.main:app", host=host, port=port, reload=reload)

if __name__ == "__main__":
    run_server()
