"""
Main Application Server for RAG Pipeline

Provides REST API endpoints for querying the RAG system,
managing the index, and retrieving system statistics.

Designed to work with external vLLM PD (Prefill-Decode) servers.
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
from src.rag_pipeline.ircot import IRCoTRAGPipeline, IRCoTConfig, IRCoTResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Agent Testbed RAG Server")

# Global state
state = {
    "retriever": None,
    "pipeline": None,
    "config": {
        "llm_endpoint": os.getenv("LLM_ENDPOINT", "http://localhost:8000"),
        "llm_model": os.getenv("LLM_MODEL", "meta-llama/Llama-2-7b-chat-hf"),
        "cache_dir": os.getenv("CACHE_DIR", "./cache/faiss_index"),
        "algorithm": os.getenv("FAISS_ALGORITHM", "hnsw"),
    }
}


class QueryRequest(BaseModel):
    """Request model for RAG query."""
    question: str
    query_id: Optional[str] = None  # For profiling tracking
    top_k: int = 5
    max_iterations: int = 3


class QueryResponse(BaseModel):
    """Response model for RAG query with detailed timing."""
    question: str
    answer: str
    context: List[str]
    num_iterations: int
    num_retrieved_docs: int
    
    # Timing information (milliseconds)
    total_time_ms: float
    retrieval_time_ms: float
    llm_time_ms: float
    
    # Server timing for queue time calculation
    server_start_time: float  # Unix timestamp when server started processing
    
    # Detailed steps
    reasoning_steps: List[Dict[str, Any]]


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    logger.info("Initializing RAG components...")
    logger.info(f"LLM Endpoint: {state['config']['llm_endpoint']}")
    logger.info(f"LLM Model: {state['config']['llm_model']}")
    logger.info(f"Cache Dir: {state['config']['cache_dir']}")
    
    # Initialize retriever
    state["retriever"] = create_retriever(
        algorithm=state["config"]["algorithm"],
        cache_dir=state["config"]["cache_dir"]
    )
    
    # Load index if exists
    if state["retriever"].load():
        logger.info(f"Loaded existing index from {state['config']['cache_dir']}")
        stats = state["retriever"].get_stats()
        logger.info(f"Index stats: {stats}")
    else:
        logger.warning(f"No index found at {state['config']['cache_dir']}. Please run indexing first.")
    
    # Initialize IRCoT pipeline
    ircot_config = IRCoTConfig(
        llm_endpoint=state["config"]["llm_endpoint"],
        model_name=state["config"]["llm_model"]
    )
    state["pipeline"] = IRCoTRAGPipeline(
        retriever=state["retriever"],
        config=ircot_config
    )
    
    logger.info("RAG components initialized.")


@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "Agent Testbed RAG Server",
        "status": "online",
        "config": state["config"]
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    if state["retriever"] is None or state["pipeline"] is None:
        raise HTTPException(status_code=503, detail="Components not initialized")
    return {"status": "healthy"}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Process a RAG query using IRCoT.
    
    Returns detailed timing information including server_start_time
    for queue time calculation by the profiler.
    """
    if state["pipeline"] is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    # Record server start time for queue time calculation
    server_start_time = time.time()
    
    try:
        # Run IRCoT pipeline - returns IRCoTResult dataclass
        result: IRCoTResult = await state["pipeline"].arun(
            question=request.question,
            max_iterations=request.max_iterations,
            retrieval_per_step=request.top_k
        )
        
        # Access IRCoTResult using attribute access (not dictionary)
        return QueryResponse(
            question=request.question,
            answer=result.answer,
            context=result.context,
            num_iterations=result.num_iterations,
            num_retrieved_docs=len(result.context),
            total_time_ms=result.total_time_ms,
            retrieval_time_ms=result.retrieval_time_ms,
            llm_time_ms=result.llm_time_ms,
            server_start_time=server_start_time,
            reasoning_steps=result.steps
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
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
