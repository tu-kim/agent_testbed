#!/usr/bin/env python3
"""
CLI Runner for Multi-hop RAG with IRCoT

Commands:
- index: Build FAISS index from HotpotQA corpus
- server: Run the RAG server (connects to external vLLM PD server)
- profile: Run query generator profiling with Poisson arrivals
- query: Run a single query
"""

import argparse
import asyncio
import os
import sys
import logging

# Add the project root to sys.path to allow imports from src
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_server(args):
    """Run the main RAG server."""
    os.environ.setdefault("LLM_ENDPOINT", args.llm_endpoint)
    os.environ.setdefault("LLM_MODEL", args.llm_model)
    os.environ.setdefault("FAISS_ALGORITHM", args.algorithm)
    os.environ.setdefault("CACHE_DIR", args.cache_dir)
    
    from src.main import run_server as start_server
    start_server(host=args.host, port=args.port, reload=args.reload)


def run_index(args):
    """Index documents from HotpotQA corpus into FAISS."""
    from src.retrieval.faiss_retriever import create_retriever
    from src.retrieval.dataset_loader import create_dataset_manager
    
    logger.info(f"Building index from HotpotQA {args.corpus_split} split")
    logger.info(f"Algorithm: {args.algorithm}")
    logger.info(f"Device: {args.device}, Batch size: {args.batch_size}")
    
    retriever = create_retriever(
        algorithm=args.algorithm,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        device=args.device
    )
    
    # Try to load existing index
    if retriever.load():
        logger.info("Loaded existing index")
        if not args.force:
            logger.info("Use --force to rebuild index")
            return
    
    # Load HotpotQA corpus
    logger.info(f"Loading HotpotQA corpus from {args.corpus_split} split...")
    manager = create_dataset_manager(
        corpus_split=args.corpus_split,
        eval_split=args.eval_split
    )
    corpus = manager.load_corpus()
    
    if not corpus:
        logger.error("No documents loaded from HotpotQA")
        sys.exit(1)
    
    logger.info(f"Loaded {len(corpus)} documents from HotpotQA")
    
    # Index documents
    logger.info(f"Indexing {len(corpus)} documents...")
    retriever.add_documents(corpus)
    
    # Save index
    retriever.save()
    logger.info(f"Index saved to {args.cache_dir}")
    logger.info(f"Index stats: {retriever.get_stats()}")


def run_profile(args):
    """Run query generator profiling with Poisson arrivals."""
    from src.frontend.query_generator import run_profiling
    from src.retrieval.dataset_loader import create_dataset_manager
    
    logger.info(f"Loading HotpotQA questions from {args.eval_split} split...")
    
    # Load questions from HotpotQA
    manager = create_dataset_manager(
        eval_split=args.eval_split
    )
    questions = manager.load_queries(max_samples=args.max_queries)
    
    if not questions:
        logger.error("No questions loaded from HotpotQA")
        sys.exit(1)
    
    logger.info(f"Loaded {len(questions)} questions")
    logger.info(f"Profiling endpoint: {args.endpoint}")
    logger.info(f"QPS range: {args.start_qps} -> {args.end_qps} (step: {args.qps_step})")
    logger.info(f"Stage duration: {args.stage_duration}s")
    
    # Run profiling
    asyncio.run(run_profiling(
        endpoint=args.endpoint,
        questions=questions,
        start_qps=args.start_qps,
        end_qps=args.end_qps,
        qps_step=args.qps_step,
        stage_duration=args.stage_duration,
        seed=args.seed,
        output_file=args.output
    ))


def run_query(args):
    """Run a single query."""
    import httpx
    
    response = httpx.post(
        f"{args.endpoint}/query",
        json={
            "question": args.question,
            "top_k": args.top_k,
            "max_iterations": args.max_iterations
        },
        timeout=120.0
    )
    
    result = response.json()
    
    print(f"\nQuestion: {result.get('question', args.question)}")
    print(f"\nAnswer: {result.get('answer', 'No answer')}")
    print(f"\nMetrics:")
    print(f"  Total time: {result.get('total_time_ms', 0):.2f}ms")
    print(f"  Retrieval time: {result.get('retrieval_time_ms', 0):.2f}ms")
    print(f"  LLM time: {result.get('llm_time_ms', 0):.2f}ms")
    print(f"  Iterations: {result.get('num_iterations', 0)}")
    print(f"  Retrieved docs: {result.get('num_retrieved_docs', 0)}")
    
    if args.verbose and result.get('reasoning_steps'):
        print(f"\nReasoning Steps:")
        for step in result['reasoning_steps']:
            print(f"\n  Step {step['step']}:")
            print(f"    {step['reasoning'][:200]}...")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-hop RAG with IRCoT CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build index from HotpotQA train split
  python run.py index --corpus-split train --algorithm hnsw

  # Run RAG server (assumes external vLLM PD server at localhost:8000)
  python run.py server --llm-endpoint http://localhost:8000 --port 8080

  # Run profiling with Poisson arrivals (QPS 1 -> 10)
  python run.py profile --endpoint http://localhost:8080 --start-qps 1 --end-qps 10

  # Run a single query
  python run.py query "What is the capital of France?"
"""
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Server command
    server_parser = subparsers.add_parser("server", help="Run the RAG server")
    server_parser.add_argument("--host", type=str, default="0.0.0.0",
                               help="Server host (default: 0.0.0.0)")
    server_parser.add_argument("--port", type=int, default=8080,
                               help="Server port (default: 8080)")
    server_parser.add_argument("--llm-endpoint", type=str, default="http://localhost:8000",
                               help="External vLLM PD server endpoint (default: http://localhost:8000)")
    server_parser.add_argument("--llm-model", type=str, default="meta-llama/Llama-2-7b-chat-hf",
                               help="LLM model name for API calls")
    server_parser.add_argument("--algorithm", type=str, default="hnsw", 
                               choices=["hnsw", "ivf", "flat"],
                               help="FAISS algorithm (default: hnsw)")
    server_parser.add_argument("--cache-dir", type=str, default="./cache/faiss_index",
                               help="Directory containing FAISS index")
    server_parser.add_argument("--reload", action="store_true",
                               help="Enable auto-reload for development")
    server_parser.set_defaults(func=run_server)
    
    # Index command (HotpotQA corpus)
    index_parser = subparsers.add_parser("index", help="Build FAISS index from HotpotQA")
    index_parser.add_argument("--corpus-split", type=str, default="train",
                              choices=["train", "validation"],
                              help="HotpotQA split for corpus (default: train)")
    index_parser.add_argument("--eval-split", type=str, default="validation",
                              choices=["train", "validation"],
                              help="HotpotQA split for evaluation (default: validation)")
    index_parser.add_argument("--algorithm", type=str, default="hnsw", 
                              choices=["hnsw", "ivf", "flat"],
                              help="FAISS algorithm (default: hnsw)")
    index_parser.add_argument("--cache-dir", type=str, default="./cache/faiss_index",
                              help="Directory to save index")
    index_parser.add_argument("--force", action="store_true", 
                              help="Force rebuild index")
    index_parser.add_argument("--device", type=str, default="cuda", 
                              choices=["auto", "cuda", "cpu"],
                              help="Device for embedding (default: cuda)")
    index_parser.add_argument("--batch-size", type=int, default=256,
                              help="Batch size for encoding (default: 256)")
    index_parser.set_defaults(func=run_index)
    
    # Profile command (Query Generator with Poisson arrivals)
    profile_parser = subparsers.add_parser("profile", 
                                           help="Run query generator profiling with Poisson arrivals")
    profile_parser.add_argument("--endpoint", type=str, default="http://localhost:8080",
                                help="RAG server endpoint (default: http://localhost:8080)")
    profile_parser.add_argument("--eval-split", type=str, default="validation",
                                choices=["train", "validation"],
                                help="HotpotQA split for queries (default: validation)")
    profile_parser.add_argument("--start-qps", type=float, default=1.0,
                                help="Starting QPS (default: 1.0)")
    profile_parser.add_argument("--end-qps", type=float, default=10.0,
                                help="Ending QPS (default: 10.0)")
    profile_parser.add_argument("--qps-step", type=float, default=1.0,
                                help="QPS increment per stage (default: 1.0)")
    profile_parser.add_argument("--stage-duration", type=float, default=30.0,
                                help="Duration of each QPS stage in seconds (default: 30)")
    profile_parser.add_argument("--max-queries", type=int, default=None,
                                help="Maximum queries to load from dataset")
    profile_parser.add_argument("--seed", type=int, default=42,
                                help="Random seed for reproducibility (default: 42)")
    profile_parser.add_argument("--output", type=str, default=None,
                                help="Output JSON file for profiling results")
    profile_parser.set_defaults(func=run_profile)
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Run a single query")
    query_parser.add_argument("question", type=str, help="Question to ask")
    query_parser.add_argument("--endpoint", type=str, default="http://localhost:8080",
                              help="RAG server endpoint")
    query_parser.add_argument("--top-k", type=int, default=5,
                              help="Number of documents to retrieve")
    query_parser.add_argument("--max-iterations", type=int, default=3,
                              help="Maximum IRCoT iterations")
    query_parser.add_argument("--verbose", "-v", action="store_true",
                              help="Show reasoning steps")
    query_parser.set_defaults(func=run_query)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()
