#!/usr/bin/env python3
"""
CLI Runner for Multi-hop RAG with IRCoT

Provides command-line interface for running different components
of the RAG system.
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
    os.environ.setdefault("FAISS_ALGORITHM", args.algorithm)
    os.environ.setdefault("CACHE_DIR", args.cache_dir)
    os.environ.setdefault("MODEL_NAME", args.model_name)
    
    from src.main import run_server as start_server
    start_server(host=args.host, port=args.port, reload=args.reload)


def run_proxy(args):
    """Run the vLLM proxy server."""
    from src.vllm_server.proxy_server import create_proxy_server
    
    prefill_workers = []
    for addr in args.prefill_workers:
        host, port = addr.split(":")
        prefill_workers.append({"host": host, "port": int(port)})
    
    decode_workers = []
    for addr in args.decode_workers:
        host, port = addr.split(":")
        decode_workers.append({"host": host, "port": int(port)})
    
    server = create_proxy_server(
        host=args.host,
        port=args.port,
        prefill_workers=prefill_workers,
        decode_workers=decode_workers,
        model_name=args.model_name
    )
    server.run()


def run_prefill(args):
    """Run the vLLM prefill worker."""
    from src.vllm_server.prefill_worker import create_prefill_worker
    
    worker = create_prefill_worker(
        model_name=args.model,
        host=args.host,
        port=args.port,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
        tokenizer=args.tokenizer
    )
    
    asyncio.run(worker.initialize())
    worker.run()


def run_decode(args):
    """Run the vLLM decode worker."""
    from src.vllm_server.decode_worker import create_decode_worker
    
    worker = create_decode_worker(
        model_name=args.model,
        host=args.host,
        port=args.port,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
        tokenizer=args.tokenizer
    )
    
    asyncio.run(worker.initialize())
    worker.run()


def run_index(args):
    """Index documents into FAISS (C4 corpus only)."""
    from src.retrieval.faiss_retriever import create_retriever
    from src.retrieval.dataset_loader import create_dataset_manager
    
    logger.info(f"Creating retriever with algorithm: {args.algorithm}")
    logger.info(f"Tiered storage: {'enabled' if args.tiered else 'disabled'}")
    if args.tiered:
        logger.info(f"RAM capacity: {args.ram_capacity}, SSD dir: {args.ssd_dir}")
    
    retriever = create_retriever(
        algorithm=args.algorithm,
        cache_dir=args.cache_dir,
        tiered=args.tiered,
        ram_capacity=args.ram_capacity,
        ssd_dir=args.ssd_dir
    )
    
    # Try to load existing index
    if retriever.load():
        logger.info("Loaded existing index")
        if not args.force:
            logger.info("Use --force to rebuild index")
            return
    
    # Load C4 corpus from local directory
    if not args.corpus_dir:
        logger.error("--corpus-dir is required. Please specify the local directory containing C4 dataset.")
        sys.exit(1)
    
    logger.info(f"Loading C4 corpus from: {args.corpus_dir}")
    manager = create_dataset_manager(corpus_local_dir=args.corpus_dir)
    corpus = manager.load_corpus()
    
    if not corpus:
        logger.error("No documents loaded from corpus directory")
        sys.exit(1)
    
    logger.info(f"Loaded {len(corpus)} documents from C4 corpus")
    
    # Index documents
    logger.info(f"Indexing {len(corpus)} documents...")
    retriever.add_documents(corpus)
    
    # Save index
    retriever.save()
    logger.info(f"Index saved to {args.cache_dir}")
    logger.info(f"Index stats: {retriever.get_stats()}")


def run_profile(args):
    """Run performance profiling using HotpotQA questions."""
    from src.frontend.profiler import run_profiling
    from src.retrieval.dataset_loader import create_dataset_manager
    
    # Load questions from HotpotQA
    logger.info("Loading HotpotQA questions for profiling...")
    manager = create_dataset_manager()
    
    total_needed = args.num_queries + args.warmup_queries + 10
    eval_data = manager.get_evaluation_data(num_samples=total_needed)
    questions = [item["question"] for item in eval_data]
    
    if len(questions) < args.num_queries:
        logger.warning(f"Only {len(questions)} questions available")
    
    logger.info(f"Loaded {len(questions)} HotpotQA questions")
    
    # Run profiling
    asyncio.run(run_profiling(
        endpoint=args.endpoint,
        questions=questions,
        num_queries=args.num_queries,
        warmup_queries=args.warmup_queries,
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
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Server command
    server_parser = subparsers.add_parser("server", help="Run the RAG server")
    server_parser.add_argument("--host", type=str, default="0.0.0.0")
    server_parser.add_argument("--port", type=int, default=8080,
                               help="Port for RAG server (default: 8080)")
    server_parser.add_argument("--llm-endpoint", type=str, default="http://localhost:8000",
                               help="LLM server endpoint (default: http://localhost:8000)")
    server_parser.add_argument("--algorithm", type=str, default="hnsw", choices=["hnsw", "ivf", "flat"])
    server_parser.add_argument("--cache-dir", type=str, default="./cache/faiss_index")
    server_parser.add_argument("--model-name", type=str, default="meta-llama/Llama-2-7b-chat-hf",
                               help="Model name or local path")
    server_parser.add_argument("--reload", action="store_true")
    server_parser.set_defaults(func=run_server)
    
    # Proxy command
    proxy_parser = subparsers.add_parser("proxy", help="Run the vLLM proxy server")
    proxy_parser.add_argument("--host", type=str, default="0.0.0.0")
    proxy_parser.add_argument("--port", type=int, default=8000,
                              help="Port for proxy server (default: 8000)")
    proxy_parser.add_argument("--prefill-workers", type=str, nargs="+", default=["localhost:8001"])
    proxy_parser.add_argument("--decode-workers", type=str, nargs="+", default=["localhost:8002"])
    proxy_parser.add_argument("--model-name", type=str, default="meta-llama/Llama-2-7b-chat-hf",
                              help="Model name or local path")
    proxy_parser.set_defaults(func=run_proxy)
    
    # Prefill worker command
    prefill_parser = subparsers.add_parser("prefill", help="Run the vLLM prefill worker")
    prefill_parser.add_argument("--host", type=str, default="0.0.0.0")
    prefill_parser.add_argument("--port", type=int, default=8001)
    prefill_parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf",
                                help="Model name (HuggingFace ID) or local directory path")
    prefill_parser.add_argument("--tokenizer", type=str, default=None,
                                help="Tokenizer path (optional, defaults to model path)")
    prefill_parser.add_argument("--tensor-parallel-size", type=int, default=1)
    prefill_parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    prefill_parser.add_argument("--trust-remote-code", action="store_true", default=True)
    prefill_parser.set_defaults(func=run_prefill)
    
    # Decode worker command
    decode_parser = subparsers.add_parser("decode", help="Run the vLLM decode worker")
    decode_parser.add_argument("--host", type=str, default="0.0.0.0")
    decode_parser.add_argument("--port", type=int, default=8002)
    decode_parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf",
                               help="Model name (HuggingFace ID) or local directory path")
    decode_parser.add_argument("--tokenizer", type=str, default=None,
                               help="Tokenizer path (optional, defaults to model path)")
    decode_parser.add_argument("--tensor-parallel-size", type=int, default=1)
    decode_parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    decode_parser.add_argument("--trust-remote-code", action="store_true", default=True)
    decode_parser.set_defaults(func=run_decode)
    
    # Index command (C4 corpus only)
    index_parser = subparsers.add_parser("index", help="Index C4 corpus into FAISS")
    index_parser.add_argument("--corpus-dir", type=str, required=True,
                              help="Local directory path containing C4 dataset (required)")
    index_parser.add_argument("--algorithm", type=str, default="hnsw", choices=["hnsw", "ivf", "flat"])
    index_parser.add_argument("--cache-dir", type=str, default="./cache/faiss_index")
    index_parser.add_argument("--force", action="store_true", help="Force rebuild index")
    index_parser.add_argument("--tiered", action="store_true",
                              help="Enable tiered storage (RAM + SSD)")
    index_parser.add_argument("--ram-capacity", type=int, default=100000,
                              help="Max vectors in RAM (hot tier) when tiered is enabled")
    index_parser.add_argument("--ssd-dir", type=str, default="./cache/ssd_index",
                              help="Directory for SSD-based cold tier")
    index_parser.set_defaults(func=run_index)
    
    # Profile command (HotpotQA questions)
    profile_parser = subparsers.add_parser("profile", help="Run performance profiling with HotpotQA")
    profile_parser.add_argument("--endpoint", type=str, default="http://localhost:8080",
                                help="RAG server endpoint (default: http://localhost:8080)")
    profile_parser.add_argument("--num-queries", type=int, default=10)
    profile_parser.add_argument("--warmup-queries", type=int, default=2)
    profile_parser.add_argument("--output", type=str, default=None)
    profile_parser.set_defaults(func=run_profile)
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Run a single query")
    query_parser.add_argument("question", type=str, help="Question to ask")
    query_parser.add_argument("--endpoint", type=str, default="http://localhost:8080")
    query_parser.add_argument("--top-k", type=int, default=5)
    query_parser.add_argument("--max-iterations", type=int, default=3)
    query_parser.add_argument("--verbose", "-v", action="store_true")
    query_parser.set_defaults(func=run_query)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()
