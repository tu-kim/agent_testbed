"""
Frontend Profiler for Multi-hop RAG Pipeline

Sends N queries to the RAG pipeline and profiles performance metrics
including retrieval latency, LLM latency, and end-to-end time.
"""

import asyncio
import logging
import time
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
import statistics

import httpx
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QueryMetrics:
    """Metrics for a single query."""
    query_id: int
    question: str
    answer: str
    success: bool
    
    # Timing metrics (in milliseconds)
    total_time_ms: float
    retrieval_time_ms: float
    llm_time_ms: float
    
    # Additional metrics
    num_iterations: int
    num_retrieved_docs: int
    
    # Error info
    error_message: Optional[str] = None
    
    # Timestamps
    start_time: str = ""
    end_time: str = ""


@dataclass
class ProfilerSummary:
    """Summary statistics from profiling run."""
    num_queries: int
    num_successful: int
    num_failed: int
    
    # Timing statistics (in milliseconds)
    avg_total_time_ms: float
    min_total_time_ms: float
    max_total_time_ms: float
    std_total_time_ms: float
    p50_total_time_ms: float
    p90_total_time_ms: float
    p99_total_time_ms: float
    
    avg_retrieval_time_ms: float
    min_retrieval_time_ms: float
    max_retrieval_time_ms: float
    
    avg_llm_time_ms: float
    min_llm_time_ms: float
    max_llm_time_ms: float
    
    # Throughput
    queries_per_second: float
    total_run_time_ms: float
    
    # Additional stats
    avg_iterations: float
    avg_retrieved_docs: float


@dataclass
class ProfilerConfig:
    """Configuration for profiler."""
    # Target endpoint
    endpoint: str = "http://localhost:8000"
    
    # Query settings
    num_queries: int = 10
    warmup_queries: int = 2
    
    # Timeout settings
    timeout_seconds: float = 120.0
    
    # Logging
    log_detailed: bool = True
    output_file: Optional[str] = None
    
    # Concurrency
    concurrent_requests: int = 1


class RAGProfiler:
    """
    Profiler for RAG pipeline performance evaluation.
    
    Sends queries to the RAG endpoint and collects detailed
    performance metrics including retrieval and LLM latencies.
    """
    
    def __init__(self, config: ProfilerConfig):
        self.config = config
        self.metrics: List[QueryMetrics] = []
        self.warmup_metrics: List[QueryMetrics] = []
        self.http_client: Optional[httpx.AsyncClient] = None
    
    async def _init_client(self):
        """Initialize HTTP client."""
        if self.http_client is None:
            self.http_client = httpx.AsyncClient(
                timeout=self.config.timeout_seconds
            )
    
    async def _close_client(self):
        """Close HTTP client."""
        if self.http_client:
            await self.http_client.aclose()
            self.http_client = None
    
    async def _send_query(
        self, 
        query_id: int, 
        question: str
    ) -> QueryMetrics:
        """
        Send a single query to the RAG endpoint.
        
        Args:
            query_id: Query identifier
            question: Question text
            
        Returns:
            QueryMetrics with timing information
        """
        start_time = datetime.now()
        start_perf = time.perf_counter()
        
        try:
            # Send request to RAG endpoint
            response = await self.http_client.post(
                f"{self.config.endpoint}/query",
                json={"question": question}
            )
            response.raise_for_status()
            
            result = response.json()
            
            end_perf = time.perf_counter()
            end_time = datetime.now()
            
            return QueryMetrics(
                query_id=query_id,
                question=question,
                answer=result.get("answer", ""),
                success=result.get("success", True),
                total_time_ms=result.get("total_time_ms", (end_perf - start_perf) * 1000),
                retrieval_time_ms=result.get("retrieval_time_ms", 0),
                llm_time_ms=result.get("llm_time_ms", 0),
                num_iterations=result.get("num_iterations", 1),
                num_retrieved_docs=result.get("num_retrieved_docs", 0),
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat()
            )
            
        except Exception as e:
            end_perf = time.perf_counter()
            end_time = datetime.now()
            
            logger.error(f"Query {query_id} failed: {e}")
            
            return QueryMetrics(
                query_id=query_id,
                question=question,
                answer="",
                success=False,
                total_time_ms=(end_perf - start_perf) * 1000,
                retrieval_time_ms=0,
                llm_time_ms=0,
                num_iterations=0,
                num_retrieved_docs=0,
                error_message=str(e),
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat()
            )
    
    async def _run_warmup(self, questions: List[str]):
        """
        Run warmup queries to initialize caches.
        
        Args:
            questions: List of warmup questions
        """
        if self.config.warmup_queries <= 0:
            return
        
        warmup_questions = questions[:self.config.warmup_queries]
        logger.info(f"Running {len(warmup_questions)} warmup queries...")
        
        for i, question in enumerate(warmup_questions):
            metrics = await self._send_query(-i - 1, question)
            self.warmup_metrics.append(metrics)
            
            if self.config.log_detailed:
                logger.info(f"Warmup {i+1}: {metrics.total_time_ms:.2f}ms")
        
        logger.info("Warmup complete")
    
    def _calculate_percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile of a list of values."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def _calculate_summary(self) -> ProfilerSummary:
        """Calculate summary statistics from collected metrics."""
        successful_metrics = [m for m in self.metrics if m.success]
        
        if not successful_metrics:
            return ProfilerSummary(
                num_queries=len(self.metrics),
                num_successful=0,
                num_failed=len(self.metrics),
                avg_total_time_ms=0, min_total_time_ms=0, max_total_time_ms=0,
                std_total_time_ms=0, p50_total_time_ms=0, p90_total_time_ms=0,
                p99_total_time_ms=0, avg_retrieval_time_ms=0, min_retrieval_time_ms=0,
                max_retrieval_time_ms=0, avg_llm_time_ms=0, min_llm_time_ms=0,
                max_llm_time_ms=0, queries_per_second=0, total_run_time_ms=0,
                avg_iterations=0, avg_retrieved_docs=0
            )
        
        total_times = [m.total_time_ms for m in successful_metrics]
        retrieval_times = [m.retrieval_time_ms for m in successful_metrics]
        llm_times = [m.llm_time_ms for m in successful_metrics]
        iterations = [m.num_iterations for m in successful_metrics]
        docs = [m.num_retrieved_docs for m in successful_metrics]
        
        total_run_time = sum(total_times)
        
        return ProfilerSummary(
            num_queries=len(self.metrics),
            num_successful=len(successful_metrics),
            num_failed=len(self.metrics) - len(successful_metrics),
            
            avg_total_time_ms=statistics.mean(total_times),
            min_total_time_ms=min(total_times),
            max_total_time_ms=max(total_times),
            std_total_time_ms=statistics.stdev(total_times) if len(total_times) > 1 else 0,
            p50_total_time_ms=self._calculate_percentile(total_times, 50),
            p90_total_time_ms=self._calculate_percentile(total_times, 90),
            p99_total_time_ms=self._calculate_percentile(total_times, 99),
            
            avg_retrieval_time_ms=statistics.mean(retrieval_times),
            min_retrieval_time_ms=min(retrieval_times),
            max_retrieval_time_ms=max(retrieval_times),
            
            avg_llm_time_ms=statistics.mean(llm_times),
            min_llm_time_ms=min(llm_times),
            max_llm_time_ms=max(llm_times),
            
            queries_per_second=len(successful_metrics) / (total_run_time / 1000) if total_run_time > 0 else 0,
            total_run_time_ms=total_run_time,
            
            avg_iterations=statistics.mean(iterations),
            avg_retrieved_docs=statistics.mean(docs)
        )
    
    async def run(self, questions: List[str]) -> ProfilerSummary:
        """
        Run profiling on a list of questions.
        
        Args:
            questions: List of questions to profile
            
        Returns:
            ProfilerSummary with statistics
        """
        await self._init_client()
        
        try:
            # Run warmup
            await self._run_warmup(questions)
            
            # Run actual profiling
            test_questions = questions[self.config.warmup_queries:self.config.num_queries + self.config.warmup_queries]
            
            if not test_questions:
                test_questions = questions[:self.config.num_queries]
            
            logger.info(f"Running {len(test_questions)} profiling queries...")
            
            if self.config.concurrent_requests > 1:
                # Concurrent execution
                semaphore = asyncio.Semaphore(self.config.concurrent_requests)
                
                async def bounded_query(query_id: int, question: str):
                    async with semaphore:
                        return await self._send_query(query_id, question)
                
                tasks = [
                    bounded_query(i, q) 
                    for i, q in enumerate(test_questions)
                ]
                self.metrics = await asyncio.gather(*tasks)
            else:
                # Sequential execution with progress bar
                for i, question in enumerate(tqdm(test_questions, desc="Profiling")):
                    metrics = await self._send_query(i, question)
                    self.metrics.append(metrics)
                    
                    if self.config.log_detailed:
                        logger.info(
                            f"Query {i+1}: total={metrics.total_time_ms:.2f}ms, "
                            f"retrieval={metrics.retrieval_time_ms:.2f}ms, "
                            f"llm={metrics.llm_time_ms:.2f}ms"
                        )
            
            # Calculate summary
            summary = self._calculate_summary()
            
            # Save results if output file specified
            if self.config.output_file:
                self._save_results(summary)
            
            return summary
            
        finally:
            await self._close_client()
    
    def _save_results(self, summary: ProfilerSummary):
        """Save profiling results to file."""
        results = {
            "summary": asdict(summary),
            "metrics": [asdict(m) for m in self.metrics],
            "warmup_metrics": [asdict(m) for m in self.warmup_metrics],
            "config": asdict(self.config)
        }
        
        with open(self.config.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {self.config.output_file}")
    
    def print_summary(self, summary: ProfilerSummary):
        """Print formatted summary to console."""
        print("\n" + "=" * 60)
        print("RAG Pipeline Profiling Summary")
        print("=" * 60)
        
        print(f"\nQueries: {summary.num_queries} total, "
              f"{summary.num_successful} successful, "
              f"{summary.num_failed} failed")
        
        print(f"\nTotal Time (ms):")
        print(f"  Average: {summary.avg_total_time_ms:.2f}")
        print(f"  Min:     {summary.min_total_time_ms:.2f}")
        print(f"  Max:     {summary.max_total_time_ms:.2f}")
        print(f"  Std Dev: {summary.std_total_time_ms:.2f}")
        print(f"  P50:     {summary.p50_total_time_ms:.2f}")
        print(f"  P90:     {summary.p90_total_time_ms:.2f}")
        print(f"  P99:     {summary.p99_total_time_ms:.2f}")
        
        print(f"\nRetrieval Time (ms):")
        print(f"  Average: {summary.avg_retrieval_time_ms:.2f}")
        print(f"  Min:     {summary.min_retrieval_time_ms:.2f}")
        print(f"  Max:     {summary.max_retrieval_time_ms:.2f}")
        
        print(f"\nLLM Time (ms):")
        print(f"  Average: {summary.avg_llm_time_ms:.2f}")
        print(f"  Min:     {summary.min_llm_time_ms:.2f}")
        print(f"  Max:     {summary.max_llm_time_ms:.2f}")
        
        print(f"\nThroughput:")
        print(f"  Queries/second: {summary.queries_per_second:.2f}")
        print(f"  Total run time: {summary.total_run_time_ms:.2f}ms")
        
        print(f"\nRAG Metrics:")
        print(f"  Avg iterations: {summary.avg_iterations:.2f}")
        print(f"  Avg retrieved docs: {summary.avg_retrieved_docs:.2f}")
        
        print("=" * 60 + "\n")


def create_profiler(
    endpoint: str = "http://localhost:8000",
    num_queries: int = 10,
    warmup_queries: int = 2,
    **kwargs
) -> RAGProfiler:
    """
    Factory function to create a RAG profiler.
    
    Args:
        endpoint: RAG service endpoint
        num_queries: Number of queries to profile
        warmup_queries: Number of warmup queries
        **kwargs: Additional configuration options
        
    Returns:
        Configured RAGProfiler instance
    """
    config = ProfilerConfig(
        endpoint=endpoint,
        num_queries=num_queries,
        warmup_queries=warmup_queries,
        **{k: v for k, v in kwargs.items() if hasattr(ProfilerConfig, k)}
    )
    
    return RAGProfiler(config)


async def run_profiling(
    endpoint: str,
    questions: List[str],
    num_queries: int = 10,
    warmup_queries: int = 2,
    output_file: Optional[str] = None
) -> ProfilerSummary:
    """
    Convenience function to run profiling.
    
    Args:
        endpoint: RAG service endpoint
        questions: List of questions to profile
        num_queries: Number of queries to profile
        warmup_queries: Number of warmup queries
        output_file: Optional file to save results
        
    Returns:
        ProfilerSummary with statistics
    """
    profiler = create_profiler(
        endpoint=endpoint,
        num_queries=num_queries,
        warmup_queries=warmup_queries,
        output_file=output_file
    )
    
    summary = await profiler.run(questions)
    profiler.print_summary(summary)
    
    return summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Pipeline Profiler")
    parser.add_argument("--endpoint", type=str, default="http://localhost:8000")
    parser.add_argument("--num-queries", type=int, default=10)
    parser.add_argument("--warmup-queries", type=int, default=2)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--concurrent", type=int, default=1)
    
    args = parser.parse_args()
    
    # Sample questions for testing
    sample_questions = [
        "What is the capital of France?",
        "Who invented the telephone?",
        "What is machine learning?",
        "When was the Eiffel Tower built?",
        "What is the largest planet in our solar system?",
        "Who wrote Romeo and Juliet?",
        "What is the speed of light?",
        "When did World War II end?",
        "What is the chemical formula for water?",
        "Who painted the Mona Lisa?",
        "What is the tallest mountain in the world?",
        "When was the Declaration of Independence signed?",
    ]
    
    asyncio.run(run_profiling(
        endpoint=args.endpoint,
        questions=sample_questions,
        num_queries=args.num_queries,
        warmup_queries=args.warmup_queries,
        output_file=args.output
    ))
