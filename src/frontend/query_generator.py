"""
Query Generator for RAG Pipeline Profiling

Generates queries following a Poisson arrival process with gradually increasing QPS.
Profiles total time, retrieval time, LLM time, P90/P99 latencies, and throughput.
"""

import asyncio
import aiohttp
import time
import random
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import logging
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Result of a single query execution."""
    query_id: str
    question: str
    answer: str
    
    # Timestamps (all in seconds since epoch)
    submit_time: float  # When query was submitted
    start_time: float   # When server started processing (from response)
    end_time: float     # When response was received
    
    # Latency breakdown (in milliseconds)
    total_time_ms: float
    retrieval_time_ms: float
    llm_time_ms: float
    
    # Queue/wait time if available (in milliseconds)
    queue_time_ms: Optional[float] = None
    
    # Metadata
    success: bool = True
    error_message: Optional[str] = None
    target_qps: float = 0.0


@dataclass
class QPSStageResult:
    """Results for a single QPS stage."""
    target_qps: float
    actual_qps: float
    num_queries: int
    duration_seconds: float
    
    # Latency statistics (in milliseconds)
    total_time_avg: float
    total_time_p50: float
    total_time_p90: float
    total_time_p99: float
    
    retrieval_time_avg: float
    retrieval_time_p90: float
    retrieval_time_p99: float
    
    llm_time_avg: float
    llm_time_p90: float
    llm_time_p99: float
    
    # Queue time (if available)
    queue_time_avg: Optional[float] = None
    queue_time_p90: Optional[float] = None
    queue_time_p99: Optional[float] = None
    
    # Throughput
    throughput_qps: float = 0.0
    success_rate: float = 1.0


@dataclass
class ProfilerConfig:
    """Configuration for the query generator profiler."""
    # Server endpoint
    endpoint: str = "http://localhost:8080"
    
    # QPS ramp configuration
    start_qps: float = 1.0
    end_qps: float = 10.0
    qps_step: float = 1.0
    
    # Duration per QPS stage (seconds)
    stage_duration: float = 30.0
    
    # Warmup configuration
    warmup_queries: int = 5
    warmup_qps: float = 1.0
    
    # Query sampling
    seed: int = 42
    
    # Timeout
    timeout_seconds: float = 120.0
    
    # Concurrent request limit
    max_concurrent: int = 100


class PoissonQueryGenerator:
    """
    Generates queries following a Poisson arrival process.
    
    Implements a profiler that:
    1. Gradually increases QPS from start_qps to end_qps
    2. Uses Poisson distribution for query arrival times
    3. Measures total/retrieval/LLM latencies with P90/P99
    4. Tracks queue/wait time when available
    5. Reports throughput (queries per second)
    """
    
    def __init__(self, config: ProfilerConfig, questions: List[Dict[str, Any]]):
        self.config = config
        self.questions = questions
        self.results: List[QueryResult] = []
        self.stage_results: List[QPSStageResult] = []
        
        # Set random seed for reproducibility
        random.seed(config.seed)
        np.random.seed(config.seed)
        
        # Semaphore for concurrent request limiting
        self._semaphore: Optional[asyncio.Semaphore] = None
        
    def _get_poisson_interval(self, qps: float) -> float:
        """
        Generate inter-arrival time following exponential distribution.
        
        For a Poisson process with rate λ (qps), inter-arrival times
        follow an exponential distribution with mean 1/λ.
        """
        if qps <= 0:
            return 1.0
        return np.random.exponential(1.0 / qps)
    
    def _sample_question(self) -> Dict[str, Any]:
        """Sample a random question from the dataset."""
        return random.choice(self.questions)
    
    async def _send_query(
        self, 
        session: aiohttp.ClientSession,
        question: Dict[str, Any],
        query_id: str,
        target_qps: float
    ) -> QueryResult:
        """
        Send a single query to the RAG server and measure latencies.
        """
        submit_time = time.time()
        
        try:
            async with self._semaphore:
                payload = {
                    "question": question["question"],
                    "query_id": query_id
                }
                
                async with session.post(
                    f"{self.config.endpoint}/query",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)
                ) as response:
                    end_time = time.time()
                    
                    if response.status != 200:
                        error_text = await response.text()
                        return QueryResult(
                            query_id=query_id,
                            question=question["question"],
                            answer="",
                            submit_time=submit_time,
                            start_time=submit_time,
                            end_time=end_time,
                            total_time_ms=(end_time - submit_time) * 1000,
                            retrieval_time_ms=0,
                            llm_time_ms=0,
                            success=False,
                            error_message=f"HTTP {response.status}: {error_text[:100]}",
                            target_qps=target_qps
                        )
                    
                    result_data = await response.json()
                    
                    # Extract latency breakdown from response
                    # Expected response format:
                    # {
                    #   "answer": "...",
                    #   "retrieval_time_ms": 123.45,
                    #   "llm_time_ms": 456.78,
                    #   "total_time_ms": 580.23,
                    #   "server_start_time": 1234567890.123,  # optional
                    #   "queue_time_ms": 10.5  # optional
                    # }
                    
                    retrieval_time_ms = result_data.get("retrieval_time_ms", 0)
                    llm_time_ms = result_data.get("llm_time_ms", 0)
                    server_total_ms = result_data.get("total_time_ms", 0)
                    
                    # Calculate total time from client perspective
                    client_total_ms = (end_time - submit_time) * 1000
                    
                    # Get server start time if available (for queue time calculation)
                    server_start_time = result_data.get("server_start_time")
                    queue_time_ms = result_data.get("queue_time_ms")
                    
                    # If server provides start time, calculate actual queue time
                    if server_start_time is not None and queue_time_ms is None:
                        # Queue time = time between submit and server start processing
                        queue_time_ms = (server_start_time - submit_time) * 1000
                        if queue_time_ms < 0:
                            queue_time_ms = None  # Clock skew, ignore
                    
                    return QueryResult(
                        query_id=query_id,
                        question=question["question"],
                        answer=result_data.get("answer", ""),
                        submit_time=submit_time,
                        start_time=server_start_time or submit_time,
                        end_time=end_time,
                        total_time_ms=client_total_ms,
                        retrieval_time_ms=retrieval_time_ms,
                        llm_time_ms=llm_time_ms,
                        queue_time_ms=queue_time_ms,
                        success=True,
                        target_qps=target_qps
                    )
                    
        except asyncio.TimeoutError:
            end_time = time.time()
            return QueryResult(
                query_id=query_id,
                question=question["question"],
                answer="",
                submit_time=submit_time,
                start_time=submit_time,
                end_time=end_time,
                total_time_ms=(end_time - submit_time) * 1000,
                retrieval_time_ms=0,
                llm_time_ms=0,
                success=False,
                error_message="Timeout",
                target_qps=target_qps
            )
        except Exception as e:
            end_time = time.time()
            return QueryResult(
                query_id=query_id,
                question=question["question"],
                answer="",
                submit_time=submit_time,
                start_time=submit_time,
                end_time=end_time,
                total_time_ms=(end_time - submit_time) * 1000,
                retrieval_time_ms=0,
                llm_time_ms=0,
                success=False,
                error_message=str(e),
                target_qps=target_qps
            )
    
    async def _run_qps_stage(
        self,
        session: aiohttp.ClientSession,
        target_qps: float,
        duration: float
    ) -> List[QueryResult]:
        """
        Run a single QPS stage with Poisson arrivals.
        """
        logger.info(f"Starting QPS stage: target={target_qps:.1f} QPS, duration={duration:.0f}s")
        
        stage_results = []
        tasks = []
        stage_start = time.time()
        query_count = 0
        
        while time.time() - stage_start < duration:
            # Sample question and generate query ID
            question = self._sample_question()
            query_id = f"q_{target_qps:.1f}_{query_count}"
            
            # Create async task for query
            task = asyncio.create_task(
                self._send_query(session, question, query_id, target_qps)
            )
            tasks.append(task)
            query_count += 1
            
            # Wait for Poisson-distributed inter-arrival time
            interval = self._get_poisson_interval(target_qps)
            await asyncio.sleep(interval)
        
        # Wait for all pending tasks to complete
        if tasks:
            completed_results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in completed_results:
                if isinstance(result, QueryResult):
                    stage_results.append(result)
                elif isinstance(result, Exception):
                    logger.warning(f"Query failed with exception: {result}")
        
        logger.info(f"QPS stage complete: sent {query_count} queries, received {len(stage_results)} results")
        return stage_results
    
    def _compute_percentile(self, values: List[float], percentile: float) -> float:
        """Compute percentile of a list of values."""
        if not values:
            return 0.0
        return float(np.percentile(values, percentile))
    
    def _compute_stage_statistics(
        self,
        results: List[QueryResult],
        target_qps: float,
        duration: float
    ) -> QPSStageResult:
        """
        Compute statistics for a QPS stage.
        """
        successful = [r for r in results if r.success]
        
        if not successful:
            return QPSStageResult(
                target_qps=target_qps,
                actual_qps=0,
                num_queries=len(results),
                duration_seconds=duration,
                total_time_avg=0, total_time_p50=0, total_time_p90=0, total_time_p99=0,
                retrieval_time_avg=0, retrieval_time_p90=0, retrieval_time_p99=0,
                llm_time_avg=0, llm_time_p90=0, llm_time_p99=0,
                throughput_qps=0,
                success_rate=0
            )
        
        # Extract latency values
        total_times = [r.total_time_ms for r in successful]
        retrieval_times = [r.retrieval_time_ms for r in successful]
        llm_times = [r.llm_time_ms for r in successful]
        queue_times = [r.queue_time_ms for r in successful if r.queue_time_ms is not None]
        
        # Calculate actual throughput
        if successful:
            first_submit = min(r.submit_time for r in successful)
            last_end = max(r.end_time for r in successful)
            actual_duration = last_end - first_submit
            throughput = len(successful) / actual_duration if actual_duration > 0 else 0
        else:
            throughput = 0
        
        stage_result = QPSStageResult(
            target_qps=target_qps,
            actual_qps=len(results) / duration if duration > 0 else 0,
            num_queries=len(results),
            duration_seconds=duration,
            
            total_time_avg=np.mean(total_times),
            total_time_p50=self._compute_percentile(total_times, 50),
            total_time_p90=self._compute_percentile(total_times, 90),
            total_time_p99=self._compute_percentile(total_times, 99),
            
            retrieval_time_avg=np.mean(retrieval_times),
            retrieval_time_p90=self._compute_percentile(retrieval_times, 90),
            retrieval_time_p99=self._compute_percentile(retrieval_times, 99),
            
            llm_time_avg=np.mean(llm_times),
            llm_time_p90=self._compute_percentile(llm_times, 90),
            llm_time_p99=self._compute_percentile(llm_times, 99),
            
            throughput_qps=throughput,
            success_rate=len(successful) / len(results) if results else 0
        )
        
        # Add queue time statistics if available
        if queue_times:
            stage_result.queue_time_avg = np.mean(queue_times)
            stage_result.queue_time_p90 = self._compute_percentile(queue_times, 90)
            stage_result.queue_time_p99 = self._compute_percentile(queue_times, 99)
        
        return stage_result
    
    async def _warmup(self, session: aiohttp.ClientSession):
        """Run warmup queries."""
        logger.info(f"Running {self.config.warmup_queries} warmup queries...")
        
        for i in range(self.config.warmup_queries):
            question = self._sample_question()
            result = await self._send_query(
                session, question, f"warmup_{i}", self.config.warmup_qps
            )
            if result.success:
                logger.info(f"Warmup {i+1}/{self.config.warmup_queries}: {result.total_time_ms:.1f}ms")
            else:
                logger.warning(f"Warmup {i+1} failed: {result.error_message}")
            
            await asyncio.sleep(1.0 / self.config.warmup_qps)
        
        logger.info("Warmup complete")
    
    async def run(self) -> Dict[str, Any]:
        """
        Run the full profiling session.
        
        Returns:
            Dictionary containing all profiling results.
        """
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        async with aiohttp.ClientSession() as session:
            # Warmup
            await self._warmup(session)
            
            # Generate QPS stages
            qps_values = np.arange(
                self.config.start_qps,
                self.config.end_qps + self.config.qps_step,
                self.config.qps_step
            )
            
            logger.info(f"Running {len(qps_values)} QPS stages: {qps_values.tolist()}")
            
            # Run each QPS stage
            for target_qps in qps_values:
                stage_results = await self._run_qps_stage(
                    session, target_qps, self.config.stage_duration
                )
                self.results.extend(stage_results)
                
                # Compute and store stage statistics
                stage_stats = self._compute_stage_statistics(
                    stage_results, target_qps, self.config.stage_duration
                )
                self.stage_results.append(stage_stats)
                
                # Log stage summary
                self._log_stage_summary(stage_stats)
        
        return self._generate_report()
    
    def _log_stage_summary(self, stage: QPSStageResult):
        """Log summary for a QPS stage."""
        logger.info(f"\n{'='*60}")
        logger.info(f"QPS Stage Summary: Target={stage.target_qps:.1f}")
        logger.info(f"{'='*60}")
        logger.info(f"  Queries: {stage.num_queries} (Success rate: {stage.success_rate*100:.1f}%)")
        logger.info(f"  Throughput: {stage.throughput_qps:.2f} QPS")
        logger.info(f"  Total Time:     Avg={stage.total_time_avg:.1f}ms, P90={stage.total_time_p90:.1f}ms, P99={stage.total_time_p99:.1f}ms")
        logger.info(f"  Retrieval Time: Avg={stage.retrieval_time_avg:.1f}ms, P90={stage.retrieval_time_p90:.1f}ms, P99={stage.retrieval_time_p99:.1f}ms")
        logger.info(f"  LLM Time:       Avg={stage.llm_time_avg:.1f}ms, P90={stage.llm_time_p90:.1f}ms, P99={stage.llm_time_p99:.1f}ms")
        if stage.queue_time_avg is not None:
            logger.info(f"  Queue Time:     Avg={stage.queue_time_avg:.1f}ms, P90={stage.queue_time_p90:.1f}ms, P99={stage.queue_time_p99:.1f}ms")
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate final profiling report."""
        successful = [r for r in self.results if r.success]
        
        # Overall statistics
        all_total = [r.total_time_ms for r in successful]
        all_retrieval = [r.retrieval_time_ms for r in successful]
        all_llm = [r.llm_time_ms for r in successful]
        all_queue = [r.queue_time_ms for r in successful if r.queue_time_ms is not None]
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "endpoint": self.config.endpoint,
                "start_qps": self.config.start_qps,
                "end_qps": self.config.end_qps,
                "qps_step": self.config.qps_step,
                "stage_duration": self.config.stage_duration,
                "seed": self.config.seed
            },
            "summary": {
                "total_queries": len(self.results),
                "successful_queries": len(successful),
                "success_rate": len(successful) / len(self.results) if self.results else 0,
                "num_stages": len(self.stage_results)
            },
            "overall_latency": {
                "total_time": {
                    "avg_ms": np.mean(all_total) if all_total else 0,
                    "p50_ms": self._compute_percentile(all_total, 50),
                    "p90_ms": self._compute_percentile(all_total, 90),
                    "p99_ms": self._compute_percentile(all_total, 99),
                    "min_ms": min(all_total) if all_total else 0,
                    "max_ms": max(all_total) if all_total else 0
                },
                "retrieval_time": {
                    "avg_ms": np.mean(all_retrieval) if all_retrieval else 0,
                    "p50_ms": self._compute_percentile(all_retrieval, 50),
                    "p90_ms": self._compute_percentile(all_retrieval, 90),
                    "p99_ms": self._compute_percentile(all_retrieval, 99)
                },
                "llm_time": {
                    "avg_ms": np.mean(all_llm) if all_llm else 0,
                    "p50_ms": self._compute_percentile(all_llm, 50),
                    "p90_ms": self._compute_percentile(all_llm, 90),
                    "p99_ms": self._compute_percentile(all_llm, 99)
                }
            },
            "stages": [
                {
                    "target_qps": s.target_qps,
                    "actual_qps": s.actual_qps,
                    "throughput_qps": s.throughput_qps,
                    "num_queries": s.num_queries,
                    "success_rate": s.success_rate,
                    "total_time": {
                        "avg_ms": s.total_time_avg,
                        "p50_ms": s.total_time_p50,
                        "p90_ms": s.total_time_p90,
                        "p99_ms": s.total_time_p99
                    },
                    "retrieval_time": {
                        "avg_ms": s.retrieval_time_avg,
                        "p90_ms": s.retrieval_time_p90,
                        "p99_ms": s.retrieval_time_p99
                    },
                    "llm_time": {
                        "avg_ms": s.llm_time_avg,
                        "p90_ms": s.llm_time_p90,
                        "p99_ms": s.llm_time_p99
                    },
                    "queue_time": {
                        "avg_ms": s.queue_time_avg,
                        "p90_ms": s.queue_time_p90,
                        "p99_ms": s.queue_time_p99
                    } if s.queue_time_avg is not None else None
                }
                for s in self.stage_results
            ]
        }
        
        # Add queue time to overall if available
        if all_queue:
            report["overall_latency"]["queue_time"] = {
                "avg_ms": np.mean(all_queue),
                "p50_ms": self._compute_percentile(all_queue, 50),
                "p90_ms": self._compute_percentile(all_queue, 90),
                "p99_ms": self._compute_percentile(all_queue, 99)
            }
        
        return report


def print_report(report: Dict[str, Any]):
    """Pretty print the profiling report."""
    print("\n" + "="*80)
    print("RAG PROFILING REPORT")
    print("="*80)
    
    print(f"\nTimestamp: {report['timestamp']}")
    print(f"Endpoint: {report['config']['endpoint']}")
    print(f"QPS Range: {report['config']['start_qps']} -> {report['config']['end_qps']} (step: {report['config']['qps_step']})")
    
    summary = report['summary']
    print(f"\n--- Summary ---")
    print(f"Total Queries: {summary['total_queries']}")
    print(f"Successful: {summary['successful_queries']} ({summary['success_rate']*100:.1f}%)")
    
    overall = report['overall_latency']
    print(f"\n--- Overall Latency (ms) ---")
    print(f"{'Metric':<20} {'Avg':>10} {'P50':>10} {'P90':>10} {'P99':>10}")
    print("-" * 60)
    
    for name, data in [("Total Time", overall['total_time']),
                       ("Retrieval Time", overall['retrieval_time']),
                       ("LLM Time", overall['llm_time'])]:
        print(f"{name:<20} {data['avg_ms']:>10.1f} {data['p50_ms']:>10.1f} {data['p90_ms']:>10.1f} {data['p99_ms']:>10.1f}")
    
    if 'queue_time' in overall:
        qt = overall['queue_time']
        print(f"{'Queue Time':<20} {qt['avg_ms']:>10.1f} {qt['p50_ms']:>10.1f} {qt['p90_ms']:>10.1f} {qt['p99_ms']:>10.1f}")
    
    print(f"\n--- Per-Stage Results ---")
    print(f"{'QPS':>6} {'Throughput':>12} {'Queries':>8} {'Success':>8} {'Total P90':>12} {'Total P99':>12}")
    print("-" * 70)
    
    for stage in report['stages']:
        print(f"{stage['target_qps']:>6.1f} {stage['throughput_qps']:>12.2f} {stage['num_queries']:>8} "
              f"{stage['success_rate']*100:>7.1f}% {stage['total_time']['p90_ms']:>12.1f} {stage['total_time']['p99_ms']:>12.1f}")
    
    print("="*80)


async def run_profiling(
    endpoint: str,
    questions: List[Dict[str, Any]],
    start_qps: float = 1.0,
    end_qps: float = 10.0,
    qps_step: float = 1.0,
    stage_duration: float = 30.0,
    seed: int = 42,
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run profiling with the query generator.
    
    Args:
        endpoint: RAG server endpoint URL.
        questions: List of questions to sample from.
        start_qps: Starting QPS.
        end_qps: Ending QPS.
        qps_step: QPS increment per stage.
        stage_duration: Duration of each QPS stage in seconds.
        seed: Random seed for reproducibility.
        output_file: Optional file path to save JSON report.
    
    Returns:
        Profiling report dictionary.
    """
    config = ProfilerConfig(
        endpoint=endpoint,
        start_qps=start_qps,
        end_qps=end_qps,
        qps_step=qps_step,
        stage_duration=stage_duration,
        seed=seed
    )
    
    generator = PoissonQueryGenerator(config, questions)
    report = await generator.run()
    
    # Print report
    print_report(report)
    
    # Save to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved to: {output_file}")
    
    return report


if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    from src.retrieval.dataset_loader import create_dataset_manager
    
    parser = argparse.ArgumentParser(description="RAG Query Generator Profiler")
    parser.add_argument("--endpoint", type=str, default="http://localhost:8080",
                        help="RAG server endpoint")
    parser.add_argument("--start-qps", type=float, default=1.0,
                        help="Starting QPS")
    parser.add_argument("--end-qps", type=float, default=10.0,
                        help="Ending QPS")
    parser.add_argument("--qps-step", type=float, default=1.0,
                        help="QPS increment per stage")
    parser.add_argument("--stage-duration", type=float, default=30.0,
                        help="Duration of each QPS stage (seconds)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path")
    parser.add_argument("--max-queries", type=int, default=None,
                        help="Maximum queries to load from dataset")
    
    args = parser.parse_args()
    
    # Load questions
    manager = create_dataset_manager()
    questions = manager.load_queries(max_samples=args.max_queries)
    
    if not questions:
        logger.error("No questions loaded. Exiting.")
        sys.exit(1)
    
    logger.info(f"Loaded {len(questions)} questions for profiling")
    
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
