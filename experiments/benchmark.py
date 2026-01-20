"""
Benchmark Automation Script

Experiment 1: QPS sweep with fixed max_iterations=3, top_k=5
Experiment 2: QPS sweep with varying max_iterations (1-5)

Generates graphs for:
- Total latency (avg, p99) vs QPS
- Queue waiting time ratio vs QPS
"""

import os
import sys
import json
import time
import argparse
import subprocess
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    endpoint: str = "http://localhost:8080"
    start_qps: float = 1.0
    end_qps: float = 10.0
    qps_step: float = 1.0
    stage_duration: int = 30
    max_iterations: int = 3
    top_k: int = 5
    eval_split: str = "validation"
    output_dir: str = "./results"


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    config: Dict[str, Any]
    qps_values: List[float] = field(default_factory=list)
    total_latency_avg: List[float] = field(default_factory=list)
    total_latency_p99: List[float] = field(default_factory=list)
    llm_latency_avg: List[float] = field(default_factory=list)
    llm_latency_p99: List[float] = field(default_factory=list)
    queue_time_avg: List[float] = field(default_factory=list)
    queue_time_p99: List[float] = field(default_factory=list)
    queue_time_ratio: List[float] = field(default_factory=list)
    throughput: List[float] = field(default_factory=list)


class BenchmarkRunner:
    """
    Runs benchmark experiments and collects results.
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results_dir = Path(config.output_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def run_single_experiment(
        self,
        max_iterations: int,
        top_k: int,
        experiment_name: str
    ) -> ExperimentResult:
        """
        Run a single experiment with given parameters.
        
        Args:
            max_iterations: Maximum IRCoT iterations
            top_k: Number of documents to retrieve
            experiment_name: Name for output files
        
        Returns:
            ExperimentResult with collected metrics
        """
        print(f"\n{'='*60}")
        print(f"Running experiment: {experiment_name}")
        print(f"  max_iterations={max_iterations}, top_k={top_k}")
        print(f"  QPS range: {self.config.start_qps} -> {self.config.end_qps}")
        print(f"{'='*60}\n")
        
        output_file = self.results_dir / f"{experiment_name}.json"
        
        # Build command
        cmd = [
            sys.executable, str(PROJECT_ROOT / "run.py"), "profile",
            "--endpoint", self.config.endpoint,
            "--start-qps", str(self.config.start_qps),
            "--end-qps", str(self.config.end_qps),
            "--qps-step", str(self.config.qps_step),
            "--stage-duration", str(self.config.stage_duration),
            "--max-iterations", str(max_iterations),
            "--top-k", str(top_k),
            "--eval-split", self.config.eval_split,
            "--output", str(output_file)
        ]
        
        print(f"Command: {' '.join(cmd)}\n")
        
        # Run the profiling command
        try:
            result = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode != 0:
                print(f"Error running experiment: {result.stderr}")
                return None
            
            print(result.stdout)
            
        except subprocess.TimeoutExpired:
            print(f"Experiment timed out after 1 hour")
            return None
        except Exception as e:
            print(f"Error: {e}")
            return None
        
        # Parse results
        return self._parse_results(output_file, max_iterations, top_k)
    
    def _parse_results(
        self,
        output_file: Path,
        max_iterations: int,
        top_k: int
    ) -> Optional[ExperimentResult]:
        """Parse results from JSON output file."""
        if not output_file.exists():
            print(f"Output file not found: {output_file}")
            return None
        
        with open(output_file, 'r') as f:
            data = json.load(f)
        
        result = ExperimentResult(
            config={
                "max_iterations": max_iterations,
                "top_k": top_k,
                "start_qps": self.config.start_qps,
                "end_qps": self.config.end_qps
            }
        )
        
        # Extract per-stage results
        stages = data.get("stages", [])
        for stage in stages:
            qps = stage.get("target_qps", 0)
            latency = stage.get("latency", {})
            
            result.qps_values.append(qps)
            
            # Total latency
            total = latency.get("total_time_ms", {})
            result.total_latency_avg.append(total.get("avg", 0))
            result.total_latency_p99.append(total.get("p99", 0))
            
            # LLM latency
            llm = latency.get("llm_time_ms", {})
            result.llm_latency_avg.append(llm.get("avg", 0))
            result.llm_latency_p99.append(llm.get("p99", 0))
            
            # Queue time
            queue = latency.get("queue_time_ms", {})
            result.queue_time_avg.append(queue.get("avg", 0))
            result.queue_time_p99.append(queue.get("p99", 0))
            
            # Queue time ratio
            total_avg = total.get("avg", 1)
            queue_avg = queue.get("avg", 0)
            ratio = (queue_avg / total_avg * 100) if total_avg > 0 else 0
            result.queue_time_ratio.append(ratio)
            
            # Throughput
            result.throughput.append(stage.get("actual_throughput", 0))
        
        return result


class GraphGenerator:
    """
    Generates graphs from experiment results.
    """
    
    def __init__(self, output_dir: str = "./results/graphs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Style settings
        plt.style.use('seaborn-v0_8-whitegrid')
        self.colors = plt.cm.tab10.colors
        self.markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h']
    
    def generate_experiment1_graphs(self, result: ExperimentResult, prefix: str = "exp1"):
        """
        Generate graphs for Experiment 1.
        
        Graph 1: Total latency avg and p99 vs QPS (separate graphs)
        Graph 2: Queue waiting time ratio vs QPS
        """
        # Graph 1a: Total latency average
        self._plot_single_line(
            x=result.qps_values,
            y=result.total_latency_avg,
            xlabel="QPS",
            ylabel="Total Latency (ms)",
            title="Total Latency (Average) vs QPS",
            filename=f"{prefix}_total_latency_avg.png",
            color=self.colors[0]
        )
        
        # Graph 1b: Total latency p99
        self._plot_single_line(
            x=result.qps_values,
            y=result.total_latency_p99,
            xlabel="QPS",
            ylabel="Total Latency (ms)",
            title="Total Latency (P99) vs QPS",
            filename=f"{prefix}_total_latency_p99.png",
            color=self.colors[1]
        )
        
        # Graph 2: Queue waiting time ratio
        self._plot_single_line(
            x=result.qps_values,
            y=result.queue_time_ratio,
            xlabel="QPS",
            ylabel="Queue Wait Time Ratio (%)",
            title="Queue Waiting Time Ratio vs QPS",
            filename=f"{prefix}_queue_time_ratio.png",
            color=self.colors[2]
        )
        
        print(f"Experiment 1 graphs saved to {self.output_dir}")
    
    def generate_experiment2_graphs(
        self,
        results: Dict[int, ExperimentResult],
        prefix: str = "exp2"
    ):
        """
        Generate graphs for Experiment 2.
        
        Compares results across different max_iterations values.
        
        Args:
            results: Dict mapping max_iterations -> ExperimentResult
        """
        # Graph 1a: Total latency average comparison
        self._plot_multi_line(
            results=results,
            y_key="total_latency_avg",
            xlabel="QPS",
            ylabel="Total Latency (ms)",
            title="Total Latency (Average) vs QPS by Max Iterations",
            filename=f"{prefix}_total_latency_avg_comparison.png",
            legend_prefix="max_iter="
        )
        
        # Graph 1b: Total latency p99 comparison
        self._plot_multi_line(
            results=results,
            y_key="total_latency_p99",
            xlabel="QPS",
            ylabel="Total Latency (ms)",
            title="Total Latency (P99) vs QPS by Max Iterations",
            filename=f"{prefix}_total_latency_p99_comparison.png",
            legend_prefix="max_iter="
        )
        
        # Graph 2: Queue waiting time ratio comparison
        self._plot_multi_line(
            results=results,
            y_key="queue_time_ratio",
            xlabel="QPS",
            ylabel="Queue Wait Time Ratio (%)",
            title="Queue Waiting Time Ratio vs QPS by Max Iterations",
            filename=f"{prefix}_queue_time_ratio_comparison.png",
            legend_prefix="max_iter="
        )
        
        print(f"Experiment 2 graphs saved to {self.output_dir}")
    
    def _plot_single_line(
        self,
        x: List[float],
        y: List[float],
        xlabel: str,
        ylabel: str,
        title: str,
        filename: str,
        color: tuple
    ):
        """Plot a single line graph."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(x, y, marker='o', linewidth=2, markersize=8, color=color)
        
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_multi_line(
        self,
        results: Dict[int, ExperimentResult],
        y_key: str,
        xlabel: str,
        ylabel: str,
        title: str,
        filename: str,
        legend_prefix: str
    ):
        """Plot multiple lines for comparison."""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        for i, (max_iter, result) in enumerate(sorted(results.items())):
            y_values = getattr(result, y_key)
            ax.plot(
                result.qps_values,
                y_values,
                marker=self.markers[i % len(self.markers)],
                linewidth=2,
                markersize=8,
                color=self.colors[i % len(self.colors)],
                label=f"{legend_prefix}{max_iter}"
            )
        
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()
    
    def generate_summary_report(
        self,
        exp1_result: Optional[ExperimentResult],
        exp2_results: Dict[int, ExperimentResult],
        filename: str = "benchmark_report.md"
    ):
        """Generate a markdown summary report."""
        report_path = self.output_dir / filename
        
        with open(report_path, 'w') as f:
            f.write("# Benchmark Results Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Experiment 1 Summary
            if exp1_result:
                f.write("## Experiment 1: Fixed max_iterations=3, top_k=5\n\n")
                f.write("### Configuration\n")
                f.write(f"- Max Iterations: {exp1_result.config.get('max_iterations', 3)}\n")
                f.write(f"- Top-K: {exp1_result.config.get('top_k', 5)}\n")
                f.write(f"- QPS Range: {exp1_result.config.get('start_qps')} -> {exp1_result.config.get('end_qps')}\n\n")
                
                f.write("### Results Summary\n\n")
                f.write("| QPS | Total Latency Avg (ms) | Total Latency P99 (ms) | Queue Ratio (%) |\n")
                f.write("|-----|------------------------|------------------------|------------------|\n")
                for i, qps in enumerate(exp1_result.qps_values):
                    f.write(f"| {qps:.1f} | {exp1_result.total_latency_avg[i]:.2f} | ")
                    f.write(f"{exp1_result.total_latency_p99[i]:.2f} | ")
                    f.write(f"{exp1_result.queue_time_ratio[i]:.2f} |\n")
                f.write("\n")
            
            # Experiment 2 Summary
            if exp2_results:
                f.write("## Experiment 2: Varying max_iterations (1-5)\n\n")
                f.write("### Configuration\n")
                f.write("- Top-K: 5 (fixed)\n")
                f.write("- Max Iterations: 1, 2, 3, 4, 5\n\n")
                
                f.write("### Results Comparison (at highest QPS)\n\n")
                f.write("| Max Iterations | Total Latency Avg (ms) | Total Latency P99 (ms) | Queue Ratio (%) |\n")
                f.write("|----------------|------------------------|------------------------|------------------|\n")
                
                for max_iter, result in sorted(exp2_results.items()):
                    if result.qps_values:
                        idx = -1  # Last (highest) QPS
                        f.write(f"| {max_iter} | {result.total_latency_avg[idx]:.2f} | ")
                        f.write(f"{result.total_latency_p99[idx]:.2f} | ")
                        f.write(f"{result.queue_time_ratio[idx]:.2f} |\n")
                f.write("\n")
            
            # Graph References
            f.write("## Generated Graphs\n\n")
            f.write("### Experiment 1\n")
            f.write("- `exp1_total_latency_avg.png`: Total latency average vs QPS\n")
            f.write("- `exp1_total_latency_p99.png`: Total latency P99 vs QPS\n")
            f.write("- `exp1_queue_time_ratio.png`: Queue waiting time ratio vs QPS\n\n")
            
            f.write("### Experiment 2\n")
            f.write("- `exp2_total_latency_avg_comparison.png`: Total latency average comparison\n")
            f.write("- `exp2_total_latency_p99_comparison.png`: Total latency P99 comparison\n")
            f.write("- `exp2_queue_time_ratio_comparison.png`: Queue time ratio comparison\n")
        
        print(f"Summary report saved to {report_path}")


def run_experiment1(runner: BenchmarkRunner) -> Optional[ExperimentResult]:
    """Run Experiment 1: Fixed max_iterations=3, top_k=5."""
    print("\n" + "="*80)
    print("EXPERIMENT 1: QPS Sweep with fixed max_iterations=3, top_k=5")
    print("="*80)
    
    return runner.run_single_experiment(
        max_iterations=3,
        top_k=5,
        experiment_name="exp1_fixed_iter3_topk5"
    )


def run_experiment2(runner: BenchmarkRunner) -> Dict[int, ExperimentResult]:
    """Run Experiment 2: Varying max_iterations (1-5)."""
    print("\n" + "="*80)
    print("EXPERIMENT 2: QPS Sweep with varying max_iterations (1-5)")
    print("="*80)
    
    results = {}
    for max_iter in [1, 2, 3, 4, 5]:
        result = runner.run_single_experiment(
            max_iterations=max_iter,
            top_k=5,
            experiment_name=f"exp2_iter{max_iter}_topk5"
        )
        if result:
            results[max_iter] = result
        
        # Brief pause between experiments
        time.sleep(5)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmark experiments for RAG system"
    )
    
    # Experiment selection
    parser.add_argument(
        "--experiment", type=str, choices=["1", "2", "all"], default="all",
        help="Which experiment to run (1, 2, or all)"
    )
    
    # QPS configuration
    parser.add_argument(
        "--start-qps", type=float, default=1.0,
        help="Starting QPS for sweep"
    )
    parser.add_argument(
        "--end-qps", type=float, default=10.0,
        help="Ending QPS for sweep"
    )
    parser.add_argument(
        "--qps-step", type=float, default=1.0,
        help="QPS increment step"
    )
    
    # Server configuration
    parser.add_argument(
        "--endpoint", type=str, default="http://localhost:8080",
        help="RAG server endpoint"
    )
    
    # Timing configuration
    parser.add_argument(
        "--stage-duration", type=int, default=30,
        help="Duration in seconds for each QPS stage"
    )
    
    # Output configuration
    parser.add_argument(
        "--output-dir", type=str, default="./results",
        help="Directory for output files"
    )
    
    # Skip profiling (use existing results)
    parser.add_argument(
        "--graphs-only", action="store_true",
        help="Only generate graphs from existing results (skip profiling)"
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = ExperimentConfig(
        endpoint=args.endpoint,
        start_qps=args.start_qps,
        end_qps=args.end_qps,
        qps_step=args.qps_step,
        stage_duration=args.stage_duration,
        output_dir=args.output_dir
    )
    
    runner = BenchmarkRunner(config)
    graph_gen = GraphGenerator(output_dir=f"{args.output_dir}/graphs")
    
    exp1_result = None
    exp2_results = {}
    
    if not args.graphs_only:
        # Run experiments
        if args.experiment in ["1", "all"]:
            exp1_result = run_experiment1(runner)
        
        if args.experiment in ["2", "all"]:
            exp2_results = run_experiment2(runner)
    else:
        # Load existing results
        results_path = Path(args.output_dir)
        
        # Load Experiment 1 results
        exp1_file = results_path / "exp1_fixed_iter3_topk5.json"
        if exp1_file.exists():
            exp1_result = runner._parse_results(exp1_file, 3, 5)
        
        # Load Experiment 2 results
        for max_iter in [1, 2, 3, 4, 5]:
            exp2_file = results_path / f"exp2_iter{max_iter}_topk5.json"
            if exp2_file.exists():
                result = runner._parse_results(exp2_file, max_iter, 5)
                if result:
                    exp2_results[max_iter] = result
    
    # Generate graphs
    print("\n" + "="*80)
    print("GENERATING GRAPHS")
    print("="*80)
    
    if exp1_result:
        graph_gen.generate_experiment1_graphs(exp1_result)
    
    if exp2_results:
        graph_gen.generate_experiment2_graphs(exp2_results)
    
    # Generate summary report
    graph_gen.generate_summary_report(exp1_result, exp2_results)
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print(f"Results saved to: {args.output_dir}")
    print(f"Graphs saved to: {args.output_dir}/graphs")
    print("="*80)


if __name__ == "__main__":
    main()
