#!/usr/bin/env python3
"""
Plot Experiment Results

Generate graphs from experiment result JSON files.

Experiment 1: Fixed max_iterations=3, top_k=5, varying QPS
Experiment 2: Varying max_iterations (1-5), fixed top_k=5, varying QPS

Input files:
- Experiment 1: ../results/exp_fixed_iter3_topk5.json
- Experiment 2: ../results/exp_iter{1,2,3,4,5}_topk5.json

Output:
- Graphs saved to ../results/graphs/

Usage:
    python plot_results.py
    python plot_results.py --results-dir ../results --output-dir ../results/graphs
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Color palette for different iterations
COLORS = {
    1: '#e74c3c',  # Red
    2: '#f39c12',  # Orange
    3: '#2ecc71',  # Green
    4: '#3498db',  # Blue
    5: '#9b59b6',  # Purple
}

LINE_STYLES = {
    1: '-',
    2: '--',
    3: '-.',
    4: ':',
    5: '-',
}

MARKERS = {
    1: 'o',
    2: 's',
    3: '^',
    4: 'D',
    5: 'v',
}


def load_json(filepath: str) -> Optional[Dict]:
    """Load JSON file, return None if not found."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: File not found: {filepath}")
        return None
    except json.JSONDecodeError as e:
        print(f"Warning: Invalid JSON in {filepath}: {e}")
        return None


def extract_stage_data(results: Dict) -> Tuple[List[float], Dict[str, List[float]]]:
    """
    Extract QPS levels and metrics from result JSON.
    
    Returns:
        Tuple of (qps_list, metrics_dict)
        metrics_dict contains: total_avg, total_p99, llm_avg, llm_p99, queue_avg, queue_p99
    """
    qps_list = []
    metrics = {
        'total_avg': [],
        'total_p99': [],
        'llm_avg': [],
        'llm_p99': [],
        'retrieval_avg': [],
        'retrieval_p99': [],
        'queue_avg': [],
        'queue_p99': [],
    }
    
    # Handle different result formats
    stages = results.get('stages', results.get('per_stage_results', []))
    
    for stage in stages:
        # QPS
        qps = stage.get('target_qps', stage.get('qps', 0))
        qps_list.append(qps)
        
        # Total latency
        metrics['total_avg'].append(stage.get('avg_total_time_ms', stage.get('total_time_avg', 0)))
        metrics['total_p99'].append(stage.get('p99_total_time_ms', stage.get('total_time_p99', 0)))
        
        # LLM latency
        metrics['llm_avg'].append(stage.get('avg_llm_time_ms', stage.get('llm_time_avg', 0)))
        metrics['llm_p99'].append(stage.get('p99_llm_time_ms', stage.get('llm_time_p99', 0)))
        
        # Retrieval latency
        metrics['retrieval_avg'].append(stage.get('avg_retrieval_time_ms', stage.get('retrieval_time_avg', 0)))
        metrics['retrieval_p99'].append(stage.get('p99_retrieval_time_ms', stage.get('retrieval_time_p99', 0)))
        
        # Queue time
        metrics['queue_avg'].append(stage.get('avg_queue_time_ms', stage.get('queue_time_avg', 0)))
        metrics['queue_p99'].append(stage.get('p99_queue_time_ms', stage.get('queue_time_p99', 0)))
    
    return qps_list, metrics


def plot_exp1_total_latency_avg(qps_list: List[float], metrics: Dict, output_dir: str):
    """Experiment 1: Total latency average vs QPS."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(qps_list, metrics['total_avg'], 'o-', color='#2ecc71', linewidth=2, markersize=8, label='Total Latency (Avg)')
    
    ax.set_xlabel('QPS (Queries Per Second)', fontsize=12)
    ax.set_ylabel('Average Latency (ms)', fontsize=12)
    ax.set_title('Experiment 1: Total Latency (Average) vs QPS\n(max_iterations=3, top_k=5)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    
    # Add value labels
    for i, (x, y) in enumerate(zip(qps_list, metrics['total_avg'])):
        ax.annotate(f'{y:.1f}', (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'exp1_total_latency_avg.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_exp1_total_latency_p99(qps_list: List[float], metrics: Dict, output_dir: str):
    """Experiment 1: Total latency P99 vs QPS."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(qps_list, metrics['total_p99'], 's-', color='#e74c3c', linewidth=2, markersize=8, label='Total Latency (P99)')
    
    ax.set_xlabel('QPS (Queries Per Second)', fontsize=12)
    ax.set_ylabel('P99 Latency (ms)', fontsize=12)
    ax.set_title('Experiment 1: Total Latency (P99) vs QPS\n(max_iterations=3, top_k=5)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    
    # Add value labels
    for i, (x, y) in enumerate(zip(qps_list, metrics['total_p99'])):
        ax.annotate(f'{y:.1f}', (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'exp1_total_latency_p99.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_exp1_queue_time_ratio(qps_list: List[float], metrics: Dict, output_dir: str):
    """Experiment 1: Queue waiting time ratio vs QPS."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate queue time ratio (queue_time / total_time * 100)
    queue_ratio = []
    for q_avg, t_avg in zip(metrics['queue_avg'], metrics['total_avg']):
        if t_avg > 0:
            queue_ratio.append((q_avg / t_avg) * 100)
        else:
            queue_ratio.append(0)
    
    bars = ax.bar(range(len(qps_list)), queue_ratio, color='#3498db', edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('QPS (Queries Per Second)', fontsize=12)
    ax.set_ylabel('Queue Time Ratio (%)', fontsize=12)
    ax.set_title('Experiment 1: Queue Waiting Time Ratio vs QPS\n(max_iterations=3, top_k=5)', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(qps_list)))
    ax.set_xticklabels([f'{qps:.1f}' for qps in qps_list])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, ratio in zip(bars, queue_ratio):
        height = bar.get_height()
        ax.annotate(f'{ratio:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'exp1_queue_time_ratio.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_exp2_total_latency_avg_comparison(all_data: Dict[int, Tuple], output_dir: str):
    """Experiment 2: Total latency average comparison across max_iterations."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for max_iter, (qps_list, metrics) in sorted(all_data.items()):
        ax.plot(qps_list, metrics['total_avg'], 
                marker=MARKERS[max_iter], linestyle=LINE_STYLES[max_iter], 
                color=COLORS[max_iter], linewidth=2, markersize=8,
                label=f'max_iter={max_iter}')
    
    ax.set_xlabel('QPS (Queries Per Second)', fontsize=12)
    ax.set_ylabel('Average Latency (ms)', fontsize=12)
    ax.set_title('Experiment 2: Total Latency (Average) vs QPS\nComparison by max_iterations (top_k=5)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', title='Max Iterations')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'exp2_total_latency_avg_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_exp2_total_latency_p99_comparison(all_data: Dict[int, Tuple], output_dir: str):
    """Experiment 2: Total latency P99 comparison across max_iterations."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for max_iter, (qps_list, metrics) in sorted(all_data.items()):
        ax.plot(qps_list, metrics['total_p99'], 
                marker=MARKERS[max_iter], linestyle=LINE_STYLES[max_iter], 
                color=COLORS[max_iter], linewidth=2, markersize=8,
                label=f'max_iter={max_iter}')
    
    ax.set_xlabel('QPS (Queries Per Second)', fontsize=12)
    ax.set_ylabel('P99 Latency (ms)', fontsize=12)
    ax.set_title('Experiment 2: Total Latency (P99) vs QPS\nComparison by max_iterations (top_k=5)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', title='Max Iterations')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'exp2_total_latency_p99_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_exp2_queue_time_ratio_comparison(all_data: Dict[int, Tuple], output_dir: str):
    """Experiment 2: Queue time ratio comparison across max_iterations."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Get common QPS list (assume all have same QPS levels)
    first_iter = min(all_data.keys())
    qps_list = all_data[first_iter][0]
    
    x = np.arange(len(qps_list))
    width = 0.15  # Width of each bar
    num_iters = len(all_data)
    
    for i, (max_iter, (_, metrics)) in enumerate(sorted(all_data.items())):
        # Calculate queue time ratio
        queue_ratio = []
        for q_avg, t_avg in zip(metrics['queue_avg'], metrics['total_avg']):
            if t_avg > 0:
                queue_ratio.append((q_avg / t_avg) * 100)
            else:
                queue_ratio.append(0)
        
        offset = (i - num_iters / 2 + 0.5) * width
        bars = ax.bar(x + offset, queue_ratio, width, 
                      color=COLORS[max_iter], edgecolor='black', linewidth=0.5,
                      label=f'max_iter={max_iter}')
    
    ax.set_xlabel('QPS (Queries Per Second)', fontsize=12)
    ax.set_ylabel('Queue Time Ratio (%)', fontsize=12)
    ax.set_title('Experiment 2: Queue Waiting Time Ratio vs QPS\nComparison by max_iterations (top_k=5)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{qps:.1f}' for qps in qps_list])
    ax.grid(axis='y', alpha=0.3)
    ax.legend(loc='upper left', title='Max Iterations')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'exp2_queue_time_ratio_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_exp2_latency_breakdown(all_data: Dict[int, Tuple], output_dir: str):
    """Experiment 2: Latency breakdown (LLM, Retrieval, Queue) by max_iterations."""
    fig, axes = plt.subplots(1, len(all_data), figsize=(4 * len(all_data), 6), sharey=True)
    
    if len(all_data) == 1:
        axes = [axes]
    
    for ax, (max_iter, (qps_list, metrics)) in zip(axes, sorted(all_data.items())):
        # Use last QPS stage for breakdown
        idx = -1  # Last stage
        
        llm = metrics['llm_avg'][idx]
        retrieval = metrics['retrieval_avg'][idx]
        queue = metrics['queue_avg'][idx]
        other = max(0, metrics['total_avg'][idx] - llm - retrieval - queue)
        
        sizes = [llm, retrieval, queue, other]
        labels = ['LLM', 'Retrieval', 'Queue', 'Other']
        colors_pie = ['#3498db', '#2ecc71', '#e74c3c', '#95a5a6']
        
        ax.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
        ax.set_title(f'max_iter={max_iter}\n(QPS={qps_list[idx]:.1f})', fontsize=11)
    
    fig.suptitle('Experiment 2: Latency Breakdown by max_iterations\n(at highest QPS)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'exp2_latency_breakdown.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate graphs from experiment results')
    parser.add_argument('--results-dir', '-r', type=str, default='../results',
                        help='Directory containing result JSON files')
    parser.add_argument('--output-dir', '-o', type=str, default='../results/graphs',
                        help='Output directory for graphs')
    
    args = parser.parse_args()
    
    # Resolve paths relative to script location
    script_dir = Path(__file__).parent
    results_dir = (script_dir / args.results_dir).resolve()
    output_dir = (script_dir / args.output_dir).resolve()
    
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # ========== Experiment 1 ==========
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Fixed max_iterations=3, top_k=5")
    print("=" * 60)
    
    exp1_file = results_dir / 'exp_fixed_iter3_topk5.json'
    exp1_data = load_json(str(exp1_file))
    
    if exp1_data:
        qps_list, metrics = extract_stage_data(exp1_data)
        print(f"Loaded {len(qps_list)} QPS stages: {qps_list}")
        
        plot_exp1_total_latency_avg(qps_list, metrics, str(output_dir))
        plot_exp1_total_latency_p99(qps_list, metrics, str(output_dir))
        plot_exp1_queue_time_ratio(qps_list, metrics, str(output_dir))
    else:
        print(f"Skipping Experiment 1 graphs (file not found: {exp1_file})")
    
    # ========== Experiment 2 ==========
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Varying max_iterations (1-5), top_k=5")
    print("=" * 60)
    
    exp2_data = {}
    for max_iter in [1, 2, 3, 4, 5]:
        exp2_file = results_dir / f'exp_iter{max_iter}_topk5.json'
        data = load_json(str(exp2_file))
        
        if data:
            qps_list, metrics = extract_stage_data(data)
            exp2_data[max_iter] = (qps_list, metrics)
            print(f"Loaded max_iter={max_iter}: {len(qps_list)} QPS stages")
        else:
            print(f"Skipping max_iter={max_iter} (file not found: {exp2_file})")
    
    if exp2_data:
        plot_exp2_total_latency_avg_comparison(exp2_data, str(output_dir))
        plot_exp2_total_latency_p99_comparison(exp2_data, str(output_dir))
        plot_exp2_queue_time_ratio_comparison(exp2_data, str(output_dir))
        plot_exp2_latency_breakdown(exp2_data, str(output_dir))
    else:
        print("Skipping Experiment 2 graphs (no data files found)")
    
    print("\n" + "=" * 60)
    print(f"All graphs saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
