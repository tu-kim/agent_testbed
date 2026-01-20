#!/usr/bin/env python3
"""
Prefix Caching Speedup Comparison

Compare two experiment results (prefix caching on vs off) and generate
speedup bar charts for total latency (P99 and average) across QPS levels.

Usage:
    python compare_prefix_caching.py \
        --baseline results/exp1_fixed_iter3_topk5.json \
        --optimized results/exp1_fixed_iter3_topk5_prefixcaching.json \
        --output-dir results/graphs
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_results(filepath: str) -> Dict:
    """Load experiment results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def safe_float(value, default=0.0) -> float:
    """Convert value to float, handling None and NaN."""
    if value is None:
        return default
    try:
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (ValueError, TypeError):
        return default


def extract_qps_latency(results: Dict) -> Tuple[List[float], List[float], List[float]]:
    """
    Extract QPS levels and corresponding latency metrics from results.
    
    Supports both nested structure (from query_generator.py) and flat structure.
    
    Expected nested structure:
    {
        "stages": [
            {
                "target_qps": 1.0,
                "total_time": {"avg_ms": 100, "p99_ms": 200}
            }
        ]
    }
    
    Returns:
        Tuple of (qps_list, avg_latency_list, p99_latency_list)
    """
    qps_list = []
    avg_latency_list = []
    p99_latency_list = []
    
    # Handle different result formats
    stages = results.get("stages", results.get("per_stage_results", []))
    
    if not stages:
        raise ValueError("Unknown result format: missing 'stages' or 'per_stage_results' key")
    
    for stage in stages:
        qps = safe_float(stage.get("target_qps", stage.get("qps", 0)))
        qps_list.append(qps)
        
        # Check for nested structure first
        total_time = stage.get("total_time")
        if total_time and isinstance(total_time, dict):
            # Nested structure: {"total_time": {"avg_ms": ..., "p99_ms": ...}}
            avg_latency_list.append(safe_float(total_time.get("avg_ms", 0)))
            p99_latency_list.append(safe_float(total_time.get("p99_ms", 0)))
        else:
            # Flat structure: {"avg_total_time_ms": ..., "p99_total_time_ms": ...}
            avg_latency_list.append(safe_float(stage.get("avg_total_time_ms", stage.get("total_time_avg", 0))))
            p99_latency_list.append(safe_float(stage.get("p99_total_time_ms", stage.get("total_time_p99", 0))))
    
    print(f"  Parsed {len(qps_list)} stages, QPS: {qps_list}")
    print(f"  Sample avg latencies: {avg_latency_list[:3]}...")
    
    return qps_list, avg_latency_list, p99_latency_list


def calculate_speedup(baseline_latency: List[float], optimized_latency: List[float]) -> List[float]:
    """
    Calculate speedup as baseline_latency / optimized_latency.
    
    Speedup > 1 means optimized is faster.
    """
    speedups = []
    for base, opt in zip(baseline_latency, optimized_latency):
        if opt > 0:
            speedups.append(base / opt)
        else:
            speedups.append(1.0)
    return speedups


def plot_speedup_bar_chart(
    qps_list: List[float],
    speedups: List[float],
    metric_name: str,
    output_path: str,
    title: str
):
    """Plot speedup bar chart for a single metric."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(qps_list))
    bars = ax.bar(x, speedups, color='#3498db', edgecolor='black', linewidth=0.5)
    
    # Add horizontal line at y=1 (no speedup)
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, label='No Speedup (1.0x)')
    
    # Add value labels on bars
    for bar, speedup in zip(bars, speedups):
        height = bar.get_height()
        ax.annotate(f'{speedup:.2f}x',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('QPS (Queries Per Second)', fontsize=12)
    ax.set_ylabel(f'Speedup ({metric_name})', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{qps:.1f}' for qps in qps_list])
    ax.grid(axis='y', alpha=0.3)
    ax.legend(loc='upper right')
    
    # Set y-axis to start from 0
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_latency_comparison(
    qps_list: List[float],
    baseline_latency: List[float],
    optimized_latency: List[float],
    metric_name: str,
    output_path: str,
    title: str
):
    """Plot side-by-side latency comparison bar chart."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(qps_list))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, baseline_latency, width, label='Baseline (No Prefix Caching)', 
                   color='#e74c3c', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, optimized_latency, width, label='Optimized (Prefix Caching)', 
                   color='#2ecc71', edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('QPS (Queries Per Second)', fontsize=12)
    ax.set_ylabel(f'{metric_name} Latency (ms)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{qps:.1f}' for qps in qps_list])
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def print_comparison_table(
    qps_list: List[float],
    baseline_avg: List[float],
    optimized_avg: List[float],
    avg_speedup: List[float],
    baseline_p99: List[float],
    optimized_p99: List[float],
    p99_speedup: List[float]
):
    """Print comparison table to console."""
    print("\n" + "=" * 100)
    print("PREFIX CACHING SPEEDUP COMPARISON")
    print("=" * 100)
    
    header = f"{'QPS':>6} | {'Baseline Avg':>12} | {'Optimized Avg':>13} | {'Avg Speedup':>11} | {'Baseline P99':>12} | {'Optimized P99':>13} | {'P99 Speedup':>11}"
    print(header)
    print("-" * 100)
    
    for i in range(len(qps_list)):
        row = f"{qps_list[i]:>6.1f} | {baseline_avg[i]:>10.2f}ms | {optimized_avg[i]:>11.2f}ms | {avg_speedup[i]:>10.2f}x | {baseline_p99[i]:>10.2f}ms | {optimized_p99[i]:>11.2f}ms | {p99_speedup[i]:>10.2f}x"
        print(row)
    
    print("-" * 100)
    
    # Print mean speedup
    mean_avg_speedup = np.mean(avg_speedup)
    mean_p99_speedup = np.mean(p99_speedup)
    mean_baseline_avg = np.mean(baseline_avg)
    mean_optimized_avg = np.mean(optimized_avg)
    mean_baseline_p99 = np.mean(baseline_p99)
    mean_optimized_p99 = np.mean(optimized_p99)
    
    mean_row = f"{'Mean':>6} | {mean_baseline_avg:>10.2f}ms | {mean_optimized_avg:>11.2f}ms | {mean_avg_speedup:>10.2f}x | {mean_baseline_p99:>10.2f}ms | {mean_optimized_p99:>11.2f}ms | {mean_p99_speedup:>10.2f}x"
    print(mean_row)
    print("=" * 100)


def save_comparison_json(
    qps_list: List[float],
    baseline_avg: List[float],
    optimized_avg: List[float],
    avg_speedup: List[float],
    baseline_p99: List[float],
    optimized_p99: List[float],
    p99_speedup: List[float],
    output_path: str
):
    """Save comparison results to JSON file."""
    comparison = {
        "qps_levels": qps_list,
        "average_latency": {
            "baseline_ms": baseline_avg,
            "optimized_ms": optimized_avg,
            "speedup": avg_speedup,
            "mean_speedup": float(np.mean(avg_speedup))
        },
        "p99_latency": {
            "baseline_ms": baseline_p99,
            "optimized_ms": optimized_p99,
            "speedup": p99_speedup,
            "mean_speedup": float(np.mean(p99_speedup))
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare prefix caching speedup')
    parser.add_argument('--baseline', '-b', type=str, required=True,
                        help='Path to baseline results JSON (prefix caching off)')
    parser.add_argument('--optimized', '-o', type=str, required=True,
                        help='Path to optimized results JSON (prefix caching on)')
    parser.add_argument('--output-dir', '-d', type=str, default='./graphs',
                        help='Output directory for graphs')
    parser.add_argument('--combined', action='store_true',
                        help='Also generate combined comparison charts')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    print(f"\nLoading baseline: {args.baseline}")
    baseline_results = load_results(args.baseline)
    baseline_qps, baseline_avg, baseline_p99 = extract_qps_latency(baseline_results)
    
    print(f"\nLoading optimized: {args.optimized}")
    optimized_results = load_results(args.optimized)
    optimized_qps, optimized_avg, optimized_p99 = extract_qps_latency(optimized_results)
    
    # Verify QPS levels match
    if baseline_qps != optimized_qps:
        print("Warning: QPS levels don't match between baseline and optimized results")
        print(f"  Baseline QPS: {baseline_qps}")
        print(f"  Optimized QPS: {optimized_qps}")
    
    qps_list = baseline_qps
    
    # Calculate speedups
    avg_speedup = calculate_speedup(baseline_avg, optimized_avg)
    p99_speedup = calculate_speedup(baseline_p99, optimized_p99)
    
    # Print comparison table
    print_comparison_table(
        qps_list, baseline_avg, optimized_avg, avg_speedup,
        baseline_p99, optimized_p99, p99_speedup
    )
    
    # Generate speedup bar charts
    plot_speedup_bar_chart(
        qps_list, avg_speedup, 'Average',
        os.path.join(args.output_dir, 'prefix_caching_speedup_avg.png'),
        'Prefix Caching Speedup: Average Latency'
    )
    
    plot_speedup_bar_chart(
        qps_list, p99_speedup, 'P99',
        os.path.join(args.output_dir, 'prefix_caching_speedup_p99.png'),
        'Prefix Caching Speedup: P99 Latency'
    )
    
    # Generate comparison charts if requested
    if args.combined:
        plot_latency_comparison(
            qps_list, baseline_avg, optimized_avg, 'Average',
            os.path.join(args.output_dir, 'prefix_caching_latency_avg_comparison.png'),
            'Latency Comparison: Average (Baseline vs Prefix Caching)'
        )
        
        plot_latency_comparison(
            qps_list, baseline_p99, optimized_p99, 'P99',
            os.path.join(args.output_dir, 'prefix_caching_latency_p99_comparison.png'),
            'Latency Comparison: P99 (Baseline vs Prefix Caching)'
        )
    
    # Save comparison JSON
    save_comparison_json(
        qps_list, baseline_avg, optimized_avg, avg_speedup,
        baseline_p99, optimized_p99, p99_speedup,
        os.path.join(args.output_dir, 'prefix_caching_comparison_summary.json')
    )
    
    print(f"\nAll outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
