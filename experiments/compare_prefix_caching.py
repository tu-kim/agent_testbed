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
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_results(filepath: str) -> Dict:
    """Load experiment results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def extract_qps_latency(results: Dict) -> Tuple[List[float], List[float], List[float]]:
    """
    Extract QPS levels and corresponding latency metrics from results.
    
    Returns:
        Tuple of (qps_list, avg_latency_list, p99_latency_list)
    """
    qps_list = []
    avg_latency_list = []
    p99_latency_list = []
    
    # Handle different result formats
    if "stages" in results:
        # Per-stage results format
        for stage in results["stages"]:
            qps_list.append(stage.get("target_qps", stage.get("qps", 0)))
            avg_latency_list.append(stage.get("avg_total_time_ms", stage.get("total_time_avg", 0)))
            p99_latency_list.append(stage.get("p99_total_time_ms", stage.get("total_time_p99", 0)))
    elif "per_stage_results" in results:
        # Alternative format
        for stage in results["per_stage_results"]:
            qps_list.append(stage.get("target_qps", stage.get("qps", 0)))
            avg_latency_list.append(stage.get("avg_total_time_ms", stage.get("total_time_avg", 0)))
            p99_latency_list.append(stage.get("p99_total_time_ms", stage.get("total_time_p99", 0)))
    else:
        raise ValueError("Unknown result format: missing 'stages' or 'per_stage_results' key")
    
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
    avg_speedup: List[float],
    p99_speedup: List[float],
    output_path: str,
    title: str = "Prefix Caching Speedup by QPS"
):
    """
    Generate a grouped bar chart comparing average and P99 speedup across QPS levels.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(qps_list))
    width = 0.35
    
    # Create bars
    bars_avg = ax.bar(x - width/2, avg_speedup, width, label='Average Latency Speedup', 
                      color='#2ecc71', edgecolor='black', linewidth=0.5)
    bars_p99 = ax.bar(x + width/2, p99_speedup, width, label='P99 Latency Speedup', 
                      color='#3498db', edgecolor='black', linewidth=0.5)
    
    # Add reference line at speedup = 1.0
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, label='No Speedup (1.0x)')
    
    # Labels and formatting
    ax.set_xlabel('QPS (Queries Per Second)', fontsize=12)
    ax.set_ylabel('Speedup (Baseline / Optimized)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{qps:.1f}' for qps in qps_list])
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    def add_bar_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}x',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    add_bar_labels(bars_avg)
    add_bar_labels(bars_p99)
    
    # Set y-axis to start from 0
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_separate_speedup_charts(
    qps_list: List[float],
    avg_speedup: List[float],
    p99_speedup: List[float],
    output_dir: str
):
    """
    Generate separate bar charts for average and P99 speedup.
    """
    # Average Latency Speedup Chart
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(qps_list)), avg_speedup, color='#2ecc71', edgecolor='black', linewidth=0.5)
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, label='No Speedup (1.0x)')
    ax.set_xlabel('QPS (Queries Per Second)', fontsize=12)
    ax.set_ylabel('Speedup', fontsize=12)
    ax.set_title('Prefix Caching Speedup - Average Latency', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(qps_list)))
    ax.set_xticklabels([f'{qps:.1f}' for qps in qps_list])
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(bottom=0)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}x',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    avg_path = os.path.join(output_dir, 'prefix_caching_speedup_avg.png')
    plt.savefig(avg_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {avg_path}")
    
    # P99 Latency Speedup Chart
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(qps_list)), p99_speedup, color='#3498db', edgecolor='black', linewidth=0.5)
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, label='No Speedup (1.0x)')
    ax.set_xlabel('QPS (Queries Per Second)', fontsize=12)
    ax.set_ylabel('Speedup', fontsize=12)
    ax.set_title('Prefix Caching Speedup - P99 Latency', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(qps_list)))
    ax.set_xticklabels([f'{qps:.1f}' for qps in qps_list])
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(bottom=0)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}x',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    p99_path = os.path.join(output_dir, 'prefix_caching_speedup_p99.png')
    plt.savefig(p99_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {p99_path}")


def plot_latency_comparison(
    qps_list: List[float],
    baseline_avg: List[float],
    baseline_p99: List[float],
    optimized_avg: List[float],
    optimized_p99: List[float],
    output_dir: str
):
    """
    Generate latency comparison charts showing actual values (not speedup).
    """
    # Average Latency Comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(qps_list))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, baseline_avg, width, label='Baseline (No Prefix Caching)', 
                   color='#e74c3c', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, optimized_avg, width, label='Optimized (Prefix Caching)', 
                   color='#2ecc71', edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('QPS (Queries Per Second)', fontsize=12)
    ax.set_ylabel('Average Latency (ms)', fontsize=12)
    ax.set_title('Average Latency Comparison: Prefix Caching On vs Off', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{qps:.1f}' for qps in qps_list])
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    avg_path = os.path.join(output_dir, 'prefix_caching_latency_avg_comparison.png')
    plt.savefig(avg_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {avg_path}")
    
    # P99 Latency Comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    
    bars1 = ax.bar(x - width/2, baseline_p99, width, label='Baseline (No Prefix Caching)', 
                   color='#e74c3c', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, optimized_p99, width, label='Optimized (Prefix Caching)', 
                   color='#2ecc71', edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('QPS (Queries Per Second)', fontsize=12)
    ax.set_ylabel('P99 Latency (ms)', fontsize=12)
    ax.set_title('P99 Latency Comparison: Prefix Caching On vs Off', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{qps:.1f}' for qps in qps_list])
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    p99_path = os.path.join(output_dir, 'prefix_caching_latency_p99_comparison.png')
    plt.savefig(p99_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {p99_path}")


def print_summary_table(
    qps_list: List[float],
    baseline_avg: List[float],
    baseline_p99: List[float],
    optimized_avg: List[float],
    optimized_p99: List[float],
    avg_speedup: List[float],
    p99_speedup: List[float]
):
    """Print a summary table of the comparison results."""
    print("\n" + "=" * 100)
    print("PREFIX CACHING SPEEDUP COMPARISON")
    print("=" * 100)
    print(f"{'QPS':>6} | {'Baseline Avg':>12} | {'Optimized Avg':>13} | {'Avg Speedup':>11} | "
          f"{'Baseline P99':>12} | {'Optimized P99':>13} | {'P99 Speedup':>11}")
    print("-" * 100)
    
    for i, qps in enumerate(qps_list):
        print(f"{qps:>6.1f} | {baseline_avg[i]:>10.2f}ms | {optimized_avg[i]:>11.2f}ms | "
              f"{avg_speedup[i]:>10.2f}x | {baseline_p99[i]:>10.2f}ms | {optimized_p99[i]:>11.2f}ms | "
              f"{p99_speedup[i]:>10.2f}x")
    
    print("-" * 100)
    print(f"{'Mean':>6} | {np.mean(baseline_avg):>10.2f}ms | {np.mean(optimized_avg):>11.2f}ms | "
          f"{np.mean(avg_speedup):>10.2f}x | {np.mean(baseline_p99):>10.2f}ms | {np.mean(optimized_p99):>11.2f}ms | "
          f"{np.mean(p99_speedup):>10.2f}x")
    print("=" * 100 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Compare prefix caching on/off results and generate speedup charts'
    )
    parser.add_argument(
        '--baseline', '-b',
        type=str,
        default='results/exp1_fixed_iter3_topk5.json',
        help='Path to baseline results (prefix caching off)'
    )
    parser.add_argument(
        '--optimized', '-o',
        type=str,
        default='results/exp1_fixed_iter3_topk5_prefixcaching.json',
        help='Path to optimized results (prefix caching on)'
    )
    parser.add_argument(
        '--output-dir', '-d',
        type=str,
        default='results/graphs',
        help='Output directory for graphs'
    )
    parser.add_argument(
        '--combined',
        action='store_true',
        help='Generate combined chart (avg and p99 in one graph)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    print(f"Loading baseline results: {args.baseline}")
    baseline_results = load_results(args.baseline)
    
    print(f"Loading optimized results: {args.optimized}")
    optimized_results = load_results(args.optimized)
    
    # Extract QPS and latency data
    baseline_qps, baseline_avg, baseline_p99 = extract_qps_latency(baseline_results)
    optimized_qps, optimized_avg, optimized_p99 = extract_qps_latency(optimized_results)
    
    # Verify QPS levels match
    if baseline_qps != optimized_qps:
        print("Warning: QPS levels do not match between baseline and optimized results")
        print(f"  Baseline QPS: {baseline_qps}")
        print(f"  Optimized QPS: {optimized_qps}")
        # Use intersection of QPS levels
        common_qps = sorted(set(baseline_qps) & set(optimized_qps))
        print(f"  Using common QPS levels: {common_qps}")
        
        # Filter to common QPS levels
        baseline_indices = [baseline_qps.index(q) for q in common_qps]
        optimized_indices = [optimized_qps.index(q) for q in common_qps]
        
        baseline_avg = [baseline_avg[i] for i in baseline_indices]
        baseline_p99 = [baseline_p99[i] for i in baseline_indices]
        optimized_avg = [optimized_avg[i] for i in optimized_indices]
        optimized_p99 = [optimized_p99[i] for i in optimized_indices]
        qps_list = common_qps
    else:
        qps_list = baseline_qps
    
    # Calculate speedup
    avg_speedup = calculate_speedup(baseline_avg, optimized_avg)
    p99_speedup = calculate_speedup(baseline_p99, optimized_p99)
    
    # Print summary table
    print_summary_table(
        qps_list, baseline_avg, baseline_p99,
        optimized_avg, optimized_p99,
        avg_speedup, p99_speedup
    )
    
    # Generate graphs
    if args.combined:
        # Combined chart
        combined_path = os.path.join(args.output_dir, 'prefix_caching_speedup_combined.png')
        plot_speedup_bar_chart(qps_list, avg_speedup, p99_speedup, combined_path)
    
    # Separate charts
    plot_separate_speedup_charts(qps_list, avg_speedup, p99_speedup, args.output_dir)
    
    # Latency comparison charts
    plot_latency_comparison(
        qps_list, baseline_avg, baseline_p99,
        optimized_avg, optimized_p99, args.output_dir
    )
    
    print(f"\nAll graphs saved to: {args.output_dir}")
    
    # Save summary to JSON
    summary = {
        "qps_levels": qps_list,
        "baseline": {
            "avg_latency_ms": baseline_avg,
            "p99_latency_ms": baseline_p99
        },
        "optimized": {
            "avg_latency_ms": optimized_avg,
            "p99_latency_ms": optimized_p99
        },
        "speedup": {
            "avg": avg_speedup,
            "p99": p99_speedup,
            "mean_avg": float(np.mean(avg_speedup)),
            "mean_p99": float(np.mean(p99_speedup))
        }
    }
    
    summary_path = os.path.join(args.output_dir, 'prefix_caching_comparison_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
