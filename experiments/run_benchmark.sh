#!/bin/bash
#
# Benchmark Runner Script
#
# Usage:
#   ./run_benchmark.sh [OPTIONS]
#
# Options:
#   --experiment 1|2|all    Run specific experiment (default: all)
#   --start-qps N           Starting QPS (default: 1)
#   --end-qps N             Ending QPS (default: 10)
#   --endpoint URL          RAG server endpoint (default: http://localhost:8080)
#   --stage-duration N      Duration per QPS stage in seconds (default: 30)
#   --output-dir DIR        Output directory (default: ./results)
#   --graphs-only           Only generate graphs from existing results
#

set -e

# Default values
EXPERIMENT="all"
START_QPS=1
END_QPS=10
QPS_STEP=1
ENDPOINT="http://localhost:8080"
STAGE_DURATION=30
OUTPUT_DIR="./results"
GRAPHS_ONLY=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --experiment)
            EXPERIMENT="$2"
            shift 2
            ;;
        --start-qps)
            START_QPS="$2"
            shift 2
            ;;
        --end-qps)
            END_QPS="$2"
            shift 2
            ;;
        --qps-step)
            QPS_STEP="$2"
            shift 2
            ;;
        --endpoint)
            ENDPOINT="$2"
            shift 2
            ;;
        --stage-duration)
            STAGE_DURATION="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --graphs-only)
            GRAPHS_ONLY="--graphs-only"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/graphs"

echo "============================================================"
echo "RAG Benchmark Runner"
echo "============================================================"
echo "Experiment:     $EXPERIMENT"
echo "QPS Range:      $START_QPS -> $END_QPS (step: $QPS_STEP)"
echo "Endpoint:       $ENDPOINT"
echo "Stage Duration: ${STAGE_DURATION}s"
echo "Output Dir:     $OUTPUT_DIR"
echo "============================================================"
echo ""

# Run benchmark
cd "$PROJECT_ROOT"
python experiments/benchmark.py \
    --experiment "$EXPERIMENT" \
    --start-qps "$START_QPS" \
    --end-qps "$END_QPS" \
    --qps-step "$QPS_STEP" \
    --endpoint "$ENDPOINT" \
    --stage-duration "$STAGE_DURATION" \
    --output-dir "$OUTPUT_DIR" \
    $GRAPHS_ONLY

echo ""
echo "============================================================"
echo "Benchmark completed!"
echo "Results: $OUTPUT_DIR"
echo "Graphs:  $OUTPUT_DIR/graphs"
echo "============================================================"
