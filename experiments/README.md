# Benchmark Experiments

This directory contains automated benchmark scripts for evaluating RAG system performance.

## Experiments

### Experiment 1: QPS Sweep (Fixed Parameters)
- **Fixed Parameters**: `max_iterations=3`, `top_k=5`
- **Variable**: QPS (queries per second)
- **Metrics**: Total latency (avg, P99), LLM latency, Queue waiting time ratio

### Experiment 2: Max Iterations Comparison
- **Fixed Parameters**: `top_k=5`
- **Variable**: `max_iterations` (1, 2, 3, 4, 5) × QPS sweep
- **Metrics**: Same as Experiment 1, compared across different max_iterations values

## Usage

### Prerequisites
1. Start the RAG server:
   ```bash
   python run.py server --llm-endpoint http://localhost:8000 --port 8080
   ```

2. Ensure the FAISS index is built:
   ```bash
   python run.py index --corpus-split train --algorithm hnsw
   ```

### Running Benchmarks

**Run all experiments:**
```bash
cd experiments
./run_benchmark.sh --start-qps 1 --end-qps 10 --stage-duration 30
```

**Run specific experiment:**
```bash
# Experiment 1 only
./run_benchmark.sh --experiment 1 --start-qps 1 --end-qps 10

# Experiment 2 only
./run_benchmark.sh --experiment 2 --start-qps 1 --end-qps 10
```

**Generate graphs from existing results:**
```bash
./run_benchmark.sh --graphs-only --output-dir ./results
```

### Python Script Direct Usage

```bash
python benchmark.py \
    --experiment all \
    --start-qps 1 \
    --end-qps 10 \
    --qps-step 1 \
    --endpoint http://localhost:8080 \
    --stage-duration 30 \
    --output-dir ./results
```

## Output Files

### Results Directory (`./results/`)
- `exp1_fixed_iter3_topk5.json`: Experiment 1 raw results
- `exp2_iter{1-5}_topk5.json`: Experiment 2 raw results per max_iterations

### Graphs Directory (`./results/graphs/`)

**Experiment 1:**
- `exp1_total_latency_avg.png`: Total latency average vs QPS
- `exp1_total_latency_p99.png`: Total latency P99 vs QPS
- `exp1_queue_time_ratio.png`: Queue waiting time ratio vs QPS

**Experiment 2:**
- `exp2_total_latency_avg_comparison.png`: Total latency average comparison across max_iterations
- `exp2_total_latency_p99_comparison.png`: Total latency P99 comparison
- `exp2_queue_time_ratio_comparison.png`: Queue time ratio comparison

### Summary Report
- `benchmark_report.md`: Markdown summary with tables and graph references

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `--experiment` | `all` | Which experiment to run (`1`, `2`, or `all`) |
| `--start-qps` | `1.0` | Starting QPS for sweep |
| `--end-qps` | `10.0` | Ending QPS for sweep |
| `--qps-step` | `1.0` | QPS increment step |
| `--endpoint` | `http://localhost:8080` | RAG server endpoint |
| `--stage-duration` | `30` | Duration (seconds) for each QPS stage |
| `--output-dir` | `./results` | Output directory for results |
| `--graphs-only` | `false` | Only generate graphs from existing results |

## Metrics Collected

| Metric | Description |
|--------|-------------|
| `total_latency_avg` | Average total query latency (ms) |
| `total_latency_p99` | 99th percentile total latency (ms) |
| `llm_latency_avg` | Average LLM inference time (ms) |
| `llm_latency_p99` | 99th percentile LLM time (ms) |
| `queue_time_avg` | Average queue waiting time (ms) |
| `queue_time_p99` | 99th percentile queue time (ms) |
| `queue_time_ratio` | Queue time as percentage of total time |
| `throughput` | Actual achieved throughput (QPS) |
