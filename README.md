# Multi-hop RAG with IRCoT

A distributed multi-hop Retrieval-Augmented Generation (RAG) system using the IRCoT (Interleaving Retrieval with Chain-of-Thought) pipeline. Built with LangChain, FAISS, and external vLLM PD (Prefill-Decode) servers for high-performance question answering.

## Features

- **IRCoT RAG Pipeline**: ReAct-style interleaving retrieval with chain-of-thought reasoning for multi-hop QA
- **External vLLM PD Server**: Connects to distributed vLLM servers with multiple Prefill/Decode workers
- **FAISS Vector Database**: Support for HNSW and IVF algorithms with GPU-accelerated embeddings
- **Dense Retrieval**: Using sentence-transformers/all-mpnet-base-v2 (768-dim, 109M params)
- **HotpotQA Dataset**: Corpus indexing and evaluation queries from HotpotQA distractor set
- **Query Generator**: Poisson distribution-based load testing with QPS sweep
- **Performance Profiling**: Detailed latency breakdown (Total/Retrieval/LLM/Queue time) with P50/P90/P99 metrics
- **Benchmark Automation**: Automated experiments for QPS and max_iterations analysis

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RAG Server (:8080)                                │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────────────────┐  │
│  │   IRCoT Pipeline │  │  FAISS Retriever │  │  LangChain LLM Client     │  │
│  │   (ReAct-style)  │──│  (all-mpnet-v2)  │──│  (OpenAI-compatible API)  │  │
│  └──────────────────┘  └──────────────────┘  └─────────────┬─────────────┘  │
└────────────────────────────────────────────────────────────┼────────────────┘
                                                             │
                                                             ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                        vLLM PD Proxy Server (:8000)                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Load Balancer / Request Router                   │   │
│  └───────────┬─────────────────┬─────────────────┬─────────────────────┘   │
└──────────────┼─────────────────┼─────────────────┼─────────────────────────┘
               │                 │                 │
       ┌───────┴───────┐ ┌───────┴───────┐ ┌───────┴───────┐
       │               │ │               │ │               │
       ▼               ▼ ▼               ▼ ▼               ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│  Prefill    │ │  Prefill    │ │  Prefill    │ │   Decode    │ │   Decode    │
│  Worker 1   │ │  Worker 2   │ │  Worker N   │ │  Worker 1   │ │  Worker M   │
│  (:8001)    │ │  (:8002)    │ │  (:800N)    │ │  (:8101)    │ │  (:810M)    │
│ ┌─────────┐ │ │ ┌─────────┐ │ │ ┌─────────┐ │ │ ┌─────────┐ │ │ ┌─────────┐ │
│ │  vLLM   │ │ │ │  vLLM   │ │ │ │  vLLM   │ │ │ │  vLLM   │ │ │ │  vLLM   │ │
│ │ Engine  │ │ │ │ Engine  │ │ │ │ Engine  │ │ │ │ Engine  │ │ │ │ Engine  │ │
│ └─────────┘ │ │ └─────────┘ │ │ └─────────┘ │ │ └─────────┘ │ │ └─────────┘ │
└──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
       │               │               │               │               │
       └───────────────┴───────────────┴───────────────┴───────────────┘
                                       │
                         ┌─────────────▼─────────────┐
                         │       GPU Cluster         │
                         │  (Multi-GPU / Multi-Node) │
                         └───────────────────────────┘
```

## Installation

```bash
# Clone the repository
git clone https://github.com/tu-kim/agent_testbed.git
cd agent_testbed

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Start External vLLM PD Server

This system assumes an external vLLM server with Prefill-Decode disaggregation is already running. The RAG server connects to it via OpenAI-compatible API.

```bash
# Example: Start vLLM server externally (not part of this codebase)
# vllm serve <model> --host 0.0.0.0 --port 8000
```

### 2. Index HotpotQA Corpus

Build the FAISS index from HotpotQA context documents:

```bash
# Index with HNSW algorithm (train split)
python run.py index --corpus-split train --algorithm hnsw

# Index with IVF algorithm
python run.py index --corpus-split train --algorithm ivf

# Index from validation split
python run.py index --corpus-split validation --algorithm hnsw

# Force rebuild existing index
python run.py index --corpus-split train --force

# Specify custom cache directory
python run.py index --corpus-split train --cache-dir /data/faiss_index
```

### 3. Start RAG Server

```bash
# Start RAG server (connects to external vLLM at localhost:8000)
python run.py server --llm-endpoint http://localhost:8000 --port 8080

# With custom model name
python run.py server --llm-endpoint http://localhost:8000 --model-name llama-3-8b
```

### 4. Run Queries

```bash
# Single query
python run.py query "Who directed Titanic and what year was it released?"

# With verbose output (shows reasoning steps and retrieved documents)
python run.py query "What is the capital of the country where the Eiffel Tower is located?" -v

# Save result to JSON file
python run.py query "Who wrote Romeo and Juliet?" -o result.json
```

### 5. Performance Profiling with Query Generator

Profile using HotpotQA questions with Poisson-distributed query arrivals:

```bash
# Basic profiling (QPS 1 → 10, 30s per stage)
python run.py profile \
    --endpoint http://localhost:8080 \
    --start-qps 1 \
    --end-qps 10 \
    --stage-duration 30

# With custom max_iterations and top_k
python run.py profile \
    --endpoint http://localhost:8080 \
    --start-qps 1 \
    --end-qps 20 \
    --max-iterations 5 \
    --top-k 10 \
    --output results.json

# Use train split for evaluation queries
python run.py profile \
    --eval-split train \
    --start-qps 1 \
    --end-qps 5
```

### 6. Run Benchmark Experiments

Automated experiments for latency analysis:

```bash
cd experiments

# Run all experiments (Experiment 1 + Experiment 2)
./run_benchmark.sh --start-qps 1 --end-qps 10 --stage-duration 30

# Run Experiment 1 only (fixed max_iterations=3, top_k=5)
./run_benchmark.sh --experiment 1 --start-qps 1 --end-qps 10

# Run Experiment 2 only (varying max_iterations 1-5)
./run_benchmark.sh --experiment 2 --start-qps 1 --end-qps 10

# Generate graphs from existing results
./run_benchmark.sh --graphs-only --output-dir ./results
```

## CLI Reference

### Commands

| Command | Description |
|---------|-------------|
| `server` | Run the RAG server |
| `index` | Index HotpotQA corpus into FAISS |
| `profile` | Run performance profiling with Query Generator |
| `query` | Run a single query with detailed output |

### Index Command Options

```bash
python run.py index --help

Options:
  --corpus-split       Dataset split for corpus: train, validation (default: train)
  --algorithm          FAISS algorithm: hnsw, ivf, flat (default: hnsw)
  --cache-dir PATH     Directory to store FAISS index (default: ./cache/faiss_index)
  --device             Embedding device: auto, cuda, cpu (default: auto)
  --batch-size N       Embedding batch size (default: 256)
  --force              Force rebuild index even if exists
```

### Server Command Options

```bash
python run.py server --help

Options:
  --host HOST          Server host (default: 0.0.0.0)
  --port PORT          Server port (default: 8080)
  --llm-endpoint URL   External vLLM server endpoint (default: http://localhost:8000)
  --algorithm          FAISS algorithm (default: hnsw)
  --cache-dir PATH     FAISS index directory (default: ./cache/faiss_index)
  --model-name NAME    Model name for LLM calls
  --reload             Enable auto-reload for development
```

### Profile Command Options

```bash
python run.py profile --help

Options:
  --endpoint URL       RAG server endpoint (default: http://localhost:8080)
  --eval-split         HotpotQA split for queries: train, validation (default: validation)
  --start-qps N        Starting QPS (default: 1.0)
  --end-qps N          Ending QPS (default: 10.0)
  --qps-step N         QPS increment step (default: 1.0)
  --stage-duration N   Duration per QPS stage in seconds (default: 30)
  --max-iterations N   Maximum IRCoT iterations (default: 3)
  --top-k N            Number of documents to retrieve (default: 5)
  --output FILE        Output file for results (JSON)
```

### Query Command Options

```bash
python run.py query --help

Options:
  -v, --verbose        Show detailed reasoning steps and retrieved documents
  -o, --output FILE    Save result to JSON file
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service info |
| `/health` | GET | Health check |
| `/query` | POST | Run RAG query |
| `/index` | POST | Index documents |
| `/config` | POST | Update configuration |
| `/stats` | GET | Get system statistics |
| `/evaluation` | GET | Get HotpotQA evaluation data |

### Query Example

```bash
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Who directed the movie Titanic?",
    "top_k": 5,
    "max_iterations": 3
  }'
```

### Response Example

```json
{
  "answer": "James Cameron",
  "question": "Who directed the movie Titanic?",
  "total_time_ms": 1245.32,
  "retrieval_time_ms": 156.78,
  "llm_time_ms": 987.54,
  "iterations": 2,
  "retrieved_docs": 10,
  "reasoning_steps": [
    {
      "step": 1,
      "action": "SEARCH",
      "query": "Titanic movie director",
      "reasoning": "I need to find information about who directed Titanic."
    },
    {
      "step": 2,
      "action": "ANSWER",
      "reasoning": "Based on the retrieved documents, Titanic was directed by James Cameron."
    }
  ]
}
```

## Datasets

### Corpus (for indexing)
- **HotpotQA (hotpotqa/hotpot_qa)**: Context documents from distractor subset
- Train split: ~90,000 questions with context passages
- Validation split: ~7,400 questions with context passages

### QA (for profiling)
- **HotpotQA**: Multi-hop QA questions requiring reasoning across multiple documents
- Deterministic sampling with fixed seed for reproducible benchmarks

## FAISS Algorithms

### HNSW (Hierarchical Navigable Small World)
- Best for high-recall requirements
- Parameters: M (32), ef_construction (200), ef_search (128)

### IVF (Inverted File Index)
- Best for large-scale datasets
- Parameters: nlist (100), nprobe (10)

## Embedding Model

- **Model**: `sentence-transformers/all-mpnet-base-v2`
- **Dimensions**: 768
- **Parameters**: 109M
- **Max Sequence Length**: 384 tokens
- **GPU Acceleration**: Automatic CUDA detection

## Benchmark Experiments

### Experiment 1: QPS Sweep (Fixed Parameters)
- **Fixed**: `max_iterations=3`, `top_k=5`
- **Variable**: QPS (queries per second)
- **Output Graphs**:
  - Total latency average vs QPS
  - Total latency P99 vs QPS
  - Queue waiting time ratio vs QPS

### Experiment 2: Max Iterations Comparison
- **Fixed**: `top_k=5`
- **Variable**: `max_iterations` (1, 2, 3, 4, 5) × QPS sweep
- **Output Graphs**:
  - Total latency comparison across max_iterations
  - Queue time ratio comparison

## Project Structure

```
agent_testbed/
├── src/
│   ├── retrieval/
│   │   ├── faiss_retriever.py   # FAISS vector DB with GPU embeddings
│   │   └── dataset_loader.py    # HotpotQA corpus and query loaders
│   ├── rag_pipeline/
│   │   └── ircot.py             # IRCoT ReAct-style implementation
│   ├── frontend/
│   │   ├── profiler.py          # Legacy profiler
│   │   └── query_generator.py   # Poisson-based load generator
│   └── main.py                  # FastAPI REST server
├── experiments/
│   ├── benchmark.py             # Automated benchmark script
│   ├── run_benchmark.sh         # Benchmark runner shell script
│   └── README.md                # Experiment documentation
├── configs/
│   └── default_config.yaml      # Default configuration
├── tests/
│   └── test_retriever.py        # Unit tests
├── run.py                       # CLI runner
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## Profiling Metrics

| Metric | Description |
|--------|-------------|
| `total_time_ms` | End-to-end query latency |
| `retrieval_time_ms` | FAISS search time (all iterations) |
| `llm_time_ms` | LLM inference time (all iterations) |
| `queue_time_ms` | Server queue waiting time |
| `iterations` | Number of IRCoT iterations |
| `retrieved_docs` | Total documents retrieved |
| `throughput` | Actual achieved QPS |

## License

MIT License
