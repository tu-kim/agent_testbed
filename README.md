# Multi-hop RAG with IRCoT

A distributed multi-hop Retrieval-Augmented Generation (RAG) system using the IRCoT (Interleaving Retrieval with Chain-of-Thought) pipeline. Built with LangChain, vLLM, and FAISS for high-performance question answering.

## Features

- **IRCoT RAG Pipeline**: Interleaving retrieval with chain-of-thought reasoning for multi-hop QA
- **Distributed vLLM Server**: Proxy server with separate prefill and decode workers
- **FAISS Vector Database**: Support for HNSW and IVF algorithms
- **Dense Retrieval**: Using sentence-transformers/all-MiniLM-L6-v2
- **Performance Profiling**: Frontend profiler for latency measurement
- **Local Model Support**: Load models from local directories or HuggingFace Hub

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAG Server (:8080)                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │   IRCoT     │  │   FAISS     │  │   LangChain LLM Client  │ │
│  │  Pipeline   │──│  Retriever  │──│                         │ │
│  └─────────────┘  └─────────────┘  └───────────┬─────────────┘ │
└────────────────────────────────────────────────┼───────────────┘
                                                 │
                    ┌────────────────────────────┼────────────────┐
                    │        vLLM Proxy Server (:8000)            │
                    │  ┌─────────────┐  ┌─────────────────────┐   │
                    │  │ Load        │  │ Request             │   │
                    │  │ Balancer    │──│ Router              │   │
                    │  └──────┬──────┘  └──────────┬──────────┘   │
                    └─────────┼────────────────────┼──────────────┘
                              │                    │
              ┌───────────────┼────────────────────┼───────────────┐
              │               │                    │               │
     ┌────────▼────────┐      │       ┌────────────▼────────┐      │
     │ Prefill Worker  │      │       │   Decode Worker     │      │
     │    (:8001)      │      │       │      (:8002)        │      │
     │  ┌───────────┐  │      │       │  ┌───────────────┐  │      │
     │  │ vLLM      │  │      │       │  │ vLLM          │  │      │
     │  │ Engine    │  │      │       │  │ Engine        │  │      │
     │  └───────────┘  │      │       │  └───────────────┘  │      │
     └─────────────────┘      │       └─────────────────────┘      │
                              │                                    │
                    ┌─────────▼────────────────────────────────────┘
                    │              GPU Cluster
                    └──────────────────────────────────────────────
```

## Installation

```bash
# Clone the repository
git clone https://github.com/tu-kim/agent_testbed.git
cd agent_testbed

# Install dependencies
pip install -r requirements.txt

# For vLLM workers (requires GPU)
pip install vllm>=0.3.0
```

## Quick Start

### 1. Index C4 Corpus

Index the C4 corpus from a local directory (all samples):

```bash
# Index with HNSW algorithm
python run.py index --corpus-dir /path/to/c4/data --algorithm hnsw

# Index with IVF algorithm
python run.py index --corpus-dir /path/to/c4/data --algorithm ivf

# Force rebuild existing index
python run.py index --corpus-dir /path/to/c4/data --force
```

The `--corpus-dir` argument is **required** and should point to a local directory containing C4 dataset files (JSONL, JSON, or text format).

### 2. Start vLLM Workers (Optional - for distributed inference)

```bash
# Terminal 1: Start prefill worker
python run.py prefill --model /path/to/model --port 8001

# Terminal 2: Start decode worker
python run.py decode --model /path/to/model --port 8002

# Terminal 3: Start proxy server
python run.py proxy --port 8000 --prefill-workers localhost:8001 --decode-workers localhost:8002
```

**Note**: The `--model` argument supports both HuggingFace model IDs (e.g., `meta-llama/Llama-2-7b-chat-hf`) and local directory paths (e.g., `/data/models/llama-7b`).

### 3. Start RAG Server

```bash
# Start RAG server (connects to LLM at localhost:8000)
python run.py server --llm-endpoint http://localhost:8000 --port 8080
```

### 4. Run Queries

```bash
# Single query
python run.py query "What is the capital of France and when was the Eiffel Tower built?"

# With verbose output
python run.py query "Who wrote Romeo and Juliet?" -v
```

### 5. Performance Profiling

Profile using HotpotQA questions:

```bash
# Run profiling with 10 queries
python run.py profile --endpoint http://localhost:8080 --num-queries 10

# Save results to file
python run.py profile --num-queries 50 --output profiling_results.json
```

## CLI Reference

### Commands

| Command | Description |
|---------|-------------|
| `server` | Run the RAG server |
| `proxy` | Run the vLLM proxy server |
| `prefill` | Run the vLLM prefill worker |
| `decode` | Run the vLLM decode worker |
| `index` | Index C4 corpus into FAISS |
| `profile` | Run performance profiling with HotpotQA |
| `query` | Run a single query |

### Index Command Options

```bash
python run.py index --help

Options:
  --corpus-dir PATH    Local directory path containing C4 dataset (required)
  --algorithm          FAISS algorithm: hnsw, ivf, flat (default: hnsw)
  --cache-dir PATH     Directory to store FAISS index (default: ./cache/faiss_index)
  --force              Force rebuild index even if exists
```

### Server Command Options

```bash
python run.py server --help

Options:
  --host HOST          Server host (default: 0.0.0.0)
  --port PORT          Server port (default: 8080)
  --llm-endpoint URL   LLM server endpoint (default: http://localhost:8000)
  --algorithm          FAISS algorithm (default: hnsw)
  --cache-dir PATH     FAISS index directory (default: ./cache/faiss_index)
  --model-name NAME    Model name or local path
  --reload             Enable auto-reload for development
```

### Profile Command Options

```bash
python run.py profile --help

Options:
  --endpoint URL       RAG server endpoint (default: http://localhost:8080)
  --num-queries N      Number of queries to run (default: 10)
  --warmup-queries N   Number of warmup queries (default: 2)
  --output FILE        Output file for results (optional)
```

## Configuration

Edit `configs/default_config.yaml`:

```yaml
# vLLM Server Configuration
vllm:
  proxy:
    host: "0.0.0.0"
    port: 8000
  model_name: "meta-llama/Llama-2-7b-chat-hf"  # or local path
  trust_remote_code: true

# FAISS Index Configuration
faiss:
  algorithm: "hnsw"  # or "ivf"
  hnsw:
    M: 32
    ef_construction: 200
    ef_search: 128
  ivf:
    nlist: 100
    nprobe: 10

# Dataset Configuration
dataset:
  corpus:
    name: "allenai/c4"
    # local_dir: "/path/to/c4/data"  # Uncomment for local data
  qa:
    name: "hotpotqa/hotpot_qa"
    subset: "fullwiki"

# IRCoT Configuration
ircot:
  max_iterations: 3
  retrieval_per_step: 3
  early_stop: true
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
    "question": "What is machine learning?",
    "top_k": 5,
    "max_iterations": 3
  }'
```

## Datasets

### Corpus (for indexing)
- **C4 (allenai/c4)**: Colossal Clean Crawled Corpus
- Loaded from local directory via `--corpus-dir` argument
- Supports JSONL, JSON, and text file formats

### QA (for profiling)
- **HotpotQA (hotpotqa/hotpot_qa)**: Multi-hop QA dataset
- Automatically loaded from HuggingFace for profiling

## FAISS Algorithms

### HNSW (Hierarchical Navigable Small World)
- Best for high-recall requirements
- Parameters: M, ef_construction, ef_search

### IVF (Inverted File Index)
- Best for large-scale datasets
- Parameters: nlist, nprobe

## Project Structure

```
agent_testbed/
├── src/
│   ├── vllm_server/
│   │   ├── proxy_server.py      # Load balancer and API gateway
│   │   ├── prefill_worker.py    # Prefill phase worker
│   │   └── decode_worker.py     # Decode phase worker
│   ├── retrieval/
│   │   ├── faiss_retriever.py   # FAISS vector DB
│   │   └── dataset_loader.py    # C4 and HotpotQA loaders
│   ├── rag_pipeline/
│   │   └── ircot.py             # IRCoT implementation
│   ├── frontend/
│   │   └── profiler.py          # Performance profiler
│   └── main.py                  # REST API server
├── configs/
│   └── default_config.yaml      # Default configuration
├── tests/
│   └── test_retriever.py        # Unit tests
├── run.py                       # CLI runner
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## License

MIT License
