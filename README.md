
---
title: Enterprise Hybrid RAG Engine
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Enterprise Hybrid RAG Engine over Unstructured Internal Docs

[![Production Ready](https://img.shields.io/badge/status-production--ready-green.svg)](https://github.com/divyyadav007/RAG-Pipeline-with-Hybrid-Search-Over-Internal-Docs)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Deployment-Docker-2496ED.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An enterprise-grade, asynchronous, and fully decoupled Retrieval-Augmented Generation (RAG) engine designed to perform accurate synthesis over unstructured internal documentation, institutional guidelines, compliance specs, and complex corporate PDFs. This engine mitigates classic RAG extraction pitfalls by implementing parallel retrieval matrices, deterministic layout text-normalization, and neural cross-attention re-ranking to achieve zero-hallucination source grounding with verified citations.

---

## 🚀 Key Features

* **Dual-Engine Retrieval & RRF Fusion:** Executes synchronous parallel candidate queries across non-homogeneous indexing structures. Keyword frequency constraints are indexed via an **Inverted BM25 Sparse Matrix**, while high-dimensional vector space mappings are indexed using an **HNSW ChromaDB Vector Store**. Scores are programmatically consolidated via the **Reciprocal Rank Fusion (RRF)** algorithm (`1 / (rrf_k + rank + 1)`).
* **Neural Re-Ranking Node:** Fused text blocks are passed into a deep **Cross-Encoder node (`ms-marco-MiniLM-L-6-v2`)** to evaluate precise sentence-pair semantic cross-attention rather than naive cosine similarity, filtering out background document noise before prompting the LLM.
* **Deterministic Text Normalization Guardrails:** Prevents pipeline ingestion failures caused by faulty open-source PDF layout text extractors. Built-in normalization parsing safely substitutes corrupt tracking bullet matrices (``) with standardized formats and dynamically solves word gluing issues (e.g., splitting `laiddownby` into `laid down by`) using strict regex boundaries.
* **Serialization Safe Filter Integration:** Injects structural byte scanning routines into the document router to clean unpaired surrogates (`\ud800` to `\udfff`) and clear trailing null bytes (`\x00`). This ensures zero runtime string panics within Rust-native Pydantic backend parameters or database validation schemas.
* **Unified Application Orchestration Governance:** Operates via a centralized configuration blueprint (`AppConfig`) decoupling storage directories from model hyper-parameters globally. Controls runtime sliding-window thresholds (`CHUNK_SIZE = 1500`, `CHUNK_OVERLAP = 300`) and evaluation top-K boundaries from a single architectural state file.

---

## 🏗️ System Architecture

```text
                             [Unstructured Documents Ingestion]
                                             │
                                             ▼
                                    DocumentParserRouter
                       (Unicode Cleansing & Regex Space Normalization)
                                             │
                                             ▼
                                      ChunkingEngine
                       (Adaptive 1500-Character Sliding Windows)
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       ▼                                           ▼
            Dense Embedding Store                       Sparse Inverted Index
        (ChromaDB Local HNSW Graph)                 (Rank-BM25 Pickled Vocabulary)
                       │                                           │
                       └─────────────────────┬─────────────────────┘
                                             │ (Query Parsing Runtime)
                                             ▼
                               Parallel Hybrid Search Retriever
                                             │
                                             ▼
                              Reciprocal Rank Fusion (RRF) Engine
                                             │
                                             ▼
                             Cross-Encoder Neural Re-Ranking Node
                                             │
                                             ▼
                              Groq Synthesis Gateway Engine 
                        (Context-Grounded Llama-3.1 Response + Citations)

```

---

## 🛠️ Tech Stack Matrix

| Domain Layer | Component Technology | Selection Justification |
| --- | --- | --- |
| **Interactive UI** | `Streamlit Framework` | Provides real-time dashboard instrumentation, session metrics caching, and document management channels. |
| **Asynchronous API** | `FastAPI (Uvicorn)` | Drives scalable async routing, endpoints isolation, and standard Swagger documentation layers. |
| **Dense Vector Space** | `ChromaDB (Local Persistent)` | Efficient local storage indexing executing low-latency Cosine HNSW spatial graph matching. |
| **Sparse Vocabulary Store** | `Rank-BM25` | Provides exact lexical token alignment, balancing deep embeddings with structural keyword lookups. |
| **Local Vectorization Core** | `SentenceTransformers` | Hosts `all-MiniLM-L6-v2` locally to ensure zero-cost semantic embedding passes. |
| **Re-ranking Framework** | `Cross-Encoder` | Leverages `ms-marco-MiniLM-L-6-v2` attention to compute exact query-context cross-attention filtering. |
| **Inference Infrastructure** | `Groq Cloud Engine` | High-throughput API gateway execution running `llama-3.1-8b-instant` models at ultra-low sub-second latency. |

---

## 📂 Repository Structural Layout

```text
├── .streamlit/                 # UI production layout configuration profiles
├── data/                       # Persistent data storage layers (git-ignored)
│   ├── chroma_db/              # ChromaDB block binary persistence layers
│   └── sparse_index.pkl        # Serialized inverted BM25 state dictionary
├── src/                        # Deep Implementation Core
│   ├── config.py               # Governance setup controlling unified variables & paths
│   ├── main.py                 # FastAPI master orchestration routers and pipelines
│   ├── evaluation/             # Self-running evaluation engines (Faithfulness scoring)
│   ├── generation/             # Groq model endpoints connectors and schema parameters
│   ├── indexing/               # Parallel retrieval execution modules (Dense & BM25)
│   ├── ingestion/              # Custom layout parsers, deduplicators, and chunkers
│   ├── reranking/              # Neural Cross-Encoder candidate routing layer
│   └── ui/                     # Dashboards structure and session caching layers
├── requirements.txt            # Explicitly pinned deployment packaging manifest
├── run.py                      # Core routing wrapper to launch localized components
└── Dockerfile                  # Multi-layer secure workspace containerization matrix

```

---

## 🛠️ Local Installation & Workspace Integration

### Prerequisites

* Python 3.11 or Python 3.12 (Environments running Python 3.14+ are strictly unsupported due to missing pre-compiled wheels binaries for core numerical models).
* Valid Groq Endpoint Key.

### 1. Project Initialization

Clone the target structure, instantiate a virtual environment boundary box, and execute packaging installations:

```bash
git clone [https://github.com/divyyadav007/RAG-Pipeline-with-Hybrid-Search-Over-Internal-Docs.git](https://github.com/divyyadav007/RAG-Pipeline-with-Hybrid-Search-Over-Internal-Docs.git)
cd RAG-Pipeline-with-Hybrid-Search-Over-Internal-Docs

# Instantiate environment tracking
python -m venv .venv
source .venv/bin/activate  # On Windows PowerShell: .venv\Scripts\activate

# Install requirements matrix natively
pip install --upgrade pip
pip install -r requirements.txt

```

### 2. Environment Variables Configuration

To prevent hardcoded data leaks, export the necessary operational environment parameters within the active terminal terminal configuration:

```bash
# Linux/macOS Shell Environment
export GROQ_API_KEY="gsk_your_actual_production_secret_key_here"
export PYTHONPATH="."

# Windows PowerShell Environment
$env:GROQ_API_KEY="gsk_your_actual_production_secret_key_here"
$env:PYTHONPATH="."

```

### 3. Creating Local Infrastructure Directories

Manually seed the persistent binary storage folders locally prior to initiating ingestion workflows to avoid system file lock conflicts:

```bash
# Windows PowerShell
mkdir data
mkdir data/uploaded_files

```

---

## 🚀 Execution Workflows

### 1. Launching the Asynchronous API Backend (FastAPI Option)

To operate the RAG pipeline as a microservice, spin up the async web container utilizing Uvicorn:

```bash
uvicorn src.main:app --reload --port 8000

```

Once initialized, access the full interactive Swagger documentation layer to run ad-hoc validation tests at `http://127.0.0.1:8000/docs`.

### 2. Launching the Unified Analytics Dashboard (Streamlit Frontend)

To operate the engine via a production UI dashboard, execute Streamlit targeting the root structural pathing wrapper:

```bash
streamlit run run.py

```

Open your default web interface at `http://localhost:8501` to access file uploads, data streaming hooks, and grounded synthesis verification tools.

---

## 🐳 Containerization & Production Deployment (Docker)

To bundle the stack cleanly inside stateless reproducible microservice containers utilizing localized Docker images:

### 1. Direct Docker Engine Compilation

```bash
# Compile and build the workspace container image
docker build -t enterprise-hybrid-rag:v1 .

# Instantiate container passing secret vectors securely
docker run -d -p 8501:8501 -e GROQ_API_KEY="gsk_secret_key_here" enterprise-hybrid-rag:v1

```

### 2. Multi-Container Composition (Docker Compose alternative)

```bash
docker-compose up --build -d

```

---

## 🤗 Hugging Face Spaces Cloud Deployment Specifications

When deploying this application natively to **Hugging Face Spaces** utilizing the `sdk: docker` profile framework:

1. **Space Initialization:** Initialize the configuration frontmatter exactly as defined at the top of this `README.md` file to configure the backend image parser.
2. **Secret Keys Ingestion:** Do not hardcode credential manifests. Navigate into the **Hugging Face Space Settings Panel -> Repository Variables / Secrets Configurations Layer** and explicitly configure:
* `GROQ_API_KEY` = `[Your Production Groq Token Ingress Vector]`


3. **Persistent Volume Strategy:** Hugging Face containers utilize ephemeral scratch disk memory boundaries. If scaling storage footprints past container recycle intervals, mount an external **Hugging Face Persistent Storage Volume Slice** and point `CHROMA_STORAGE_DIR` to the path of the persistent network mount inside `src/config.py`.

---

## 📊 Performance & Evaluation Architecture

The repository incorporates an isolated diagnostic scoring module (`src/evaluation/metrics_runner.py`) running programmatic evaluation loops using LLM-as-a-Judge criteria across three core evaluation pillars:

* **Faithfulness Metric Check:** Evaluates whether the generated text tokens are *strictly* grounded within retrieved context segments, proving the complete neutralization of hallucination rates.
* **Answer Relevancy Scoring:** Measures semantic cohesion between output matrices and user queries, eliminating conversational drifts.
* **Context Recall Precision:** Verifies whether the unified RRF sparse/dense search layers correctly capture critical values located in deep nested tables and list fields.

---

## 🔒 License

Distributed under the MIT Operational Software License. See `LICENSE` inside the repository root for detailed permission parameters.

```
