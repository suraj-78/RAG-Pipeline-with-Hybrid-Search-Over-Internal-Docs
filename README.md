---
title: ProductionRAG
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.46.1
app_file: src\ui\app.py
pinned: false
---


# Enterprise Hybrid RAG Engine over Internal Docs
BTech CSE Specialization in AI - Placement Query Verification Core.

This application is containerized using Docker and deployed seamlessly on Hugging Face Spaces using direct module orchestration inside Streamlit.

---

## 🏗️ Architectural Blueprint

The system processing layers decouple document ingestion from query-time synthesis to maintain data isolation and horizontal scaling compatibility:

```text
[Document Ingestion] 
   └── DocumentParserRouter (PDF / HTML / TXT) ──> Strict Unicode & Regex Sanitizer
   └── ChunkingEngine (Sliding Character Windows via AppConfig Parameters)
   └── Dual Index Ingestion 
          ├── Dense Vector Space (ChromaDB Vector Store)
          └── Sparse Index State (Inverted BM25 Serialized Pickles)

[Query Runtime Execution Pipeline]
   └── User Query 
          ├── Parallel Retrieval Matrix 
          │     ├── Dense Search (all-MiniLM-L6-v2 Space Top-2K Vector Lookup)
          │     └── Sparse Search (BM25 Vocabulary TF-IDF Frequency Matching)
          ├── Fusion Engine (Reciprocal Rank Fusion - RRF Algorithm)
          ├── Re-Ranking Router (Cross-Encoder / ms-marco-MiniLM-L-6-v2 Node)
          └── Synthesis Gateway (Context-Stuffed Llama-3.1-8B Inference Session via Groq)

```

---

## 🚀 Key Engineering Implementations

* **Dual-Engine Search & Reciprocal Rank Fusion (RRF):** Executes parallel candidates generation across non-homogeneous index topologies. Scores are mathematically fused using the reciprocal of inverted ranking profiles (`1 / (rrf_k + rank + 1)`) to balance precise keyword indexing against deep contextual semantics.
* **Deterministic Text Normalization Guardrails:** Solves standard PDF text extraction failure modes where legacy layout parsers introduce broken bullet matrices (``) or cause lowercase word gluing (e.g., `laiddownby` instead of `laid down by`). Dynamic regex parsing breaks up token gluing, optimizing vocabulary accuracy for BM25 mapping.
* **Strict Rust/Pydantic Serialization Safe Filter:** Injects deep byte scanning inside the `DocumentParserRouter` to forcefully strip unpaired surrogates (`\ud800` to `\udfff`) and hidden trailing null bytes (`\x00`). This eliminates runtime data parsing panics inside Rust-native Pydantic backend models and Vector DB clients.
* **Centralized Governance Configurations:** Configured with an extensible configurations class (`AppConfig`) decoupling infrastructure storage directories from model architectural states. Controls runtime sliding-window parameters (`CHUNK_SIZE`, `CHUNK_OVERLAP`, and `RERANK_TOP_N`) from a unified systemic point.

---

## 🛠️ Tech Stack & Models

* **Frontend UI:** Streamlit
* **Asynchronous Backend API:** FastAPI
* **Vector Database:** ChromaDB (HNSW Graph Layout)
* **Sparse Token Index:** Rank-BM25
* **Dense Embedding Core:** `all-MiniLM-L6-v2`
* **Neural Re-ranker Node:** Cross-Encoder (`ms-marco-MiniLM-L-6-v2`)
* **LLM Core Gateway:** Groq Inference Engine (`llama-3.1-8b-instant`)

---

## 📂 Repository Directory Tree

```text
├── .streamlit/                 # Frontend UI theme constraints configuration
├── data/                       # Dynamic system database layer (git-ignored)
│   ├── chroma_db/              # Persistent vector database space
│   └── sparse_index.pkl        # Serialized BM25 inverted vocabulary storage
├── src/                        # Isolated Core Application Source
│   ├── config.py               # Governance layer managing system states & variables
│   ├── main.py                 # FastAPI master routing and endpoint configurations
│   ├── evaluation/             # Pipeline metrics runner matrices (Ragas/Scoring)
│   ├── generation/             # Text synthesis modules & Groq proxy connectors
│   ├── indexing/               # Dense vector indexes and BM25 implementations
│   ├── ingestion/              # Extensible parsers, deduplicators, and chunkers
│   ├── reranking/              # Cross-Encoder model execution layer
│   └── ui/                     # UI visual interface architecture
├── requirements.txt            # Production pinned package dependencies manifest
├── run.py                      # Root pathing wrapper executing application nodes
└── Dockerfile                  # Application orchestration containerization layout

```

---

## 🛠️ Local Installation & Environment Setup

### Prerequisites

* Python 3.11 or Python 3.12 (Do not deploy on Python 3.14+ due to missing native wheels compiler support for `pillow` and `pydantic-core`).
* Valid Groq API Key (obtained from the Groq console).

### 1. Initialize Code Environment

Clone the repository path, instantiate a clean virtual environment, and pull all dependencies:

```bash
git clone [https://github.com/divyyadav007/RAG-Pipeline-with-Hybrid-Search-Over-Internal-Docs.git](https://github.com/divyyadav007/RAG-Pipeline-with-Hybrid-Search-Over-Internal-Docs.git)
cd RAG-Pipeline-with-Hybrid-Search-Over-Internal-Docs

# Setup isolation environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install requirements matrix
pip install --upgrade pip
pip install -r requirements.txt

```

### 2. Configure Environment Tokens

Create a `.env` file at the root repository location or pass terminal environmental configurations:

```bash
# Linux/macOS Shell
export GROQ_API_KEY="gsk_your_production_secret_key_here"
export PYTHONPATH="."

# Windows PowerShell
$env:GROQ_API_KEY="gsk_your_production_secret_key_here"
$env:PYTHONPATH="."

```

### 3. Launch API Backend Core (FastAPI)

```bash
uvicorn src.main:app --reload --port 8000

```

Verify the backend deployment status by launching the interactive Swagger interface at `http://127.0.0.1:8000/docs`.

### 4. Launch Application UI Interface (Streamlit)

Open a separate execution terminal, ensure paths are exported, and run via the root wrapper configuration:

```bash
streamlit run run.py

```

---

## 🐳 Containerization & Deployment (Docker)

To deploy the unified microservice stack into cloud clusters or Hugging Face Spaces using isolated Docker structures:

```bash
# Build the production Docker image locally
docker build -t hybrid-rag-engine .

# Run the container mapping ports natively
docker run -p 8501:8501 -e GROQ_API_KEY="your_key" hybrid-rag-engine

```

---

## 📊 Evaluation Metrics Matrix

The pipeline incorporates an internal automated evaluations framework (`src/evaluation/metrics_runner.py`) running automated verification test benches across standard parameters:

* **Faithfulness Evaluation:** Assesses whether synthesized statements are strictly grounded inside retrieved contexts, neutralizing hallucination rates.
* **Answer Relevance Metrics:** Validates systemic synthesis logic against literal semantic questions prompted by users.
* **Context Recall Precision:** Verifies whether the combined Sparse + Dense retrieval layers successfully fetch complete required rows from complex tables and bullet metrics.

---

## 🔒 License

Distributed under the MIT License. See `LICENSE` for more explicit system permissions structures.

```

```