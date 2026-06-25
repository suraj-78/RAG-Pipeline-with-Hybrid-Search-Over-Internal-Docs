## Enterprise Hybrid RAG Engine

A production-ready Retrieval-Augmented Generation (RAG) system for querying internal documents using **Hybrid Search (Dense + Sparse Retrieval)** with reranking and LLM-based response generation.

---

## 📌 Features

- ⚡ Hybrid Search (BM25 + Dense Embeddings)
- 🧠 Cross-Encoder Reranking (`ms-marco-MiniLM-L-6-v2`)
- 🔍 Semantic Search (`all-MiniLM-L6-v2`)
- 🔗 Reciprocal Rank Fusion (RRF)
- 📄 Multi-format Document Support (PDF / TXT / HTML)
- 🧹 Advanced Text Cleaning & Normalization
- 💬 LLM-powered Answers via Groq (LLaMA 3.1 8B)
- 🌐 Streamlit UI
- ⚙️ FastAPI Backend
- 🐳 Docker Support

---

## 🏗️ System Architecture

```text
Document Ingestion
    ├── PDF / HTML / TXT Parser
    ├── Text Sanitization
    └── Chunking Engine

Indexing Layer
    ├── Dense Index (ChromaDB)
    └── Sparse Index (BM25)

Retrieval Layer
    ├── Dense Vector Search
    ├── Keyword Search
    └── RRF Fusion

Reranking Layer
    └── Cross Encoder Model

Generation Layer
    └── Groq LLM (LLaMA 3.1 8B)

UI Layer
    └── Streamlit App
````

---

## 🧰 Tech Stack

* **Frontend:** Streamlit
* **Backend:** FastAPI
* **Vector DB:** ChromaDB
* **Sparse Search:** BM25
* **Embeddings:** all-MiniLM-L6-v2
* **Reranker:** Cross-Encoder (MiniLM)
* **LLM:** Groq API (LLaMA 3.1 8B)
* **Deployment:** Docker / Hugging Face Spaces

---

## 📂 Project Structure

```text
├── app.py                  # Streamlit entrypoint
├── run.py                  # Runtime wrapper
├── src/
│   ├── main.py            # FastAPI backend
│   ├── config.py          # Config management
│   ├── ingestion/         # Data ingestion pipeline
│   ├── indexing/          # Vector + BM25 indexing
│   ├── retrieval/         # Hybrid retrieval logic
│   ├── reranking/         # Cross-encoder reranker
│   ├── generation/        # LLM response generation
│   ├── evaluation/        # Metrics & evaluation
│   └── ui/                # Streamlit UI modules
├── data/                  # Local DB storage (git ignored)
├── requirements.txt
└── Dockerfile
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone Repository

```bash
git clone https://github.com/<username>/RAG-Pipeline.git
cd RAG-Pipeline
```

---

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Set Environment Variables

```bash
# Linux / Mac
export GROQ_API_KEY="your_api_key"

# Windows
$env:GROQ_API_KEY="your_api_key"
```

---

### 5️⃣ Run Application

```bash
streamlit run app.py
```

---

## 🌐 FastAPI Backend (Optional)

```bash
uvicorn src.main:app --reload --port 8000
```

Docs:

```
http://127.0.0.1:8000/docs
```

---

## 🐳 Docker Deployment

```bash
docker build -t hybrid-rag-engine .

docker run -p 8501:8501 \
  -e GROQ_API_KEY="your_api_key" \
  hybrid-rag-engine
```

---

## 📊 Capabilities

* Hybrid semantic + keyword retrieval
* Context-aware LLM generation
* Reduced hallucinations via reranking
* Scalable modular architecture
* Production-ready deployment design

---

## 📜 License

MIT License

```

