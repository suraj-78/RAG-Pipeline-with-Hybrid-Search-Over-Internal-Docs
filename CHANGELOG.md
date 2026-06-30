# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-06-30

### Added
- Decoupled document ingestion and query runtime processing APIs in FastAPI.
- Hybrid search fusion utilizing Reciprocal Rank Fusion (RRF) algorithm.
- Local vector storage using HNSW index topologies (ChromaDB) with `all-MiniLM-L6-v2`.
- Inverted sparse index leveraging BM25 lexical keyword scoring.
- Deep neural re-ranking layer utilizing Cross-Encoder `ms-marco-MiniLM-L-6-v2`.
- Grounded synthesis generation using Llama 3.1 on Groq with JSON Mode.
- Deterministic Citation Verifier to prevent LLM citation alignment drift.
- LLM-as-a-Judge RAG metrics runner evaluating Faithfulness and Relevancy.
- 50-sample Golden Test Suite spanning institutional policies.
- Unit and integration tests for chunking, sanitization, and citation tracing.
- Unified configuration class `AppConfig` and Docker orchestration configs.
- Streamlit interactive web dashboard UI.

### Fixed
- Stripped unpaired unicode surrogates and null bytes to prevent Pydantic string validation panics.
- Resolved word gluing and text bullet extraction corruptions on PDF parser layers.
- Patched SQLite/ChromaDB concurrent indexing thread collisions during file ingestion.
- Pre-warmed neural models weights on Streamlit startup to avoid WebSocket timeouts.
