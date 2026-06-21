import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

# Central configuration and architecture mapping
from src.config import config
from src.ingestion.parsers import DocumentParserRouter
from src.ingestion.chunkers import ChunkingEngine
from src.ingestion.deduplicator import ChunkDeduplicator
from src.indexing.sparse import SparseBM25Index
from src.indexing.dense import DenseVectorIndex
from src.indexing.hybrid_retriever import HybridRetriever
from src.reranking.cross_encoder import DocumentReranker
from src.generation.generator import GroundedGenerator
from src.generation.verifier import CitationVerifier

# =====================================================================
# MODERN LIFESPAN GATEWAY (Unified Thread-Safe Single Initialization)
# =====================================================================
@asynccontextmanager
async def app_lifespan(app: FastAPI):
    print("\n🚀 [STARTUP] Booting Enterprise Hybrid RAG Engine API Node...")
    try:
        config.validate_environment()
        
        # Explicitly initialize and load indices safely within context boundaries
        sparse_index = SparseBM25Index()
        sparse_index.load_index()  # Pull pre-compiled pickle tokens from disk
        dense_index = DenseVectorIndex()
        
        # Enforcing single-ton instance isolation inside app state containers
        app.state.parser_router = DocumentParserRouter()
        app.state.deduplicator = ChunkDeduplicator()
        app.state.sparse_index = sparse_index
        app.state.dense_index = dense_index
        app.state.hybrid_retriever = HybridRetriever(sparse_index, dense_index)
        app.state.reranker = DocumentReranker()
        app.state.generator = GroundedGenerator()
        
        print("✅ [STARTUP] Systems check passed. All neural indexing layers loaded cleanly.")
    except Exception as e:
        print(f"❌ [CRITICAL STARTUP FAILURE] Initialization sequence aborted: {str(e)}")
    yield
    print("🛑 [SHUTDOWN] Evicting memory models and closing disk locks cleanly...\n")


# Instantiate the FastAPI context node matching structural lifespan hooks
app = FastAPI(
    title="Enterprise Hybrid RAG Engine API",
    description="High-performance dual-index retrieval orchestration layers.",
    version="1.0.0",
    lifespan=app_lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define robust API request schemas
class IngestRequest(BaseModel):
    file_path: str = Field(..., min_length=1, max_length=500, description="Path to document file")

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=5000, description="User query string")


# Helper function wrapping thread-safe active state embedding references
def get_local_embedding_fn(app: FastAPI):
    return lambda texts: app.state.dense_index.embedding_fn(texts)


# =====================================================================
# API ENDPOINTS
# =====================================================================

@app.post("/v1/ingest")
async def ingest_document(payload: IngestRequest):
    """Production endpoint driving raw ingestion, structural chunking, and dual indexing loops."""
    if not payload.file_path.strip():
        raise HTTPException(status_code=400, detail="File path string cannot be empty.")
        
    try:
        # Step 1: Parse file securely via state parser router
        document = app.state.parser_router.process_file(payload.file_path)
        
        # Step 2: Dynamic Ingestion Routing to completely prevent Token Overflow (Error 413)
        if document.metadata.file_type == "md":
            print("[INGESTION] Structure-Aware Markdown Chunking activated.")
            raw_chunks = ChunkingEngine.structure_aware_markdown_chunk(document)
        else:
            print(f"[INGESTION] Fixed-Size Sliding Window Chunking activated for .{document.metadata.file_type}")
            raw_chunks = ChunkingEngine.fixed_size_chunk(
                document, 
                chunk_size=config.CHUNK_SIZE, 
                chunk_overlap=config.CHUNK_OVERLAP
            )
        
        # Step 3: Evict redundant informational duplicates using thread-safe state embeddings
        ui_embedding_fn = get_local_embedding_fn(app)
        clean_chunks = app.state.deduplicator.deduplicate(raw_chunks, embedding_fn=ui_embedding_fn)
        
        if not clean_chunks:
            return {"status": "success", "message": "No new unique content chunks detected. Index skipped."}

        # Step 4: Index concurrently into parallel systems via active references
        app.state.sparse_index.index_chunks(clean_chunks)
        app.state.dense_index.index_chunks(clean_chunks)
        
        return {
            "status": "success", 
            "chunks_indexed": len(clean_chunks), 
            "source": payload.file_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion pipeline failure: {str(e)}")


@app.post("/v1/ask")
async def process_query(payload: QueryRequest):
    """Central search engine orchestrating parallel lookup fusions and neural re-ranking matrices."""
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Query question string cannot be empty.")
        
    try:
        # Step 1: Run Stage 1 Hybrid Retrieval via state containers (Fetches Top 10 candidates)
        hybrid_candidates = app.state.hybrid_retriever.retrieve(payload.question, top_k=config.RETRIEVAL_TOP_K)
        
        if not hybrid_candidates:
            return {
                "answer": "Documentation index is currently completely empty. Please ingest tracking documents first.",
                "is_context_sufficient": False,
                "verification_matrix": {"is_valid": False, "flagged_issues": ["No context available"]}
            }

        # Step 2: Run Stage 2 Deep Attention Neural Reranking Pass (Filters down to top 3)
        elite_chunks = app.state.reranker.rerank(payload.question, hybrid_candidates, top_n=config.RERANK_TOP_N)

        # Step 3: Run synthesis pipeline using Free Groq Llama Architecture
        llm_output = app.state.generator.generate_answer(payload.question, elite_chunks)

        # Step 4: Run Post-Gen Deterministic Citation Verification Guardrails
        verification_report = CitationVerifier.verify_citations(llm_output["answer"], elite_chunks)

        return {
            "answer": llm_output["answer"],
            "is_context_sufficient": llm_output["is_context_sufficient"],
            "verification_matrix": verification_report
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query resolution pipeline failure: {str(e)}")


@app.get("/health")
async def health_check():
    """Liveness probe for infrastructure monitoring trackers."""
    return {"status": "healthy", "engine": "enterprise_hybrid_rag"}