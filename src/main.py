import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from contextlib import asynccontextmanager  # <-- NEW IMPORT

# Central configuration mapping
from src.config import config
from src.ingestion.parsers import DocumentParserRouter
from src.ingestion.chunkers import ChunkingEngine
from src.ingestion.deduplicator import ChunkDeduplicator
from src.indexing.sparse import SparseBM25Index
from src.indexing.dense import DenseVectorIndex
from src.indexing.hybrid_retriever import HybridRetriever
from src.reranking.cross_encoder import DocumentReranker
from src.generation.generator import GroundedGenerator

# =====================================================================
# MODERN LIFESPAN GATEWAY (Replaces @app.on_event)
# =====================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Executes structural environment audits before processing incoming traffic."""
    # 1. Everything before the 'yield' executes at STARTUP
    print("[STARTUP] Running system compliance checks...")
    config.validate_environment()
    
    yield  # The app is now live and serving requests
    
    # 2. Everything after the 'yield' executes at SHUTDOWN
    print("[SHUTDOWN] Tearing down operational microservices...")

# Inject the lifespan manager directly into the FastAPI application core
app = FastAPI(
    title="Production Hybrid RAG Pipeline Engine", 
    version="1.0.0",
    lifespan=lifespan  
)

# =====================================================================
# Caching heavy components once app is initialized
# =====================================================================
parser_router = DocumentParserRouter()
deduplicator = ChunkDeduplicator()
sparse_index = SparseBM25Index()
dense_index = DenseVectorIndex()
hybrid_retriever = HybridRetriever(sparse_index, dense_index)
reranker = DocumentReranker()
generator = GroundedGenerator()

# Helper mock embedding wrapper to feed local Chroma setups safely
def local_embedding_fn(texts: List[str]) -> List[List[float]]:
    # Redirect calls directly onto Chroma's active embedding algorithm abstraction
    return dense_index.embedding_fn(texts)

# Define API request data schemas
class IngestRequest(BaseModel):
    file_path: str

class QueryRequest(BaseModel):
    question: str

@app.post("/v1/ingest")
async def ingest_document(payload: IngestRequest):
    """Production endpoint driving raw ingestion, structural chunking, and dual indexing loops."""
    try:
        # Step 1: Parse file securely
        document = parser_router.process_file(payload.file_path)
        
        # Step 2: Slice text into structural segments
        raw_chunks = ChunkingEngine.structure_aware_markdown_chunk(document)
        
        # Step 3: Evict redundant information profiles
        clean_chunks = deduplicator.deduplicate(raw_chunks, embedding_fn=local_embedding_fn)
        
        if not clean_chunks:
            return {"status": "success", "message": "No new unique content chunks detected. Index skipped."}

        # Step 4: Index into parallel systems
        sparse_index.index_chunks(clean_chunks)
        dense_index.index_chunks(clean_chunks)
        
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
    try:
        # Step 1: Run Stage 1 Hybrid Retrieval (Fetches Top 10 combined candidates)
        hybrid_candidates = hybrid_retriever.retrieve(payload.question, top_k=10)
        
        if not hybrid_candidates:
            return {
                "answer": "Documentation index is currently completely empty.",
                "is_context_sufficient": False,
                "verified_citations": []
            }

        # Step 2: Run Stage 2 Deep Attention Neural Reranking Pass (Filters down to top 3)
        elite_chunks = reranker.rerank(payload.question, hybrid_candidates, top_n=3)

        # Step 3: Run synthesis pipeline using Free Groq Llama Architecture
        llm_output = generator.generate_answer(payload.question, elite_chunks)

        # Step 4: Run Post-Gen Deterministic Citation Verification Guardrails
        verification_report = CitationVerifier.verify_citations(llm_output["answer"], elite_chunks)

        return {
            "answer": llm_output["answer"],
            "is_context_sufficient": llm_output["is_context_sufficient"],
            "verification_matrix": verification_report
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query resolution pipeline failure: {str(e)}")
    
