import streamlit as st
import os
from pathlib import Path

# Direct Module Imports (No FastAPI network overhead needed for cloud)
from src.config import config
from src.ingestion.parsers import DocumentParserRouter
from src.ingestion.chunkers import ChunkingEngine
from src.indexing.sparse import SparseBM25Index
from src.indexing.dense import DenseVectorIndex
from src.indexing.hybrid_retriever import HybridRetriever
from src.reranking.cross_encoder import DocumentReranker
from src.generation.generator import GroundedGenerator

st.set_page_config(page_title="Enterprise Hybrid RAG Dashboard", layout="wide")
st.title("🚀 Enterprise Hybrid RAG Engine over Internal Docs")
st.subheader("BTech CSE Specialization in AI - Placement Verification Center")
st.markdown("---")

# Initialize Engines and cache them in Streamlit state so they don't reload on every click
if "pipeline" not in st.session_state:
    with st.spinner("Initializing neural search weights and HNSW graph matrices..."):
        config.validate_environment()
        sparse_idx = SparseBM25Index()
        dense_idx = DenseVectorIndex()
        st.session_state.retriever = HybridRetriever(sparse_idx, dense_idx)
        st.session_state.reranker = DocumentReranker()
        st.session_state.generator = GroundedGenerator()
        st.session_state.parser = DocumentParserRouter()
        st.session_state.chunker = ChunkingEngine()
        st.session_state.pipeline = True

col1, col2 = st.columns([2, 1])

with col1:
    st.header("🔍 Ask the Documentation Space")
    user_query = st.text_input("Enter your policy or architectural infrastructure question:", 
                               placeholder="e.g., What happens if a student has less than 75% attendance?")
    
    if st.button("Execute Pipeline Search", type="primary"):
        if not user_query.strip():
            st.warning("Please enter a valid non-empty question string.")
        else:
            with st.spinner("Executing parallel hybrid lookups & neural re-ranking matrices..."):
                try:
                    # Direct Python pipeline invocation
                    raw_chunks = st.session_state.retriever.get_relevant_chunks(user_query, top_k=config.RETRIEVAL_TOP_K)
                    reranked = st.session_state.reranker.rerank(user_query, raw_chunks, top_n=config.RERANK_TOP_N)
                    payload = st.session_state.generator.generate_response(user_query, reranked)
                    
                    st.markdown("### 🤖 Synthesized Answer")
                    st.success(payload["answer"])
                    st.markdown(f"**Context Sufficiency Boundary:** `{payload['is_context_sufficient']}`")
                    
                    v_matrix = payload.get("verification_matrix", {})
                    st.markdown("### 🛡️ Citation Trace Integrity Check")
                    if v_matrix.get("is_valid", False):
                        st.info("✅ All generated text assertions match the physical document chunk anchors securely.")
                    else:
                        st.error("⚠️ Potential Alignment Drift Detected! References failed regex mapping.")
                except Exception as e:
                    st.error(f"Pipeline Runtime Failure: {str(e)}")

with col2:
    st.header("📂 Document Ingestion Node")
    ingest_path = st.text_input("Absolute/Relative Document File Path:", placeholder="e.g., data/test_policy.txt")
    
    if st.button("Trigger Pipeline Ingestion"):
        if not ingest_path.strip():
            st.warning("Provide a valid file sequence mapping.")
        else:
            with st.spinner("Processing structural chunking and dual-indexing runs..."):
                try:
                    if not os.path.exists(ingest_path):
                        st.error(f"File not found tracking path: {ingest_path}")
                    else:
                        raw_text = st.session_state.parser.parse(ingest_path)
                        chunks = st.session_state.chunker.chunk_document(raw_text, ingest_path)
                        
                        # Indexing into both sparse and dense layers
                        st.session_state.retriever.sparse_index.index_chunks(chunks)
                        st.session_state.retriever.dense_index.index_chunks(chunks)
                        
                        st.balloons()
                        st.success(f"Successfully processed! Unique Chunks Indexed: {len(chunks)}")
                except Exception as e:
                    st.error(f"Ingestion worker failure: {str(e)}")