import streamlit as st
import os
from pathlib import Path

# Direct Module Imports aligned perfectly with repository architectures
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

st.set_page_config(page_title="Enterprise Hybrid RAG Dashboard", layout="wide")
st.title("🚀 Enterprise Hybrid RAG Engine over Internal Docs")
st.subheader("BTech CSE Specialization in AI - Placement Verification Center")
st.markdown("---")

# Initialize and Cache Engines into Streamlit Session State
if "pipeline" not in st.session_state:
    with st.spinner("Initializing neural search weights and HNSW graph matrices..."):
        config.validate_environment()
        
        # Instantiate index references safely
        sparse_idx = SparseBM25Index()
        sparse_idx.load_index()  # Load pre-built indices if available
        dense_idx = DenseVectorIndex()
        
        # Bind pipeline nodes onto state cache
        st.session_state.retriever = HybridRetriever(sparse_idx, dense_idx)
        st.session_state.reranker = DocumentReranker()
        st.session_state.generator = GroundedGenerator()
        st.session_state.parser = DocumentParserRouter()
        st.session_state.deduplicator = ChunkDeduplicator()
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
                    # FIXED: Called .retrieve() instead of non-existent get_relevant_chunks
                    hybrid_candidates = st.session_state.retriever.retrieve(user_query, top_k=config.RETRIEVAL_TOP_K)
                    
                    if not hybrid_candidates:
                        st.warning("The documentation index is currently completely empty.")
                    else:
                        # FIXED: Driven through the true attention cross-encoder layer
                        reranked = st.session_state.reranker.rerank(user_query, hybrid_candidates, top_n=config.RERANK_TOP_N)
                        
                        # FIXED: Called .generate_answer() matching generator.py signatures
                        payload = st.session_state.generator.generate_answer(user_query, reranked)
                        
                        st.markdown("### 🤖 Synthesized Answer")
                        st.success(payload["answer"])
                        st.markdown(f"**Context Sufficiency Boundary:** `{payload['is_context_sufficient']}`")
                        
                        # FIXED: Invoked the proper CitationVerifier matching verifier.py blueprint
                        v_matrix = CitationVerifier.verify_citations(payload["answer"], reranked)
                        
                        st.markdown("### 🛡️ Citation Trace Integrity Check")
                        if v_matrix.get("is_valid", False):
                            st.info("✅ All generated text assertions match the physical document chunk anchors securely.")
                        else:
                            st.error("⚠️ Potential Alignment Drift Detected! References failed index mapping.")
                            if v_matrix.get("flagged_issues"):
                                st.json(v_matrix["flagged_issues"])
                except Exception as e:
                    st.error(f"Pipeline Runtime Failure: {str(e)}")

with col2:
    st.header("📂 Document Ingestion Node")
    ingest_path = st.text_input("Absolute/Relative Document File Path:", placeholder="e.g., README.md")
    
    if st.button("Trigger Pipeline Ingestion"):
        if not ingest_path.strip():
            st.warning("Provide a valid file sequence mapping.")
        else:
            with st.spinner("Processing structural chunking and dual-indexing runs..."):
                try:
                    if not os.path.exists(ingest_path):
                        st.error(f"File not found tracking path: {ingest_path}")
                    else:
                        # FIXED: Returns a validated Pydantic Document model object
                        document = st.session_state.parser.process_file(ingest_path)
                        
                        # FIXED: Static method invocation passing the Document instance
                        raw_chunks = ChunkingEngine.structure_aware_markdown_chunk(document)
                        
                        # FIXED: Deduplication layer injection using the cached embeddings lambda
                        def ui_embedding_fn(texts):
                            return st.session_state.retriever.dense_index.embedding_fn(texts)
                            
                        clean_chunks = st.session_state.deduplicator.deduplicate(raw_chunks, embedding_fn=ui_embedding_fn)
                        
                        if not clean_chunks:
                            st.info("No new unique chunks detected. Ingestion skipped to protect vector weights.")
                        else:
                            # Indexing safely into both system segments
                            st.session_state.retriever.sparse_index.index_chunks(clean_chunks)
                            st.session_state.retriever.dense_index.index_chunks(clean_chunks)
                            
                            st.balloons()
                            st.success(f"Successfully processed! Unique Chunks Indexed: {len(clean_chunks)}")
                except Exception as e:
                    st.error(f"Ingestion worker failure: {str(e)}")