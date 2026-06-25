import os
import gc  # Critical garbage collection utility to flush volatile memory arrays instantly
import sys
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

# =====================================================================
# SYSTEM GUARD: Dynamic Root Path Injection to prevent ModuleNotFoundError
# =====================================================================
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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

# Professional Startup Layout Branding Setup
st.set_page_config(
    page_title="Enterprise Hybrid-RAG Dashboard", 
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enterprise Modern UI CSS Styling Injection
st.markdown("""
    <style>
    .main-title { font-size: 2.4rem !important; font-weight: 700 !important; color: #1E3A8A; margin-bottom: 0.2rem; }
    .sub-title { font-size: 1.1rem !important; color: #4B5563; margin-bottom: 2rem; }
    .section-header { font-size: 1.4rem !important; font-weight: 600 !important; color: #1F2937; border-bottom: 2px solid #E5E7EB; padding-bottom: 0.5rem; margin-top: 1rem; }
    .metric-card { background-color: #F9FAFB; padding: 1rem; border-radius: 0.5rem; border: 1px solid #E5E7EB; }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="main-title">🚀 Enterprise Hybrid RAG Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Production-Grade Knowledge Synthesis Framework Over Complex Unstructured Documentation</div>', unsafe_allow_html=True)

# =====================================================================
# INITIALIZATION & FORCED MODEL EAGER WARMUP (Prevents Thread Freeze)
# =====================================================================
if "pipeline" not in st.session_state:
    with st.spinner("📦 [INFRASTRUCTURE] Initializing neural search layers & caching transformer weights into RAM..."):
        config.validate_environment()
        
        sparse_idx = SparseBM25Index()
        sparse_idx.load_index()  # Load pre-built indices if available on disk
        dense_idx = DenseVectorIndex()
        
        # Binding pipeline nodes onto state cache cleanly
        st.session_state.retriever = HybridRetriever(sparse_idx, dense_idx)
        st.session_state.reranker = DocumentReranker()
        st.session_state.generator = GroundedGenerator()
        st.session_state.parser = DocumentParserRouter()
        st.session_state.deduplicator = ChunkDeduplicator()
        
        # FIXED: Forced Model Warmup to prevent lazy-loading crashes during file uploads
        try:
            # 1. Warm up Dense Vector Encoder
            st.session_state.retriever.dense_index.embedding_fn(["warmup text matrix sequence"])
            
            # 2. Warm up Neural Cross-Encoder Reranker using a mock chunk object
            class MockChunk:
                def __init__(self, content): self.page_content = content
            st.session_state.reranker.rerank("warmup", [MockChunk("warmup cache mapping verification")], top_n=1)
            
            print("✅ [WARMUP] All heavy transformers are eagerly pre-loaded and baked into RAM.")
        except Exception as warmup_err:
            print(f"⚠️ [WARMUP WARNING] Pre-loading diagnostic layer bypassed: {str(warmup_err)}")
            
        st.session_state.pipeline = True

# Dashboard Grid Framework Realignment
col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.markdown('<div class="section-header">🔍 Query Interface</div>', unsafe_allow_html=True)
    st.write("")
    
    user_query = st.text_input(
        "Enter your policy, compliance, or structural infrastructure inquiry:", 
        placeholder="e.g., What are the clear compliance requirements regarding internal regulatory parameters?",
        label_visibility="visible"
    )
    
    if st.button("Execute Intelligence Query", type="primary", use_container_width=True):
        if not user_query.strip():
            st.warning("Query validation error: Input string cannot be empty.")
        else:
            with st.spinner("Executing parallel lookups, score distributions, and neural cross-attention routing..."):
                try:
                    # Parallel Index Extraction Ingress
                    hybrid_candidates = st.session_state.retriever.retrieve(user_query, top_k=config.RETRIEVAL_TOP_K)
                    
                    if not hybrid_candidates:
                        st.info("System Notification: The database index space is currently completely empty. Please upload documents.")
                    else:
                        # Cross-Attention Neural Filtration
                        reranked = st.session_state.reranker.rerank(user_query, hybrid_candidates, top_n=config.RERANK_TOP_N)
                        
                        # LLM Direct Grounded Generation Token Stream
                        payload = st.session_state.generator.generate_answer(user_query, reranked)
                        
                        st.markdown("### 🤖 Synthesized Knowledge Output")
                        st.success(payload["answer"])
                        
                        # Metadata Analysis Layout
                        m_col1, m_col2 = st.columns(2)
                        with m_col1:
                            st.markdown(f"**Context Grounding Sufficiency Boundary:** `{payload['is_context_sufficient']}`")
                        
                        # Dynamic Trace Matrix Check
                        v_matrix = CitationVerifier.verify_citations(payload["answer"], reranked)
                        
                        st.markdown("### 🛡️ Citation Trace Integrity Diagnostics")
                        if v_matrix.get("is_valid", False):
                            st.info("✅ System Verification Complete: All response metrics map accurately to document chunk source anchors.")
                        else:
                            st.error("⚠️ Alignment Drift Warning! Synthesized claims failed index validation constraints.")
                            if v_matrix.get("flagged_issues"):
                                st.json(v_matrix["flagged_issues"])
                                
                except Exception as e:
                    st.error(f"Critical Runtime Exception within Pipeline Gateway: {str(e)}")

with col2:
    st.markdown('<div class="section-header">📂 Ingestion Control Panel</div>', unsafe_allow_html=True)
    st.write("")
    
    uploaded_file = st.file_uploader(
        "Ingest Corporate Knowledge Bases:", 
        type=["txt", "md", "pdf", "html", "htm"],
        help="Supported production document layouts: PDF, Markdown, TXT, HTML"
    )
    
    if st.button("Trigger Asset Ingestion Pipeline", use_container_width=True):
        if uploaded_file is None:
            st.warning("Action Aborted: Please reference a valid physical file target first.")
        else:
            # Enforcing dynamic st.status to hold real-time connection locks during operations
            with st.status("Initializing Ingestion Engine Core Processors...", expanded=True) as status_box:
                try:
                    temp_dir = Path(config.DATA_DIR) / "uploaded_files"
                    temp_dir.mkdir(parents=True, exist_ok=True)
                    temp_file_path = temp_dir / uploaded_file.name

                    with open(temp_file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    status_box.write("📄 Layout Analyzer: Sweeping document structural matrices...")
                    document = st.session_state.parser.process_file(str(temp_file_path))
                    
                    if document.metadata.file_type == "md":
                        status_box.write("✂️ Content Tokenizer: Structure-Aware Markdown processing initialized.")
                        raw_chunks = ChunkingEngine.structure_aware_markdown_chunk(document)
                    else:
                        status_box.write(f"✂️ Content Tokenizer: Segmenting .{document.metadata.file_type} layout via character sliding windows...")
                        raw_chunks = ChunkingEngine.fixed_size_chunk(
                            document, 
                            chunk_size=config.CHUNK_SIZE, 
                            chunk_overlap=config.CHUNK_OVERLAP
                        )
                    
                    # Memory-Safe Mini-Batching Embedding Adapter Strategy
                    def ui_embedding_fn(texts):
                        batch_size = 16
                        all_embeddings = []
                        for i in range(0, len(texts), batch_size):
                            batch_texts = texts[i:i + batch_size]
                            batch_res = st.session_state.retriever.dense_index.embedding_fn(batch_texts)
                            all_embeddings.extend(batch_res)
                        return all_embeddings
                        
                    status_box.write("⚡ Entropy Guard: Scanning for internal duplicate hash matches...")
                    clean_chunks = st.session_state.deduplicator.deduplicate(raw_chunks, embedding_fn=ui_embedding_fn)
                    
                    if not clean_chunks:
                        status_box.update(label="ℹ️ Ingestion Notice: Duplicate content footprint neutralized. Indexing skipped.", state="complete")
                    else:
                        status_box.write(f"📥 Matrix Registrar: Concurrent batch loading {len(clean_chunks)} data slices into local matrices...")
                        
                        # Synchronize onto Sparse Matrix Engine
                        st.session_state.retriever.sparse_index.index_chunks(clean_chunks)
                        
                        # Sequential Incremental Array Writes to Completely Avoid SQLite/ChromaDB Thread Collisions
                        vector_batch_size = 25
                        for j in range(0, len(clean_chunks), vector_batch_size):
                            sub_batch = clean_chunks[j:j + vector_batch_size]
                            st.session_state.retriever.dense_index.index_chunks(sub_batch)
                        
                        status_box.update(label="✅ Asset Processing Succeeded: Index Topologies Fully Updated.", state="complete")
                        st.balloons()
                        
                except Exception as e:
                    status_box.update(label=f"❌ Ingestion Worker Routine Terminated: {str(e)}", state="error")
                finally:
                    # Deterministic explicit sweep of systemic file buffers from volatility pools
                    if 'uploaded_file' in locals():
                        del uploaded_file
                    gc.collect()

# Dashboard System Diagnostic Layer Footer
st.markdown("---")
st.caption("Enterprise Hybrid RAG Engine Node v1.0.0 • Architecture Topology: Cosine Space HNSW + Inverted BM25 Token Grid")