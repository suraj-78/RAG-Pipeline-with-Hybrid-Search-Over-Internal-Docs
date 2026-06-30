import os
import gc  # Critical garbage collection utility to flush volatile memory arrays instantly
import sys
import requests
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

# Check for Microservices Mode configuration
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "").strip().rstrip("/")

# Professional Startup Layout Branding Setup
st.set_page_config(
    page_title="Enterprise Hybrid-RAG Dashboard", 
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enterprise Modern UI CSS Styling Injection (Curated Dark theme, Outfit Font, Glassmorphism elements)
st.markdown("""
    <style>
    /* Google Fonts import */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;600;700;800&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%) !important;
        color: #f8fafc !important;
    }
    
    /* Typography */
    .main-title { 
        font-family: 'Outfit', sans-serif;
        font-size: 2.8rem !important; 
        font-weight: 800 !important; 
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        padding-top: 0.5rem;
    }
    
    .sub-title { 
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem !important; 
        color: #94a3b8; 
        margin-bottom: 2rem; 
        font-weight: 400;
    }
    
    .section-header { 
        font-family: 'Outfit', sans-serif;
        font-size: 1.5rem !important; 
        font-weight: 700 !important; 
        color: #f1f5f9; 
        border-bottom: 2px solid #334155; 
        padding-bottom: 0.6rem; 
        margin-top: 1rem;
        margin-bottom: 1.2rem;
    }
    
    /* Glassmorphic Container Cards */
    .custom-card {
        background: rgba(30, 41, 59, 0.45) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        margin-bottom: 1.5rem !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
    }
    
    .metric-value {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #38bdf8 !important;
    }
    
    /* Streamlit overrides for custom buttons */
    .stButton>button {
        background: linear-gradient(90deg, #2563eb 0%, #4f46e5 100%) !important;
        color: #ffffff !important;
        border: none !important;
        padding: 0.6rem 1.5rem !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3) !important;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #1d4ed8 0%, #4338ca 100%) !important;
        box-shadow: 0 6px 16px rgba(79, 70, 229, 0.45) !important;
        transform: translateY(-1px);
    }
    
    /* Footer styles */
    .footer-text {
        text-align: center;
        color: #64748b;
        font-size: 0.85rem;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid #1e293b;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="main-title">🚀 Enterprise Hybrid RAG Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Production-Grade Knowledge Synthesis & Dual-Index Retrieval Architecture</div>', unsafe_allow_html=True)

# Show operational telemetry state in the sidebar
with st.sidebar:
    st.markdown("### 🖥️ Engine Status")
    if BACKEND_API_URL:
        st.success(f"🌐 **Microservice Mode**")
        st.markdown(f"- **API Node**: `{BACKEND_API_URL}`")
        st.markdown("- **Resource Footprint**: Minimal (API-Delegated)")
    else:
        st.info("🔌 **Standalone Mode**")
        st.markdown("- **Execution**: Local Process (In-Memory)")
        st.markdown("- **Models**: Loaded into local RAM")
    
    st.markdown("---")
    st.markdown("### ⚙️ Engine Configurations")
    st.markdown(f"- **Chunk Size**: `{config.CHUNK_SIZE} chars`")
    st.markdown(f"- **Chunk Overlap**: `{config.CHUNK_OVERLAP} chars`")
    st.markdown(f"- **Retrieval Candidates**: `{config.RETRIEVAL_TOP_K}`")
    st.markdown(f"- **Rerank Candidates**: `{config.RERANK_TOP_N}`")

# =====================================================================
# INITIALIZATION & PIPELINE SETUP (Divergent Modes)
# =====================================================================
if "pipeline" not in st.session_state:
    if BACKEND_API_URL:
        # Microservice mode doesn't need to load heavy local models
        st.session_state.pipeline_mode = "microservice"
        st.session_state.pipeline = True
    else:
        st.session_state.pipeline_mode = "standalone"
        with st.spinner("📦 [INFRASTRUCTURE] Initializing local neural search layers & caching weights into RAM..."):
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
            
            # Forced Model Warmup to prevent lazy-loading crashes during file uploads
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

# Dashboard Grid Framework
col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.markdown('<div class="section-header">🔍 Query Interface</div>', unsafe_allow_html=True)
    st.write("")
    
    user_query = st.text_input(
        "Enter your policy, compliance, or structural infrastructure inquiry:", 
        placeholder="e.g., What are the rules regarding campus Wi-Fi network utilization?",
        label_visibility="visible"
    )
    
    if st.button("Execute Intelligence Query", type="primary", use_container_width=True):
        if not user_query.strip():
            st.warning("Query validation error: Input string cannot be empty.")
        else:
            with st.spinner("Executing parallel lookups, score distributions, and neural cross-attention routing..."):
                try:
                    if st.session_state.pipeline_mode == "microservice":
                        # Call API Backend
                        response = requests.post(
                            f"{BACKEND_API_URL}/v1/ask",
                            json={"question": user_query},
                            timeout=60
                        )
                        if response.status_code == 200:
                            payload = response.json()
                            st.markdown("### 🤖 Synthesized Knowledge Output")
                            st.success(payload["answer"])
                            
                            st.markdown("### 🛡️ Citation Trace Integrity Diagnostics")
                            v_matrix = payload["verification_matrix"]
                            if v_matrix.get("is_valid", False):
                                st.info("✅ System Verification Complete: All response metrics map accurately to document chunk source anchors.")
                            else:
                                st.error("⚠️ Alignment Drift Warning! Synthesized claims failed index validation constraints.")
                                if v_matrix.get("flagged_issues"):
                                    st.json(v_matrix["flagged_issues"])
                        else:
                            st.error(f"Backend API Error ({response.status_code}): {response.text}")
                    else:
                        # Local Standalone Processing
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
            with st.status("Initializing Ingestion Engine Core Processors...", expanded=True) as status_box:
                try:
                    temp_dir = Path(config.DATA_DIR) / "uploaded_files"
                    temp_dir.mkdir(parents=True, exist_ok=True)
                    temp_file_path = temp_dir / uploaded_file.name

                    with open(temp_file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    if st.session_state.pipeline_mode == "microservice":
                        status_box.write("🌐 Microservice: Uploading and initiating remote ingestion pipeline...")
                        response = requests.post(
                            f"{BACKEND_API_URL}/v1/ingest",
                            json={"file_path": str(temp_file_path)},
                            timeout=120
                        )
                        if response.status_code == 200:
                            res_data = response.json()
                            status_box.update(label=f"✅ Asset Processing Succeeded: Indexed {res_data.get('chunks_indexed', 0)} chunks on remote node.", state="complete")
                            st.balloons()
                        else:
                            status_box.update(label=f"❌ Remote Ingestion Worker Routine Terminated: {response.text}", state="error")
                    else:
                        # Local Standalone Ingestion
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
st.markdown('<div class="footer-text">Enterprise Hybrid RAG Engine Node v1.0.0 • Architecture Topology: Cosine Space HNSW (ChromaDB) + Inverted BM25 Token Grid</div>', unsafe_allow_html=True)