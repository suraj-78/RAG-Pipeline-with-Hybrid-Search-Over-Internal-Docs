import streamlit as st
import os
import gc  # Critical garbage collection utility to flush volatile memory arrays instantly
from pathlib import Path
from dotenv import load_dotenv
import sys
load_dotenv()
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
# =====================================================================
# SYSTEM GUARD: Dynamic Root Path Injection to prevent ModuleNotFoundError
# =====================================================================
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# =====================================================================
# INITIALIZATION & FORCED MODEL EAGER WARMUP (Prevents Thread Freeze)
# =====================================================================
if "pipeline" not in st.session_state:
    with st.spinner("🚀 [STARTUP] Booting neural search layers & eager loading model weights into RAM..."):
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
                    hybrid_candidates = st.session_state.retriever.retrieve(user_query, top_k=config.RETRIEVAL_TOP_K)
                    
                    if not hybrid_candidates:
                        st.warning("The documentation index is currently completely empty. Please upload documents first.")
                    else:
                        reranked = st.session_state.reranker.rerank(user_query, hybrid_candidates, top_n=config.RERANK_TOP_N)
                        payload = st.session_state.generator.generate_answer(user_query, reranked)
                        
                        st.markdown("### 🤖 Synthesized Answer")
                        st.success(payload["answer"])
                        st.markdown(f"**Context Sufficiency Boundary:** `{payload['is_context_sufficient']}`")
                        
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
    
    uploaded_file = st.file_uploader(
        "Upload Internal Document:", 
        type=["txt", "md", "pdf", "html", "htm"],
        help="Supported formats: PDF, TXT, Markdown, HTML"
    )
    
    if st.button("Trigger Pipeline Ingestion"):
        if uploaded_file is None:
            st.warning("Please upload a file first before triggering the pipeline.")
        else:
            # FIXED: Using dynamic st.status to stream continuous real-time handshake tokens 
            # and completely stop browser WebSocket connection dropped loops during heavy computations.
            with st.status("Processing Document Ingestion Pipeline...", expanded=True) as status_box:
                try:
                    temp_dir = Path(config.DATA_DIR) / "uploaded_files"
                    temp_dir.mkdir(parents=True, exist_ok=True)
                    temp_file_path = temp_dir / uploaded_file.name

                    with open(temp_file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    status_box.write("📄 Executing secure document layout parsing...")
                    document = st.session_state.parser.process_file(str(temp_file_path))
                    
                    if document.metadata.file_type == "md":
                        status_box.write("✂️ Structure-Aware Markdown Chunking activated.")
                        raw_chunks = ChunkingEngine.structure_aware_markdown_chunk(document)
                    else:
                        status_box.write(f"✂️ Slicing .{document.metadata.file_type} text blocks via sliding windows...")
                        raw_chunks = ChunkingEngine.fixed_size_chunk(
                            document, 
                            chunk_size=config.CHUNK_SIZE, 
                            chunk_overlap=config.CHUNK_OVERLAP
                        )
                    
                    # Memory-Safe Mini-Batching Embedding Driver 
                    def ui_embedding_fn(texts):
                        batch_size = 16
                        all_embeddings = []
                        for i in range(0, len(texts), batch_size):
                            batch_texts = texts[i:i + batch_size]
                            batch_res = st.session_state.retriever.dense_index.embedding_fn(batch_texts)
                            all_embeddings.extend(batch_res)
                        return all_embeddings
                        
                    status_box.write("⚡ Scanning for duplicate content profiles...")
                    clean_chunks = st.session_state.deduplicator.deduplicate(raw_chunks, embedding_fn=ui_embedding_fn)
                    
                    if not clean_chunks:
                        status_box.update(label="ℹ️ Duplicate text structure filtered out. Index skipped.", state="complete")
                    else:
                        status_box.write(f"📥 Concurrently indexing {len(clean_chunks)} unique items...")
                        
                        # Index into sparse matrix
                        st.session_state.retriever.sparse_index.index_chunks(clean_chunks)
                        
                        # Incremental batch load to avoid vector database serialization locks
                        vector_batch_size = 25
                        for j in range(0, len(clean_chunks), vector_batch_size):
                            sub_batch = clean_chunks[j:j + vector_batch_size]
                            st.session_state.retriever.dense_index.index_chunks(sub_batch)
                        
                        status_box.update(label="✅ Pipeline Ingestion Successful! Dual-Indices Updated.", state="complete")
                        st.balloons()
                        
                except Exception as e:
                    status_box.update(label=f"❌ Ingestion Worker Aborted: {str(e)}", state="error")
                finally:
                    # Explicit forced RAM flush execution
                    if 'uploaded_file' in locals():
                        del uploaded_file
                    gc.collect()