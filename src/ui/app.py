import streamlit as st
import os
import gc  # FIXED: Critical garbage collection utility to flush volatile memory variables
from pathlib import Path
from dotenv import load_dotenv

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

# Initialize and Cache Engines into Streamlit Session State securely
if "pipeline" not in st.session_state:
    with st.spinner("Initializing neural search weights and HNSW graph matrices..."):
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
            with st.spinner("Processing structural chunking and dual-indexing runs..."):
                try:
                    temp_dir = Path(config.DATA_DIR) / "uploaded_files"
                    temp_dir.mkdir(parents=True, exist_ok=True)
                    temp_file_path = temp_dir / uploaded_file.name

                    with open(temp_file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    document = st.session_state.parser.process_file(str(temp_file_path))
                    
                    if document.metadata.file_type == "md":
                        st.info("Structure-Aware Markdown Chunking activated.")
                        raw_chunks = ChunkingEngine.structure_aware_markdown_chunk(document)
                    else:
                        st.info(f"Fixed-Size Character Overlap Chunking activated for .{document.metadata.file_type} file.")
                        raw_chunks = ChunkingEngine.fixed_size_chunk(
                            document, 
                            chunk_size=config.CHUNK_SIZE, 
                            chunk_overlap=config.CHUNK_OVERLAP
                        )
                    
                    # =====================================================================
                    # FIXED: Memory-Safe Mini-Batching Embedding Driver to prevent HF OOM
                    # =====================================================================
                    def ui_embedding_fn(texts):
                        """
                        Throttles text embeddings operations into safe mini-batches of 16 arrays.
                        Keeps RAM footprint tightly flattened to protect free-tier cloud container limits.
                        """
                        batch_size = 16
                        all_embeddings = []
                        for i in range(0, len(texts), batch_size):
                            batch_texts = texts[i:i + batch_size]
                            batch_res = st.session_state.retriever.dense_index.embedding_fn(batch_texts)
                            all_embeddings.extend(batch_res)
                        return all_embeddings
                        
                    # Step 3: Run deduplication loop using our batch-throttled driver
                    st.info("⚡ Running pairwise text deduplication pass in small safe batches...")
                    clean_chunks = st.session_state.deduplicator.deduplicate(raw_chunks, embedding_fn=ui_embedding_fn)
                    
                    if not clean_chunks:
                        st.info("No new unique chunks detected. Ingestion skipped to protect database index integrity.")
                    else:
                        st.info(f"📥 Loading {len(clean_chunks)} clean chunks concurrently into Dual-Indices...")
                        
                        # Step 4a: Index into sparse inverted arrays
                        st.session_state.retriever.sparse_index.index_chunks(clean_chunks)
                        
                        # FIXED: Step 4b: Vector DB injection split into incremental steps of 25 chunks
                        # to prevent massive array matrix sizing allocation payload choking on cloud CPU.
                        vector_batch_size = 25
                        for j in range(0, len(clean_chunks), vector_batch_size):
                            sub_batch = clean_chunks[j:j + vector_batch_size]
                            st.session_state.retriever.dense_index.index_chunks(sub_batch)
                        
                        st.balloons()
                        st.success(f"🚀 Successfully processed! Unique Chunks Indexed: {len(clean_chunks)}")
                        
                except Exception as e:
                    st.error(f"Ingestion worker failure: {str(e)}")
                finally:
                    # FIXED: Hard cleanup pass releasing system descriptors instantly
                    if 'uploaded_file' in locals():
                        del uploaded_file
                    gc.collect()  # Forces garbage collection arrays purge from active RAM threads