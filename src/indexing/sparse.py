import os
import pickle
import re
import logging
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from src.ingestion.schemas import Chunk

logger = logging.getLogger(__name__)

class SparseBM25Index:
    """The enterprise sparse index engine managing local BM25 keyword matching states."""
    
    def __init__(self, storage_path: str = "data/sparse_index.pkl"):
        self.storage_path = storage_path
        self.bm25: BM25Okapi = None
        self.indexed_chunks: List[Chunk] = []

    def _tokenize(self, text: str) -> List[str]:
        """Internal worker to process raw text arrays into uniform keyword tokens."""
        if not text:
            return []
        text = text.lower()
        # Remove extra whitespace and structural formatting artifacts
        text = re.sub(r'\s+', ' ', text).strip()
        # Captures alphanumeric characters and safely retains hyphenated technical terms
        tokens = re.findall(r'\b[a-z0-9]+(?:-[a-z0-9]+)*\b', text)
        
        # Domain-agnostic high-frequency English stopword filtering
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'is', 'are', 'be', 'by', 'with', 'as'}
        return [t for t in tokens if len(t) > 1 and t not in stopwords]

    def index_chunks(self, chunks: List[Chunk]) -> None:
        """Builds and fits the BM25 structural inverted token matrix model map."""
        if not chunks:
            return

        self.indexed_chunks.extend(chunks)
        
        # Formulate token streams across entire incoming corpus array matrix
        corpus_tokenized = [self._tokenize(chunk.page_content) for chunk in self.indexed_chunks]
        
        # Instantiate the Core BM25 scoring configuration block
        self.bm25 = BM25Okapi(corpus_tokenized)
        self.save_index()

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Executes exact keyword text lookups and extracts matched document scoring distributions."""
        if not self.bm25 or not self.indexed_chunks:
            logger.warning("Sparse index not initialized. Ensure documents have been indexed first.")
            return []

        tokenized_query = self._tokenize(query)
        # Calculate raw term frequency inverse ranking weights arrays
        raw_scores = self.bm25.get_scores(tokenized_query)
        
        results = []
        for idx, score in enumerate(raw_scores):
            if score > 0.0:  # Filter completely unrelated assets out early
                results.append({
                    "chunk": self.indexed_chunks[idx],
                    "sparse_score": float(score)
                })
        
        # Sort structural arrays based descending priority score metric metrics
        results.sort(key=lambda x: x["sparse_score"], reverse=True)
        return results[:top_k]

    # FIXED: Integration Bridge mapping added to prevent AttributeError inside HybridRetriever
    def retrieve(self, query: str, top_k: int = 10) -> List[Chunk]:
        """
        Unified retrieval abstraction signature consumed directly by HybridRetriever layer.
        Unpacks scored mappings and returns a standard array sequence of raw Chunk nodes.
        """
        search_results = self.search(query, top_k=top_k)
        # Extract and return just the pure Chunk objects from score envelopes
        return [item["chunk"] for item in search_results]

    def save_index(self) -> None:
        """Serializes current operational indices safely out onto physical disk arrays."""
        try:
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            with open(self.storage_path, "wb") as f:
                pickle.dump({"chunks": self.indexed_chunks, "model": self.bm25}, f)
            logger.info(f"Successfully baked index states to disk: {self.storage_path}")
        except Exception as e:
            logger.error(f"Failed serialization disk dump sequence: {str(e)}")

    def load_index(self) -> None:
        """Deserializes local storage blocks back into production active runtime memories."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "rb") as f:
                    data = pickle.load(f)
                    self.indexed_chunks = data["chunks"]
                    self.bm25 = data["model"]
                    logger.info(f"✅ [SPARSE] Sparse index loaded: {len(self.indexed_chunks)} chunks matched from storage.")
            except (pickle.UnpicklingError, EOFError, KeyError) as e:
                logger.error(f"⚠️ [SPARSE] Failed to deserialize pickle block, file might be malformed: {str(e)}")
                self.bm25 = None
                self.indexed_chunks = []
        else:
            # FIXED: Graceful telemetry handling matching your Hugging Face initialization warn behaviors
            logger.info(f"ℹ️ [SPARSE] Empty initial state context: No index found at '{self.storage_path}'. Ready for cold ingestion.")
            self.bm25 = None
            self.indexed_chunks = []