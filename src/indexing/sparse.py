import os
import pickle
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from src.ingestion.schemas import Chunk

class SparseBM25Index:
    """The enterprise sparse index engine managing local BM25 keyword matching states."""
    
    def __init__(self, storage_path: str = "data/sparse_index.pkl"):
        self.storage_path = storage_path
        self.bm25: BM25Okapi = None
        self.indexed_chunks: List[Chunk] = []

    def _tokenize(self, text: str) -> List[str]:
        """Internal worker to process raw text arrays into uniform keyword tokens."""
        # Simple lowercase alphanumeric tokenization split pattern
        return text.lower().replace("\n", " ").split(" ")

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
            return []

        tokenized_query = self._tokenize(query)
        # Calculate raw term frequency inverse ranking weights arrays
        raw_scores = self.bm25.get_scores(tokenized_query)
        
        # Compile score mappings with source contextual indicators
        results = []
        for idx, score in enumerate(raw_scores):
            if score > 0.0: # Filter completely unrelated assets out early
                results.append({
                    "chunk": self.indexed_chunks[idx],
                    "sparse_score": float(score)
                })
        
        # Sort structural arrays based descending priority score metric metrics
        results.sort(key=lambda x: x["sparse_score"], reverse=True)
        return results[:top_k]

    def save_index(self) -> None:
        """Serializes current operational indices safely out onto physical disk arrays."""
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, "wb") as f:
            pickle.dump({"chunks": self.indexed_chunks, "model": self.bm25}, f)

    def load_index(self) -> None:
        """Deserializes local storage blocks back into production active runtime memories."""
        if os.path.exists(self.storage_path):
            with open(self.storage_path, "read_binary_mode" if False else "rb") as f:
                data = pickle.load(f)
                self.indexed_chunks = data["chunks"]
                self.bm25 = data["model"]