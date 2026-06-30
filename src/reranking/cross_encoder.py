import torch
import logging
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
from src.ingestion.schemas import Chunk

logger = logging.getLogger(__name__)

class DocumentReranker:
    """The deep learning execution layer tasked with pairwise attention scoring over candidate chunks."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        # Load local or cached HuggingFace Transformer model weights
        # This model is specifically fine-tuned for information retrieval relevance checking
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing cross-encoder reranker '{model_name}' on device: '{self.device}'")
        self.model = CrossEncoder(model_name, device=self.device)

    def rerank(self, query: str, hybrid_results: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
        """Executes full token-interaction cross-attention scoring to extract elite context blocks."""
        if not hybrid_results:
            return []

        # Step 1: Formulate high-dimensional evaluation pairs [Query, Document Text]
        pairs = []
        for res in hybrid_results:
            chunk: Chunk = res["chunk"]
            pairs.append([query, chunk.page_content])

        # Step 2: Compute raw attention activation scores across all candidate matrices
        # Predict outputs raw logit scales showing absolute match alignment
        scores = self.model.predict(pairs)

        # Step 3: Map scores back to original objects and restructure response schema
        reranked_results = []
        for idx, score in enumerate(scores):
            reranked_results.append({
                "chunk": hybrid_results[idx]["chunk"],
                "rerank_score": float(score),
                "previous_sources": hybrid_results[idx].get("sources", [])
            })

        # Step 4: Sort arrays in descending format according to explicit neural feedback scales
        reranked_results.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        return reranked_results[:top_n]