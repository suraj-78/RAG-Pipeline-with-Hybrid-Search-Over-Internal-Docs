import os
import logging
from typing import List, Dict, Any
import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings  # FIXED: Imported core typing interfaces
from sentence_transformers import SentenceTransformer  # FIXED: Direct clean loading layer
from src.ingestion.schemas import Chunk
from src.config import config  # Centralized configuration gateway

logger = logging.getLogger(__name__)

class LocalSentenceTransformerEmbeddingFunction(EmbeddingFunction):
    """Custom high-performance embedding driver to completely bypass PyTorch 2.x meta-tensor bugs on Windows."""
    def __init__(self, model_name: str):
        import torch
        # Explicitly determine physical hardware targets, preventing empty meta device initialization deadlocks
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing local embedding tensors securely on device target: '{self.device}'")
        
        # Instantiate directly using SentenceTransformer to enforce clean weight allocation mappings
        self.model = SentenceTransformer(model_name, device=self.device)

    def __call__(self, input: Documents) -> Embeddings:
        """Fulfills ChromaDB's operational contract for batch text vectorizations."""
        embeddings = self.model.encode(input, convert_to_numpy=True)
        return embeddings.tolist()


class DenseVectorIndex:
    """The engineering interface driving dense vector storage and local HNSW graph lookups."""

    def __init__(self, collection_name: str = "internal_docs"):
        # Pulled the storage path directly from the central config state
        self.client = chromadb.PersistentClient(path=config.CHROMA_STORAGE_DIR)
        
        # FIXED: Injected our bulletproof custom embedding function instead of Chroma's faulty utility
        self.embedding_fn = LocalSentenceTransformerEmbeddingFunction(
            model_name=config.EMBEDDING_MODEL_NAME
        )
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

    def index_chunks(self, chunks: List[Chunk]) -> None:
        """Upserts structured text slices and metadata tags into the dense vector index.
        
        Using upsert prevents unique constraint violations if documents are re-ingested.
        """
        if not chunks:
            logger.info("Empty chunk list provided to dense vector index. Ingestion skipped.")
            return

        ids: List[str] = [chunk.id for chunk in chunks]
        documents: List[str] = [chunk.page_content for chunk in chunks]
        
        metadatas: List[Dict[str, Any]] = []
        for chunk in chunks:
            meta_dict = {
                "source_path": chunk.metadata.source_path,
                "file_type": chunk.metadata.file_type,
                "chunk_index": chunk.metadata.chunk_index,
                "parent_document_id": chunk.metadata.parent_document_id
            }
            metadatas.append(meta_dict)

        logger.info(f"Upserting {len(chunks)} chunks into dense vector database (ChromaDB)...")
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Queries the HNSW graph execution layers for the closest semantic neighbor vectors."""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )

        formatted_results = []
        if not results or not results["ids"] or not results["ids"][0]:
            return []

        for idx in range(len(results["ids"][0])):
            distance = results["distances"][0][idx] if results["distances"] else 1.0
            # For Cosine space: Similarity = 1.0 - Cosine Distance
            similarity_score = float(1.0 - distance)

            formatted_results.append({
                "chunk_id": results["ids"][0][idx],
                "text": results["documents"][0][idx],
                "metadata": results["metadatas"][0][idx],
                "dense_score": similarity_score
            })

        formatted_results.sort(key=lambda x: x["dense_score"], reverse=True)
        return formatted_results