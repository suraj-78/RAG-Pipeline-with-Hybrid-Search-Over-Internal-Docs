import os
from typing import List, Dict, Any
import chromadb
from chromadb.utils import embedding_functions
from src.ingestion.schemas import Chunk
from src.config import config  # Centralized configuration gateway

class DenseVectorIndex:
    """The engineering interface driving dense vector storage and local HNSW graph lookups."""

    def __init__(self, collection_name: str = "internal_docs"):
        # Fixed: Pulled the storage path directly from the central config state
        self.client = chromadb.PersistentClient(path=config.CHROMA_STORAGE_DIR)
        
        # Fixed: Replaced hardcoded string with central configuration parameter
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=config.EMBEDDING_MODEL_NAME
        )
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

    def index_chunks(self, chunks: List[Chunk]) -> None:
        """Pushes structured text slices and structural metadata tags into the vector index."""
        if not chunks:
            return

        ids = [chunk.id for chunk in chunks]
        documents = [chunk.page_content for chunk in chunks]
        
        metadatas = []
        for chunk in chunks:
            meta_dict = {
                "source_path": chunk.metadata.source_path,
                "file_type": chunk.metadata.file_type,
                "chunk_index": chunk.metadata.chunk_index,
                "parent_document_id": chunk.metadata.parent_document_id
            }
            metadatas.append(meta_dict)

        self.collection.add(
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