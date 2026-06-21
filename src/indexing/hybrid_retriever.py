from typing import List, Dict, Any
from src.indexing.sparse import SparseBM25Index
from src.indexing.dense import DenseVectorIndex
from src.ingestion.schemas import Chunk, ChunkMetadata

class HybridRetriever:
    """The orchestration engine that executes parallel search flows and fuses result spaces via RRF."""

    def __init__(self, sparse_index: SparseBM25Index, dense_index: DenseVectorIndex):
        self.sparse_index = sparse_index
        self.dense_index = dense_index

    def retrieve(self, query: str, top_k: int = 10, rrf_k: int = 60) -> List[Dict[str, Any]]:
        """Executes dual retrieval calls and balances ranking metrics using Reciprocal Rank Fusion."""
        
        # Step 1: Run parallel executions across both isolated search grids
        # We fetch top_k * 2 to give the fusion layer enough candidate pool overlap
        sparse_results = self.sparse_index.search(query, top_k=top_k * 2)
        dense_results = self.dense_index.search(query, top_k=top_k * 2)

        rrf_scores: Dict[str, Dict[str, Any]] = {}

        # Step 2: Compute reciprocal score profiles for Sparse (BM25) candidates
        for rank, res in enumerate(sparse_results):
            chunk: Chunk = res["chunk"]
            chunk_id = chunk.id
            
            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = {"chunk": chunk, "score": 0.0, "sources": ["sparse"]}
            
            # Apply standard 1 / (k + rank) calculation
            rrf_scores[chunk_id]["score"] += 1.0 / (rrf_k + (rank + 1))

        # Step 3: Compute reciprocal score profiles for Dense (ChromaDB) candidates
        for rank, res in enumerate(dense_results):
            chunk_id = res["chunk_id"]
            
            if chunk_id not in rrf_scores:
                # Reconstruct Chunk properties from database return elements if missing
                # If it wasn't caught by sparse, we instantiate its placement here
                meta = res["metadata"]
                metadata = ChunkMetadata(
                    source_path=meta["source_path"],
                    file_type=meta["file_type"],
                    chunk_index=meta["chunk_index"],
                    parent_document_id=meta["parent_document_id"]
                )
                chunk = Chunk(id=chunk_id, page_content=res["text"], metadata=metadata)
                
                rrf_scores[chunk_id] = {"chunk": chunk, "score": 0.0, "sources": ["dense"]}
            else:
                rrf_scores[chunk_id]["sources"].append("dense")

            # Accumulate the rank inversion value onto the identity tracking node
            rrf_scores[chunk_id]["score"] += 1.0 / (rrf_k + (rank + 1))

        # Step 4: Flatten database dictionaries out and execute final rank ordering configurations
        fused_results = list(rrf_scores.values())
        fused_results.sort(key=lambda x: x["score"], reverse=True)

        return fused_results[:top_k]