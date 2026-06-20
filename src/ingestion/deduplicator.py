import numpy as np
from typing import List, Callable
from src.ingestion.schemas import Chunk

class ChunkDeduplicator:
    """The enterprise engine tasked with purging redundant semantic vectors from ingestion arrays."""

    @staticmethod
    def calculate_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Computes the angular distance between two dense numeric embeddings."""
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        return float(dot_product / (norm_a * norm_b))

    def deduplicate(
        self, 
        chunks: List[Chunk], 
        embedding_fn: Callable[[List[str]], List[List[float]]], 
        threshold: float = 0.95
    ) -> List[Chunk]:
        """Scans multi-chunk payloads and drops elements crossing the similarity threshold barrier."""
        if not chunks:
            return []

        # Step 1: Bulk extraction of textual body payloads for network efficiency
        texts = [chunk.page_content for chunk in chunks]
        
        # Step 2: Compute dense representations via the injected vector embedding function
        embeddings_list = embedding_fn(texts)
        embeddings = [np.array(vec) for vec in embeddings_list]

        unique_chunks = []
        seen_vectors: List[np.ndarray] = []

        # Step 3: Iterate and perform pairwise mathematical verification passes
        for idx, current_vector in enumerate(embeddings):
            is_duplicate = False
            
            for seen_vector in seen_vectors:
                similarity = self.calculate_cosine_similarity(current_vector, seen_vector)
                
                if similarity > threshold:
                    is_duplicate = True
                    break # Break early if redundancy is verified
            
            if not is_duplicate:
                unique_chunks.append(chunks[idx])
                seen_vectors.append(current_vector)

        return unique_chunks