import re
from typing import List
from src.ingestion.schemas import Document, Chunk, ChunkMetadata

class ChunkingEngine:
    """The engineering matrix that fractionalizes raw text maps into optimized token contexts."""

    @staticmethod
    def fixed_size_chunk(document: Document, chunk_size: int = 500, chunk_overlap: int = 100) -> List[Chunk]:
        """Splits data strictly by character size windows with sliding overlaps."""
        text = document.page_content
        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            # Define end point boundary
            end = start + chunk_size
            slice_text = text[start:end]
            
            # Formulate metadata trace bindings
            metadata = ChunkMetadata(
                source_path=document.metadata.source_path,
                file_type=document.metadata.file_type,
                chunk_index=chunk_index,
                parent_document_id=document.id
            )
            
            # Compute static linear hash tracking key
            deterministic_id = f"{document.id}-chunk-{chunk_index}"
            
            chunks.append(Chunk(id=deterministic_id, page_content=slice_text, metadata=metadata))
            
            # Progress window forward subtracting overlap configuration
            start += (chunk_size - chunk_overlap)
            chunk_index += 1
            
        return chunks

    @staticmethod
    def structure_aware_markdown_chunk(document: Document) -> List[Chunk]:
        """Segments text files dynamically using markdown section boundaries."""
        text = document.page_content
        # Split text wherever a Markdown Header level 1, 2, or 3 appears
        sections = re.split(r'(^#+\s+.*$)', text, flags=re.MULTILINE)
        
        chunks = []
        chunk_index = 0
        current_header = "Introduction"
        
        for section in sections:
            if not section.strip():
                continue
                
            # Track current section context headers
            if section.startswith("#"):
                current_header = section.strip()
                continue
                
            # Glue context header text onto sub-paragraph items for retention
            contextualized_content = f"{current_header}\n\n{section.strip()}"
            
            metadata = ChunkMetadata(
                source_path=document.metadata.source_path,
                file_type=document.metadata.file_type,
                chunk_index=chunk_index,
                parent_document_id=document.id,
                custom_attributes={"markdown_header": current_header}
            )
            
            chunks.append(Chunk(
                id=f"{document.id}-markdown-{chunk_index}",
                page_content=contextualized_content,
                metadata=metadata
            ))
            chunk_index += 1
            
        return chunks