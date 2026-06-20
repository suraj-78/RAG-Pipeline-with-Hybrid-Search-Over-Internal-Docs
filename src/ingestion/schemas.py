from pydantic import BaseModel, Field, HttpUrl
from typing import Dict, Any, Optional
from datetime import datetime
import uuid

class DocumentMetadata(BaseModel):
    """Strict schema enforcement for source document context metadata tracking."""
    source_path: str = Field(..., description="Absolute path or unique identifier of the source file.")
    file_type: str = Field(..., description="The extension format of the parsed asset (e.g., pdf, md, html).")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Ingestion validation timestamp.")
    custom_attributes: Dict[str, Any] = Field(default_factory=dict, description="Extensible dictionary for vendor-specific metrics.")

class Document(BaseModel):
    """The master data transfer container representing a raw ingestion resource."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Primary lookup key identifier.")
    page_content: str = Field(..., min_length=1, description="The structural plaintext extraction payload.")
    metadata: DocumentMetadata = Field(..., description="The context object verified for systemic tracking lineage.")

class ChunkMetadata(DocumentMetadata):
    """Extended tracking metadata properties specific to down-stream token chunks."""
    chunk_index: int = Field(..., ge=0, description="The sequential positional index sequence inside the parent asset.")
    parent_document_id: str = Field(..., description="Foreign key mapping connection tracking back to the Document ID.")

class Chunk(BaseModel):
    """The granular computational split object optimized for dense and sparse indexing passes."""
    id: str = Field(..., description="Deterministic or unique hash string identifying this specific token slice.")
    page_content: str = Field(..., min_length=1, description="The distinct contextual slice of text data.")
    metadata: ChunkMetadata = Field(..., description="Linage tracking parameters for citation reconstruction loops.")