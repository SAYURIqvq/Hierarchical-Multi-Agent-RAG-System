"""
Ingestion components for document processing.
"""

from src.ingestion.document_loader import DocumentLoader, Document, DocumentLoadError
from src.ingestion.hierarchical_chunker import HierarchicalChunker
from src.models.chunk import Chunk  # ← Import from unified location

from src.ingestion.embedder import (
    EmbeddingGenerator,
    CachedEmbeddingGenerator,
    EmbeddingError
)

__all__ = [
    # Document loading
    "DocumentLoader",
    "Document",
    "DocumentLoadError",
    
    # Chunking
    "HierarchicalChunker",  # ← CHANGE from DocumentChunker
    "Chunk",
    
    # Embedding
    "EmbeddingGenerator",
    "CachedEmbeddingGenerator",
    "EmbeddingError",
]