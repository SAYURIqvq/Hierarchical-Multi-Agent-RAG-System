"""
Unified Chunk Model - Single source of truth for all chunk representations.

Supports both flat and hierarchical chunking strategies.
Compatible with all ingestion, storage, and retrieval components.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import hashlib


@dataclass
class Chunk:
    """
    Unified chunk representation for the entire RAG system.
    
    Supports:
    - Flat chunking (simple text splitting)
    - Hierarchical chunking (parent-child relationships)
    - Vector embeddings
    - Metadata tracking
    - Retrieval scoring
    
    Attributes:
        chunk_id: Unique chunk identifier
        text: Chunk text content
        doc_id: Source document ID
        
        # Hierarchical support
        parent_id: Parent chunk ID (None if this is parent)
        children_ids: List of child chunk IDs (empty if this is child)
        
        # Tokenization
        tokens: Token IDs from tokenizer
        token_count: Number of tokens in chunk
        
        # Position in document
        start_idx: Start position (character or token index)
        end_idx: End position (character or token index)
        start_char: Start character position (for compatibility)
        end_char: End character position (for compatibility)
        
        # Type
        chunk_type: 'parent' or 'child'
        
        # Embeddings
        embedding: Vector embedding (from Voyage AI or other)
        
        # Metadata
        metadata: Additional metadata (filename, page, etc.)
        
        # Retrieval
        score: Relevance score from retrieval (0.0-1.0)
    
    Example:
        >>> # Create a simple chunk
        >>> chunk = Chunk(
        ...     chunk_id="chunk_001",
        ...     text="Python is a programming language.",
        ...     doc_id="doc_123",
        ...     chunk_type="child"
        ... )
        
        >>> # Create hierarchical chunk
        >>> parent = Chunk(
        ...     chunk_id="parent_001",
        ...     text="Long parent text...",
        ...     doc_id="doc_123",
        ...     chunk_type="parent",
        ...     children_ids=["child_001", "child_002"]
        ... )
        
        >>> child = Chunk(
        ...     chunk_id="child_001",
        ...     text="Child text...",
        ...     doc_id="doc_123",
        ...     chunk_type="child",
        ...     parent_id="parent_001"
        ... )
    """
    
    # Required fields
    chunk_id: str
    text: str
    doc_id: str = "unknown"
    
    # Hierarchical relationships
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    
    # Tokenization
    tokens: List[int] = field(default_factory=list)
    token_count: int = 0
    
    # Position tracking (support both character and token indices)
    start_idx: int = 0
    end_idx: int = 0
    start_char: int = 0  # For compatibility with chunker.py
    end_char: int = 0    # For compatibility with chunker.py
    
    # Chunk classification
    chunk_type: str = "child"  # 'parent' or 'child'
    
    # Embeddings
    embedding: Optional[List[float]] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Retrieval scoring
    score: float = 0.0
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Ensure metadata is initialized
        if self.metadata is None:
            self.metadata = {}
        
        # Ensure children_ids is a list
        if self.children_ids is None:
            self.children_ids = []
        
        # Auto-calculate token_count if tokens provided but count is 0
        if self.tokens and self.token_count == 0:
            self.token_count = len(self.tokens)
        
        # Sync position indices if only one set provided
        if self.start_char == 0 and self.start_idx > 0:
            self.start_char = self.start_idx
        if self.end_char == 0 and self.end_idx > 0:
            self.end_char = self.end_idx
        
        if self.start_idx == 0 and self.start_char > 0:
            self.start_idx = self.start_char
        if self.end_idx == 0 and self.end_char > 0:
            self.end_idx = self.end_char
    
    def __len__(self) -> int:
        """Get chunk text length in characters."""
        return len(self.text)
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"Chunk(id={self.chunk_id}, type={self.chunk_type}, "
            f"tokens={self.token_count}, score={self.score:.3f})"
        )
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return f"[{self.chunk_type.upper()}] {preview}"
    
    # ========== HELPER METHODS ==========
    
    def is_parent(self) -> bool:
        """Check if this is a parent chunk."""
        return self.chunk_type == "parent" or self.parent_id is None
    
    def is_child(self) -> bool:
        """Check if this is a child chunk."""
        return self.chunk_type == "child" or self.parent_id is not None
    
    def has_embedding(self) -> bool:
        """Check if chunk has an embedding."""
        return self.embedding is not None and len(self.embedding) > 0
    
    def has_children(self) -> bool:
        """Check if chunk has children."""
        return len(self.children_ids) > 0
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension."""
        if not self.has_embedding():
            return 0
        return len(self.embedding)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert chunk to dictionary.
        
        Returns:
            Dictionary representation of chunk
        
        Example:
            >>> chunk_dict = chunk.to_dict()
            >>> print(chunk_dict['chunk_id'])
        """
        return {
            'chunk_id': self.chunk_id,
            'text': self.text,
            'doc_id': self.doc_id,
            'parent_id': self.parent_id,
            'children_ids': self.children_ids,
            'tokens': self.tokens,
            'token_count': self.token_count,
            'start_idx': self.start_idx,
            'end_idx': self.end_idx,
            'start_char': self.start_char,
            'end_char': self.end_char,
            'chunk_type': self.chunk_type,
            'embedding': self.embedding,
            'metadata': self.metadata,
            'score': self.score
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Chunk':
        """
        Create chunk from dictionary.
        
        Args:
            data: Dictionary with chunk data
        
        Returns:
            Chunk instance
        
        Example:
            >>> chunk = Chunk.from_dict(chunk_dict)
        """
        return cls(**data)
    
    def clone(self) -> 'Chunk':
        """
        Create a deep copy of the chunk.
        
        Returns:
            New Chunk instance with same data
        """
        return Chunk.from_dict(self.to_dict())


# ========== HELPER FUNCTIONS ==========

def generate_chunk_id(doc_id: str, chunk_key: str) -> str:
    """
    Generate unique chunk ID using MD5 hash.
    
    Args:
        doc_id: Document ID
        chunk_key: Unique key for this chunk
    
    Returns:
        Chunk ID (MD5 hash)
    
    Example:
        >>> chunk_id = generate_chunk_id("doc_123", "parent_0")
        >>> print(chunk_id)  # '5d41402abc4b2a76b9719d911017c592'
    """
    combined = f"{doc_id}_{chunk_key}"
    return hashlib.md5(combined.encode()).hexdigest()


def filter_by_type(chunks: List[Chunk], chunk_type: str) -> List[Chunk]:
    """
    Filter chunks by type.
    
    Args:
        chunks: List of chunks
        chunk_type: 'parent' or 'child'
    
    Returns:
        Filtered list of chunks
    
    Example:
        >>> parent_chunks = filter_by_type(all_chunks, 'parent')
        >>> child_chunks = filter_by_type(all_chunks, 'child')
    """
    return [c for c in chunks if c.chunk_type == chunk_type]


def get_parent_chunks(chunks: List[Chunk]) -> List[Chunk]:
    """
    Get only parent chunks.
    
    Args:
        chunks: List of all chunks
    
    Returns:
        List of parent chunks
    """
    return filter_by_type(chunks, 'parent')


def get_child_chunks(chunks: List[Chunk]) -> List[Chunk]:
    """
    Get only child chunks.
    
    Args:
        chunks: List of all chunks
    
    Returns:
        List of child chunks
    """
    return filter_by_type(chunks, 'child')


def find_chunk_by_id(chunks: List[Chunk], chunk_id: str) -> Optional[Chunk]:
    """
    Find chunk by ID.
    
    Args:
        chunks: List of chunks
        chunk_id: Chunk ID to find
    
    Returns:
        Chunk if found, None otherwise
    
    Example:
        >>> chunk = find_chunk_by_id(all_chunks, "chunk_001")
    """
    return next((c for c in chunks if c.chunk_id == chunk_id), None)


def get_chunk_with_parent(
    chunk_id: str,
    all_chunks: List[Chunk]
) -> Dict[str, Optional[Chunk]]:
    """
    Get chunk and its parent for context.
    
    Args:
        chunk_id: ID of child chunk
        all_chunks: List of all chunks
    
    Returns:
        Dictionary with 'chunk' and 'parent' keys
    
    Example:
        >>> result = get_chunk_with_parent("child_001", all_chunks)
        >>> print(result['chunk'].text)  # Child text
        >>> print(result['parent'].text)  # Parent context
    """
    chunk = find_chunk_by_id(all_chunks, chunk_id)
    
    if not chunk:
        return {"chunk": None, "parent": None}
    
    parent = None
    if chunk.parent_id:
        parent = find_chunk_by_id(all_chunks, chunk.parent_id)
    
    return {"chunk": chunk, "parent": parent}