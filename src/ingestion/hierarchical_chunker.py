"""
Hierarchical Chunking System
Parent chunks: 2000 tokens (full context)
Child chunks: 500 tokens (searchable units)

This provides flexibility:
- Search in child chunks (fast, precise)
- Retrieve parent chunks (full context)
"""

from src.models.chunk import Chunk
from typing import List, Dict, Any, Tuple
import tiktoken

class HierarchicalChunker:
    """
    Create hierarchical chunks with parent-child relationships.
    
    Strategy:
    1. Create large parent chunks (2000 tokens)
    2. Split each parent into smaller child chunks (500 tokens)
    3. Maintain relationships between parents and children
    """
    
    def __init__(
        self,
        parent_size: int = 2000,
        child_size: int = 500,
        child_overlap: int = 50,
        encoding_name: str = "cl100k_base"
    ):
        """
        Initialize hierarchical chunker.
        
        Args:
            parent_size: Parent chunk size in tokens
            child_size: Child chunk size in tokens
            child_overlap: Overlap between child chunks
            encoding_name: Tokenizer encoding
        """
        self.parent_size = parent_size
        self.child_size = child_size
        self.child_overlap = child_overlap
        self.encoding = tiktoken.get_encoding(encoding_name)
    
    def chunk_text(
        self, 
        text: str,
        doc_id: str = "unknown",  # ← ADD
        metadata: Dict[str, Any] = None  # ← ADD
    ) -> Tuple[List[Chunk], List[Chunk]]:
        """
        Create hierarchical chunks from text.
        
        Args:
            text: Input text to chunk
            
        Returns:
            Tuple of (parent_chunks, child_chunks)
        """
        print(f"\n📝 Creating hierarchical chunks...")
        print(f"   Parent size: {self.parent_size} tokens")
        print(f"   Child size: {self.child_size} tokens")
        print(f"   Child overlap: {self.child_overlap} tokens")
        
        # Tokenize entire text
        tokens = self.encoding.encode(text)
        total_tokens = len(tokens)
        
        print(f"   Total tokens: {total_tokens:,}")
        
        parent_chunks = []
        child_chunks = []
        
        # Create parent chunks
        parent_num = 0
        start_idx = 0
        
        while start_idx < total_tokens:
            # Parent chunk boundaries
            end_idx = min(start_idx + self.parent_size, total_tokens)
            parent_tokens = tokens[start_idx:end_idx]
            parent_text = self.encoding.decode(parent_tokens)
            
            parent_id = f"parent_{parent_num}"
            
            # Create parent chunk
            parent = Chunk(
                chunk_id=parent_id,
                text=parent_text,
                doc_id=doc_id,  # ← USE
                tokens=parent_tokens,
                token_count=len(parent_tokens),
                start_idx=start_idx,
                end_idx=end_idx,
                chunk_type='parent',
                children_ids=[],
                metadata=metadata or {}  # ← USE
            )
            
            # ✅ PASS doc_id and metadata to _create_children
            children = self._create_children(
                parent_tokens=parent_tokens,
                parent_id=parent_id,
                parent_start_idx=start_idx,
                doc_id=doc_id,              # ← ADD
                metadata=metadata           # ← ADD
            )

            
            # Link children to parent
            parent.children_ids = [child.chunk_id for child in children]
            
            parent_chunks.append(parent)
            child_chunks.extend(children)
            
            parent_num += 1
            start_idx = end_idx  # No overlap for parents
        
        print(f"✅ Created {len(parent_chunks)} parent chunks")
        print(f"✅ Created {len(child_chunks)} child chunks")
        print(f"   Average children per parent: {len(child_chunks)/len(parent_chunks):.1f}")
        
        return parent_chunks, child_chunks
    
    def _create_children(
        self,
        parent_tokens: List[int],
        parent_id: str,
        parent_start_idx: int,
        doc_id: str = "unknown",          # ← ADD
        metadata: Dict[str, Any] = None   # ← ADD
    ) -> List[Chunk]:
        """
        Create child chunks from a parent chunk.
        
        Args:
            parent_tokens: Parent's token list
            parent_id: Parent chunk ID
            parent_start_idx: Parent's start index in full text
            
        Returns:
            List of child chunks
        """
        children = []
        child_num = 0
        start_idx = 0
        
        while start_idx < len(parent_tokens):
            # Child chunk boundaries
            end_idx = min(start_idx + self.child_size, len(parent_tokens))
            child_tokens = parent_tokens[start_idx:end_idx]
            child_text = self.encoding.decode(child_tokens)
            
            child_id = f"{parent_id}_child_{child_num}"
            
            # Create child chunk
            child = Chunk(
                chunk_id=child_id,
                text=child_text,
                doc_id=doc_id,                      # ← USE doc_id
                tokens=child_tokens,
                token_count=len(child_tokens),
                start_idx=parent_start_idx + start_idx,
                end_idx=parent_start_idx + end_idx,
                chunk_type='child',
                parent_id=parent_id,
                metadata=metadata or {}              # ← USE metadata
            )
            
            children.append(child)
            child_num += 1
            
            # Move to next child with overlap
            start_idx += self.child_size - self.child_overlap
        
        return children
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens
        """
        return len(self.encoding.encode(text))
    
    def get_parent_context(
        self,
        child_chunk: Chunk,
        parent_chunks: List[Chunk]
    ) -> Chunk:
        """
        Get parent chunk for a child chunk.
        
        Args:
            child_chunk: Child chunk
            parent_chunks: List of all parent chunks
            
        Returns:
            Parent chunk or None
        """
        if child_chunk.chunk_type != 'child':
            return child_chunk
        
        for parent in parent_chunks:
            if parent.chunk_id == child_chunk.parent_id:
                return parent
        
        return None