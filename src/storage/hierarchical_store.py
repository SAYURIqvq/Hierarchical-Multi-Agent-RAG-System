"""
Storage for hierarchical chunks.
Manages both parent and child chunks with relationships.
"""

from typing import List, Dict, Any
import numpy as np
from src.ingestion.hierarchical_chunker import Chunk


class HierarchicalVectorStore:
    """
    Vector store that handles parent-child chunk relationships.
    
    Strategy:
    - Store child chunks for search (fast, precise)
    - Store parent chunks for context (full information)
    - Maintain relationships for retrieval
    """
    
    def __init__(self):
        """Initialize hierarchical vector store."""
        self.child_chunks = []
        self.parent_chunks = []
        self.parent_map = {}  # parent_id -> parent_chunk
        print("💾 HierarchicalVectorStore initialized")
    
    def add_chunks(
        self,
        parent_chunks: List[Chunk],
        child_chunks: List[Chunk]
    ) -> None:
        """
        Add parent and child chunks to storage.
        
        Args:
            parent_chunks: List of parent chunks with embeddings
            child_chunks: List of child chunks with embeddings
        """
        self.parent_chunks.extend(parent_chunks)
        self.child_chunks.extend(child_chunks)
        
        # Build parent map for quick lookup
        for parent in parent_chunks:
            self.parent_map[parent.chunk_id] = parent
        
        print(f"✅ Added {len(parent_chunks)} parents, {len(child_chunks)} children")
        print(f"   Total in store: {len(self.parent_chunks)} parents, {len(self.child_chunks)} children")
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        return_parent: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results
            return_parent: If True, return parent chunks for context
            
        Returns:
            List of chunks with scores
        """
        if not self.child_chunks:
            print("⚠️  No chunks in storage")
            return []
        
        print(f"\n🔍 Searching in {len(self.child_chunks)} child chunks...")
        
        # Search in CHILD chunks (more granular, better precision)
        similarities = []
        for chunk in self.child_chunks:
            if not hasattr(chunk, 'embedding'):
                continue
            
            similarity = self._cosine_similarity(
                query_embedding,
                chunk.embedding
            )
            
            similarities.append({
                'child_chunk': chunk,
                'score': similarity
            })
        
        # Sort by score
        similarities.sort(key=lambda x: x['score'], reverse=True)
        top_results = similarities[:top_k]
        
        print(f"✅ Found {len(top_results)} results")
        
        # Return parent chunks for context if requested
        if return_parent:
            results = []
            for result in top_results:
                child = result['child_chunk']
                parent = self.parent_map.get(child.parent_id)
                
                results.append({
                    'chunk': parent if parent else child,  # Use parent for context
                    'child_chunk': child,  # Keep child for reference
                    'score': result['score'],
                    'chunk_type': 'parent' if parent else 'child',
                    'text': parent.text if parent else child.text
                })
                
                print(f"   {len(results)}. Child: {child.chunk_id} → Parent: {parent.chunk_id if parent else 'N/A'} (score: {result['score']:.4f})")
        else:
            # Return child chunks directly
            results = []
            for result in top_results:
                child = result['child_chunk']
                results.append({
                    'chunk': child,
                    'score': result['score'],
                    'chunk_type': 'child',
                    'text': child.text
                })
        
        return results
    
    def _cosine_similarity(
        self,
        vec1: List[float],
        vec2: List[float]
    ) -> float:
        """Calculate cosine similarity."""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        return dot_product / (norm1 * norm2)