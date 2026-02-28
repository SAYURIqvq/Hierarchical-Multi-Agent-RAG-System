"""
ChromaDB persistent vector storage.
Replaces in-memory storage with persistent database.
"""

import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from src.ingestion.hierarchical_chunker import Chunk


class ChromaVectorStore:
    """
    Persistent vector storage using ChromaDB.
    
    Features:
    - Persistent storage (survives restarts)
    - Efficient similarity search
    - Metadata filtering
    - Separate collections for parents and children
    """
    
    def __init__(self, persist_directory: str = "data/chroma_db"):
        """
        Initialize ChromaDB vector store.
        
        Args:
            persist_directory: Directory for ChromaDB storage
        """
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Create collections
        self._init_collections()
        
        print(f"💾 ChromaDB initialized: {persist_directory}")
        print(f"   Collections: parent_chunks, child_chunks")
    
    def _init_collections(self):
        """Initialize or get existing collections."""
        
        # Parent chunks collection
        try:
            self.parent_collection = self.client.get_collection("parent_chunks")
            print(f"   ✅ Loaded existing parent collection ({self.parent_collection.count()} vectors)")
        except:
            self.parent_collection = self.client.create_collection(
                name="parent_chunks",
                metadata={"description": "Parent chunks for hierarchical retrieval"}
            )
            print(f"   ✅ Created new parent collection")
        
        # Child chunks collection
        try:
            self.child_collection = self.client.get_collection("child_chunks")
            print(f"   ✅ Loaded existing child collection ({self.child_collection.count()} vectors)")
        except:
            self.child_collection = self.client.create_collection(
                name="child_chunks",
                metadata={"description": "Child chunks for precise search"}
            )
            print(f"   ✅ Created new child collection")
    
    def add_chunks(
        self,
        parent_chunks: List[Chunk],
        child_chunks: List[Chunk],
        filename: str = "unknown"  # ← ADD PARAMETER
    ) -> None:
        """Add chunks with filename metadata."""
        
        print(f"\n💾 Adding chunks to ChromaDB...")
        
        # Add parents
        if parent_chunks:
            parent_metadatas = [
                {
                    "chunk_type": "parent",
                    "token_count": p.token_count,
                    "start_idx": p.start_idx,
                    "end_idx": p.end_idx,
                    "filename": filename  # ← ADD THIS
                }
                for p in parent_chunks
            ]
            
            self.parent_collection.add(
                ids=[p.chunk_id for p in parent_chunks],
                embeddings=[p.embedding for p in parent_chunks],
                documents=[p.text for p in parent_chunks],
                metadatas=parent_metadatas
            )
        
        # Add children
        if child_chunks:
            child_metadatas = [
                {
                    "chunk_type": "child",
                    "parent_id": c.parent_id if c.parent_id else "",
                    "token_count": c.token_count,
                    "start_idx": c.start_idx,
                    "end_idx": c.end_idx,
                    "filename": filename  # ← ADD THIS
                }
                for c in child_chunks
            ]
            
            self.child_collection.add(
                ids=[c.chunk_id for c in child_chunks],
                embeddings=[c.embedding for c in child_chunks],
                documents=[c.text for c in child_chunks],
                metadatas=child_metadatas
            )    
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
        print(f"\n🔍 Searching ChromaDB (top_k={top_k})...")
        
        # Search in child chunks
        results = self.child_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        if not results['ids'][0]:
            print("   ⚠️  No results found")
            return []
        
        print(f"   ✅ Found {len(results['ids'][0])} results")
        
        # Convert to our format
        formatted_results = []
        
        for i, chunk_id in enumerate(results['ids'][0]):
            child_text = results['documents'][0][i]
            child_metadata = results['metadatas'][0][i]
            distance = results['distances'][0][i]
            
            # Convert distance to similarity score (ChromaDB uses L2 distance)
            # Lower distance = higher similarity
            # Normalize to 0-1 range
            similarity = 1 / (1 + distance)
            
            if return_parent and child_metadata.get('parent_id'):
                # Get parent chunk
                parent_id = child_metadata['parent_id']
                try:
                    parent_results = self.parent_collection.get(
                        ids=[parent_id],
                        include=["documents", "metadatas"]
                    )
                    
                    if parent_results['ids']:
                        parent_text = parent_results['documents'][0]
                        parent_metadata = parent_results['metadatas'][0]
                        
                        formatted_results.append({
                            'chunk_id': parent_id,
                            'text': parent_text,
                            'score': similarity,
                            'chunk_type': 'parent',
                            'child_chunk_id': chunk_id,
                            'child_text': child_text,
                            'metadata': parent_metadata
                        })
                    else:
                        # Parent not found, return child
                        formatted_results.append({
                            'chunk_id': chunk_id,
                            'text': child_text,
                            'score': similarity,
                            'chunk_type': 'child',
                            'metadata': child_metadata
                        })
                except:
                    # Error getting parent, return child
                    formatted_results.append({
                        'chunk_id': chunk_id,
                        'text': child_text,
                        'score': similarity,
                        'chunk_type': 'child',
                        'metadata': child_metadata
                    })
            else:
                # Return child chunks directly
                formatted_results.append({
                    'chunk_id': chunk_id,
                    'text': child_text,
                    'score': similarity,
                    'chunk_type': 'child',
                    'metadata': child_metadata
                })
            
            print(f"   {i+1}. {formatted_results[-1]['chunk_type']}: {formatted_results[-1]['chunk_id'][:30]} (score: {similarity:.4f})")
        
        return formatted_results
    
    def delete_document_chunks(self, document_id: str) -> None:
        """
        Delete all chunks for a document.
        
        Args:
            document_id: Document identifier
        """
        # This requires querying by metadata
        # ChromaDB doesn't support delete by metadata directly
        # So we need to get all chunks first, then delete
        
        # For now, we can delete by ID pattern
        # Better: Store document_id in metadata during add
        pass
    
    def clear_all(self) -> None:
        """Clear all collections (dangerous!)."""
        self.client.delete_collection("parent_chunks")
        self.client.delete_collection("child_chunks")
        self._init_collections()
        print("⚠️  All collections cleared")
    
    def get_stats(self) -> Dict[str, int]:
        """Get storage statistics."""
        return {
            'total_parents': self.parent_collection.count(),
            'total_children': self.child_collection.count(),
            'total_vectors': self.parent_collection.count() + self.child_collection.count()
        }