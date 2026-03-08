"""
BM25 Index - Build and search inverted index for keyword retrieval.

Uses rank-bm25 library for efficient BM25 scoring.
"""

from typing import List, Dict, Any, Optional
import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.models.chunk import Chunk
from src.storage.chroma_store import ChromaVectorStore

from src.utils.logger import setup_logger
from src.utils.exceptions import AgenticRAGException


class BM25IndexError(AgenticRAGException):
    """Error during BM25 index operations."""
    pass


class BM25Index:
    """
    BM25 inverted index for keyword search.
    
    Features:
    - Build index from ChromaDB chunks
    - Fast keyword-based retrieval
    - BM25 scoring (Okapi BM25 variant)
    - Persistent storage
    
    Example:
        >>> # Build index
        >>> index = BM25Index()
        >>> index.build_from_vector_store()
        
        >>> # Search
        >>> results = index.search("python programming", top_k=5)
        >>> for result in results:
        ...     print(result['chunk_id'], result['score'])
    """
    
    def __init__(self, index_path: str = "data/bm25_index.pkl"):
        """
        Initialize BM25 index.
        
        Args:
            index_path: Path to save/load index
        """
        self.logger = setup_logger("bm25_index")
        self.index_path = Path(index_path)
        
        self.bm25: Optional[BM25Okapi] = None
        self.chunk_ids: List[str] = []
        self.chunk_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Try to load existing index
        if self.index_path.exists():
            self.load()
    
    def build_from_vector_store(
        self,
        vector_store: Optional[ChromaVectorStore] = None  # ← CHANGE type
    ) -> None:
        """
        Build BM25 index from chunks in vector store.
        
        Args:
            vector_store: ChromaVectorStore instance (creates new if None)
        
        Raises:
            BM25IndexError: If index building fails
        
        Example:
            >>> index = BM25Index()
            >>> index.build_from_vector_store()
            >>> index.save()
        """
        self.logger.info("Building BM25 index from vector store...")
        
        try:
            # Get vector store
            if vector_store is None:
                vector_store = ChromaVectorStore()  # ← CHANGE
            
            # Get all chunks from ChromaDB
            total_chunks = vector_store.count()
            
            if total_chunks == 0:
                raise BM25IndexError(
                    message="No chunks in vector store to index",
                    details={"collection": vector_store.collection_name}
                )
            
            self.logger.info(f"Indexing {total_chunks} chunks...")
            
            # Fetch all chunks (ChromaDB doesn't have scan, so query with large limit)
            # This is a workaround - in production, use batch fetching
            results = vector_store.collection.get(
                include=["documents", "metadatas"],
                limit=total_chunks
            )
            
            if not results or not results['ids']:
                raise BM25IndexError(
                    message="Failed to fetch chunks from vector store",
                    details={}
                )
            
            # Prepare data for BM25
            documents = []
            chunk_ids = []
            chunk_metadata = {}
            
            for i, chunk_id in enumerate(results['ids']):
                text = results['documents'][i]
                metadata = results['metadatas'][i]
                
                # Tokenize (simple word splitting)
                tokens = self._tokenize(text)
                documents.append(tokens)
                
                chunk_ids.append(chunk_id)
                chunk_metadata[chunk_id] = {
                    'text': text,
                    'metadata': metadata
                }
            
            # Build BM25 index
            self.bm25 = BM25Okapi(documents)
            self.chunk_ids = chunk_ids
            self.chunk_metadata = chunk_metadata
            
            self.logger.info(
                f"✅ Built BM25 index with {len(chunk_ids)} chunks"
            )
            
        except BM25IndexError:
            raise
        except Exception as e:
            raise BM25IndexError(
                message=f"Failed to build BM25 index: {str(e)}",
                details={"error": str(e)}
            ) from e
    
    def search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search index with BM25 scoring.
        
        Args:
            query: Search query
            top_k: Number of results to return
        
        Returns:
            List of results with chunk_id, text, score, metadata
        
        Raises:
            BM25IndexError: If search fails
        
        Example:
            >>> results = index.search("machine learning", top_k=5)
            >>> for r in results:
            ...     print(f"{r['score']:.3f}: {r['text'][:50]}...")
        """
        if self.bm25 is None:
            raise BM25IndexError(
                message="BM25 index not built. Call build_from_vector_store() first.",
                details={}
            )
        
        try:
            # Tokenize query
            query_tokens = self._tokenize(query)
            
            # Get BM25 scores
            scores = self.bm25.get_scores(query_tokens)
            
            # Get top-k indices
            top_indices = scores.argsort()[::-1][:top_k]
            
            # Format results
            results = []
            for idx in top_indices:
                chunk_id = self.chunk_ids[idx]
                chunk_data = self.chunk_metadata[chunk_id]
                score = float(scores[idx])
                
                # Only return chunks with non-zero scores
                if score > 0:
                    results.append({
                        'chunk_id': chunk_id,
                        'text': chunk_data['text'],
                        'score': score,
                        'metadata': chunk_data['metadata']
                    })
            
            self.logger.debug(f"BM25 search returned {len(results)} results")
            
            return results
            
        except Exception as e:
            raise BM25IndexError(
                message=f"BM25 search failed: {str(e)}",
                details={"query": query}
            ) from e
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25.
        
        Simple word-based tokenization:
        - Lowercase
        - Split on whitespace
        - Remove punctuation
        
        Args:
            text: Text to tokenize
        
        Returns:
            List of tokens
        """
        # Lowercase
        text = text.lower()
        
        # Remove common punctuation (keep alphanumeric and spaces)
        import re
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Split and filter empty
        tokens = [t for t in text.split() if t]
        
        return tokens
    
    def save(self) -> None:
        """
        Save BM25 index to disk.
        
        Example:
            >>> index.build_from_vector_store()
            >>> index.save()
        """
        if self.bm25 is None:
            self.logger.warning("No index to save")
            return
        
        try:
            # Create directory if needed
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save index data
            data = {
                'bm25': self.bm25,
                'chunk_ids': self.chunk_ids,
                'chunk_metadata': self.chunk_metadata
            }
            
            with open(self.index_path, 'wb') as f:
                pickle.dump(data, f)
            
            self.logger.info(f"💾 Saved BM25 index to {self.index_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save index: {str(e)}")
            raise BM25IndexError(
                message=f"Failed to save index: {str(e)}",
                details={"path": str(self.index_path)}
            ) from e
    
    def load(self) -> bool:
        """
        Load BM25 index from disk.
        
        Returns:
            True if loaded successfully, False otherwise
        
        Example:
            >>> index = BM25Index()
            >>> if index.load():
            ...     results = index.search("query")
        """
        if not self.index_path.exists():
            self.logger.debug(f"Index file not found: {self.index_path}")
            return False
        
        try:
            with open(self.index_path, 'rb') as f:
                data = pickle.load(f)
            
            self.bm25 = data['bm25']
            self.chunk_ids = data['chunk_ids']
            self.chunk_metadata = data['chunk_metadata']
            
            self.logger.info(
                f"📂 Loaded BM25 index with {len(self.chunk_ids)} chunks"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load index: {str(e)}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get index statistics.
        
        Returns:
            Dictionary with index stats
        
        Example:
            >>> stats = index.get_stats()
            >>> print(stats['total_chunks'])
        """
        if self.bm25 is None:
            return {
                'built': False,
                'total_chunks': 0
            }
        
        return {
            'built': True,
            'total_chunks': len(self.chunk_ids),
            'index_path': str(self.index_path),
            'index_exists': self.index_path.exists()
        }
    
    def rebuild(self) -> None:
        """
        Rebuild index from vector store.
        
        Convenience method to rebuild and save index.
        
        Example:
            >>> index = BM25Index()
            >>> index.rebuild()  # Build and save in one step
        """
        self.build_from_vector_store()
        self.save()
        self.logger.info("✅ Index rebuilt and saved")