"""
Keyword Search Agent - Operational Level 3 Agent.

Performs keyword-based search using BM25 algorithm.
Part of retrieval swarm coordinated by RetrievalCoordinator.
"""

from typing import List, Optional
from pathlib import Path

from src.agents.base_agent import BaseAgent
from src.models.agent_state import AgentState, Chunk
from src.retrieval.bm25_index import BM25Index
from src.storage.chroma_store import ChromaVectorStore
from src.utils.exceptions import AgentExecutionError


class KeywordSearchAgent(BaseAgent):
    """
    Keyword Search Agent - BM25-based retrieval.
    
    Uses BM25 algorithm for keyword matching.
    Part of the retrieval swarm (Level 3 operational agent).
    
    Attributes:
        bm25_index: BM25 inverted index
        vector_store: ChromaDB store (for building index)
        
    Example:
        >>> keyword_agent = KeywordSearchAgent(
        ...     vector_store=chroma_store
        ... )
        >>> 
        >>> state = AgentState(query="Python programming")
        >>> result = keyword_agent.run(state)
        >>> print(len(result.chunks))  # Retrieved chunks
    """
    
    def __init__(
        self,
        vector_store: ChromaVectorStore,
        index_path: str = "data/bm25_index.pkl"
    ):
        """
        Initialize Keyword Search Agent.
        
        Args:
            vector_store: ChromaDB vector store (for building index)
            index_path: Path to BM25 index file
        """
        super().__init__(name="keyword_search", version="1.0.0")
        
        self.vector_store = vector_store
        self.bm25_index = BM25Index(index_path=index_path)
        
        # Build index if not exists
        if not self.bm25_index.bm25:
            self.log("Building BM25 index...", level="info")
            try:
                self.bm25_index.build_from_vector_store(vector_store)
                self.bm25_index.save()
                self.log("BM25 index built and saved", level="info")
            except Exception as e:
                self.log(f"Failed to build BM25 index: {e}", level="warning")
        
        self.log("Initialized with BM25 index", level="debug")
    
    def execute(self, state: AgentState) -> AgentState:
        """
        Execute keyword search.
        
        Args:
            state: Current agent state with query
        
        Returns:
            Updated state with chunks
        
        Raises:
            AgentExecutionError: If search fails
        """
        try:
            query = state.query
            
            self.log(f"Keyword search for: {query[:50]}...", level="info")
            
            # Check if index exists
            if not self.bm25_index.bm25:
                self.log("BM25 index not available, returning empty", level="warning")
                state.chunks = []
                return state
            
            # Search BM25 index
            results = self.bm25_index.search(query, top_k=10)
            
            # Convert to Chunk objects
            chunks = []
            for result in results:
                chunk = Chunk(
                    text=result['text'],
                    doc_id='unknown',
                    chunk_id=result['chunk_id'],
                    score=result['score'],
                    metadata={
                        'filename': result['metadata'].get('filename', 'unknown'),
                        'chunk_type': result['metadata'].get('chunk_type', 'parent'),
                        'source': 'keyword',  # ← Mark source
                        **result['metadata']
                    }
                )
                chunks.append(chunk)
            
            state.chunks = chunks
            
            self.log(f"Keyword search retrieved {len(chunks)} chunks", level="info")
            
            return state
            
        except Exception as e:
            self.log(f"Keyword search failed: {str(e)}", level="error")
            raise AgentExecutionError(
                agent_name=self.name,
                message=f"Keyword search failed: {str(e)}",
                details={"query": state.query}
            ) from e
    
    def search_async(self, query: str, top_k: int = 10) -> List[Chunk]:
        """
        Async-compatible search method for swarm coordination.
        
        Args:
            query: Search query
            top_k: Number of chunks to return
        
        Returns:
            List of Chunk objects
        """
        try:
            # Check index
            if not self.bm25_index.bm25:
                self.log("BM25 index not available", level="warning")
                return []
            
            # Search
            results = self.bm25_index.search(query, top_k=top_k)
            
            # Convert to chunks
            chunks = []
            for result in results:
                chunk = Chunk(
                    text=result['text'],
                    doc_id='unknown',
                    chunk_id=result['chunk_id'],
                    score=result['score'],
                    metadata={
                        'filename': result['metadata'].get('filename', 'unknown'),
                        'chunk_type': result['metadata'].get('chunk_type', 'parent'),
                        'source': 'keyword',
                        **result['metadata']
                    }
                )
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            self.log(f"Async keyword search failed: {str(e)}", level="error")
            return []
    
    def rebuild_index(self) -> None:
        """
        Rebuild BM25 index from vector store.
        
        Useful when documents are added/removed.
        
        Example:
            >>> keyword_agent.rebuild_index()
        """
        self.log("Rebuilding BM25 index...", level="info")
        
        try:
            self.bm25_index.rebuild()
            self.log("✅ BM25 index rebuilt", level="info")
        except Exception as e:
            self.log(f"Failed to rebuild index: {e}", level="error")
            raise AgentExecutionError(
                agent_name=self.name,
                message=f"Index rebuild failed: {str(e)}",
                details={}
            ) from e