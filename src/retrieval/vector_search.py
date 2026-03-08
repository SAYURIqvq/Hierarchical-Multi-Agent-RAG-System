"""
Vector Search Agent - Operational Level 3 Agent.

Performs semantic search using vector embeddings.
Part of retrieval swarm coordinated by RetrievalCoordinator.
"""

from typing import List, Optional
import numpy as np

from src.agents.base_agent import BaseAgent
from src.models.agent_state import AgentState, Chunk
from src.storage.chroma_store import ChromaVectorStore
from src.ingestion.embedder import EmbeddingGenerator
from src.utils.exceptions import AgentExecutionError


class VectorSearchAgent(BaseAgent):
    """
    Vector Search Agent - Semantic similarity search.
    
    Uses embeddings to find semantically similar chunks.
    Part of the retrieval swarm (Level 3 operational agent).
    
    Attributes:
        vector_store: ChromaDB vector store
        embedder: Embedding generator
        
    Example:
        >>> vector_agent = VectorSearchAgent(
        ...     vector_store=chroma_store,
        ...     embedder=embedder
        ... )
        >>> 
        >>> state = AgentState(query="What is Python?")
        >>> result = vector_agent.run(state)
        >>> print(len(result.chunks))  # Retrieved chunks
    """
    
    def __init__(
        self,
        vector_store: ChromaVectorStore,
        embedder: EmbeddingGenerator
    ):
        """
        Initialize Vector Search Agent.
        
        Args:
            vector_store: ChromaDB vector store instance
            embedder: Embedding generator instance
        """
        super().__init__(name="vector_search", version="1.0.0")
        
        self.vector_store = vector_store
        self.embedder = embedder
        
        self.log("Initialized with ChromaDB vector store", level="debug")
    
    def execute(self, state: AgentState) -> AgentState:
        """
        Execute vector search.
        
        Args:
            state: Current agent state with query
        
        Returns:
            Updated state with chunks
        
        Raises:
            AgentExecutionError: If search fails
        """
        try:
            query = state.query
            
            self.log(f"Vector search for: {query[:50]}...", level="info")
            
            # Generate query embedding
            query_embedding = self.embedder.generate_query_embedding(query)
            
            # Search vector store
            results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=10,
                return_parent=True
            )
            
            # Convert to Chunk objects
            chunks = []
            for result in results:
                chunk = Chunk(
                    text=result['text'],
                    doc_id='unknown',
                    chunk_id=result['chunk_id'],
                    score=result['score'],
                    metadata={
                        'filename': result.get('metadata', {}).get('filename', 'unknown'),
                        'chunk_type': result.get('chunk_type', 'parent'),
                        'source': 'vector',  # ← Mark source
                        **result.get('metadata', {})
                    }
                )
                chunks.append(chunk)
            
            state.chunks = chunks
            
            self.log(f"Vector search retrieved {len(chunks)} chunks", level="info")
            
            return state
            
        except Exception as e:
            self.log(f"Vector search failed: {str(e)}", level="error")
            raise AgentExecutionError(
                agent_name=self.name,
                message=f"Vector search failed: {str(e)}",
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
            # Generate embedding
            query_embedding = self.embedder.generate_query_embedding(query)
            
            # Search
            results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
                return_parent=True
            )
            
            # Convert to chunks
            chunks = []
            for result in results:
                chunk = Chunk(
                    text=result['text'],
                    doc_id='unknown',
                    chunk_id=result['chunk_id'],
                    score=result['score'],
                    metadata={
                        'filename': result.get('metadata', {}).get('filename', 'unknown'),
                        'chunk_type': result.get('chunk_type', 'parent'),
                        'source': 'vector',
                        **result.get('metadata', {})
                    }
                )
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            self.log(f"Async vector search failed: {str(e)}", level="error")
            return []