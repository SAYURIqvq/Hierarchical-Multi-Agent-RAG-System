"""
Vector Search Agent - Operational Level 3 Agent.

Performs semantic search using vector embeddings.
Uses ChromaDB for storage and Voyage AI for embeddings.
"""

from typing import List, Optional

from src.agents.base_agent import BaseAgent
from src.models.agent_state import AgentState, Chunk
from src.storage.vector_store import VectorStore, VectorStoreError
from src.ingestion.embedder import EmbeddingGenerator, EmbeddingError
from src.utils.exceptions import RetrievalError
from src.config import get_settings


class VectorSearchAgent(BaseAgent):
    """
    Vector Search Agent - Semantic retrieval using ChromaDB.
    
    Performs semantic search using vector embeddings to find
    contextually similar chunks from ChromaDB.
    
    Attributes:
        top_k: Number of chunks to retrieve
        vector_store: ChromaDB vector store instance
        embedder: Voyage AI embedding generator
        mock_mode: If True, returns dummy chunks (for testing)
        
    Example:
        >>> # Real mode
        >>> agent = VectorSearchAgent(top_k=10, mock_mode=False)
        >>> state = AgentState(query="What is Python?")
        >>> result = agent.run(state)
        >>> print(len(result.chunks))  # 10
        
        >>> # Mock mode (for testing)
        >>> agent = VectorSearchAgent(top_k=5, mock_mode=True)
    """
    
    def __init__(
        self,
        top_k: int = None,
        mock_mode: bool = True,
        vector_store: VectorStore = None,
        embedder: EmbeddingGenerator = None
    ):
        """
        Initialize Vector Search Agent.
        
        Args:
            top_k: Number of chunks to retrieve (default from config)
            mock_mode: Use mock data (default: True for backward compatibility)
            vector_store: VectorStore instance (created if None)
            embedder: EmbeddingGenerator instance (created if None)
        
        Example:
            >>> # Real mode with auto-initialization
            >>> agent = VectorSearchAgent(top_k=10, mock_mode=False)
            
            >>> # Real mode with custom instances
            >>> store = VectorStore()
            >>> embedder = EmbeddingGenerator()
            >>> agent = VectorSearchAgent(
            ...     vector_store=store,
            ...     embedder=embedder,
            ...     mock_mode=False
            ... )
        """
        super().__init__(name="vector_search", version="2.0.0")
        
        settings = get_settings()
        self.top_k = top_k or settings.vector_search_top_k
        self.mock_mode = mock_mode
        
        # Initialize vector store and embedder (only if not mock mode)
        if not mock_mode:
            self.vector_store = vector_store or VectorStore()
            self.embedder = embedder or EmbeddingGenerator()
            
            self.log(
                f"Initialized in REAL mode: "
                f"top_k={self.top_k}, "
                f"collection={self.vector_store.collection_name}",
                level="info"
            )
        else:
            self.vector_store = None
            self.embedder = None
            
            self.log(
                f"Initialized in MOCK mode with top_k={self.top_k}",
                level="debug"
            )
    
    def execute(self, state: AgentState) -> AgentState:
        """
        Execute vector search.
        
        Args:
            state: Current agent state with query
        
        Returns:
            Updated state with retrieved chunks
        
        Raises:
            RetrievalError: If search fails
        
        Example:
            >>> state = AgentState(query="machine learning basics")
            >>> result = agent.execute(state)
            >>> print(result.chunks[0].text)
        """
        try:
            query = state.query
            
            self.log(f"Performing vector search for: {query[:50]}...", level="info")
            
            if self.mock_mode:
                chunks = self._mock_search(query)
            else:
                chunks = self._real_search(query)
            
            self.log(f"Retrieved {len(chunks)} chunks via vector search", level="info")
            
            # Update state
            state.chunks = chunks
            
            # Add metadata
            for chunk in chunks:
                chunk.metadata["source"] = "vector"
                chunk.metadata["method"] = "semantic_search"
            
            return state
            
        except Exception as e:
            self.log(f"Vector search failed: {str(e)}", level="error")
            raise RetrievalError(
                retrieval_type="vector",
                message=f"Vector search failed: {str(e)}",
                details={"query": state.query}
            ) from e
    
    def _real_search(self, query: str) -> List[Chunk]:
        """
        Real vector search using ChromaDB and Voyage AI.
        
        Args:
            query: User query string
        
        Returns:
            List of chunks from vector database
        
        Raises:
            EmbeddingError: If embedding generation fails
            VectorStoreError: If search fails
        """
        try:
            # Step 1: Generate query embedding
            self.log("Generating query embedding...", level="debug")
            query_embedding = self.embedder.generate_query_embedding(query)
            
            # Step 2: Search in vector store
            self.log(f"Searching vector store (top_k={self.top_k})...", level="debug")
            search_results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=self.top_k
            )
            
            # Step 3: Convert to Chunk objects
            chunks = []
            for result in search_results:
                chunk = Chunk(
                    text=result['text'],
                    doc_id=result['metadata'].get('doc_id', 'unknown'),
                    chunk_id=result['chunk_id'],
                    score=result['score'],
                    metadata=result['metadata']
                )
                chunks.append(chunk)
            
            return chunks
            
        except EmbeddingError as e:
            self.log(f"Embedding generation failed: {str(e)}", level="error")
            raise RetrievalError(
                retrieval_type="vector",
                message=f"Failed to generate query embedding: {str(e)}",
                details={"query": query}
            ) from e
            
        except VectorStoreError as e:
            self.log(f"Vector store search failed: {str(e)}", level="error")
            raise RetrievalError(
                retrieval_type="vector",
                message=f"Vector store search failed: {str(e)}",
                details={"query": query}
            ) from e
    
    def _mock_search(self, query: str) -> List[Chunk]:
        """
        Mock vector search returning dummy chunks.
        
        Generates realistic-looking chunks for testing.
        
        Args:
            query: User query string
        
        Returns:
            List of mock chunks
        """
        chunks = []
        
        # Generate mock chunks based on query
        query_lower = query.lower()
        
        # Mock chunk templates
        if "python" in query_lower:
            templates = [
                "Python is a high-level, interpreted programming language known for its simplicity and readability.",
                "Python was created by Guido van Rossum and first released in 1991.",
                "Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
                "Python has a comprehensive standard library and extensive ecosystem of third-party packages.",
                "Python is widely used in web development, data science, machine learning, and automation.",
            ]
        elif "machine learning" in query_lower or "ml" in query_lower:
            templates = [
                "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
                "Supervised learning uses labeled data to train models for prediction tasks.",
                "Unsupervised learning finds patterns in unlabeled data through clustering and dimensionality reduction.",
                "Neural networks are inspired by biological neurons and form the basis of deep learning.",
                "Common ML algorithms include decision trees, random forests, SVMs, and neural networks.",
            ]
        else:
            templates = [
                f"This is relevant information about {query} from document A.",
                f"Key concepts related to {query} include various important aspects.",
                f"Research shows that {query} has significant implications in the field.",
                f"Experts agree that understanding {query} requires comprehensive knowledge.",
                f"Recent developments in {query} have led to new insights and applications.",
            ]
        
        # Create chunks with decreasing scores
        for i in range(min(self.top_k, len(templates))):
            chunk = Chunk(
                text=templates[i],
                doc_id=f"mock_doc_{i // 2}",  # Multiple chunks per doc
                chunk_id=f"vector_chunk_{i}",
                score=0.9 - (i * 0.05),  # Decreasing scores
                metadata={}
            )
            chunks.append(chunk)
        
        # If need more chunks, generate generic ones
        while len(chunks) < self.top_k:
            i = len(chunks)
            chunk = Chunk(
                text=f"Additional context about {query} (chunk {i}).",
                doc_id=f"mock_doc_{i // 2}",
                chunk_id=f"vector_chunk_{i}",
                score=max(0.5, 0.9 - (i * 0.05)),
                metadata={}
            )
            chunks.append(chunk)
        
        return chunks