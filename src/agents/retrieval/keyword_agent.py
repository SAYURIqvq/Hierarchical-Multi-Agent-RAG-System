"""
Keyword Search Agent - Operational Level 3 Agent.

Performs keyword-based search using BM25 algorithm.
Uses BM25Index for fast inverted index search.
"""

from typing import List, Optional

from src.agents.base_agent import BaseAgent
from src.models.agent_state import AgentState, Chunk
from src.retrieval.bm25_index import BM25Index, BM25IndexError
from src.utils.exceptions import RetrievalError
from src.config import get_settings


class KeywordSearchAgent(BaseAgent):
    """
    Keyword Search Agent - BM25 retrieval.
    
    Performs keyword-based search using BM25 algorithm for
    exact term matching and relevance scoring.
    
    Attributes:
        top_k: Number of chunks to retrieve
        bm25_index: BM25 index instance
        mock_mode: If True, returns dummy chunks (for testing)
        
    Example:
        >>> # Real mode
        >>> agent = KeywordSearchAgent(top_k=10, mock_mode=False)
        >>> state = AgentState(query="Python programming")
        >>> result = agent.run(state)
        >>> print(len(result.chunks))  # 10
        
        >>> # Mock mode (for testing)
        >>> agent = KeywordSearchAgent(top_k=5, mock_mode=True)
    """
    
    def __init__(
        self,
        top_k: int = None,
        mock_mode: bool = True,
        bm25_index: BM25Index = None
    ):
        """
        Initialize Keyword Search Agent.
        
        Args:
            top_k: Number of chunks to retrieve (default from config)
            mock_mode: Use mock data (default: True for backward compatibility)
            bm25_index: BM25Index instance (created if None)
        
        Example:
            >>> # Real mode with auto-initialization
            >>> agent = KeywordSearchAgent(top_k=10, mock_mode=False)
            
            >>> # Real mode with custom index
            >>> index = BM25Index()
            >>> agent = KeywordSearchAgent(bm25_index=index, mock_mode=False)
        """
        super().__init__(name="keyword_search", version="2.0.0")
        
        settings = get_settings()
        self.top_k = top_k or settings.retrieval_top_k
        self.mock_mode = mock_mode
        
        # Initialize BM25 index (only if not mock mode)
        if not mock_mode:
            self.bm25_index = bm25_index or BM25Index()
            
            # Check if index is built
            stats = self.bm25_index.get_stats()
            if not stats['built']:
                self.log(
                    "BM25 index not built. Call build_index() or rebuild() first.",
                    level="warning"
                )
            
            self.log(
                f"Initialized in REAL mode: "
                f"top_k={self.top_k}, "
                f"indexed_chunks={stats.get('total_chunks', 0)}",
                level="info"
            )
        else:
            self.bm25_index = None
            
            self.log(
                f"Initialized in MOCK mode with top_k={self.top_k}",
                level="debug"
            )
    
    def execute(self, state: AgentState) -> AgentState:
        """
        Execute keyword search.
        
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
            
            self.log(f"Performing keyword search for: {query[:50]}...", level="info")
            
            if self.mock_mode:
                chunks = self._mock_search(query)
            else:
                chunks = self._real_search(query)
            
            self.log(f"Retrieved {len(chunks)} chunks via keyword search", level="info")
            
            # Update state
            state.chunks = chunks
            
            # Add metadata
            for chunk in chunks:
                chunk.metadata["source"] = "keyword"
                chunk.metadata["method"] = "bm25"
            
            return state
            
        except Exception as e:
            self.log(f"Keyword search failed: {str(e)}", level="error")
            raise RetrievalError(
                retrieval_type="keyword",
                message=f"Keyword search failed: {str(e)}",
                details={"query": state.query}
            ) from e
    
    def _real_search(self, query: str) -> List[Chunk]:
        """
        Real keyword search using BM25 index.
        
        Args:
            query: User query string
        
        Returns:
            List of chunks from BM25 search
        
        Raises:
            BM25IndexError: If search fails
        """
        try:
            # Check if index is built
            if self.bm25_index.bm25 is None:
                raise RetrievalError(
                    retrieval_type="keyword",
                    message="BM25 index not built. Run build_index() first.",
                    details={"query": query}
                )
            
            # Search with BM25
            self.log("Searching BM25 index...", level="debug")
            search_results = self.bm25_index.search(
                query=query,
                top_k=self.top_k
            )
            
            # Normalize scores to 0-1 range
            if search_results:
                max_score = max(r['score'] for r in search_results)
                if max_score > 0:
                    for result in search_results:
                        result['score'] = result['score'] / max_score
            
            # Convert to Chunk objects
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
            
        except BM25IndexError as e:
            self.log(f"BM25 index error: {str(e)}", level="error")
            raise RetrievalError(
                retrieval_type="keyword",
                message=f"BM25 search failed: {str(e)}",
                details={"query": query}
            ) from e
    
    def _mock_search(self, query: str) -> List[Chunk]:
        """
        Mock keyword search returning dummy chunks.
        
        Simulates BM25 by looking for exact keyword matches.
        
        Args:
            query: User query string
        
        Returns:
            List of mock chunks
        """
        chunks = []
        
        # Extract keywords from query
        keywords = query.lower().split()
        
        # Mock chunk templates with keyword emphasis
        query_lower = query.lower()
        
        if "python" in query_lower:
            templates = [
                "Python programming language documentation: Python is versatile and Python is easy to learn.",
                "The official Python website provides Python tutorials and Python resources for developers.",
                "Python software foundation maintains Python core development and Python community standards.",
                "Learn Python basics: Python syntax, Python data types, and Python functions.",
                "Python applications span from Python web frameworks to Python data analysis tools.",
            ]
        elif "machine learning" in query_lower:
            templates = [
                "Machine learning algorithms: machine learning models use machine learning techniques for predictions.",
                "Introduction to machine learning: supervised machine learning and unsupervised machine learning methods.",
                "Machine learning frameworks like TensorFlow enable efficient machine learning development.",
                "Machine learning applications in industry leverage machine learning for automation.",
                "Deep machine learning with neural networks advances machine learning capabilities.",
            ]
        else:
            # Generic keyword-rich templates
            first_keyword = keywords[0] if keywords else "topic"
            templates = [
                f"Comprehensive guide to {first_keyword}: {first_keyword} basics and {first_keyword} applications.",
                f"Understanding {first_keyword}: {first_keyword} fundamentals explained with {first_keyword} examples.",
                f"Advanced {first_keyword} topics: {first_keyword} techniques and {first_keyword} best practices.",
                f"The complete {first_keyword} reference: {first_keyword} documentation and {first_keyword} tutorials.",
                f"Practical {first_keyword} guide: {first_keyword} implementation and {first_keyword} usage patterns.",
            ]
        
        # Create chunks with scores based on keyword frequency
        for i in range(min(self.top_k, len(templates))):
            # Score based on keyword matches (mock)
            keyword_count = sum(kw in templates[i].lower() for kw in keywords)
            score = min(0.85 - (i * 0.06), 0.85)  # Slightly lower than vector
            
            chunk = Chunk(
                text=templates[i],
                doc_id=f"mock_doc_{i // 2 + 10}",  # Different doc IDs from vector
                chunk_id=f"keyword_chunk_{i}",
                score=score,
                metadata={"keyword_matches": keyword_count}
            )
            chunks.append(chunk)
        
        # Generate more if needed
        while len(chunks) < self.top_k:
            i = len(chunks)
            chunk = Chunk(
                text=f"Document mentioning {query} with relevant keywords.",
                doc_id=f"mock_doc_{i // 2 + 10}",
                chunk_id=f"keyword_chunk_{i}",
                score=max(0.4, 0.85 - (i * 0.06)),
                metadata={"keyword_matches": 1}
            )
            chunks.append(chunk)
        
        return chunks
    
    def build_index(self) -> None:
        """
        Build BM25 index from vector store.
        
        Convenience method to build index.
        
        Example:
            >>> agent = KeywordSearchAgent(mock_mode=False)
            >>> agent.build_index()  # Build index from ChromaDB
        """
        if self.mock_mode:
            self.log("Cannot build index in mock mode", level="warning")
            return
        
        self.log("Building BM25 index...", level="info")
        self.bm25_index.build_from_vector_store()
        self.bm25_index.save()
        self.log("✅ BM25 index built and saved", level="info")