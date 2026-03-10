"""
Synthesis Agent - Tactical Level 2 Agent.

Combines and reranks results from multiple retrieval sources.
Uses hybrid ranking and optional reranking for optimal results.
"""

from typing import List, Dict, Any, Optional
import hashlib
from collections import defaultdict

from src.agents.base_agent import BaseAgent
from src.models.agent_state import AgentState, Chunk
from src.config import get_settings
from src.utils.logger import setup_logger


class SynthesisAgent(BaseAgent):
    """
    Synthesis Agent - Result fusion and reranking.
    
    Combines results from multiple retrieval agents:
    - Vector search (semantic similarity)
    - Keyword search (BM25)
    - Graph search (relationships)
    
    Performs:
    - Deduplication (content-based)
    - Hybrid ranking (weighted fusion)
    - Diversity filtering
    - Top-K selection
    - Optional Cohere reranking
    
    Attributes:
        top_k: Number of final results to return
        vector_weight: Weight for vector search scores (0.0-1.0)
        keyword_weight: Weight for keyword search scores (0.0-1.0)
        use_reranker: Whether to use Cohere reranking
        
    Example:
        >>> agent = SynthesisAgent(top_k=10)
        >>> state = AgentState(query="query", chunks=[...])
        >>> result = agent.run(state)
        >>> print(len(result.chunks))  # 10 best chunks
    """
    
    def __init__(
        self,
        top_k: int = None,
        vector_weight: float = None,
        keyword_weight: float = None,
        use_reranker: bool = False
    ):
        """
        Initialize Synthesis Agent.
        
        Args:
            top_k: Number of final results (default from config)
            vector_weight: Weight for vector scores (default from config)
            keyword_weight: Weight for keyword scores (default from config)
            use_reranker: Enable Cohere reranking (default: False)
        
        Example:
            >>> agent = SynthesisAgent(
            ...     top_k=10,
            ...     vector_weight=0.7,
            ...     keyword_weight=0.3
            ... )
        """
        super().__init__(name="synthesis", version="1.0.0")
        
        settings = get_settings()
        
        self.top_k = top_k or settings.retrieval_top_k
        self.vector_weight = vector_weight or settings.vector_search_weight
        self.keyword_weight = keyword_weight or settings.keyword_search_weight
        self.use_reranker = use_reranker
        
        # Validate weights sum to 1.0
        total_weight = self.vector_weight + self.keyword_weight
        if abs(total_weight - 1.0) > 0.01:
            self.log(
                f"Warning: Weights sum to {total_weight}, normalizing to 1.0",
                level="warning"
            )
            # Normalize
            self.vector_weight = self.vector_weight / total_weight
            self.keyword_weight = self.keyword_weight / total_weight
        
        self.log(
            f"Initialized with top_k={self.top_k}, "
            f"weights=(vector:{self.vector_weight:.2f}, "
            f"keyword:{self.keyword_weight:.2f}), "
            f"reranker={use_reranker}",
            level="info"
        )
    
    def execute(self, state: AgentState) -> AgentState:
        """
        Execute synthesis: deduplicate, rank, select top-K.
        
        Args:
            state: Current state with chunks from multiple sources
        
        Returns:
            Updated state with synthesized top-K chunks
        
        Example:
            >>> state = AgentState(query="query", chunks=[30 chunks])
            >>> result = agent.execute(state)
            >>> len(result.chunks)  # 10 (top_k)
        """
        try:
            chunks = state.chunks
            
            if not chunks:
                self.log("No chunks to synthesize", level="warning")
                return state
            
            self.log(
                f"Synthesizing {len(chunks)} chunks from multiple sources",
                level="info"
            )
            
            # Step 1: Deduplicate
            unique_chunks = self._deduplicate(chunks)
            self.log(
                f"After deduplication: {len(unique_chunks)} unique chunks",
                level="debug"
            )
            
            # Step 2: Hybrid ranking
            ranked_chunks = self._hybrid_rank(unique_chunks)
            self.log(
                f"Ranked {len(ranked_chunks)} chunks by hybrid score",
                level="debug"
            )
            
            # Step 3: Optional reranking
            if self.use_reranker and len(ranked_chunks) > 0:
                reranked_chunks = self._rerank_with_cohere(
                    state.query,
                    ranked_chunks
                )
                self.log("Applied Cohere reranking", level="debug")
            else:
                reranked_chunks = ranked_chunks
            
            # Step 4: Select top-K
            final_chunks = reranked_chunks[:self.top_k]
            
            self.log(
                f"Final synthesis: {len(final_chunks)} chunks selected",
                level="info"
            )
            
            # Update state
            state.chunks = final_chunks
            
            # Add metadata
            state.metadata["synthesis"] = {
                "input_count": len(chunks),
                "unique_count": len(unique_chunks),
                "final_count": len(final_chunks),
                "deduplication_rate": 1 - (len(unique_chunks) / len(chunks)),
                "reranker_used": self.use_reranker
            }
            
            return state
            
        except Exception as e:
            self.log(f"Synthesis failed: {str(e)}", level="error")
            # Return original state on failure
            return state
    
    def _deduplicate(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Deduplicate chunks based on content similarity.
        
        Uses MD5 hash of normalized text to detect duplicates.
        Keeps chunk with highest score.
        
        Args:
            chunks: List of chunks (possibly with duplicates)
        
        Returns:
            List of unique chunks
        """
        if not chunks:
            return []
        
        # Group by content hash
        hash_groups: Dict[str, List[Chunk]] = defaultdict(list)
        
        for chunk in chunks:
            content_hash = self._compute_hash(chunk.text)
            hash_groups[content_hash].append(chunk)
        
        # Keep highest scored chunk per group
        unique_chunks = []
        for group in hash_groups.values():
            # Sort by score (highest first)
            sorted_group = sorted(group, key=lambda c: c.score or 0, reverse=True)
            unique_chunks.append(sorted_group[0])
        
        return unique_chunks
    
    def _hybrid_rank(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Rank chunks using hybrid scoring.
        
        Combines scores from different sources with weights:
        - Vector search: semantic similarity
        - Keyword search: BM25 relevance
        - Graph search: relationship strength
        
        Final score = (vector_score * vector_weight) + 
                      (keyword_score * keyword_weight)
        
        Args:
            chunks: List of chunks to rank
        
        Returns:
            Sorted list of chunks by hybrid score
        """
        if not chunks:
            return []
        
        # Calculate hybrid scores
        for chunk in chunks:
            source = chunk.metadata.get("source", "unknown")
            base_score = chunk.score or 0.0
            
            # Weight by source
            if source == "vector":
                hybrid_score = base_score * self.vector_weight
            elif source == "keyword":
                hybrid_score = base_score * self.keyword_weight
            elif source == "graph":
                # Graph uses average of vector/keyword weights
                hybrid_score = base_score * (
                    (self.vector_weight + self.keyword_weight) / 2
                )
            else:
                # Unknown source, use base score
                hybrid_score = base_score
            
            # Store in metadata
            chunk.metadata["hybrid_score"] = hybrid_score
            chunk.metadata["original_score"] = base_score
        
        # Sort by hybrid score
        ranked = sorted(
            chunks,
            key=lambda c: c.metadata.get("hybrid_score", 0),
            reverse=True
        )
        
        # Update chunk scores to hybrid scores
        for chunk in ranked:
            chunk.score = chunk.metadata["hybrid_score"]
        
        return ranked
    
    def _rerank_with_cohere(
        self,
        query: str,
        chunks: List[Chunk]
    ) -> List[Chunk]:
        """
        Rerank chunks using Cohere Rerank API.
        
        Uses Cohere's rerank-english-v3.0 model for improved ranking.
        Falls back to hybrid ranking if API fails or key not available.
        
        Args:
            query: User query
            chunks: Chunks to rerank
        
        Returns:
            Reranked chunks
        """
        if not chunks:
            return []
        
        try:
            import cohere
            settings = get_settings()
            
            # Check if API key available
            if not settings.cohere_api_key:
                self.log(
                    "Cohere API key not found, skipping reranking",
                    level="debug"
                )
                return chunks
            
            # Initialize Cohere client
            co = cohere.Client(api_key=settings.cohere_api_key)
            
            # Prepare documents for reranking
            documents = [chunk.text for chunk in chunks]
            
            # Call Cohere Rerank API
            self.log(f"Reranking {len(documents)} chunks with Cohere...", level="debug")
            
            response = co.rerank(
                model="rerank-english-v3.0",
                query=query,
                documents=documents,
                top_n=len(chunks),  # Rerank all, we'll select top_k later
                return_documents=False  # We already have the texts
            )
            
            # Reorder chunks based on Cohere results
            reranked_chunks = []
            for result in response.results:
                original_index = result.index
                rerank_score = result.relevance_score
                
                # Get original chunk
                chunk = chunks[original_index]
                
                # Update score with rerank score
                chunk.metadata["rerank_score"] = rerank_score
                chunk.metadata["pre_rerank_score"] = chunk.score
                chunk.score = rerank_score
                
                reranked_chunks.append(chunk)
            
            self.log(
                f"✅ Cohere reranking complete: {len(reranked_chunks)} chunks",
                level="debug"
            )
            
            return reranked_chunks
            
        except ImportError:
            self.log(
                "Cohere library not installed. Install with: pip install cohere",
                level="warning"
            )
            return chunks
            
        except Exception as e:
            self.log(
                f"Cohere reranking failed: {str(e)}, using hybrid ranking",
                level="warning"
            )
            return chunks
    
    def _compute_hash(self, text: str) -> str:
        """
        Compute content hash for deduplication.
        
        Normalizes text before hashing:
        - Lowercase
        - Strip whitespace
        - Remove extra spaces
        
        Args:
            text: Text to hash
        
        Returns:
            MD5 hash string
        """
        # Normalize
        normalized = text.lower().strip()
        normalized = " ".join(normalized.split())  # Collapse whitespace
        
        # Hash
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def get_synthesis_stats(self, state: AgentState) -> Dict[str, Any]:
        """
        Get synthesis statistics from state.
        
        Args:
            state: AgentState with synthesis metadata
        
        Returns:
            Dictionary with synthesis stats
        
        Example:
            >>> stats = agent.get_synthesis_stats(state)
            >>> print(stats['deduplication_rate'])
        """
        return state.metadata.get("synthesis", {})