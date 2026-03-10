"""
Retrieval Coordinator Agent - Tactical Level 2 Agent.

Manages parallel retrieval swarm (Vector, Keyword, Graph agents).
Coordinates retrieval, aggregates results, and deduplicates chunks.

Swarm Pattern:
- Spawns 3 retrieval agents in parallel
- Each agent uses different retrieval method
- Aggregates all results
- Deduplicates by content similarity
- Returns top-k unique chunks
"""

import asyncio
from typing import List, Dict, Any
from collections import defaultdict
import hashlib

from src.agents.base_agent import BaseAgent
from src.models.agent_state import AgentState, Chunk
from src.utils.exceptions import AgentExecutionError, RetrievalError
from src.config import get_settings


class RetrievalCoordinator(BaseAgent):
    """
    Retrieval Coordinator - Manages parallel retrieval swarm.
    
    Spawns multiple retrieval agents to search for relevant chunks
    using different methods (vector, keyword, graph). Aggregates
    and deduplicates results.
    
    Attributes:
        vector_agent: Vector search agent (semantic)
        keyword_agent: Keyword search agent (BM25)
        graph_agent: Graph search agent (relationships)
        top_k: Number of chunks to return
        parallel: Whether to execute agents in parallel
        
    Example:
        >>> coordinator = RetrievalCoordinator(
        ...     vector_agent=vector_agent,
        ...     keyword_agent=keyword_agent,
        ...     graph_agent=graph_agent
        ... )
        >>> 
        >>> state = AgentState(query="What is Python?")
        >>> result = coordinator.run(state)
        >>> 
        >>> print(len(result.chunks))  # 10 (top_k)
        >>> print(result.retrieval_round)  # 0 or incremented
    """
    
    def __init__(
        self,
        vector_agent: BaseAgent = None,
        keyword_agent: BaseAgent = None,
        graph_agent: BaseAgent = None,
        top_k: int = None,
        parallel: bool = None
    ):
        """
        Initialize Retrieval Coordinator.
        
        Args:
            vector_agent: Vector search agent instance
            keyword_agent: Keyword search agent instance
            graph_agent: Graph search agent instance
            top_k: Number of top chunks to return (default: from config)
            parallel: Execute agents in parallel (default: from config)
        
        Example:
            >>> coordinator = RetrievalCoordinator(
            ...     vector_agent=VectorAgent(),
            ...     keyword_agent=KeywordAgent(),
            ...     graph_agent=GraphAgent(),
            ...     top_k=15
            ... )
        """
        super().__init__(name="retrieval_coordinator", version="1.0.0")
        
        self.vector_agent = vector_agent
        self.keyword_agent = keyword_agent
        self.graph_agent = graph_agent
        
        # Load settings from config
        settings = get_settings()
        self.top_k = top_k if top_k is not None else settings.retrieval_top_k
        self.parallel = (
            parallel if parallel is not None else settings.parallel_retrieval
        )
        
        self.log(
            f"Initialized with top_k={self.top_k}, parallel={self.parallel}",
            level="debug"
        )
    
    def execute(self, state: AgentState) -> AgentState:
        """
        Execute retrieval coordination: spawn swarm and aggregate results.
        
        Args:
            state: Current agent state with query
        
        Returns:
            Updated state with chunks
        
        Raises:
            RetrievalError: If retrieval fails
        
        Example:
            >>> state = AgentState(query="What is machine learning?")
            >>> result = coordinator.execute(state)
            >>> print(len(result.chunks))  # Top-k chunks
        """
        try:
            query = state.query
            current_round = state.retrieval_round
            
            self.log(
                f"Starting retrieval round {current_round} for query: {query[:50]}...",
                level="info"
            )
            
            # Step 1: Spawn retrieval swarm
            all_results = self._spawn_swarm(query)
            
            self.log(
                f"Retrieved {len(all_results)} total chunks from swarm",
                level="info"
            )
            
            # Step 2: Deduplicate
            unique_chunks = self._deduplicate(all_results)
            
            self.log(
                f"Deduplication: {len(all_results)} → {len(unique_chunks)} unique chunks",
                level="info"
            )
            
            # Step 3: Select top-k
            top_chunks = self._select_top_k(unique_chunks, self.top_k)
            
            self.log(
                f"Selected top {len(top_chunks)} chunks",
                level="info"
            )
            
            # Step 4: Update state
            state.chunks = top_chunks
            state.retrieval_round = current_round + 1
            
            # Step 5: Add metadata
            state.metadata["retrieval_coordinator"] = {
                "round": current_round,
                "total_retrieved": len(all_results),
                "unique_chunks": len(unique_chunks),
                "final_chunks": len(top_chunks),
                "parallel": self.parallel
            }
            
            return state
            
        except Exception as e:
            self.log(f"Retrieval coordination failed: {str(e)}", level="error")
            raise RetrievalError(
                retrieval_type="coordination",
                message=f"Failed to coordinate retrieval: {str(e)}",
                details={"query": state.query, "round": state.retrieval_round}
            ) from e
    
    def _spawn_swarm(self, query: str) -> List[Chunk]:
        """Spawn retrieval swarm (private method)."""
        
        self.log(f"Spawning retrieval swarm for: {query}")
        
        # Collect available agents
        agents = []
        
        if self.vector_agent:
            agents.append(('vector', self.vector_agent))
        
        if self.keyword_agent:
            agents.append(('keyword', self.keyword_agent))
        
        if self.graph_agent:  # ← Just check if exists!
            agents.append(('graph', self.graph_agent))
            self.log("Graph search agent included in swarm")
        else:
            self.log("Graph search unavailable", level="warning")
        
        # Execute agents
        all_results = []
        
        for agent_name, agent in agents:
            self.log(f"Executing {agent_name} agent...")
            try:
                results = agent.search_async(query, top_k=self.top_k)
                self.log(f"{agent_name}: {len(results)} chunks")
                all_results.extend(results)
            except Exception as e:
                self.log(f"{agent_name} failed: {e}", level="error")
        
        self.log(f"Swarm complete: {len(all_results)} chunks from {len(agents)} agents")
        
        return all_results
    
    def _execute_parallel(self, query: str) -> List[Chunk]:
        """
        Execute retrieval agents in parallel using asyncio.
        
        Args:
            query: User query string
        
        Returns:
            Combined chunks from all agents
        """
        self.log("Executing swarm in parallel", level="debug")
        
        # Create event loop if needed
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Run async retrieval
        all_chunks = loop.run_until_complete(
            self._retrieve_async(query)
        )
        
        return all_chunks
    
    async def _retrieve_async(self, query: str) -> List[Chunk]:
        """
        Async retrieval from all agents.
        
        Args:
            query: User query string
        
        Returns:
            Combined chunks from all agents
        """
        tasks = []
        
        # Spawn tasks for each agent
        if self.vector_agent:
            tasks.append(self._retrieve_from_agent(self.vector_agent, query))
        
        if self.keyword_agent:
            tasks.append(self._retrieve_from_agent(self.keyword_agent, query))
        
        if self.graph_agent:
            tasks.append(self._retrieve_from_agent(self.graph_agent, query))
        
        # Wait for all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results (filter out exceptions)
        all_chunks = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.log(
                    f"Agent {i} failed: {str(result)}",
                    level="warning"
                )
            else:
                all_chunks.extend(result)
        
        return all_chunks
    
    async def _retrieve_from_agent(
        self,
        agent: BaseAgent,
        query: str
    ) -> List[Chunk]:
        """
        Async wrapper for agent retrieval.
        
        Args:
            agent: Retrieval agent instance
            query: User query string
        
        Returns:
            Chunks from agent
        """
        # Create temporary state for agent
        temp_state = AgentState(query=query)
        
        # Execute agent (synchronously, but in async context)
        result_state = agent.run(temp_state)
        
        return result_state.chunks
    
    def _execute_sequential(self, query: str) -> List[Chunk]:
        """
        Execute retrieval agents sequentially.
        
        Args:
            query: User query string
        
        Returns:
            Combined chunks from all agents
        """
        self.log("Executing swarm sequentially", level="debug")
        
        all_chunks = []
        
        # Execute each agent
        agents = [
            ("vector", self.vector_agent),
            ("keyword", self.keyword_agent),
            ("graph", self.graph_agent)
        ]
        
        for name, agent in agents:
            if agent is None:
                continue
            
            try:
                temp_state = AgentState(query=query)
                result_state = agent.run(temp_state)
                all_chunks.extend(result_state.chunks)
                
                self.log(
                    f"{name} agent retrieved {len(result_state.chunks)} chunks",
                    level="debug"
                )
            except Exception as e:
                self.log(
                    f"{name} agent failed: {str(e)}",
                    level="warning"
                )
        
        return all_chunks
    
    def _deduplicate(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Remove duplicate chunks based on content similarity.
        
        Uses text hashing to identify duplicates. Keeps chunk with
        highest score when duplicates found.
        
        Args:
            chunks: List of chunks (may contain duplicates)
        
        Returns:
            List of unique chunks
        
        Example:
            >>> duplicates = [chunk1, chunk1_copy, chunk2]
            >>> unique = coordinator._deduplicate(duplicates)
            >>> print(len(unique))  # 2
        """
        if not chunks:
            return []
        
        # Group by content hash
        hash_groups = defaultdict(list)
        
        for chunk in chunks:
            content_hash = self._hash_content(chunk.text)
            hash_groups[content_hash].append(chunk)
        
        # Keep best chunk from each group
        unique_chunks = []
        for group in hash_groups.values():
            # Sort by score (descending)
            sorted_group = sorted(
                group,
                key=lambda c: c.score if c.score is not None else 0.0,
                reverse=True
            )
            # Keep highest scored
            unique_chunks.append(sorted_group[0])
        
        return unique_chunks
    
    def _hash_content(self, text: str) -> str:
        """
        Generate hash for content similarity.
        
        Uses MD5 hash of normalized text.
        
        Args:
            text: Chunk text
        
        Returns:
            Content hash string
        """
        # Normalize text (lowercase, strip whitespace)
        normalized = text.lower().strip()
        
        # Remove extra whitespace
        normalized = " ".join(normalized.split())
        
        # Generate hash
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _select_top_k(self, chunks: List[Chunk], k: int) -> List[Chunk]:
        """
        Select top-k chunks by score.
        
        Args:
            chunks: List of chunks
            k: Number to select
        
        Returns:
            Top-k chunks sorted by score (descending)
        
        Example:
            >>> top_10 = coordinator._select_top_k(chunks, 10)
            >>> print(len(top_10))  # 10
            >>> print(top_10[0].score >= top_10[-1].score)  # True
        """
        if not chunks:
            return []
        
        # Sort by score (descending)
        sorted_chunks = sorted(
            chunks,
            key=lambda c: c.score if c.score is not None else 0.0,
            reverse=True
        )
        
        # Return top-k
        return sorted_chunks[:k]
    
    def retrieve_with_details(self, query: str) -> Dict[str, Any]:
        """
        Detailed retrieval for debugging/analysis.
        
        Returns breakdown of retrieval from each agent and
        deduplication statistics.
        
        Args:
            query: User query string
        
        Returns:
            Dictionary with detailed retrieval info
        
        Example:
            >>> details = coordinator.retrieve_with_details("What is X?")
            >>> print(details["vector_count"])
            >>> print(details["dedup_stats"])
        """
        # Execute retrieval
        all_chunks = self._spawn_swarm(query)
        
        # Count by source (if agents tag chunks)
        source_counts = defaultdict(int)
        for chunk in all_chunks:
            source = chunk.metadata.get("source", "unknown")
            source_counts[source] += 1
        
        # Deduplicate
        unique_chunks = self._deduplicate(all_chunks)
        top_chunks = self._select_top_k(unique_chunks, self.top_k)
        
        return {
            "query": query,
            "total_retrieved": len(all_chunks),
            "source_counts": dict(source_counts),
            "unique_chunks": len(unique_chunks),
            "duplicates_removed": len(all_chunks) - len(unique_chunks),
            "final_chunks": len(top_chunks),
            "top_k": self.top_k,
            "parallel": self.parallel,
            "chunks": top_chunks
        }