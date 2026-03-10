"""
Graph Search Agent - Week 10 Day 3
Real implementation using GraphRetrieval.
"""

from typing import List
from src.agents.base_agent import BaseAgent
from src.models.agent_state import AgentState, Chunk
from src.retrieval.graph_retrieval import GraphRetrieval
from src.graph.graph_builder import KnowledgeGraph


class GraphSearchAgent(BaseAgent):
    """
    Agent that searches using knowledge graph.
    
    Uses graph paths to find relationship-based results.
    """
    
    def __init__(
        self,
        knowledge_graph: KnowledgeGraph = None,
        vector_store = None
    ):
        """
        Initialize Graph Search Agent.
        
        Args:
            knowledge_graph: KnowledgeGraph instance
            vector_store: Vector store for chunk retrieval
        """
        super().__init__(name="graph_search")
        
        self.graph_retrieval = None
        if knowledge_graph and vector_store:
            self.graph_retrieval = GraphRetrieval(
                knowledge_graph=knowledge_graph,
                vector_store=vector_store
            )
    
    def search_async(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Chunk]:
        """
        Async search using graph (for swarm compatibility).
        
        Args:
            query: Search query
            top_k: Number of chunks to return
        
        Returns:
            List of Chunk objects
        """
        if not self.graph_retrieval:
            self.log("Graph retrieval not initialized", level="warning")
            return []
        
        self.log(f"Searching graph for: {query}")
        
        try:
            chunks = self.graph_retrieval.search(
                query=query,
                top_k=top_k,
                expand_neighbors=True
            )
            
            self.log(f"Found {len(chunks)} chunks via graph search")
            
            # Mark retrieval method
            for chunk in chunks:
                chunk.metadata['retrieval_method'] = 'graph'
            
            return chunks
            
        except Exception as e:
            self.log(f"Graph search failed: {e}", level="error")
            return []