"""
Graph Search Agent - Operational Level 3 Agent.

Performs relationship-based search using knowledge graph.
Part of retrieval swarm coordinated by RetrievalCoordinator.
"""

from typing import List, Optional

from src.agents.base_agent import BaseAgent
from src.models.agent_state import AgentState, Chunk
from src.retrieval.graph_retrieval import GraphRetrieval
from src.graph.graph_builder import KnowledgeGraph
from src.storage.chroma_store import ChromaVectorStore
from src.utils.exceptions import AgentExecutionError


class GraphSearchAgent(BaseAgent):
    """
    Graph Search Agent - Knowledge graph-based retrieval.
    
    Uses knowledge graph relationships to find relevant chunks.
    Part of the retrieval swarm (Level 3 operational agent).
    
    Attributes:
        graph_retrieval: GraphRetrieval instance
        knowledge_graph: KnowledgeGraph instance
        
    Example:
        >>> graph_agent = GraphSearchAgent(
        ...     knowledge_graph=kg,
        ...     vector_store=chroma_store
        ... )
        >>> 
        >>> state = AgentState(query="Compare Python and Java")
        >>> result = graph_agent.run(state)
        >>> print(len(result.chunks))  # Retrieved chunks
    """
    
    def __init__(
        self,
        knowledge_graph: Optional[KnowledgeGraph],
        vector_store: ChromaVectorStore
    ):
        """
        Initialize Graph Search Agent.
        
        Args:
            knowledge_graph: KnowledgeGraph instance (can be None)
            vector_store: ChromaDB vector store
        """
        super().__init__(name="graph_search", version="1.0.0")
        
        self.knowledge_graph = knowledge_graph
        self.vector_store = vector_store
        
        # Initialize graph retrieval if KG available
        if self.knowledge_graph:
            self.graph_retrieval = GraphRetrieval(
                knowledge_graph=knowledge_graph,
                vector_store=vector_store
            )
            self.log("Initialized with knowledge graph", level="debug")
        else:
            self.graph_retrieval = None
            self.log("Initialized without knowledge graph (will skip)", level="debug")
    
    def execute(self, state: AgentState) -> AgentState:
        """
        Execute graph search.
        
        Args:
            state: Current agent state with query
        
        Returns:
            Updated state with chunks
        
        Raises:
            AgentExecutionError: If search fails
        """
        try:
            query = state.query
            
            # Check if graph available
            if not self.graph_retrieval:
                self.log("Knowledge graph not available, skipping", level="warning")
                state.chunks = []
                return state
            
            self.log(f"Graph search for: {query[:50]}...", level="info")
            
            # Search using graph
            chunks = self.graph_retrieval.search(
                query=query,
                top_k=10,
                expand_neighbors=True
            )
            
            # Mark source
            for chunk in chunks:
                chunk.metadata['source'] = 'graph'
            
            state.chunks = chunks
            
            self.log(f"Graph search retrieved {len(chunks)} chunks", level="info")
            
            return state
            
        except Exception as e:
            self.log(f"Graph search failed: {str(e)}", level="error")
            # Don't raise - graph search is optional
            state.chunks = []
            return state
    
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
            # Check if graph available
            if not self.graph_retrieval:
                self.log("Knowledge graph not available", level="debug")
                return []
            
            # Search
            chunks = self.graph_retrieval.search(
                query=query,
                top_k=top_k,
                expand_neighbors=True
            )
            
            # Mark source
            for chunk in chunks:
                chunk.metadata['source'] = 'graph'
            
            return chunks
            
        except Exception as e:
            self.log(f"Async graph search failed: {str(e)}", level="error")
            return []