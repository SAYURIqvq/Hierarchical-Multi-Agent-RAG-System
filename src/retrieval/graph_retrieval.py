"""
Graph-based Retrieval - Week 10 Day 2
Retrieve chunks based on knowledge graph paths.
"""

from typing import List, Dict, Set, Tuple
from src.models.agent_state import Chunk
from src.agents.graph_traversal_agent import GraphTraversalAgent
from src.graph.graph_builder import KnowledgeGraph


class GraphRetrieval:
    """
    Retrieve chunks using knowledge graph paths.
    
    Strategy:
    1. Find paths between query entities
    2. Expand path with neighbors
    3. Retrieve chunks mentioning path entities
    4. Rank by path relevance
    """
    
    def __init__(
        self,
        knowledge_graph: KnowledgeGraph,
        vector_store
    ):
        """
        Initialize graph retrieval.
        
        Args:
            knowledge_graph: KnowledgeGraph instance
            vector_store: Vector store for chunk retrieval
        """
        self.kg = knowledge_graph
        self.vector_store = vector_store
        self.graph_agent = GraphTraversalAgent(knowledge_graph)
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        expand_neighbors: bool = True
    ) -> List[Chunk]:
        """
        Search using knowledge graph.
        
        Args:
            query: Search query
            top_k: Number of chunks to return
            expand_neighbors: Include neighbor entities
        
        Returns:
            List of Chunk objects
        """
        # Find paths using graph agent
        from src.models.agent_state import AgentState
        
        state = AgentState(query=query)
        result = self.graph_agent.execute(state)
        
        graph_search = result.metadata.get("graph_search", {})
        
        if graph_search.get("status") != "success":
            print(f"⚠️  Graph search failed: {graph_search.get('status')}")
            return []
        
        paths = graph_search.get("paths", [])
        
        if not paths:
            print("⚠️  No paths found in graph")
            return []
        
        # Collect entities from paths
        path_entities = self._collect_path_entities(paths)
        
        # Optionally expand with neighbors
        if expand_neighbors:
            path_entities = self._expand_with_neighbors(path_entities, k=1)
        
        print(f"🔍 Graph search targeting {len(path_entities)} entities: {list(path_entities)[:5]}...")
        
        # Retrieve chunks mentioning these entities
        chunks = self._retrieve_chunks_by_entities(
            path_entities,
            top_k=top_k * 2  # Get more, then filter
        )
        
        # Rank chunks by path relevance
        ranked_chunks = self._rank_by_path_relevance(
            chunks,
            paths,
            path_entities
        )
        
        return ranked_chunks[:top_k]
    
    def _collect_path_entities(self, paths: List[Dict]) -> Set[str]:
        """
        Collect all unique entities from paths.
        
        Args:
            paths: List of path dictionaries
        
        Returns:
            Set of entity strings
        """
        entities = set()
        
        for path_dict in paths:
            path = path_dict.get('path', [])
            entities.update(path)
        
        return entities
    
    def _expand_with_neighbors(
        self,
        entities: Set[str],
        k: int = 1
    ) -> Set[str]:
        """
        Expand entity set with k-hop neighbors.
        
        Args:
            entities: Initial entity set
            k: Number of hops
        
        Returns:
            Expanded entity set
        """
        expanded = set(entities)
        
        for _ in range(k):
            new_entities = set()
            for entity in expanded:
                if entity in self.kg.graph:
                    # Add neighbors
                    neighbors = list(self.kg.graph.successors(entity))
                    neighbors.extend(list(self.kg.graph.predecessors(entity)))
                    new_entities.update(neighbors)
            
            expanded.update(new_entities)
        
        return expanded
    
    def _retrieve_chunks_by_entities(
        self,
        entities: Set[str],
        top_k: int = 20
    ) -> List[Chunk]:
        """
        Retrieve chunks that mention the given entities.
        
        Args:
            entities: Set of entity strings
            top_k: Number of chunks to retrieve
        
        Returns:
            List of Chunk objects
        """
        # Build search query from entities
        query_text = " ".join(entities)
        
        # Generate embedding
        from src.ingestion.embedder import EmbeddingGenerator
        embedder = EmbeddingGenerator()
        query_embedding = embedder.generate_query_embedding(query_text)
        
        # Vector search
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
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
                    'chunk_type': result.get('chunk_type', 'child'),
                    'retrieval_method': 'graph',
                    **result.get('metadata', {})
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _rank_by_path_relevance(
        self,
        chunks: List[Chunk],
        paths: List[Dict],
        path_entities: Set[str]
    ) -> List[Chunk]:
        """
        Re-rank chunks by how many path entities they mention.
        
        Args:
            chunks: List of chunks
            paths: List of path dictionaries
            path_entities: Set of entity strings
        
        Returns:
            Ranked list of chunks
        """
        scored_chunks = []
        
        for chunk in chunks:
            text_lower = chunk.text.lower()
            
            # Count entity mentions
            entity_count = sum(
                1 for entity in path_entities
                if entity in text_lower
            )
            
            # Bonus for mentioning high-score path entities
            path_bonus = 0.0
            for path_dict in paths[:3]:  # Top 3 paths
                path = path_dict.get('path', [])
                path_score = path_dict.get('score', 0)
                
                # Check if chunk mentions path
                path_mentions = sum(1 for entity in path if entity in text_lower)
                if path_mentions >= 2:  # Mentions 2+ entities from path
                    path_bonus += path_score * 0.5
            
            # Combine scores
            final_score = (
                chunk.score +  # Original vector score
                entity_count * 0.3 +  # Entity coverage
                path_bonus  # Path relevance
            )
            
            chunk.score = final_score
            scored_chunks.append(chunk)
        
        # Sort by final score
        return sorted(scored_chunks, key=lambda c: c.score, reverse=True)