"""
Graph Traversal Agent - Week 10 Day 1
Navigate knowledge graph to find relationships between entities.
"""

from typing import List, Dict, Optional, Tuple
import networkx as nx

from src.agents.base_agent import BaseAgent
from src.models.agent_state import AgentState, Chunk
from src.graph.graph_builder import KnowledgeGraph


class GraphTraversalAgent(BaseAgent):
    """
    Agent that traverses knowledge graph to find relationships.
    
    Capabilities:
    - Find paths between entities
    - Extract subgraphs
    - Rank paths by relevance
    - Return graph-informed chunks
    """
    
    def __init__(self, knowledge_graph: Optional[KnowledgeGraph] = None):
        """
        Initialize Graph Traversal Agent.
        
        Args:
            knowledge_graph: KnowledgeGraph instance
        """
        super().__init__(name="graph_traversal")
        self.kg = knowledge_graph
    
    def execute(self, state: AgentState) -> AgentState:
        """
        Execute graph traversal.
        
        Args:
            state: Current agent state with query and graph
        
        Returns:
            Updated state with graph paths and metadata
        """
        if not self.kg or self.kg.graph.number_of_nodes() == 0:
            self.log("No knowledge graph available", level="warning")
            state.metadata["graph_search"] = {
                "status": "no_graph",
                "paths": [],
                "entities_found": []
            }
            return state
        
        query = state.query
        self.log(f"Traversing graph for query: {query}")
        
        # Extract entities from query
        query_entities = self._extract_query_entities(query)
        
        if len(query_entities) < 2:
            self.log(f"Need 2+ entities for graph search, found {len(query_entities)}", level="warning")
            state.metadata["graph_search"] = {
                "status": "insufficient_entities",
                "entities_found": query_entities,
                "paths": []
            }
            return state
        
        # Find paths between entity pairs
        all_paths = []
        for i, source in enumerate(query_entities):
            for target in query_entities[i+1:]:
                paths = self._find_paths(source, target, max_length=3)
                all_paths.extend(paths)
        
        # Rank paths
        ranked_paths = self._rank_paths(all_paths, query)
        
        # Store in state
        state.metadata["graph_search"] = {
            "status": "success",
            "entities_found": query_entities,
            "paths": ranked_paths[:5],  # Top 5 paths
            "path_count": len(all_paths)
        }
        
        self.log(f"Found {len(all_paths)} paths, returning top {min(5, len(ranked_paths))}")
        
        return state
    
    def _extract_query_entities(self, query: str) -> List[str]:
        """
        Extract entities from query text.
        
        Args:
            query: Query string
        
        Returns:
            List of entity strings (normalized)
        """
        from src.graph.entity_extractor import EntityExtractor
        
        extractor = EntityExtractor()
        entities = extractor.extract(query)
        
        # Get normalized entity texts that exist in graph
        query_entities = []
        for entity in entities:
            if entity.normalized in self.kg.graph:
                query_entities.append(entity.normalized)
        
        return query_entities
    
    def _find_paths(
        self,
        source: str,
        target: str,
        max_length: int = 3
    ) -> List[Dict]:
        """
        Find paths between two entities.
        
        Args:
            source: Source entity
            target: Target entity
            max_length: Maximum path length
        
        Returns:
            List of path dictionaries
        """
        if source not in self.kg.graph or target not in self.kg.graph:
            return []
        
        try:
            # Find all simple paths
            paths = list(nx.all_simple_paths(
                self.kg.graph,
                source,
                target,
                cutoff=max_length
            ))
            
            # Convert to dictionaries with metadata
            path_dicts = []
            for path in paths:
                # Get relationships along path
                relations = []
                for i in range(len(path) - 1):
                    edge_data = self.kg.graph[path[i]][path[i+1]]
                    relations.append({
                        'from': path[i],
                        'to': path[i+1],
                        'relation': edge_data.get('relation', 'related_to'),
                        'confidence': edge_data.get('confidence', 0.5)
                    })
                
                path_dicts.append({
                    'path': path,
                    'length': len(path),
                    'relations': relations
                })
            
            return path_dicts
            
        except nx.NetworkXNoPath:
            return []
    
    def _rank_paths(
        self,
        paths: List[Dict],
        query: str
    ) -> List[Dict]:
        """
        Rank paths by relevance.
        
        Scoring factors:
        - Shorter paths = higher score
        - Higher confidence relations = higher score
        - Specific relations (not "related_to") = bonus
        
        Args:
            paths: List of path dictionaries
            query: Original query
        
        Returns:
            Sorted list of paths
        """
        scored_paths = []
        
        for path_dict in paths:
            score = 0.0
            
            # Shorter paths are better (inverse of length)
            length_score = 1.0 / path_dict['length']
            score += length_score * 2.0
            
            # Average confidence of relations
            relations = path_dict['relations']
            if relations:
                avg_confidence = sum(r['confidence'] for r in relations) / len(relations)
                score += avg_confidence
            
            # Bonus for specific relations (not "related_to")
            specific_relations = [r for r in relations if r['relation'] != 'related_to']
            if specific_relations:
                score += len(specific_relations) * 0.5
            
            path_dict['score'] = score
            scored_paths.append(path_dict)
        
        # Sort by score descending
        return sorted(scored_paths, key=lambda x: x['score'], reverse=True)
    
    def get_path_description(self, path_dict: Dict) -> str:
        """
        Generate human-readable path description.
        
        Args:
            path_dict: Path dictionary
        
        Returns:
            String description
        """
        path = path_dict['path']
        relations = path_dict['relations']
        
        # Build description
        parts = []
        for i, rel in enumerate(relations):
            parts.append(f"{rel['from']} --[{rel['relation']}]--> {rel['to']}")
        
        return " → ".join(parts)