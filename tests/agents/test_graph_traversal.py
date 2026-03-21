"""Test Graph Traversal Agent - Week 10 Day 1"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agents.graph_traversal_agent import GraphTraversalAgent
from src.graph.graph_builder import KnowledgeGraph
from src.models.agent_state import AgentState


def test_graph_traversal():
    """Test graph traversal agent."""
    
    print("\n🧪 Testing Graph Traversal Agent\n")
    
    # Load test graph
    kg = KnowledgeGraph()
    kg.load('data/graphs/test_graph.pkl')
    
    print(f"📂 Loaded graph: {kg}")
    print(f"   Nodes: {kg.graph.number_of_nodes()}")
    print(f"   Edges: {kg.graph.number_of_edges()}\n")
    
    # Create agent
    agent = GraphTraversalAgent(knowledge_graph=kg)
    
    # Test queries
    test_queries = [
        "How does Google relate to machine learning?",
        "What is the connection between Python and neural networks?",
        "Relationship between TensorFlow and AI?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"{'='*60}")
        print(f"🔍 Test {i}: {query}\n")
        
        # Create state
        state = AgentState(query=query)
        
        # Execute agent
        result = agent.execute(state)
        
        # Display results
        graph_search = result.metadata.get("graph_search", {})
        
        print(f"Status: {graph_search.get('status')}")
        print(f"Entities found: {graph_search.get('entities_found', [])}")
        print(f"Total paths: {graph_search.get('path_count', 0)}")
        
        paths = graph_search.get('paths', [])
        if paths:
            print(f"\nTop {len(paths)} paths:")
            for j, path_dict in enumerate(paths, 1):
                desc = agent.get_path_description(path_dict)
                score = path_dict.get('score', 0)
                print(f"\n{j}. Score: {score:.2f}")
                print(f"   {desc}")
        else:
            print("\nNo paths found")
        
        print()
    
    print(f"{'='*60}")
    print("✅ All tests complete!\n")


if __name__ == "__main__":
    test_graph_traversal()