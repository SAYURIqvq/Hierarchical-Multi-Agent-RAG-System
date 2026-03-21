"""Test Query Decomposer - Week 5 Day 4"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agents.query_decomposer import QueryDecomposer
from src.models.agent_state import AgentState


def test_decomposer():
    """Test query decomposition."""
    
    print("\n🧪 Testing Query Decomposer\n")
    
    decomposer = QueryDecomposer()
    
    # Test cases
    queries = [
        "What is machine learning?",  # Simple - no decomposition
        "Compare Python and Java for web development",  # Should decompose
        "Explain the advantages and disadvantages of microservices versus monolithic architecture"  # Complex
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        state = AgentState(query=query)
        result = decomposer.run(state)
        
        print(f"\nSub-queries ({len(result.sub_queries)}):")
        for i, sq in enumerate(result.sub_queries, 1):
            print(f"  {i}. {sq}")


if __name__ == "__main__":
    test_decomposer()