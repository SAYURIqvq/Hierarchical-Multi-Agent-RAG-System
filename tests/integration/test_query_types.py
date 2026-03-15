"""
Test Query Type Coverage - Week 10 Day 4
Test graph search with various query types.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.graph.graph_builder import KnowledgeGraph
from src.storage.chroma_store import ChromaVectorStore
from src.retrieval.graph_retrieval import GraphRetrieval


def test_query_types():
    """Test graph retrieval with different query types."""
    
    print("\n🧪 Testing Query Type Coverage\n")
    
    # Setup
    kg = KnowledgeGraph()
    kg.load('data/graphs/machine_learning.txt_graph.pkl')
    
    vector_store = ChromaVectorStore(persist_directory="data/chroma_db")
    
    graph_retrieval = GraphRetrieval(
        knowledge_graph=kg,
        vector_store=vector_store
    )
    
    # Test cases
    test_cases = [
        {
            "type": "Relationship Query",
            "query": "How does TensorFlow relate to machine learning?",
            "expected_entities": 2,
            "expected_paths": True
        },
        {
            "type": "Definition Query",
            "query": "What is neural network?",
            "expected_entities": 1,
            "expected_paths": False  # Single entity
        },
        {
            "type": "Comparison Query",
            "query": "Compare supervised and unsupervised learning",
            "expected_entities": 2,
            "expected_paths": True
        },
        {
            "type": "Multi-entity Query",
            "query": "Explain the relationship between Python, TensorFlow, and neural networks",
            "expected_entities": 3,
            "expected_paths": True
        },
        {
            "type": "No Entity Query",
            "query": "Tell me something interesting",
            "expected_entities": 0,
            "expected_paths": False
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"{'='*60}")
        print(f"Test {i}: {test_case['type']}")
        print(f"Query: {test_case['query']}\n")
        
        try:
            chunks = graph_retrieval.search(test_case['query'], top_k=5)
            
            # Extract graph search metadata from state
            from src.models.agent_state import AgentState
            state = AgentState(query=test_case['query'])
            state = graph_retrieval.graph_agent.execute(state)
            
            graph_search = state.metadata.get('graph_search', {})
            
            entities_found = len(graph_search.get('entities_found', []))
            paths_found = graph_search.get('path_count', 0)
            
            status = "✅ PASS" if chunks or not test_case['expected_paths'] else "⚠️  WARN"
            
            print(f"Status: {status}")
            print(f"Entities found: {entities_found}")
            print(f"Paths found: {paths_found}")
            print(f"Chunks retrieved: {len(chunks)}")
            
            results.append({
                'type': test_case['type'],
                'status': status,
                'entities': entities_found,
                'paths': paths_found,
                'chunks': len(chunks)
            })
            
        except Exception as e:
            print(f"❌ FAIL: {e}")
            results.append({
                'type': test_case['type'],
                'status': '❌ FAIL',
                'error': str(e)
            })
        
        print()
    
    # Summary
    print(f"{'='*60}")
    print("📊 Test Summary:\n")
    
    for result in results:
        print(f"{result['status']} {result['type']}")
        if 'chunks' in result:
            print(f"   Entities: {result['entities']}, Paths: {result['paths']}, Chunks: {result['chunks']}")
    
    passed = sum(1 for r in results if '✅' in r['status'])
    total = len(results)
    
    print(f"\n✅ Passed: {passed}/{total} ({passed/total*100:.0f}%)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    test_query_types()