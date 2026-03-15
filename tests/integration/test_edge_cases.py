"""
Test Edge Cases - Week 10 Day 4
Test error handling and edge cases.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.graph.graph_builder import KnowledgeGraph
from src.storage.chroma_store import ChromaVectorStore
from src.retrieval.graph_retrieval import GraphRetrieval


def test_edge_cases():
    """Test edge cases and error handling."""
    
    print("\n🧪 Testing Edge Cases\n")
    
    # Setup
    kg = KnowledgeGraph()
    kg.load('data/graphs/machine_learning.txt_graph.pkl')
    
    vector_store = ChromaVectorStore(persist_directory="data/chroma_db")
    
    graph_retrieval = GraphRetrieval(
        knowledge_graph=kg,
        vector_store=vector_store
    )
    
    test_cases = [
        {
            "name": "Empty Query",
            "query": "",
            "should_handle": True
        },
        {
            "name": "Very Long Query",
            "query": "explain " * 100,  # 100 words
            "should_handle": True
        },
        {
            "name": "Special Characters",
            "query": "What is ML??? @#$%",
            "should_handle": True
        },
        {
            "name": "Non-existent Entities",
            "query": "How does Unicorn relate to Dragon?",
            "should_handle": True  # Should return gracefully
        },
        {
            "name": "Single Word",
            "query": "TensorFlow",
            "should_handle": True
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"{'='*60}")
        print(f"Test {i}: {test_case['name']}")
        print(f"Query: '{test_case['query'][:50]}...'\n")
        
        try:
            chunks = graph_retrieval.search(test_case['query'], top_k=5)
            
            status = "✅ HANDLED"
            print(f"{status} - Returned {len(chunks)} chunks")
            
            results.append({
                'name': test_case['name'],
                'status': status,
                'chunks': len(chunks)
            })
            
        except Exception as e:
            status = "❌ ERROR"
            print(f"{status} - {e}")
            
            results.append({
                'name': test_case['name'],
                'status': status,
                'error': str(e)
            })
        
        print()
    
    # Summary
    print(f"{'='*60}")
    print("📊 Edge Case Summary:\n")
    
    for result in results:
        print(f"{result['status']} {result['name']}")
    
    handled = sum(1 for r in results if '✅' in r['status'])
    total = len(results)
    
    print(f"\n✅ Handled: {handled}/{total} ({handled/total*100:.0f}%)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    test_edge_cases()