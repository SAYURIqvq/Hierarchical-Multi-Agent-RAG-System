"""Test Graph Retrieval - Week 10 Day 2"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.retrieval.graph_retrieval import GraphRetrieval
from src.graph.graph_builder import KnowledgeGraph
from src.storage.chroma_store import ChromaVectorStore


def test_graph_retrieval():
    """Test graph-based retrieval."""
    
    print("\n🧪 Testing Graph Retrieval\n")
    
    # Load knowledge graph
    kg = KnowledgeGraph()
    kg.load('data/graphs/machine_learning.txt_graph.pkl')
    print(f"📂 Loaded graph: {kg}")
    
    # Load vector store
    vector_store = ChromaVectorStore(persist_directory="data/chroma_db")
    try:
        parent_count = vector_store.parent_collection.count()
        child_count = vector_store.child_collection.count()
        print(f"📂 Vector store: {parent_count} parents, {child_count} children\n")
    except:
        print(f"📂 Vector store loaded\n")
    
    # Create graph retrieval (CRITICAL - must be here!)
    graph_retrieval = GraphRetrieval(
        knowledge_graph=kg,
        vector_store=vector_store
    )
    print("✅ Graph retrieval initialized\n")
    
    # Test queries
    test_queries = [
        "How does Google relate to machine learning?",
        "What is the connection between Python and neural networks?",
        "Explain the relationship between TensorFlow and AI"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"{'='*60}")
        print(f"🔍 Test {i}: {query}\n")
        
        try:
            # Search
            chunks = graph_retrieval.search(query, top_k=5)
            
            print(f"✅ Retrieved {len(chunks)} chunks\n")
            
            if chunks:
                print("Top 3 chunks:")
                for j, chunk in enumerate(chunks[:3], 1):
                    print(f"\n{j}. Score: {chunk.score:.3f}")
                    print(f"   File: {chunk.metadata.get('filename', 'unknown')}")
                    print(f"   Text: {chunk.text[:150]}...")
            else:
                print("No chunks retrieved")
        
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
        print()
    
    print(f"{'='*60}")
    print("✅ All tests complete!\n")


if __name__ == "__main__":
    test_graph_retrieval()