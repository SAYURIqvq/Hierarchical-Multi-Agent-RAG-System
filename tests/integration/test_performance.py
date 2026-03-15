"""
Test Performance - Week 10 Day 4
Benchmark graph search performance.
"""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.graph.graph_builder import KnowledgeGraph
from src.storage.chroma_store import ChromaVectorStore
from src.retrieval.graph_retrieval import GraphRetrieval


def test_performance():
    """Benchmark graph search performance."""
    
    print("\n🧪 Testing Performance\n")
    
    # Setup
    kg = KnowledgeGraph()
    kg.load('data/graphs/machine_learning.txt_graph.pkl')
    
    vector_store = ChromaVectorStore(persist_directory="data/chroma_db")
    
    graph_retrieval = GraphRetrieval(
        knowledge_graph=kg,
        vector_store=vector_store
    )
    
    queries = [
        "What is machine learning?",
        "How does TensorFlow work?",
        "Explain neural networks",
        "Compare supervised vs unsupervised learning",
        "What is the relationship between AI and deep learning?"
    ]
    
    print("Running 5 queries...\n")
    
    timings = []
    
    for i, query in enumerate(queries, 1):
        print(f"{i}. Query: {query}")
        
        start = time.time()
        chunks = graph_retrieval.search(query, top_k=5)
        elapsed = time.time() - start
        
        timings.append(elapsed)
        
        print(f"   Time: {elapsed:.2f}s, Chunks: {len(chunks)}")
    
    # Statistics
    avg_time = sum(timings) / len(timings)
    min_time = min(timings)
    max_time = max(timings)
    
    print(f"\n{'='*60}")
    print("📊 Performance Statistics:\n")
    print(f"Average time: {avg_time:.2f}s")
    print(f"Min time: {min_time:.2f}s")
    print(f"Max time: {max_time:.2f}s")
    
    # Goals
    print(f"\n🎯 Performance Goals:")
    print(f"   Target avg: <3s")
    print(f"   Current avg: {avg_time:.2f}s")
    
    if avg_time < 3:
        print(f"   ✅ PASS - Meeting target!")
    else:
        print(f"   ⚠️  WARN - Above target (+{avg_time-3:.2f}s)")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    test_performance()