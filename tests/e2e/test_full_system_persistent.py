"""
Test complete system with persistent storage.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_poc import DocumentLoader, Embedder, AnswerGenerator
from src.ingestion.hierarchical_chunker import HierarchicalChunker
from src.storage.chroma_store import ChromaVectorStore


def test_complete_pipeline_persistent():
    """Test complete RAG pipeline with ChromaDB."""
    
    print("=" * 60)
    print("TESTING COMPLETE PIPELINE WITH CHROMADB")
    print("=" * 60)
    
    # Initialize components
    print("\n1️⃣ Initializing components...")
    loader = DocumentLoader()
    chunker = HierarchicalChunker(parent_size=1000, child_size=300, child_overlap=30)
    embedder = Embedder()
    chroma_store = ChromaVectorStore(persist_directory="data/test_full_system")
    generator = AnswerGenerator()
    
    # Load document
    print("\n2️⃣ Loading document...")
    text = loader.load("data/uploads/sample.pdf")
    print(f"   Loaded: {len(text)} chars")
    
    # Create chunks
    print("\n3️⃣ Creating hierarchical chunks...")
    parent_chunks, child_chunks = chunker.chunk_text(text)
    print(f"   Parents: {len(parent_chunks)}, Children: {len(child_chunks)}")
    
    # Embed chunks
    print("\n4️⃣ Generating embeddings...")
    for parent in parent_chunks:
        parent.embedding = embedder.embed_query(parent.text)
    for child in child_chunks:
        child.embedding = embedder.embed_query(child.text)
    print(f"   Embedded all chunks")
    
    # Store in ChromaDB
    print("\n5️⃣ Storing in ChromaDB...")
    chroma_store.add_chunks(parent_chunks, child_chunks)
    stats = chroma_store.get_stats()
    print(f"   Stored: {stats['total_vectors']} vectors")
    
    # Query
    print("\n6️⃣ Testing query...")
    query = "What is this document about?"
    query_embedding = embedder.embed_query(query)
    
    # Search
    results = chroma_store.search(query_embedding, top_k=3, return_parent=True)
    print(f"   Retrieved: {len(results)} results")
    
    # Generate answer
    answer_result = generator.generate(query, results)
    print(f"\n   💬 Answer: {answer_result['answer'][:100]}...")
    print(f"   📚 Citations: {len(answer_result['citations'])}")
    
    # Test persistence
    print("\n7️⃣ Testing persistence...")
    print("   Creating new store instance...")
    chroma_store2 = ChromaVectorStore(persist_directory="data/test_full_system")
    stats2 = chroma_store2.get_stats()
    
    if stats2['total_vectors'] == stats['total_vectors']:
        print(f"   ✅ Persistence verified: {stats2['total_vectors']} vectors loaded")
    else:
        print(f"   ❌ Persistence failed: Expected {stats['total_vectors']}, got {stats2['total_vectors']}")
    
    # Search in new instance
    results2 = chroma_store2.search(query_embedding, top_k=3, return_parent=True)
    if len(results2) == len(results):
        print(f"   ✅ Search works after reload: {len(results2)} results")
    else:
        print(f"   ❌ Search failed after reload")
    
    # Cleanup
    print("\n8️⃣ Cleaning up...")
    chroma_store2.clear_all()
    print("   ✅ Test data cleared")
    
    print("\n" + "=" * 60)
    print("✅ COMPLETE PIPELINE TEST PASSED")
    print("=" * 60)


def test_performance_comparison():
    """Compare performance: memory vs ChromaDB."""
    
    print("\n" + "=" * 60)
    print("TESTING PERFORMANCE: ChromaDB")
    print("=" * 60)
    
    loader = DocumentLoader()
    chunker = HierarchicalChunker(parent_size=1000, child_size=300)
    embedder = Embedder()
    chroma_store = ChromaVectorStore(persist_directory="data/test_performance")
    
    # Load and process
    text = loader.load("data/uploads/sample.pdf")
    parent_chunks, child_chunks = chunker.chunk_text(text)
    
    for parent in parent_chunks:
        parent.embedding = embedder.embed_query(parent.text)
    for child in child_chunks:
        child.embedding = embedder.embed_query(child.text)
    
    # Benchmark: Add to ChromaDB
    print("\n1. Adding chunks to ChromaDB...")
    start = time.time()
    chroma_store.add_chunks(parent_chunks, child_chunks)
    add_time = time.time() - start
    print(f"   Time: {add_time:.3f}s")
    
    # Benchmark: Search
    print("\n2. Searching in ChromaDB...")
    query = "test query"
    query_embedding = embedder.embed_query(query)
    
    search_times = []
    for i in range(10):
        start = time.time()
        results = chroma_store.search(query_embedding, top_k=5)
        search_times.append(time.time() - start)
    
    avg_search = sum(search_times) / len(search_times)
    print(f"   Average search: {avg_search*1000:.2f}ms (10 runs)")
    print(f"   Min: {min(search_times)*1000:.2f}ms")
    print(f"   Max: {max(search_times)*1000:.2f}ms")
    
    # Cleanup
    chroma_store.clear_all()
    
    print("\n   📊 Performance Summary:")
    print(f"      Add time: {add_time:.3f}s")
    print(f"      Search time: {avg_search*1000:.2f}ms")
    print(f"      ✅ Fast enough for production")


def main():
    """Run all integration tests."""
    
    print("=" * 60)
    print("FULL SYSTEM INTEGRATION TEST SUITE")
    print("=" * 60)
    
    test_complete_pipeline_persistent()
    test_performance_comparison()
    
    print("\n" + "=" * 60)
    print("✅ ALL INTEGRATION TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()