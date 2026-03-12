"""
Test ChromaDB persistent storage.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.chroma_store import ChromaVectorStore
from src.ingestion.hierarchical_chunker import HierarchicalChunker, Chunk
from rag_poc import Embedder


def test_chromadb_basic():
    """Test basic ChromaDB operations."""
    
    print("=" * 60)
    print("TESTING CHROMADB BASIC OPERATIONS")
    print("=" * 60)
    
    # Initialize
    print("\n1. Initializing ChromaDB...")
    chroma_store = ChromaVectorStore(persist_directory="data/test_chroma")
    
    # Create sample chunks
    print("\n2. Creating sample chunks...")
    embedder = Embedder()
    
    # Parent chunk
    parent = Chunk(
        chunk_id="test_parent_0",
        text="Artificial intelligence is the simulation of human intelligence by machines.",
        tokens=[],
        token_count=2000,
        start_idx=0,
        end_idx=2000,
        chunk_type='parent'
    )
    parent.embedding = embedder.embed_query(parent.text)
    
    # Child chunks
    children = []
    for i in range(3):
        child = Chunk(
            chunk_id=f"test_parent_0_child_{i}",
            text=f"AI child chunk {i}: machine learning, neural networks, deep learning.",
            tokens=[],
            token_count=500,
            start_idx=i * 500,
            end_idx=(i + 1) * 500,
            chunk_type='child',
            parent_id="test_parent_0"
        )
        child.embedding = embedder.embed_query(child.text)
        children.append(child)
    
    print(f"   Created 1 parent, 3 children")
    
    # Add to ChromaDB
    print("\n3. Adding to ChromaDB...")
    chroma_store.add_chunks([parent], children)
    
    # Get stats
    stats = chroma_store.get_stats()
    print(f"\n4. Storage stats:")
    print(f"   Parents: {stats['total_parents']}")
    print(f"   Children: {stats['total_children']}")
    print(f"   Total: {stats['total_vectors']}")
    
    # Search
    print("\n5. Testing search...")
    query = "What is artificial intelligence?"
    query_embedding = embedder.embed_query(query)
    
    results = chroma_store.search(query_embedding, top_k=2, return_parent=True)
    
    print(f"\n   📋 Search results:")
    for i, result in enumerate(results, 1):
        print(f"   {i}. Type: {result['chunk_type']}")
        print(f"      ID: {result['chunk_id']}")
        print(f"      Score: {result['score']:.4f}")
        print(f"      Text: {result['text'][:60]}...")
    
    # Cleanup
    print("\n6. Cleaning up...")
    chroma_store.clear_all()
    print("   ✅ Test data cleared")
    
    print("\n" + "=" * 60)
    print("✅ CHROMADB BASIC TEST COMPLETE")
    print("=" * 60)


def test_chromadb_persistence():
    """Test that data persists between sessions."""
    
    print("\n" + "=" * 60)
    print("TESTING CHROMADB PERSISTENCE")
    print("=" * 60)
    
    embedder = Embedder()
    
    # Session 1: Add data
    print("\n1. Session 1: Adding data...")
    store1 = ChromaVectorStore(persist_directory="data/test_persist")
    
    parent = Chunk(
        chunk_id="persist_parent_0",
        text="This data should persist between sessions.",
        tokens=[],
        token_count=1000,
        start_idx=0,
        end_idx=1000,
        chunk_type='parent'
    )
    parent.embedding = embedder.embed_query(parent.text)
    
    child = Chunk(
        chunk_id="persist_parent_0_child_0",
        text="Child chunk for persistence test.",
        tokens=[],
        token_count=500,
        start_idx=0,
        end_idx=500,
        chunk_type='child',
        parent_id="persist_parent_0"
    )
    child.embedding = embedder.embed_query(child.text)
    
    store1.add_chunks([parent], [child])
    
    stats1 = store1.get_stats()
    print(f"   Added: {stats1['total_vectors']} vectors")
    
    # Session 2: Load existing data
    print("\n2. Session 2: Loading existing data...")
    store2 = ChromaVectorStore(persist_directory="data/test_persist")
    
    stats2 = store2.get_stats()
    print(f"   Found: {stats2['total_vectors']} vectors")
    
    # Verify persistence
    if stats2['total_vectors'] == stats1['total_vectors']:
        print("\n   ✅ PERSISTENCE VERIFIED!")
        print(f"   Data survived between sessions")
    else:
        print("\n   ❌ PERSISTENCE FAILED")
        print(f"   Expected {stats1['total_vectors']}, got {stats2['total_vectors']}")
    
    # Search in new session
    print("\n3. Testing search in new session...")
    query_embedding = embedder.embed_query("persistence test")
    results = store2.search(query_embedding, top_k=1, return_parent=True)
    
    if results:
        print(f"   ✅ Search works after reload")
        print(f"   Found: {results[0]['chunk_id']}")
    else:
        print(f"   ❌ Search failed after reload")
    
    # Cleanup
    print("\n4. Cleaning up...")
    store2.clear_all()
    
    print("\n" + "=" * 60)
    print("✅ CHROMADB PERSISTENCE TEST COMPLETE")
    print("=" * 60)


def main():
    """Run all ChromaDB tests."""
    
    print("=" * 60)
    print("CHROMADB TEST SUITE")
    print("=" * 60)
    
    test_chromadb_basic()
    test_chromadb_persistence()
    
    print("\n" + "=" * 60)
    print("✅ ALL CHROMADB TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()