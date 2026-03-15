"""
Test hierarchical chunking integrated with embeddings and search.
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # ← ADD THIS

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.hierarchical_chunker import HierarchicalChunker
from src.storage.hierarchical_store import HierarchicalVectorStore
from rag_poc import DocumentLoader, Embedder


def test_full_pipeline():
    """Test complete hierarchical RAG pipeline."""
    
    print("=" * 60)
    print("TESTING HIERARCHICAL RAG PIPELINE")
    print("=" * 60)
    
    # 1. Load document
    print("\n1️⃣ Loading document...")
    loader = DocumentLoader()
    text = loader.load("data/uploads/sample.pdf")
    print(f"   Loaded: {len(text)} characters")
    
    # 2. Create hierarchical chunks
    print("\n2️⃣ Creating hierarchical chunks...")
    chunker = HierarchicalChunker(
        parent_size=1000,  # Smaller for testing
        child_size=300,
        child_overlap=30
    )
    parent_chunks, child_chunks = chunker.chunk_text(text)
    
    # 3. Generate embeddings
    print("\n3️⃣ Generating embeddings...")
    embedder = Embedder()
    
    # Embed parents
    print("   Embedding parents...")
    for parent in parent_chunks:
        parent.embedding = embedder.embed_query(parent.text)
    
    # Embed children
    print("   Embedding children...")
    for child in child_chunks:
        child.embedding = embedder.embed_query(child.text)
    
    print(f"   ✅ Embedded {len(parent_chunks)} parents, {len(child_chunks)} children")
    
    # 4. Store in hierarchical store
    print("\n4️⃣ Storing in vector database...")
    store = HierarchicalVectorStore()
    store.add_chunks(parent_chunks, child_chunks)
    
    # 5. Test search
    print("\n5️⃣ Testing search...")
    query = "What is this document about?"
    print(f"   Query: '{query}'")
    
    query_embedding = embedder.embed_query(query)
    results = store.search(query_embedding, top_k=3, return_parent=True)
    
    print(f"\n   📋 Top 3 Results:")
    for i, result in enumerate(results, 1):
        print(f"\n   {i}. Score: {result['score']:.4f}")
        print(f"      Type: {result['chunk_type']}")
        print(f"      Text: {result['text'][:100]}...")
        
        if 'child_chunk' in result:
            print(f"      (Searched in child, returned parent for context)")
    
    # 6. Compare: child only vs parent context
    print("\n6️⃣ Comparing retrieval strategies...")
    
    # Child only
    results_child = store.search(query_embedding, top_k=3, return_parent=False)
    child_text_length = sum(len(r['text']) for r in results_child)
    
    # Parent context
    results_parent = store.search(query_embedding, top_k=3, return_parent=True)
    parent_text_length = sum(len(r['text']) for r in results_parent)
    
    print(f"   Child only: {child_text_length} chars total")
    print(f"   With parent: {parent_text_length} chars total")
    print(f"   Context gain: {parent_text_length - child_text_length} chars (+{(parent_text_length/child_text_length - 1)*100:.0f}%)")
    
    print("\n" + "=" * 60)
    print("✅ HIERARCHICAL PIPELINE TEST COMPLETE")
    print("=" * 60)
    print("\nBenefits demonstrated:")
    print("  ✅ Search in child chunks (precise)")
    print("  ✅ Return parent chunks (full context)")
    print("  ✅ Flexible retrieval strategy")
    print("  ✅ Better for LLM generation")


if __name__ == "__main__":
    test_full_pipeline()