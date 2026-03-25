"""
Test Script: Complete Ingestion Pipeline

Tests the full pipeline:
1. Load documents (PDF, DOCX, TXT)
2. Chunk documents (hierarchical)
3. Generate embeddings (Voyage AI)
4. Store in ChromaDB
5. Search and retrieve

Usage:
    python test_ingestion_pipeline.py
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ingestion import DocumentLoader, DocumentChunker, EmbeddingGenerator
from src.storage import VectorStore
from src.agents.retrieval.vector_agent import VectorSearchAgent
from src.models.agent_state import AgentState


def test_ingestion_pipeline():
    """Test complete ingestion pipeline."""
    
    print("=" * 60)
    print("INGESTION PIPELINE TEST")
    print("=" * 60)
    
    # Step 1: Load Documents
    print("\n📄 Step 1: Loading Documents...")
    print("-" * 60)
    
    loader = DocumentLoader()
    test_dir = Path("data/test_documents")
    
    documents = []
    for file_path in test_dir.glob("*.txt"):
        print(f"Loading: {file_path.name}")
        doc = loader.load(str(file_path))
        documents.append(doc)
        print(f"  ✅ Loaded: {len(doc.text)} chars, doc_id={doc.doc_id[:8]}...")
    
    print(f"\n✅ Total documents loaded: {len(documents)}")
    
    # Step 2: Chunk Documents
    print("\n✂️  Step 2: Chunking Documents...")
    print("-" * 60)
    
    chunker = DocumentChunker()
    
    all_chunks = []
    all_child_chunks = []
    
    for doc in documents:
        print(f"\nChunking: {doc.metadata.get('filename', 'unknown')}")
        chunks = chunker.chunk(doc)
        
        parent_chunks = chunker.get_parent_chunks_only(chunks)
        child_chunks = chunker.get_child_chunks_only(chunks)
        
        print(f"  Parents: {len(parent_chunks)}")
        print(f"  Children: {len(child_chunks)}")
        
        all_chunks.extend(chunks)
        all_child_chunks.extend(child_chunks)
    
    print(f"\n✅ Total chunks created:")
    print(f"   Parents: {len(all_chunks) - len(all_child_chunks)}")
    print(f"   Children: {len(all_child_chunks)}")
    print(f"   Total: {len(all_chunks)}")
    
    # Step 3: Generate Embeddings
    print("\n🧮 Step 3: Generating Embeddings...")
    print("-" * 60)
    
    print("⚠️  This will call Voyage AI API (may take time + cost money)")
    response = input("Continue? (y/n): ")
    
    if response.lower() != 'y':
        print("❌ Embedding generation skipped")
        print("   Use mock mode for VectorSearchAgent instead")
        return
    
    embedder = EmbeddingGenerator()
    
    # Generate embeddings for child chunks only (for retrieval)
    print(f"\nGenerating embeddings for {len(all_child_chunks)} child chunks...")
    texts = [chunk.text for chunk in all_child_chunks]
    
    embeddings = embedder.generate(texts)
    
    print(f"✅ Generated {len(embeddings)} embeddings")
    print(f"   Dimension: {len(embeddings[0])}")
    print(f"   Stats: {embedder.get_stats()}")
    
    # Step 4: Store in ChromaDB
    print("\n💾 Step 4: Storing in ChromaDB...")
    print("-" * 60)
    
    store = VectorStore()
    
    print(f"Collection: {store.collection_name}")
    print(f"Before: {store.count()} chunks")
    
    # Add chunks with embeddings
    store.add_chunks(all_child_chunks, embeddings)
    
    print(f"After: {store.count()} chunks")
    print(f"✅ Stored {len(all_child_chunks)} chunks in vector store")
    
    # Step 5: Test Vector Search
    print("\n🔍 Step 5: Testing Vector Search...")
    print("-" * 60)
    
    # Initialize agent in REAL mode
    agent = VectorSearchAgent(top_k=5, mock_mode=False)
    
    # Test queries
    test_queries = [
        "What is Python programming?",
        "Explain machine learning",
        "How does supervised learning work?"
    ]
    
    for query in test_queries:
        print(f"\n🔎 Query: {query}")
        print("-" * 40)
        
        state = AgentState(query=query)
        result = agent.run(state)
        
        print(f"Retrieved: {len(result.chunks)} chunks")
        
        for i, chunk in enumerate(result.chunks[:3], 1):
            print(f"\n{i}. Score: {chunk.score:.4f}")
            print(f"   Text: {chunk.text[:100]}...")
            print(f"   Source: {chunk.metadata.get('filename', 'unknown')}")
    
    # Final Stats
    print("\n" + "=" * 60)
    print("PIPELINE TEST COMPLETE")
    print("=" * 60)
    print(f"\n✅ Documents: {len(documents)}")
    print(f"✅ Chunks: {len(all_child_chunks)}")
    print(f"✅ Embeddings: {len(embeddings)}")
    print(f"✅ Vector Store: {store.count()} chunks")
    print(f"✅ Search: Working in REAL mode")
    
    print("\n💡 Next steps:")
    print("   1. Try more queries with the VectorSearchAgent")
    print("   2. Upload your own documents")
    print("   3. Integrate with full workflow (Planner → Coordinator → Validator)")


if __name__ == "__main__":
    try:
        test_ingestion_pipeline()
    except KeyboardInterrupt:
        print("\n\n❌ Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()