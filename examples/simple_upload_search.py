"""
Example: Simple Document Upload & Search

Demonstrates basic usage:
1. Upload a document
2. Search with queries
3. Get results

Usage:
    python examples/simple_upload_search.py path/to/document.pdf
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion import DocumentLoader, DocumentChunker, EmbeddingGenerator
from src.storage import VectorStore
from src.agents.retrieval.vector_agent import VectorSearchAgent
from src.models.agent_state import AgentState


def upload_document(file_path: str):
    """Upload and index a document."""
    
    print(f"📄 Uploading: {file_path}")
    print("-" * 60)
    
    # Load
    loader = DocumentLoader()
    doc = loader.load(file_path)
    print(f"✅ Loaded: {len(doc.text)} characters")
    
    # Chunk
    chunker = DocumentChunker()
    chunks = chunker.chunk(doc)
    child_chunks = chunker.get_child_chunks_only(chunks)
    print(f"✅ Created: {len(child_chunks)} chunks")
    
    # Embed
    embedder = EmbeddingGenerator()
    texts = [c.text for c in child_chunks]
    embeddings = embedder.generate(texts)
    print(f"✅ Generated: {len(embeddings)} embeddings")
    
    # Store
    store = VectorStore()
    store.add_chunks(child_chunks, embeddings)
    print(f"✅ Stored in ChromaDB (total: {store.count()} chunks)")
    
    return store


def search_documents(query: str, top_k: int = 5):
    """Search indexed documents."""
    
    print(f"\n🔍 Searching: {query}")
    print("-" * 60)
    
    # Search
    agent = VectorSearchAgent(top_k=top_k, mock_mode=False)
    state = AgentState(query=query)
    result = agent.run(state)
    
    # Display results
    print(f"Found {len(result.chunks)} results:\n")
    
    for i, chunk in enumerate(result.chunks, 1):
        print(f"{i}. Score: {chunk.score:.4f}")
        print(f"   {chunk.text[:200]}...")
        print(f"   Source: {chunk.metadata.get('filename', 'unknown')}")
        print()
    
    return result.chunks


def main():
    """Main function."""
    
    if len(sys.argv) < 2:
        print("Usage: python simple_upload_search.py <document_path>")
        print("\nExample:")
        print("  python simple_upload_search.py data/test_documents/python_guide.txt")
        return
    
    file_path = sys.argv[1]
    
    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        return
    
    # Upload
    store = upload_document(file_path)
    
    # Interactive search
    print("\n" + "=" * 60)
    print("READY TO SEARCH!")
    print("=" * 60)
    print("Type your questions (or 'quit' to exit)\n")
    
    while True:
        query = input("🔎 Query: ").strip()
        
        if not query or query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        search_documents(query, top_k=3)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()