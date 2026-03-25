"""
Example: Batch Upload Multiple Documents

Upload entire folder of documents at once.

Usage:
    python examples/batch_upload.py data/test_documents/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion import DocumentLoader, DocumentChunker, EmbeddingGenerator
from src.storage import VectorStore


def batch_upload(folder_path: str):
    """Upload all documents in a folder."""
    
    folder = Path(folder_path)
    
    if not folder.exists() or not folder.is_dir():
        print(f"❌ Invalid folder: {folder_path}")
        return
    
    print(f"📂 Scanning: {folder_path}")
    print("=" * 60)
    
    # Find all documents
    supported_extensions = ['.txt', '.pdf', '.docx', '.md']
    files = []
    for ext in supported_extensions:
        files.extend(folder.glob(f"*{ext}"))
    
    if not files:
        print(f"❌ No documents found in {folder_path}")
        return
    
    print(f"Found {len(files)} documents:\n")
    for f in files:
        print(f"  - {f.name}")
    
    print("\n" + "=" * 60)
    print("UPLOADING DOCUMENTS")
    print("=" * 60)
    
    # Initialize components
    loader = DocumentLoader()
    chunker = DocumentChunker()
    embedder = EmbeddingGenerator()
    store = VectorStore()
    
    total_docs = 0
    total_chunks = 0
    
    # Process each file
    for i, file_path in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] Processing: {file_path.name}")
        print("-" * 40)
        
        try:
            # Load
            doc = loader.load(str(file_path))
            print(f"  Loaded: {len(doc.text)} chars")
            
            # Chunk
            chunks = chunker.chunk(doc)
            child_chunks = chunker.get_child_chunks_only(chunks)
            print(f"  Chunks: {len(child_chunks)}")
            
            # Embed (in batches)
            texts = [c.text for c in child_chunks]
            embeddings = embedder.generate(texts)
            print(f"  Embeddings: {len(embeddings)}")
            
            # Store
            store.add_chunks(child_chunks, embeddings)
            print(f"  ✅ Stored")
            
            total_docs += 1
            total_chunks += len(child_chunks)
            
        except Exception as e:
            print(f"  ❌ Failed: {str(e)}")
            continue
    
    # Summary
    print("\n" + "=" * 60)
    print("UPLOAD COMPLETE")
    print("=" * 60)
    print(f"✅ Documents uploaded: {total_docs}/{len(files)}")
    print(f"✅ Total chunks: {total_chunks}")
    print(f"✅ ChromaDB total: {store.count()}")
    
    # Show stats
    stats = embedder.get_stats()
    print(f"\n📊 Embedding Stats:")
    print(f"   Total embeddings: {stats['total_embeddings']}")
    print(f"   Model: {stats['model']}")
    print(f"   Dimension: {stats['embedding_dimension']}")


def main():
    """Main function."""
    
    if len(sys.argv) < 2:
        print("Usage: python batch_upload.py <folder_path>")
        print("\nExample:")
        print("  python batch_upload.py data/test_documents/")
        return
    
    folder_path = sys.argv[1]
    batch_upload(folder_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()