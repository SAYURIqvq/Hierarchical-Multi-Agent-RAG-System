"""
Compare hierarchical vs flat chunking performance.
Measure accuracy improvement.
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # ← ADD THIS LINE

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_poc import DocumentLoader, TextChunker, SimpleVectorStore, Embedder, AnswerGenerator
from src.ingestion.hierarchical_chunker import HierarchicalChunker
from src.storage.hierarchical_store import HierarchicalVectorStore


def test_flat_chunking(text: str, test_queries: list):
    """Test with flat chunking (old way)."""
    
    print("\n" + "=" * 60)
    print("TESTING FLAT CHUNKING")
    print("=" * 60)
    
    # Setup
    chunker = TextChunker(chunk_size=500, chunk_overlap=50)
    embedder = Embedder()
    vector_store = SimpleVectorStore()
    generator = AnswerGenerator()
    
    # Process
    print("\n1. Chunking...")
    chunks = chunker.chunk_text(text)
    print(f"   Created {len(chunks)} chunks")
    
    print("\n2. Embedding...")
    texts = [c['text'] for c in chunks]
    embeddings = [embedder.embed_query(t) for t in texts]
    
    # Add to store
    for chunk, embedding in zip(chunks, embeddings):
        chunk['embedding'] = embedding
    vector_store.chunks = chunks
    
    print(f"   Embedded {len(chunks)} chunks")
    
    # Test queries
    print("\n3. Testing queries...")
    results = []
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n   Query {i}: {query}")
        
        query_emb = embedder.embed_query(query)
        relevant = vector_store.search(query_emb, top_k=5)
        
        answer = generator.generate(query, relevant)
        
        avg_score = sum(r['score'] for r in relevant) / len(relevant)
        answer_length = len(answer['answer'])
        
        print(f"   → Avg relevance: {avg_score:.4f}")
        print(f"   → Answer length: {answer_length} chars")
        
        results.append({
            'query': query,
            'avg_score': avg_score,
            'answer_length': answer_length,
            'answer': answer['answer']
        })
    
    return results


def test_hierarchical_chunking(text: str, test_queries: list):
    """Test with hierarchical chunking (new way)."""
    
    print("\n" + "=" * 60)
    print("TESTING HIERARCHICAL CHUNKING")
    print("=" * 60)
    
    # Setup
    chunker = HierarchicalChunker(parent_size=2000, child_size=500, child_overlap=50)
    embedder = Embedder()
    vector_store = HierarchicalVectorStore()
    generator = AnswerGenerator()
    
    # Process
    print("\n1. Chunking...")
    parent_chunks, child_chunks = chunker.chunk_text(text)
    print(f"   Created {len(parent_chunks)} parents, {len(child_chunks)} children")
    
    print("\n2. Embedding...")
    for p in parent_chunks:
        p.embedding = embedder.embed_query(p.text)
    for c in child_chunks:
        c.embedding = embedder.embed_query(c.text)
    
    vector_store.add_chunks(parent_chunks, child_chunks)
    print(f"   Embedded {len(parent_chunks)} parents, {len(child_chunks)} children")
    
    # Test queries
    print("\n3. Testing queries...")
    results = []
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n   Query {i}: {query}")
        
        query_emb = embedder.embed_query(query)
        relevant = vector_store.search(query_emb, top_k=5, return_parent=True)
        
        answer = generator.generate(query, relevant)
        
        avg_score = sum(r['score'] for r in relevant) / len(relevant)
        answer_length = len(answer['answer'])
        
        print(f"   → Avg relevance: {avg_score:.4f}")
        print(f"   → Answer length: {answer_length} chars")
        
        results.append({
            'query': query,
            'avg_score': avg_score,
            'answer_length': answer_length,
            'answer': answer['answer']
        })
    
    return results


def main():
    """Compare both chunking methods."""
    
    print("=" * 60)
    print("CHUNKING MODE COMPARISON TEST")
    print("=" * 60)
    
    # Load document
    print("\nLoading document...")
    loader = DocumentLoader()
    text = loader.load("data/uploads/sample.pdf")
    print(f"Loaded: {len(text)} characters")
    
    # Test queries
    test_queries = [
        "What is this document about?",
        "Summarize the main content",
        "Give me specific details"
    ]
    
    # Test both modes
    flat_results = test_flat_chunking(text, test_queries)
    hierarchical_results = test_hierarchical_chunking(text, test_queries)
    
    # Compare
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    
    for i, query in enumerate(test_queries):
        flat = flat_results[i]
        hier = hierarchical_results[i]
        
        print(f"\n📝 Query {i+1}: {query}")
        print(f"   Flat chunking:")
        print(f"      Relevance: {flat['avg_score']:.4f}")
        print(f"      Answer length: {flat['answer_length']} chars")
        
        print(f"   Hierarchical chunking:")
        print(f"      Relevance: {hier['avg_score']:.4f}")
        print(f"      Answer length: {hier['answer_length']} chars")
        
        # Calculate improvement
        score_diff = hier['avg_score'] - flat['avg_score']
        length_diff = hier['answer_length'] - flat['answer_length']
        
        print(f"   📈 Improvement:")
        print(f"      Relevance: {score_diff:+.4f} ({score_diff/flat['avg_score']*100:+.1f}%)")
        print(f"      Length: {length_diff:+d} chars ({length_diff/flat['answer_length']*100:+.1f}%)")
    
    # Overall summary
    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)
    
    flat_avg_score = sum(r['avg_score'] for r in flat_results) / len(flat_results)
    hier_avg_score = sum(r['avg_score'] for r in hierarchical_results) / len(hierarchical_results)
    
    flat_avg_length = sum(r['answer_length'] for r in flat_results) / len(flat_results)
    hier_avg_length = sum(r['answer_length'] for r in hierarchical_results) / len(hierarchical_results)
    
    print(f"\nAverage Relevance Score:")
    print(f"  Flat: {flat_avg_score:.4f}")
    print(f"  Hierarchical: {hier_avg_score:.4f}")
    print(f"  Improvement: {(hier_avg_score - flat_avg_score)/flat_avg_score*100:+.1f}%")
    
    print(f"\nAverage Answer Length:")
    print(f"  Flat: {flat_avg_length:.0f} chars")
    print(f"  Hierarchical: {hier_avg_length:.0f} chars")
    print(f"  Improvement: {(hier_avg_length - flat_avg_length)/flat_avg_length*100:+.1f}%")
    
    print("\n✅ COMPARISON TEST COMPLETE")


if __name__ == "__main__":
    main()