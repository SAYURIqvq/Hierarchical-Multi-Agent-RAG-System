"""
Performance benchmarking for RAG system.
Measures latency and throughput.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_poc import (
    DocumentLoader,
    TextChunker,
    Embedder,
    SimpleVectorStore,
    AnswerGenerator
)


def benchmark_loading():
    """Benchmark document loading."""
    print("\n" + "=" * 60)
    print("BENCHMARK: DOCUMENT LOADING")
    print("=" * 60)
    
    loader = DocumentLoader()
    file_path = "data/uploads/sample.pdf"
    
    # Warmup
    loader.load(file_path)
    
    # Benchmark
    iterations = 5
    times = []
    
    for i in range(iterations):
        start = time.time()
        text = loader.load(file_path)
        end = time.time()
        times.append(end - start)
    
    avg_time = sum(times) / len(times)
    print(f"Average loading time: {avg_time:.3f}s ({iterations} runs)")
    print(f"Min: {min(times):.3f}s | Max: {max(times):.3f}s")


def benchmark_chunking():
    """Benchmark chunking."""
    print("\n" + "=" * 60)
    print("BENCHMARK: TEXT CHUNKING")
    print("=" * 60)
    
    loader = DocumentLoader()
    chunker = TextChunker(chunk_size=500, chunk_overlap=50)
    
    text = loader.load("data/uploads/sample.pdf")
    
    # Benchmark
    iterations = 10
    times = []
    
    for i in range(iterations):
        start = time.time()
        chunks = chunker.chunk_text(text)
        end = time.time()
        times.append(end - start)
    
    avg_time = sum(times) / len(times)
    print(f"Average chunking time: {avg_time:.3f}s ({iterations} runs)")
    print(f"Chunks created: {len(chunks)}")


def benchmark_embedding():
    """Benchmark embedding generation."""
    print("\n" + "=" * 60)
    print("BENCHMARK: EMBEDDING GENERATION")
    print("=" * 60)
    
    embedder = Embedder()
    
    # Single query
    start = time.time()
    embedding = embedder.embed_query("Test query")
    end = time.time()
    
    print(f"Single query embedding: {end - start:.3f}s")


def benchmark_search():
    """Benchmark vector search."""
    print("\n" + "=" * 60)
    print("BENCHMARK: VECTOR SEARCH")
    print("=" * 60)
    
    # Setup
    embedder = Embedder()
    vector_store = SimpleVectorStore()
    
    # Create chunks
    loader = DocumentLoader()
    chunker = TextChunker()
    text = loader.load("data/uploads/sample.pdf")
    chunks = chunker.chunk_text(text)
    chunks = embedder.embed_chunks(chunks)
    vector_store.add_chunks(chunks)
    
    # Benchmark search
    query = "What is this document about?"
    query_embedding = embedder.embed_query(query)
    
    iterations = 100
    times = []
    
    for i in range(iterations):
        start = time.time()
        results = vector_store.search(query_embedding, top_k=5)
        end = time.time()
        times.append(end - start)
    
    avg_time = sum(times) / len(times)
    print(f"Average search time: {avg_time*1000:.2f}ms ({iterations} runs)")
    print(f"Throughput: {1/avg_time:.0f} searches/second")


def benchmark_end_to_end():
    """Benchmark complete RAG pipeline."""
    print("\n" + "=" * 60)
    print("BENCHMARK: END-TO-END RAG QUERY")
    print("=" * 60)
    
    # Setup
    loader = DocumentLoader()
    chunker = TextChunker()
    embedder = Embedder()
    vector_store = SimpleVectorStore()
    generator = AnswerGenerator()
    
    # Load and process
    text = loader.load("data/uploads/sample.pdf")
    chunks = chunker.chunk_text(text)
    chunks = embedder.embed_chunks(chunks)
    vector_store.add_chunks(chunks)
    
    # Benchmark query
    query = "What is this document about?"
    
    iterations = 3
    times = []
    
    for i in range(iterations):
        start = time.time()
        
        # Full pipeline
        query_embedding = embedder.embed_query(query)
        relevant_chunks = vector_store.search(query_embedding, top_k=5)
        result = generator.generate(query, relevant_chunks)
        
        end = time.time()
        times.append(end - start)
        
        print(f"  Run {i+1}: {times[-1]:.2f}s")
    
    avg_time = sum(times) / len(times)
    print(f"\nAverage end-to-end time: {avg_time:.2f}s")
    print(f"Target: <5s for complex queries ✅" if avg_time < 5 else "Target: <5s ⚠️")


if __name__ == "__main__":
    print("=" * 60)
    print("RAG SYSTEM PERFORMANCE BENCHMARKS")
    print("=" * 60)
    
    benchmark_loading()
    benchmark_chunking()
    benchmark_embedding()
    benchmark_search()
    benchmark_end_to_end()
    
    print("\n" + "=" * 60)
    print("BENCHMARKS COMPLETE")
    print("=" * 60)