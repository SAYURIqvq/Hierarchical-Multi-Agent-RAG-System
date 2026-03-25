"""
Test Synthesis Agent

Test deduplication, hybrid ranking, and result fusion.

Usage:
    python examples/test_synthesis.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.synthesis import SynthesisAgent
from src.agents.retrieval.vector_agent import VectorSearchAgent
from src.agents.retrieval.keyword_agent import KeywordSearchAgent
from src.models.agent_state import AgentState, Chunk


def test_synthesis():
    """Test synthesis agent with multi-source results."""
    
    print("=" * 60)
    print("TEST SYNTHESIS AGENT")
    print("=" * 60)
    
    # Test query
    query = "Python machine learning"
    
    print(f"\n🔎 Query: {query}")
    print("-" * 60)
    
    # Step 1: Get results from multiple sources
    print("\n📥 Step 1: Retrieving from multiple sources...")
    
    # Vector search
    print("\n   Vector Search:")
    vector_agent = VectorSearchAgent(top_k=5, mock_mode=False)
    vector_state = AgentState(query=query)
    vector_result = vector_agent.run(vector_state)
    print(f"   ✅ Retrieved: {len(vector_result.chunks)} chunks")
    
    # Keyword search
    print("\n   Keyword Search:")
    keyword_agent = KeywordSearchAgent(top_k=5, mock_mode=False)
    keyword_state = AgentState(query=query)
    keyword_result = keyword_agent.run(keyword_state)
    print(f"   ✅ Retrieved: {len(keyword_result.chunks)} chunks")
    
    # Step 2: Combine results
    print("\n📦 Step 2: Combining results...")
    all_chunks = vector_result.chunks + keyword_result.chunks
    print(f"   Total chunks: {len(all_chunks)}")
    
    # Show sources
    sources = {}
    for chunk in all_chunks:
        source = chunk.metadata.get('source', 'unknown')
        sources[source] = sources.get(source, 0) + 1
    
    print(f"   Sources breakdown:")
    for source, count in sources.items():
        print(f"     - {source}: {count}")
    
    # Step 3: Synthesize
    print("\n🔬 Step 3: Synthesis (dedup + hybrid rank)...")
    
    synthesis_agent = SynthesisAgent(
        top_k=5,
        vector_weight=0.7,
        keyword_weight=0.3
    )
    
    combined_state = AgentState(query=query, chunks=all_chunks)
    result = synthesis_agent.run(combined_state)
    
    print(f"   ✅ Final chunks: {len(result.chunks)}")
    
    # Show stats
    stats = synthesis_agent.get_synthesis_stats(result)
    print(f"\n📊 Synthesis Stats:")
    print(f"   Input: {stats.get('input_count', 0)}")
    print(f"   Unique: {stats.get('unique_count', 0)}")
    print(f"   Final: {stats.get('final_count', 0)}")
    print(f"   Dedup rate: {stats.get('deduplication_rate', 0):.1%}")
    
    # Step 4: Display results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    
    for i, chunk in enumerate(result.chunks, 1):
        print(f"\n[{i}] Hybrid Score: {chunk.score:.4f}")
        print(f"    Original Score: {chunk.metadata.get('original_score', 0):.4f}")
        print(f"    Source: {chunk.metadata.get('source', 'unknown')}")
        print(f"    Text: {chunk.text[:150]}...")
    
    # Test with mock data (if no real results)
    print("\n" + "=" * 60)
    print("TEST WITH MOCK DATA")
    print("=" * 60)
    
    test_mock_synthesis()


def test_mock_synthesis():
    """Test synthesis with mock chunks."""
    
    # Create mock chunks with duplicates
    chunks = [
        # Vector results
        Chunk(
            text="Python is a programming language",
            doc_id="doc1",
            chunk_id="v1",
            score=0.9,
            metadata={"source": "vector", "method": "semantic"}
        ),
        Chunk(
            text="Machine learning with Python",
            doc_id="doc2",
            chunk_id="v2",
            score=0.85,
            metadata={"source": "vector", "method": "semantic"}
        ),
        # Keyword results
        Chunk(
            text="Python is a programming language",  # Duplicate
            doc_id="doc1",
            chunk_id="k1",
            score=0.8,
            metadata={"source": "keyword", "method": "bm25"}
        ),
        Chunk(
            text="Python programming tutorial",
            doc_id="doc3",
            chunk_id="k2",
            score=0.75,
            metadata={"source": "keyword", "method": "bm25"}
        ),
    ]
    
    print(f"\n📦 Mock chunks: {len(chunks)}")
    print("   - 2 from vector (score: 0.9, 0.85)")
    print("   - 2 from keyword (score: 0.8, 0.75)")
    print("   - 1 duplicate")
    
    # Synthesize
    agent = SynthesisAgent(top_k=3, vector_weight=0.7, keyword_weight=0.3)
    state = AgentState(query="test", chunks=chunks)
    result = agent.run(state)
    
    print(f"\n✅ After synthesis: {len(result.chunks)} unique chunks")
    
    # Display
    print("\nResults (by hybrid score):")
    for i, chunk in enumerate(result.chunks, 1):
        source = chunk.metadata.get('source')
        orig_score = chunk.metadata.get('original_score', 0)
        hybrid_score = chunk.score
        
        print(f"\n{i}. {chunk.text}")
        print(f"   Source: {source}")
        print(f"   Original: {orig_score:.3f}")
        print(f"   Hybrid: {hybrid_score:.3f}")
        
        # Show calculation
        if source == "vector":
            expected = orig_score * 0.7
        else:
            expected = orig_score * 0.3
        print(f"   Expected: {expected:.3f} ({'✓' if abs(hybrid_score - expected) < 0.001 else '✗'})")


def main():
    """Main function."""
    
    try:
        test_synthesis()
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")