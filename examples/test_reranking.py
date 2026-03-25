"""
Test Cohere Reranking

Compare hybrid ranking vs Cohere reranking.

Usage:
    python examples/test_reranking.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.synthesis import SynthesisAgent
from src.agents.retrieval.vector_agent import VectorSearchAgent
from src.agents.retrieval.keyword_agent import KeywordSearchAgent
from src.models.agent_state import AgentState
from src.config import get_settings


def test_reranking():
    """Test with and without Cohere reranking."""
    
    print("=" * 60)
    print("TEST COHERE RERANKING")
    print("=" * 60)
    
    # Check Cohere API key
    settings = get_settings()
    has_cohere = bool(settings.cohere_api_key)
    
    print(f"\n🔑 Cohere API Key: {'Found' if has_cohere else 'Not found'}")
    
    if not has_cohere:
        print("\n⚠️  Cohere reranking disabled (no API key)")
        print("   Set COHERE_API_KEY in .env to enable")
        print("   Continuing with hybrid ranking only...\n")
    
    # Test query
    query = "Python machine learning libraries"
    
    print(f"🔎 Query: {query}")
    print("-" * 60)
    
    # Step 1: Get multi-source results
    print("\n📥 Retrieving from sources...")
    
    vector_agent = VectorSearchAgent(top_k=5, mock_mode=False)
    keyword_agent = KeywordSearchAgent(top_k=5, mock_mode=False)
    
    vector_result = vector_agent.run(AgentState(query=query))
    keyword_result = keyword_agent.run(AgentState(query=query))
    
    all_chunks = vector_result.chunks + keyword_result.chunks
    
    print(f"   Vector: {len(vector_result.chunks)}")
    print(f"   Keyword: {len(keyword_result.chunks)}")
    print(f"   Total: {len(all_chunks)}")
    
    # Step 2: Hybrid ranking (no reranker)
    print("\n🔬 Test 1: Hybrid Ranking Only")
    print("-" * 60)
    
    agent_no_rerank = SynthesisAgent(
        top_k=5,
        vector_weight=0.7,
        keyword_weight=0.3,
        use_reranker=False
    )
    
    state1 = AgentState(query=query, chunks=all_chunks.copy())
    result1 = agent_no_rerank.run(state1)
    
    print(f"Results: {len(result1.chunks)}")
    for i, chunk in enumerate(result1.chunks, 1):
        print(f"\n{i}. Score: {chunk.score:.4f}")
        print(f"   Source: {chunk.metadata.get('source')}")
        print(f"   {chunk.text[:100]}...")
    
    # Step 3: With Cohere reranking (if available)
    if has_cohere:
        print("\n🔬 Test 2: Hybrid + Cohere Reranking")
        print("-" * 60)
        
        agent_with_rerank = SynthesisAgent(
            top_k=5,
            vector_weight=0.7,
            keyword_weight=0.3,
            use_reranker=True
        )
        
        state2 = AgentState(query=query, chunks=all_chunks.copy())
        result2 = agent_with_rerank.run(state2)
        
        print(f"Results: {len(result2.chunks)}")
        for i, chunk in enumerate(result2.chunks, 1):
            rerank_score = chunk.metadata.get('rerank_score', chunk.score)
            pre_score = chunk.metadata.get('pre_rerank_score', 0)
            
            print(f"\n{i}. Rerank Score: {rerank_score:.4f}")
            print(f"   Pre-rerank: {pre_score:.4f}")
            print(f"   Source: {chunk.metadata.get('source')}")
            print(f"   {chunk.text[:100]}...")
        
        # Compare results
        print("\n" + "=" * 60)
        print("COMPARISON")
        print("=" * 60)
        
        # Check if order changed
        ids1 = [c.chunk_id for c in result1.chunks]
        ids2 = [c.chunk_id for c in result2.chunks]
        
        order_changed = ids1 != ids2
        
        print(f"Order changed: {order_changed}")
        
        if order_changed:
            print("\n📊 Ranking Differences:")
            for i, (id1, id2) in enumerate(zip(ids1, ids2), 1):
                if id1 != id2:
                    print(f"   Position {i}: Different chunks")
                else:
                    print(f"   Position {i}: Same chunk")
        
        # Score comparison
        print("\n📈 Score Comparison:")
        for i in range(min(len(result1.chunks), len(result2.chunks))):
            hybrid = result1.chunks[i].score
            rerank = result2.chunks[i].score
            diff = rerank - hybrid
            
            print(f"   {i+1}. Hybrid: {hybrid:.4f} | Rerank: {rerank:.4f} | Diff: {diff:+.4f}")
    
    else:
        print("\n⚠️  Skipping Cohere reranking test (no API key)")
        print("\n💡 To enable Cohere reranking:")
        print("   1. Get API key from https://cohere.com")
        print("   2. Add to .env: COHERE_API_KEY=your_key_here")
        print("   3. Install: pip install cohere")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


def main():
    """Main function."""
    test_reranking()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()