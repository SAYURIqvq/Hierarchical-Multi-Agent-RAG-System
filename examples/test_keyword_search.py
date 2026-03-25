"""
Test Keyword Search Agent

Test BM25 keyword search with various queries.

Usage:
    python examples/test_keyword_search.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.retrieval.keyword_agent import KeywordSearchAgent
from src.models.agent_state import AgentState
from src.retrieval import BM25Index


def test_keyword_search():
    """Test keyword search agent."""
    
    print("=" * 60)
    print("TEST KEYWORD SEARCH AGENT")
    print("=" * 60)
    
    # Check if index exists
    print("\n📊 Checking BM25 index...")
    index = BM25Index()
    stats = index.get_stats()
    
    if not stats['built']:
        print("❌ BM25 index not built!")
        print("\nPlease build index first:")
        print("  python build_bm25_index.py")
        return
    
    print(f"✅ Index ready: {stats['total_chunks']} chunks indexed")
    
    # Initialize agent
    print("\n🤖 Initializing KeywordSearchAgent...")
    agent = KeywordSearchAgent(top_k=5, mock_mode=False)
    
    # Test queries
    test_queries = [
        "Python programming language",
        "machine learning algorithms",
        "supervised unsupervised learning",
        "neural networks deep learning",
        "data science libraries"
    ]
    
    print("\n" + "=" * 60)
    print("RUNNING TEST QUERIES")
    print("=" * 60)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n[{i}/{len(test_queries)}] Query: {query}")
        print("-" * 60)
        
        state = AgentState(query=query)
        result = agent.run(state)
        
        print(f"Retrieved: {len(result.chunks)} chunks")
        
        if result.chunks:
            print(f"\nTop 3 results:")
            for j, chunk in enumerate(result.chunks[:3], 1):
                print(f"\n{j}. Score: {chunk.score:.4f}")
                print(f"   {chunk.text[:150]}...")
                print(f"   Source: {chunk.metadata.get('filename', 'unknown')}")
                print(f"   Method: {chunk.metadata.get('method', 'unknown')}")
        else:
            print("⚠️  No results found")
    
    # Agent metrics
    print("\n" + "=" * 60)
    print("AGENT METRICS")
    print("=" * 60)
    metrics = agent.get_metrics()
    print(f"Total calls: {metrics['total_calls']}")
    print(f"Successful: {metrics['successful_calls']}")
    print(f"Failed: {metrics['failed_calls']}")
    print(f"Success rate: {metrics['success_rate']:.1%}")
    
    # Interactive mode
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE")
    print("=" * 60)
    print("Type queries to test (or 'quit' to exit)\n")
    
    while True:
        query = input("🔎 Query: ").strip()
        
        if not query or query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        state = AgentState(query=query)
        result = agent.run(state)
        
        print(f"\n📄 Found {len(result.chunks)} results:")
        print("-" * 40)
        
        for i, chunk in enumerate(result.chunks[:3], 1):
            print(f"\n[{i}] Score: {chunk.score:.1%}")
            print(f"{chunk.text[:200]}...")
        
        print()


def main():
    """Main function."""
    test_keyword_search()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()