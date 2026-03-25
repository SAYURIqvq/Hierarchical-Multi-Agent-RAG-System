"""
Example: Interactive Q&A Session

Interactive question-answering over your documents.

Usage:
    python examples/interactive_qa.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.retrieval.vector_agent import VectorSearchAgent
from src.models.agent_state import AgentState
from src.storage import VectorStore


def interactive_qa():
    """Interactive Q&A session."""
    
    # Check if vector store has data
    store = VectorStore()
    count = store.count()
    
    if count == 0:
        print("❌ No documents in vector store!")
        print("\nPlease upload documents first:")
        print("  python examples/batch_upload.py data/test_documents/")
        return
    
    print("=" * 60)
    print("INTERACTIVE Q&A SESSION")
    print("=" * 60)
    print(f"📚 Documents indexed: {count} chunks")
    print("\nType your questions (or 'quit' to exit)")
    print("Commands: 'quit', 'stats', 'help'")
    print("-" * 60)
    
    # Initialize agent
    agent = VectorSearchAgent(top_k=5, mock_mode=False)
    
    query_count = 0
    
    while True:
        print()
        query = input("🤔 Question: ").strip()
        
        if not query:
            continue
        
        # Commands
        if query.lower() in ['quit', 'exit', 'q']:
            print(f"\n👋 Goodbye! Answered {query_count} questions.")
            break
        
        if query.lower() == 'stats':
            print(f"\n📊 Stats:")
            print(f"   Queries asked: {query_count}")
            print(f"   Chunks in DB: {store.count()}")
            print(f"   Agent metrics: {agent.get_metrics()}")
            continue
        
        if query.lower() == 'help':
            print("\n💡 Commands:")
            print("   quit  - Exit the session")
            print("   stats - Show statistics")
            print("   help  - Show this help")
            continue
        
        # Search
        try:
            print(f"\n🔍 Searching...")
            
            state = AgentState(query=query)
            result = agent.run(state)
            
            query_count += 1
            
            # Display results
            print(f"\n📄 Found {len(result.chunks)} relevant chunks:")
            print("-" * 60)
            
            for i, chunk in enumerate(result.chunks, 1):
                print(f"\n[{i}] Relevance: {chunk.score:.1%}")
                print(f"Source: {chunk.metadata.get('filename', 'unknown')}")
                print(f"\n{chunk.text[:300]}...")
                
                if len(chunk.text) > 300:
                    print(f"[...{len(chunk.text) - 300} more characters]")
            
            # Show summary
            print("\n" + "-" * 60)
            avg_score = sum(c.score for c in result.chunks) / len(result.chunks)
            print(f"📊 Average relevance: {avg_score:.1%}")
            
            sources = set(c.metadata.get('filename', 'unknown') for c in result.chunks)
            print(f"📚 Sources: {', '.join(sources)}")
            
        except Exception as e:
            print(f"❌ Search failed: {str(e)}")


def main():
    """Main function."""
    interactive_qa()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()