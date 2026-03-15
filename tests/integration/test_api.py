"""Test API connections for Anthropic Claude and Voyage AI."""

import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_voyageai import VoyageAIEmbeddings

# Load environment variables
load_dotenv()


def test_anthropic():
    """Test Anthropic Claude API."""
    print("Testing Anthropic API...")
    try:
        llm = ChatAnthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            model="claude-3-haiku-20240307",  # ← UPDATED MODEL NAME
            temperature=0
        )
        response = llm.invoke("Say 'Hello, RAG!'")
        print(f"✅ Anthropic works! Response: {response.content}")
        return True
    except Exception as e:
        print(f"❌ Anthropic failed: {e}")
        return False


def test_voyage():
    """Test Voyage AI embeddings."""
    print("\nTesting Voyage AI...")
    try:
        embedder = VoyageAIEmbeddings(
            voyage_api_key=os.getenv("VOYAGE_API_KEY"),
            model="voyage-large-2"
        )
        embedding = embedder.embed_query("Hello, world!")
        print(f"✅ Voyage AI works! Embedding dimension: {len(embedding)}")
        return True
    except Exception as e:
        print(f"❌ Voyage AI failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("API CONNECTION TESTS - Phase 1 Day 1")
    print("=" * 60)
    
    anthropic_ok = test_anthropic()
    voyage_ok = test_voyage()
    
    print("\n" + "=" * 60)
    if anthropic_ok and voyage_ok:
        print("✅ ALL TESTS PASSED! Ready to build RAG.")
        print("\nNext steps:")
        print("  1. Implement PDF loading")
        print("  2. Add text chunking")
        print("  3. Generate embeddings")
    else:
        print("❌ Some tests failed. Check your API keys in .env file")
    print("=" * 60)