"""
Test Writer Agent

Test answer generation with citations.

Usage:
    python examples/test_writer.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.writer import WriterAgent
from src.agents.retrieval.vector_agent import VectorSearchAgent
from src.agents.retrieval.keyword_agent import KeywordSearchAgent
from src.agents.synthesis import SynthesisAgent
from src.models.agent_state import AgentState
from src.utils.citation_utils import CitationUtils

from langchain_anthropic import ChatAnthropic
from src.config import get_settings


def test_writer():
    """Test Writer Agent with real retrieval."""
    
    print("=" * 60)
    print("TEST WRITER AGENT")
    print("=" * 60)
    
    # Initialize
    settings = get_settings()
    llm = ChatAnthropic(
        model=settings.llm_model,
        temperature=0.3,
        api_key=settings.anthropic_api_key
    )
    
    # Test queries
    test_queries = [
        "What is Python and what is it used for?",
        "Explain machine learning and its types",
        "What are the main features of Python?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'=' * 60}")
        print(f"TEST {i}/{len(test_queries)}")
        print("=" * 60)
        print(f"\n🔎 Query: {query}")
        print("-" * 60)
        
        # Step 1: Retrieve chunks
        print("\n📥 Step 1: Retrieving chunks...")
        
        vector_agent = VectorSearchAgent(top_k=3, mock_mode=False)
        keyword_agent = KeywordSearchAgent(top_k=3, mock_mode=False)
        
        vector_result = vector_agent.run(AgentState(query=query))
        keyword_result = keyword_agent.run(AgentState(query=query))
        
        all_chunks = vector_result.chunks + keyword_result.chunks
        
        print(f"   Retrieved: {len(all_chunks)} chunks")
        
        # Step 2: Synthesize
        print("\n🔬 Step 2: Synthesis...")
        
        synthesis = SynthesisAgent(top_k=5)
        state = AgentState(query=query, chunks=all_chunks)
        state = synthesis.run(state)
        
        print(f"   Final chunks: {len(state.chunks)}")
        
        # Step 3: Generate answer
        print("\n✍️  Step 3: Generating answer...")
        
        writer = WriterAgent(llm=llm, include_sources=True)
        state = writer.run(state)
        
        # Display answer
        print("\n" + "=" * 60)
        print("GENERATED ANSWER")
        print("=" * 60)
        print(f"\n{state.answer}")
        
        # Validate citations
        print("\n" + "-" * 60)
        print("CITATION VALIDATION")
        print("-" * 60)
        
        utils = CitationUtils()
        validation = utils.validate_citations(
            state.answer,
            max_citation=len(state.chunks)
        )
        
        print(f"Valid: {validation['valid']}")
        print(f"Citations: {validation['citations']}")
        print(f"Count: {validation['citation_count']}")
        
        if validation['errors']:
            print(f"Errors: {validation['errors']}")
        if validation['warnings']:
            print(f"Warnings: {validation['warnings']}")
        
        # Citation usage
        citation_counts = utils.count_citations_per_source(state.answer)
        print(f"\nCitation usage:")
        for cite_num, count in sorted(citation_counts.items()):
            print(f"   [{cite_num}]: used {count} time(s)")
        
        # Writer metadata
        writer_meta = state.metadata.get('writer', {})
        print(f"\nWriter stats:")
        print(f"   Chunks used: {writer_meta.get('chunks_used', 0)}")
        print(f"   Answer length: {writer_meta.get('answer_length', 0)} chars")
        print(f"   Citations: {writer_meta.get('citations_count', 0)}")
        
        # Wait for next query
        if i < len(test_queries):
            input("\n[Press Enter for next query...]")


def test_writer_with_mock():
    """Test Writer with mock chunks."""
    
    print("\n" + "=" * 60)
    print("TEST WITH MOCK CHUNKS")
    print("=" * 60)
    
    from src.models.agent_state import Chunk
    
    # Create mock chunks
    chunks = [
        Chunk(
            text="Python is a high-level programming language known for its simplicity and readability.",
            doc_id="doc1",
            chunk_id="chunk1",
            score=0.9,
            metadata={'filename': 'python_guide.txt'}
        ),
        Chunk(
            text="Python is widely used in web development, data science, and machine learning.",
            doc_id="doc1",
            chunk_id="chunk2",
            score=0.85,
            metadata={'filename': 'python_guide.txt'}
        ),
        Chunk(
            text="Machine learning is a subset of AI that enables systems to learn from data.",
            doc_id="doc2",
            chunk_id="chunk3",
            score=0.8,
            metadata={'filename': 'ml_guide.txt'}
        )
    ]
    
    query = "What is Python used for?"
    
    print(f"\n🔎 Query: {query}")
    print(f"📦 Chunks: {len(chunks)}")
    
    # Generate answer
    settings = get_settings()
    llm = ChatAnthropic(
        model=settings.llm_model,
        temperature=0.3,
        api_key=settings.anthropic_api_key
    )
    
    writer = WriterAgent(llm=llm)
    state = AgentState(query=query, chunks=chunks)
    state = writer.run(state)
    
    print("\n" + "=" * 60)
    print("ANSWER WITH MOCK CHUNKS")
    print("=" * 60)
    print(f"\n{state.answer}")


def main():
    """Main function."""
    
    try:
        # Test with real retrieval
        test_writer()
        
        # Test with mock
        test_writer_with_mock()
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")