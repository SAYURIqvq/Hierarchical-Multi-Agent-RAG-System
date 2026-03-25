"""
End-to-End Test: Complete RAG Pipeline with Synthesis

Tests full pipeline:
Query → Planner → Coordinator → (Vector + Keyword + Graph) → Synthesis → Validator

Usage:
    python examples/test_e2e_synthesis.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.planner import PlannerAgent
from src.agents.validator import ValidatorAgent
from src.agents.synthesis import SynthesisAgent
from src.agents.retrieval.vector_agent import VectorSearchAgent
from src.agents.retrieval.keyword_agent import KeywordSearchAgent
from src.agents.retrieval.graph_agent import GraphSearchAgent
from src.models.agent_state import AgentState

from langchain_anthropic import ChatAnthropic
from src.config import get_settings


def test_e2e_with_synthesis():
    """Test complete pipeline with synthesis."""
    
    print("=" * 60)
    print("END-TO-END TEST: RAG PIPELINE WITH SYNTHESIS")
    print("=" * 60)
    
    # Initialize LLM
    settings = get_settings()
    llm = ChatAnthropic(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        api_key=settings.anthropic_api_key
    )
    
    # Test query
    query = "What is Python used for in machine learning?"
    
    print(f"\n🔎 Query: {query}")
    print("-" * 60)
    
    # Step 1: Planner
    print("\n📋 Step 1: Planner (Complexity Analysis)")
    print("-" * 40)
    
    planner = PlannerAgent(llm=llm)
    state = AgentState(query=query)
    state = planner.run(state)
    
    print(f"   Complexity: {state.complexity:.3f}")
    print(f"   Strategy: {state.strategy.value}")
    
    # Step 2: Multi-source Retrieval
    print("\n🔍 Step 2: Multi-Source Retrieval")
    print("-" * 40)
    
    # Spawn retrieval agents
    vector_agent = VectorSearchAgent(top_k=5, mock_mode=False)
    keyword_agent = KeywordSearchAgent(top_k=5, mock_mode=False)
    graph_agent = GraphSearchAgent(top_k=5, mock_mode=True)  # Still mock
    
    # Retrieve from each source
    print("   Vector search...")
    vector_state = AgentState(query=query)
    vector_result = vector_agent.run(vector_state)
    print(f"   ✅ Vector: {len(vector_result.chunks)} chunks")
    
    print("   Keyword search...")
    keyword_state = AgentState(query=query)
    keyword_result = keyword_agent.run(keyword_state)
    print(f"   ✅ Keyword: {len(keyword_result.chunks)} chunks")
    
    print("   Graph search (mock)...")
    graph_state = AgentState(query=query)
    graph_result = graph_agent.run(graph_state)
    print(f"   ✅ Graph: {len(graph_result.chunks)} chunks")
    
    # Combine all results
    all_chunks = (
        vector_result.chunks +
        keyword_result.chunks +
        graph_result.chunks
    )
    
    print(f"\n   Total retrieved: {len(all_chunks)} chunks")
    
    # Step 3: Synthesis
    print("\n🔬 Step 3: Synthesis (Fusion + Reranking)")
    print("-" * 40)
    
    synthesis_agent = SynthesisAgent(
        top_k=10,
        vector_weight=0.7,
        keyword_weight=0.3,
        use_reranker=False  # Set True if you have Cohere key
    )
    
    state.chunks = all_chunks
    state = synthesis_agent.run(state)
    
    # Show synthesis stats
    stats = synthesis_agent.get_synthesis_stats(state)
    print(f"   Input chunks: {stats.get('input_count', 0)}")
    print(f"   Unique chunks: {stats.get('unique_count', 0)}")
    print(f"   Final chunks: {stats.get('final_count', 0)}")
    print(f"   Dedup rate: {stats.get('deduplication_rate', 0):.1%}")
    
    # Step 4: Validation
    print("\n✅ Step 4: Validation (Quality Check)")
    print("-" * 40)
    
    validator = ValidatorAgent(llm=llm, threshold=0.7, max_retries=2)
    state = validator.run(state)
    
    print(f"   Validation score: {state.validation_score:.3f}")
    print(f"   Decision: {state.validation_status}")
    
    # Step 5: Display Results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    
    print(f"\n📊 Pipeline Summary:")
    print(f"   Query: {query}")
    print(f"   Complexity: {state.complexity:.3f}")
    print(f"   Strategy: {state.strategy.value}")
    print(f"   Total retrieved: {len(all_chunks)}")
    print(f"   After synthesis: {len(state.chunks)}")
    print(f"   Validation: {state.validation_status}")
    
    print(f"\n📄 Top 5 Chunks:")
    for i, chunk in enumerate(state.chunks[:5], 1):
        print(f"\n[{i}] Score: {chunk.score:.4f}")
        print(f"    Source: {chunk.metadata.get('source', 'unknown')}")
        print(f"    Method: {chunk.metadata.get('method', 'unknown')}")
        print(f"    Text: {chunk.text[:150]}...")
    
    # Agent metrics
    print(f"\n📈 Agent Metrics:")
    print(f"   Planner: {planner.get_metrics()['total_calls']} calls")
    print(f"   Vector: {vector_agent.get_metrics()['total_calls']} calls")
    print(f"   Keyword: {keyword_agent.get_metrics()['total_calls']} calls")
    print(f"   Graph: {graph_agent.get_metrics()['total_calls']} calls")
    print(f"   Synthesis: {synthesis_agent.get_metrics()['total_calls']} calls")
    print(f"   Validator: {validator.get_metrics()['total_calls']} calls")
    
    print("\n" + "=" * 60)
    print("✅ PIPELINE TEST COMPLETE")
    print("=" * 60)
    
    return state


def main():
    """Main function."""
    
    try:
        result = test_e2e_with_synthesis()
        
        print("\n💡 Next Steps:")
        print("   1. Add Writer Agent (generate formatted answer)")
        print("   2. Add Critic Agent (review answer quality)")
        print("   3. Complete full agentic pipeline!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")