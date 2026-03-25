"""
Test Critic Agent & Self-Reflection Loop

Tests answer quality review and iterative improvement.

Usage:
    python examples/test_critic.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.writer import WriterAgent
from src.agents.critic import CriticAgent, CriticDecision
from src.agents.self_reflection import SelfReflectionLoop
from src.agents.retrieval.vector_agent import VectorSearchAgent
from src.agents.synthesis import SynthesisAgent
from src.models.agent_state import AgentState

from langchain_anthropic import ChatAnthropic
from src.config import get_settings


def test_critic_agent():
    """Test Critic Agent standalone."""
    
    print("=" * 60)
    print("TEST 1: CRITIC AGENT")
    print("=" * 60)
    
    settings = get_settings()
    llm = ChatAnthropic(
        model=settings.llm_model,
        temperature=0.0,
        api_key=settings.anthropic_api_key
    )
    
    # Test with good answer
    print("\n📝 Test Case 1: Good Answer")
    print("-" * 60)
    
    good_answer = """Python is a high-level programming language [1] known for its simplicity and readability. It is widely used in web development, data science, and machine learning [2].

---

**Sources:**

[1] python_guide.txt
[2] python_guide.txt"""
    
    critic = CriticAgent(llm=llm, quality_threshold=0.7)
    
    state = AgentState(
        query="What is Python?",
        answer=good_answer,
        chunks=[]  # Simplified
    )
    
    result = critic.run(state)
    
    print(f"Overall Score: {result.critic_score:.3f}")
    print(f"Decision: {result.critic_decision.value}")
    print(f"Scores: {result.critic_scores}")
    print(f"Feedback: {result.critic_feedback[:150]}...")
    
    # Test with poor answer
    print("\n📝 Test Case 2: Poor Answer")
    print("-" * 60)
    
    poor_answer = "Python is a thing."
    
    state2 = AgentState(
        query="What is Python and what is it used for?",
        answer=poor_answer,
        chunks=[]
    )
    
    result2 = critic.run(state2)
    
    print(f"Overall Score: {result2.critic_score:.3f}")
    print(f"Decision: {result2.critic_decision.value}")
    print(f"Scores: {result2.critic_scores}")
    print(f"Feedback: {result2.critic_feedback[:150]}...")


def test_self_reflection():
    """Test self-reflection loop."""
    
    print("\n" + "=" * 60)
    print("TEST 2: SELF-REFLECTION LOOP")
    print("=" * 60)
    
    settings = get_settings()
    llm = ChatAnthropic(
        model=settings.llm_model,
        temperature=0.3,
        api_key=settings.anthropic_api_key
    )
    
    # Test query
    query = "What is Python used for?"
    
    print(f"\n🔎 Query: {query}")
    print("-" * 60)
    
    # Step 1: Retrieve chunks
    print("\n📥 Step 1: Retrieving chunks...")
    
    vector_agent = VectorSearchAgent(top_k=3, mock_mode=False)
    vector_result = vector_agent.run(AgentState(query=query))
    
    synthesis = SynthesisAgent(top_k=3)
    state = AgentState(query=query, chunks=vector_result.chunks)
    state = synthesis.run(state)
    
    print(f"   Chunks: {len(state.chunks)}")
    
    # Step 2: Self-reflection loop
    print("\n🔄 Step 2: Self-Reflection Loop...")
    print("-" * 60)
    
    writer = WriterAgent(llm=llm)
    critic = CriticAgent(llm=llm, quality_threshold=0.75, max_iterations=3)
    
    loop = SelfReflectionLoop(
        writer=writer,
        critic=critic,
        max_iterations=3
    )
    
    final_state = loop.run(state)
    
    # Display results
    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)
    print(f"\n{final_state.answer}")
    
    # Show improvement stats
    print("\n" + "-" * 60)
    print("IMPROVEMENT STATS")
    print("-" * 60)
    
    stats = loop.get_stats(final_state)
    print(f"Iterations: {stats.get('iterations', 0)}")
    print(f"Final Score: {stats.get('final_score', 0):.3f}")
    print(f"Decision: {stats.get('final_decision', 'unknown')}")
    print(f"Improved: {stats.get('improved', False)}")
    
    # Detailed scores
    if hasattr(final_state, 'critic_scores'):
        print(f"\nDetailed Scores:")
        for criterion, score in final_state.critic_scores.items():
            print(f"   {criterion.capitalize()}: {score:.3f}")


def test_full_pipeline():
    """Test complete pipeline with self-reflection."""
    
    print("\n" + "=" * 60)
    print("TEST 3: FULL PIPELINE WITH SELF-REFLECTION")
    print("=" * 60)
    
    settings = get_settings()
    llm = ChatAnthropic(
        model=settings.llm_model,
        temperature=0.3,
        api_key=settings.anthropic_api_key
    )
    
    query = "Explain machine learning and its main types"
    
    print(f"\n🔎 Query: {query}")
    print("-" * 60)
    
    # Retrieval
    print("\n1️⃣  Multi-source Retrieval...")
    vector_agent = VectorSearchAgent(top_k=5, mock_mode=False)
    vector_result = vector_agent.run(AgentState(query=query))
    print(f"   ✅ Retrieved: {len(vector_result.chunks)} chunks")
    
    # Synthesis
    print("\n2️⃣  Synthesis...")
    synthesis = SynthesisAgent(top_k=5)
    state = AgentState(query=query, chunks=vector_result.chunks)
    state = synthesis.run(state)
    print(f"   ✅ Final: {len(state.chunks)} chunks")
    
    # Self-reflection loop
    print("\n3️⃣  Self-Reflection (Writer + Critic)...")
    writer = WriterAgent(llm=llm)
    critic = CriticAgent(llm=llm, quality_threshold=0.8)
    loop = SelfReflectionLoop(writer, critic, max_iterations=2)
    
    final_state = loop.run(state)
    
    # Results
    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    print(f"\n{final_state.answer}")
    
    print("\n" + "=" * 60)
    print("PIPELINE STATS")
    print("=" * 60)
    
    reflection_stats = loop.get_stats(final_state)
    print(f"Quality Score: {reflection_stats.get('final_score', 0):.1%}")
    print(f"Iterations: {reflection_stats.get('iterations', 0)}")
    print(f"Status: {reflection_stats.get('final_decision', 'unknown')}")


def main():
    """Main function."""
    
    try:
        # Test 1: Critic standalone
        test_critic_agent()
        
        input("\n[Press Enter to continue to Test 2...]")
        
        # Test 2: Self-reflection loop
        test_self_reflection()
        
        input("\n[Press Enter to continue to Test 3...]")
        
        # Test 3: Full pipeline
        test_full_pipeline()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS COMPLETE!")
        print("=" * 60)
        print("\n🎉 Week 4 Complete - All Agents Working!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")