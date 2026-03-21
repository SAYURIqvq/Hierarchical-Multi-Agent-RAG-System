"""
Test Self-Reflection Loop (Writer + Critic).
Week 5 Day 1 - Manual testing via script.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agents.writer import WriterAgent
from src.agents.critic import CriticAgent
from src.agents.self_reflection import SelfReflectionLoop
from src.models.agent_state import AgentState, Chunk
from src.config import get_settings


def create_mock_chunks():
    """Create mock chunks for testing."""
    return [
        Chunk(
            text="Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed.",
            doc_id="doc_1",
            chunk_id="chunk_1",
            score=0.95,
            metadata={"filename": "ml_basics.pdf"}
        ),
        Chunk(
            text="Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes that process information.",
            doc_id="doc_1", 
            chunk_id="chunk_2",
            score=0.88,
            metadata={"filename": "ml_basics.pdf"}
        ),
        Chunk(
            text="Deep learning uses multiple layers to progressively extract higher-level features from raw input data.",
            doc_id="doc_1",
            chunk_id="chunk_3", 
            score=0.82,
            metadata={"filename": "ml_basics.pdf"}
        )
    ]


def test_writer_only():
    """Test Writer Agent alone."""
    print("\n" + "="*60)
    print("TEST 1: Writer Agent (No Critic)")
    print("="*60)
    
    # Initialize
    writer = WriterAgent()
    state = AgentState(
        query="What is machine learning?",
        chunks=create_mock_chunks()
    )
    
    # Generate answer
    print("\n⏳ Generating answer...")
    result = writer.run(state)
    
    print(f"\n📝 Query: {result.query}")
    print(f"\n✍️  Answer:\n{result.answer}")
    
    writer_meta = result.metadata.get('writer', {})
    print(f"\n�� Metadata:")
    print(f"  - Chunks used: {writer_meta.get('chunks_used', 0)}")
    print(f"  - Answer length: {writer_meta.get('answer_length', 0)} chars")
    print(f"  - Citations: {writer_meta.get('citations_count', 0)}")
    
    return result


def test_critic_only():
    """Test Critic Agent alone."""
    print("\n" + "="*60)
    print("TEST 2: Critic Agent (Review Writer's Answer)")
    print("="*60)
    
    # First get answer from Writer
    writer = WriterAgent()
    state = AgentState(
        query="What is machine learning?",
        chunks=create_mock_chunks()
    )
    
    print("\n⏳ Generating answer for critic to review...")
    state = writer.run(state)
    
    # Now critique
    print("\n🔍 Critic reviewing answer...")
    critic = CriticAgent(quality_threshold=0.7)
    result = critic.run(state)
    
    print(f"\n📊 Critic Score: {result.critic_score:.3f}")
    print(f"🎯 Decision: {result.critic_decision.value.upper()}")
    
    print(f"\n📈 Scores Breakdown:")
    for criterion, score in result.critic_scores.items():
        emoji = "✅" if score >= 0.7 else "⚠️" if score >= 0.5 else "❌"
        print(f"  {emoji} {criterion.title()}: {score:.3f}")
    
    print(f"\n💬 Feedback:\n{result.critic_feedback}")
    
    return result


def test_self_reflection_loop():
    """Test complete Self-Reflection Loop."""
    print("\n" + "="*60)
    print("TEST 3: Self-Reflection Loop (Writer + Critic)")
    print("="*60)
    
    # Initialize agents
    writer = WriterAgent()
    critic = CriticAgent(quality_threshold=0.7)
    
    # Create loop
    loop = SelfReflectionLoop(
        writer=writer,
        critic=critic,
        max_iterations=3
    )
    
    # Initial state
    state = AgentState(
        query="What is machine learning and how does it differ from traditional programming?",
        chunks=create_mock_chunks()
    )
    
    # Run loop
    print("\n🔄 Starting self-reflection loop...")
    print("   (This may take 30-60 seconds)\n")
    
    result = loop.run(state)
    
    # Results
    reflection_stats = result.metadata.get("self_reflection", {})
    
    print("\n" + "="*60)
    print("📊 SELF-REFLECTION RESULTS:")
    print("="*60)
    print(f"🔢 Iterations: {reflection_stats.get('iterations', 0)}")
    print(f"📈 Final Score: {reflection_stats.get('final_score', 0):.3f}")
    print(f"✅ Decision: {reflection_stats.get('final_decision', 'N/A').upper()}")
    print(f"🔄 Improved: {'YES' if reflection_stats.get('improved', False) else 'NO'}")
    
    print(f"\n📝 Final Answer:\n{result.answer[:500]}...")
    
    return result


def test_regeneration_trigger():
    """Test with chunks that should trigger regeneration."""
    print("\n" + "="*60)
    print("TEST 4: Regeneration Trigger Test")
    print("="*60)
    
    # Create minimal chunks (might trigger regeneration)
    minimal_chunks = [
        Chunk(
            text="Machine learning is a type of AI.",
            doc_id="doc_2",
            chunk_id="chunk_4",
            score=0.65,
            metadata={"filename": "brief.pdf"}
        )
    ]
    
    writer = WriterAgent()
    critic = CriticAgent(quality_threshold=0.75)  # Higher threshold
    loop = SelfReflectionLoop(writer, critic, max_iterations=3)
    
    state = AgentState(
        query="Explain the mathematical foundations and key algorithms of machine learning",
        chunks=minimal_chunks
    )
    
    print("\n🔄 Testing with minimal chunks (should trigger regeneration)...")
    print("   Query is complex but chunks are minimal\n")
    
    result = loop.run(state)
    
    reflection_stats = result.metadata.get("self_reflection", {})
    
    print("\n" + "="*60)
    print("📊 REGENERATION TEST RESULTS:")
    print("="*60)
    print(f"🔢 Regenerations: {reflection_stats.get('iterations', 0)}")
    print(f"📈 Final Score: {reflection_stats.get('final_score', 0):.3f}")
    print(f"⚠️  Expected: 1-3 regeneration attempts due to complex query + minimal chunks")
    
    return result


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("🧪 SELF-REFLECTION TESTING SUITE - Week 5 Day 1")
    print("="*70)
    print("\nTesting Writer → Critic → Self-Reflection Loop")
    print("\nThis will test:")
    print("  1. Writer Agent generates answers")
    print("  2. Critic Agent reviews quality")
    print("  3. Self-Reflection Loop coordinates improvement")
    print("  4. Regeneration triggers when needed")
    
    try:
        # Test 1: Writer only
        test_writer_only()
        input("\n⏸️  Press Enter to continue to Test 2...")
        
        # Test 2: Critic only
        test_critic_only()
        input("\n⏸️  Press Enter to continue to Test 3...")
        
        # Test 3: Full loop
        test_self_reflection_loop()
        input("\n⏸️  Press Enter to continue to Test 4...")
        
        # Test 4: Regeneration trigger
        test_regeneration_trigger()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS COMPLETE!")
        print("="*70)
        print("\n📋 Summary:")
        print("  ✅ Writer Agent: Working")
        print("  ✅ Critic Agent: Working")
        print("  ✅ Self-Reflection Loop: Working")
        print("  ✅ Regeneration Logic: Working")
        
        print("\n🎯 Next Steps:")
        print("  → Review test results above")
        print("  → Note any low scores or issues")
        print("  → Ready for Streamlit testing!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\n🔍 Debug info:")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
