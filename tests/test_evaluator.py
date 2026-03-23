"""
Test custom evaluator - Week 5 Day 2
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.simple_evaluator import SimpleEvaluator
from src.models.agent_state import Chunk


def test_evaluator():
    """Test evaluator with dummy data."""
    
    print("\n🧪 Testing Simple Evaluator\n")
    
    # Create evaluator
    evaluator = SimpleEvaluator()
    
    # Dummy data
    query = "What is machine learning?"
    answer = "Machine learning is a subset of AI [1]. It uses algorithms to learn from data [2]."
    
    chunks = [
        Chunk(
            text="Machine learning is a subset of artificial intelligence.",
            doc_id="doc1",
            chunk_id="chunk1",
            score=0.9
        ),
        Chunk(
            text="Algorithms learn patterns from training data.",
            doc_id="doc1",
            chunk_id="chunk2",
            score=0.85
        )
    ]
    
    metadata = {
        'iterations': 1,
        'final_score': 0.82,
        'improved': True
    }
    
    # Evaluate
    scores = evaluator.evaluate_answer(query, answer, chunks, metadata)
    
    # Print results
    print("📊 Evaluation Scores:")
    print(f"  Has Citations: {scores['has_citations']}")
    print(f"  Citation Count: {scores['citation_count']}")
    print(f"  Word Count: {scores['word_count']}")
    print(f"  Is Substantial: {scores['is_substantial']}")
    print(f"  Chunks Used: {scores['chunks_used']}")
    print(f"  Context Usage Rate: {scores['context_usage_rate']:.2%}")
    print(f"  Critic Score: {scores.get('critic_score', 0.0):.2%}")
    print(f"  Was Improved: {scores['was_improved']}")
    print(f"  Overall: {scores['overall']:.2%}")
    
    print("\n✅ Evaluator working correctly!\n")


if __name__ == "__main__":
    test_evaluator()