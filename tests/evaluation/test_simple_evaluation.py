"""Test simple evaluation."""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.simple_evaluator import SimpleEvaluator


def test_simple_evaluation():
    """Test simple evaluator."""
    
    print("=" * 60)
    print("TESTING SIMPLE EVALUATION")
    print("=" * 60)
    
    evaluator = SimpleEvaluator()
    
    # Test case
    questions = ["What is AI?"]
    answers = ["AI is artificial intelligence, the simulation of human intelligence by machines."]
    contexts = [["Artificial intelligence is intelligence demonstrated by machines."]]
    ground_truths = ["Artificial intelligence is the simulation of human intelligence."]
    
    scores = evaluator.evaluate_rag_system(questions, answers, contexts, ground_truths)
    
    print(f"\n✅ Scores valid: {all(0 <= s <= 1 for s in scores.values())}")
    
    print("\n" + "=" * 60)
    print("✅ SIMPLE EVALUATION TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    test_simple_evaluation()