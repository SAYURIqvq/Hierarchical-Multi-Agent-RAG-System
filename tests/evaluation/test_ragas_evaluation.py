"""
Test RAGAS evaluation framework.
All external calls (ChatAnthropic, evaluate) are fully mocked.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ========== MOCK RESPONSES ==========

GOOD_SCORES = {
    "answer_relevancy": 0.85,
    "faithfulness": 0.90,
    "context_precision": 0.80,
    "context_recall": 0.78,
}

BAD_SCORES = {
    "answer_relevancy": 0.40,
    "faithfulness": 0.20,
    "context_precision": 0.60,
    "context_recall": 0.50,
}


# ========== FIXTURES ==========

@pytest.fixture
def evaluator():
    """
    Mock ChatAnthropic dan LangchainLLMWrapper
    SEBELUM import RAGASEvaluator.
    """
    with patch("src.evaluation.ragas_evaluator.ChatAnthropic") as mock_chat, \
         patch("src.evaluation.ragas_evaluator.LangchainLLMWrapper") as mock_wrapper:

        # Mock ChatAnthropic return value
        mock_chat.return_value = MagicMock()
        mock_wrapper.return_value = MagicMock()

        from src.evaluation.ragas_evaluator import RAGASEvaluator
        evaluator = RAGASEvaluator()

        yield evaluator


@pytest.fixture
def good_case():
    return {
        "question": "What is artificial intelligence?",
        "answer": (
            "Artificial intelligence is the simulation of human "
            "intelligence processes by machines and computer systems."
        ),
        "contexts": [
            "Artificial intelligence (AI) is intelligence demonstrated by machines.",
            "AI systems simulate human intelligence processes."
        ],
        "ground_truth": (
            "Artificial intelligence is the simulation of "
            "human intelligence by machines."
        )
    }


@pytest.fixture
def bad_case():
    return {
        "question": "What is artificial intelligence?",
        "answer": (
            "AI was invented in Paris in 1850 by Jacques Moreau. "
            "It is powered by quantum crystals found in the ocean."
        ),
        "contexts": [
            "Artificial intelligence (AI) is intelligence demonstrated by machines.",
            "AI systems simulate human intelligence processes."
        ],
        "ground_truth": (
            "Artificial intelligence is the simulation of "
            "human intelligence by machines."
        )
    }


@pytest.fixture
def sample_dataset(tmp_path):
    data = {
        "test_cases": [
            {
                "question": "What is Python?",
                "answer": "Python is a high-level programming language.",
                "contexts": ["Python is a high-level, general-purpose programming language."],
                "ground_truth": "Python is a high-level programming language."
            },
            {
                "question": "Who created Python?",
                "answer": "Python was created by Guido van Rossum.",
                "contexts": ["Python was created by Guido van Rossum in 1991."],
                "ground_truth": "Guido van Rossum created Python."
            }
        ]
    }
    file_path = tmp_path / "test_dataset.json"
    with open(file_path, 'w') as f:
        json.dump(data, f)
    return str(file_path)


# ========== TESTS ==========

class TestRAGASInitialization:
    """Test initialization."""

    def test_init_metrics(self, evaluator):
        assert len(evaluator.metrics) == 4

    def test_init_llm_exists(self, evaluator):
        assert evaluator.llm is not None


class TestRAGASBasic:
    """Basic evaluation — mock evaluate() return GOOD_SCORES."""

    @patch("src.evaluation.ragas_evaluator.evaluate")
    def test_single_case_returns_all_scores(self, mock_eval, evaluator, good_case):
        mock_eval.return_value = GOOD_SCORES

        scores = evaluator.evaluate_single_case(
            question=good_case["question"],
            answer=good_case["answer"],
            contexts=good_case["contexts"],
            ground_truth=good_case["ground_truth"]
        )

        assert "answer_relevancy" in scores
        assert "faithfulness" in scores
        assert "context_precision" in scores
        assert "context_recall" in scores
        assert "overall" in scores

    @patch("src.evaluation.ragas_evaluator.evaluate")
    def test_all_scores_valid_range(self, mock_eval, evaluator, good_case):
        mock_eval.return_value = GOOD_SCORES

        scores = evaluator.evaluate_single_case(
            question=good_case["question"],
            answer=good_case["answer"],
            contexts=good_case["contexts"],
            ground_truth=good_case["ground_truth"]
        )

        for metric, score in scores.items():
            assert 0.0 <= score <= 1.0, f"{metric} out of range: {score}"

    @patch("src.evaluation.ragas_evaluator.evaluate")
    def test_overall_is_average(self, mock_eval, evaluator, good_case):
        mock_eval.return_value = GOOD_SCORES

        scores = evaluator.evaluate_single_case(
            question=good_case["question"],
            answer=good_case["answer"],
            contexts=good_case["contexts"],
            ground_truth=good_case["ground_truth"]
        )

        expected = (0.85 + 0.90 + 0.80 + 0.78) / 4
        assert abs(scores["overall"] - expected) < 0.001

    @patch("src.evaluation.ragas_evaluator.evaluate")
    def test_multiple_cases(self, mock_eval, evaluator):
        mock_eval.return_value = GOOD_SCORES

        scores = evaluator.evaluate_rag_system(
            questions=["Q1?", "Q2?"],
            answers=["A1", "A2"],
            contexts=[["C1"], ["C2"]],
            ground_truths=["GT1", "GT2"]
        )

        assert scores["overall"] > 0.0


class TestRAGASQualityDetection:
    """Verify system bisa detect good vs bad answers."""

    @patch("src.evaluation.ragas_evaluator.evaluate")
    def test_good_answer_high_faithfulness(self, mock_eval, evaluator, good_case):
        mock_eval.return_value = GOOD_SCORES

        scores = evaluator.evaluate_single_case(
            question=good_case["question"],
            answer=good_case["answer"],
            contexts=good_case["contexts"],
            ground_truth=good_case["ground_truth"]
        )

        assert scores["faithfulness"] >= 0.7

    @patch("src.evaluation.ragas_evaluator.evaluate")
    def test_bad_answer_low_faithfulness(self, mock_eval, evaluator, bad_case):
        mock_eval.return_value = BAD_SCORES  # ← Return BAD scores

        scores = evaluator.evaluate_single_case(
            question=bad_case["question"],
            answer=bad_case["answer"],
            contexts=bad_case["contexts"],
            ground_truth=bad_case["ground_truth"]
        )

        assert scores["faithfulness"] < 0.5

    def test_good_vs_bad_comparison(self, evaluator, good_case, bad_case):
        """Good answer harus score lebih tinggi dari bad."""
        # Good
        with patch("src.evaluation.ragas_evaluator.evaluate") as mock_eval:
            mock_eval.return_value = GOOD_SCORES
            good_scores = evaluator.evaluate_single_case(
                question=good_case["question"],
                answer=good_case["answer"],
                contexts=good_case["contexts"],
                ground_truth=good_case["ground_truth"]
            )

        # Bad
        with patch("src.evaluation.ragas_evaluator.evaluate") as mock_eval:
            mock_eval.return_value = BAD_SCORES
            bad_scores = evaluator.evaluate_single_case(
                question=bad_case["question"],
                answer=bad_case["answer"],
                contexts=bad_case["contexts"],
                ground_truth=bad_case["ground_truth"]
            )

        assert good_scores["faithfulness"] > bad_scores["faithfulness"]
        assert good_scores["overall"] > bad_scores["overall"]


class TestRAGASEdgeCases:
    """Edge cases — tidak crash."""

    @patch("src.evaluation.ragas_evaluator.evaluate")
    def test_empty_context(self, mock_eval, evaluator):
        mock_eval.return_value = GOOD_SCORES

        scores = evaluator.evaluate_single_case(
            question="What is AI?",
            answer="AI is artificial intelligence.",
            contexts=[""],  # ← Empty
            ground_truth="AI is artificial intelligence."
        )

        for metric, score in scores.items():
            assert 0.0 <= score <= 1.0

    @patch("src.evaluation.ragas_evaluator.evaluate")
    def test_short_answer(self, mock_eval, evaluator):
        mock_eval.return_value = GOOD_SCORES

        scores = evaluator.evaluate_single_case(
            question="What is AI?",
            answer="AI.",  # ← Very short
            contexts=["Artificial intelligence is intelligence by machines."],
            ground_truth="AI is artificial intelligence."
        )

        for metric, score in scores.items():
            assert 0.0 <= score <= 1.0

    @patch("src.evaluation.ragas_evaluator.evaluate")
    def test_many_contexts(self, mock_eval, evaluator):
        mock_eval.return_value = GOOD_SCORES

        scores = evaluator.evaluate_single_case(
            question="What is Python?",
            answer="Python is a high-level programming language.",
            contexts=[
                "Python is a high-level programming language.",
                "Java is a popular programming language.",
                "The weather today is sunny.",
                "JavaScript is used for web development.",
                "Python was created by Guido van Rossum."
            ],
            ground_truth="Python is a high-level programming language."
        )

        for metric, score in scores.items():
            assert 0.0 <= score <= 1.0


class TestRAGASDataset:
    """Dataset loading tests — tidak butuh evaluate()."""

    def test_load_dataset(self, evaluator, sample_dataset):
        test_cases = evaluator.load_test_dataset(sample_dataset)

        assert len(test_cases) == 2
        assert "question" in test_cases[0]
        assert "ground_truth" in test_cases[0]

    def test_load_nonexistent_file(self, evaluator):
        with pytest.raises(FileNotFoundError):
            evaluator.load_test_dataset("nonexistent.json")

    @patch("src.evaluation.ragas_evaluator.evaluate")
    def test_full_flow_load_then_evaluate(self, mock_eval, evaluator, sample_dataset):
        mock_eval.return_value = GOOD_SCORES

        test_cases = evaluator.load_test_dataset(sample_dataset)

        scores = evaluator.evaluate_rag_system(
            questions=[tc["question"] for tc in test_cases],
            answers=[tc["answer"] for tc in test_cases],
            contexts=[tc["contexts"] for tc in test_cases],
            ground_truths=[tc["ground_truth"] for tc in test_cases]
        )

        assert scores["overall"] > 0.0


class TestRAGASThresholds:
    """Production threshold gates."""

    PRODUCTION_THRESHOLD = 0.7

    @patch("src.evaluation.ragas_evaluator.evaluate")
    def test_good_case_passes(self, mock_eval, evaluator, good_case):
        mock_eval.return_value = GOOD_SCORES

        scores = evaluator.evaluate_single_case(
            question=good_case["question"],
            answer=good_case["answer"],
            contexts=good_case["contexts"],
            ground_truth=good_case["ground_truth"]
        )

        assert scores["overall"] >= self.PRODUCTION_THRESHOLD

    @patch("src.evaluation.ragas_evaluator.evaluate")
    def test_bad_case_fails(self, mock_eval, evaluator, bad_case):
        mock_eval.return_value = BAD_SCORES

        scores = evaluator.evaluate_single_case(
            question=bad_case["question"],
            answer=bad_case["answer"],
            contexts=bad_case["contexts"],
            ground_truth=bad_case["ground_truth"]
        )

        assert scores["overall"] < self.PRODUCTION_THRESHOLD


# ========== RUN ==========

if __name__ == "__main__":
    pytest.main([__file__, "-v"])