"""
RAGAS-based evaluation for RAG system.
Override default OpenAI LLM ke Claude (Anthropic).
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv
load_dotenv()

from typing import List, Dict, Any
import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_precision,
    context_recall
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_anthropic import ChatAnthropic
from langchain_core.embeddings import Embeddings

class VoyageEmbeddings(Embeddings):
    """
    Custom Embeddings class untuk RAGAS.
    Pakai Voyage AI (sama seperti retrieval kita).
    """

    def __init__(self):
        import voyageai
        api_key = os.environ.get("VOYAGE_API_KEY")
        if not api_key:
            raise ValueError("VOYAGE_API_KEY not found in .env")
        self.client = voyageai.Client(api_key=api_key)
        self.model = "voyage-large-2-instruct"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed list of documents."""
        if not texts:
            return []
        response = self.client.embed(
            texts,
            model=self.model,
            input_type="document"
        )
        return response.embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed single query."""
        response = self.client.embed(
            [text],
            model=self.model,
            input_type="query"
        )
        return response.embeddings[0]

# ========== RAGAS EVALUATOR ==========

class RAGASEvaluator:
    """
    Evaluate RAG system using RAGAS framework.
    Uses Claude (LLM) + Voyage AI (Embeddings).
    """

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        # Validate API keys
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        if not anthropic_key:
            raise ValueError("ANTHROPIC_API_KEY not found in .env")

        voyage_key = os.environ.get("VOYAGE_API_KEY")
        if not voyage_key:
            raise ValueError("VOYAGE_API_KEY not found in .env")

        # Override LLM → Claude
        llm = ChatAnthropic(
            model=model,
            api_key=anthropic_key,
            temperature=0.0
        )
        self.llm = LangchainLLMWrapper(llm)

        # Override Embeddings → Voyage AI
        self.embeddings = LangchainEmbeddingsWrapper(VoyageEmbeddings())

        self.metrics = [
            answer_relevancy,
            faithfulness,
            context_precision,
            context_recall
        ]

        print("📊 RAGAS Evaluator initialized")
        print(f"   LLM: Claude ({model})")
        print(f"   Embeddings: Voyage AI ({self.embeddings})")
        print(f"   Metrics: {len(self.metrics)}")

    def evaluate_rag_system(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str]
    ) -> Dict[str, float]:
        """Evaluate RAG system performance."""
        print(f"\n📊 Evaluating {len(questions)} test cases...")

        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        }

        dataset = Dataset.from_dict(data)

        print("   Running RAGAS evaluation with Claude + Voyage...")

        # ← Pass BOTH llm and embeddings
        results = evaluate(
            dataset,
            metrics=self.metrics,
            llm=self.llm,
            embeddings=self.embeddings  # ← KEY FIX
        )

        scores = {
            'answer_relevancy': float(results['answer_relevancy']),
            'faithfulness': float(results['faithfulness']),
            'context_precision': float(results['context_precision']),
            'context_recall': float(results['context_recall']),
        }
        scores['overall'] = sum(scores.values()) / len(scores)

        print(f"\n   ✅ Evaluation complete!")
        print(f"      Answer Relevancy:  {scores['answer_relevancy']:.3f}")
        print(f"      Faithfulness:      {scores['faithfulness']:.3f}")
        print(f"      Context Precision: {scores['context_precision']:.3f}")
        print(f"      Context Recall:    {scores['context_recall']:.3f}")
        print(f"      Overall Score:     {scores['overall']:.3f}")

        return scores

    def evaluate_single_case(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str
    ) -> Dict[str, float]:
        """Evaluate single Q&A case."""
        return self.evaluate_rag_system(
            questions=[question],
            answers=[answer],
            contexts=[contexts],
            ground_truths=[ground_truth]
        )

    def load_test_dataset(self, filepath: str) -> List[Dict[str, Any]]:
        """Load test dataset dari JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data['test_cases']
    
    HONESTY_PHRASES = [
        "do not contain information",
        "not found in",
        "not available in",
        "does not mention",
        "no information about",
        "not covered in",
    ]

    def check_production_gate(self, scores: Dict[str, float], answer: str = "") -> Dict[str, Any]:
        """
        Production gate with honesty detection.

        Logic:
        - Faithfulness >= 0.5       → Hard gate
        - Overall >= 0.7            → Soft gate
        - Honest non-answer detected → Relevancy tidak di-penalize
        """
        FAITHFULNESS_THRESHOLD = 0.5
        OVERALL_THRESHOLD = 0.7

        passed = True
        reasons = []

        # Detect apakah answer honest tentang missing info
        is_honest_non_answer = any(
            phrase in answer.lower()
            for phrase in self.HONESTY_PHRASES
        )

        # Hard gate: Faithfulness
        if scores["faithfulness"] < FAITHFULNESS_THRESHOLD:
            passed = False
            reasons.append(
                f"❌ Faithfulness too low: {scores['faithfulness']:.3f} "
                f"(min: {FAITHFULNESS_THRESHOLD}) — possible hallucination"
            )

        # Soft gate: Overall
        # Kalau honest non-answer → skip overall check
        if scores["overall"] < OVERALL_THRESHOLD and not is_honest_non_answer:
            passed = False
            reasons.append(
                f"❌ Overall too low: {scores['overall']:.3f} "
                f"(min: {OVERALL_THRESHOLD})"
            )

        # Build reasons
        if is_honest_non_answer:
            reasons.append(
                "ℹ️ Honest non-answer detected — "
                "model correctly flagged missing info"
            )

        if passed:
            reasons.append("✅ All checks passed")

        return {
            "passed": passed,
            "reasons": reasons,
            "scores": scores,
            "is_honest_non_answer": is_honest_non_answer
        }