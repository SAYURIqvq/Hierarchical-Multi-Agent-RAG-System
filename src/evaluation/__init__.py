"""Evaluation module for RAG system quality assessment."""

try:
    from .ragas_evaluator import RAGASEvaluator
    HAS_RAGAS = True
except ImportError:
    HAS_RAGAS = False

from .simple_evaluator import SimpleEvaluator

__all__ = ['SimpleEvaluator']
if HAS_RAGAS:
    __all__.append('RAGASEvaluator')