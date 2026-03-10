"""
Validator Agent - Tactical Level 2 Agent.

Validates if retrieved chunks are sufficient to answer the query.
Acts as a quality gate before answer generation.

Validation Factors:
- Relevance: Do chunks relate to the query? (50%)
- Coverage: Do chunks cover all query aspects? (30%)
- Confidence: Are chunks reliable and complete? (20%)

Decision Logic:
- score >= threshold → PROCEED to answer generation
- score < threshold AND retries < max → RETRIEVE_MORE
- retries >= max → PROCEED anyway (force)
"""

from typing import List, Tuple
from langchain_anthropic import ChatAnthropic

from src.agents.base_agent import BaseAgent
from src.models.agent_state import AgentState, Chunk
from src.utils.exceptions import AgentExecutionError, ValidationError
from src.config import get_settings


class ValidatorAgent(BaseAgent):
    """
    Validator Agent - Quality control for retrieved chunks.
    
    Checks if retrieved chunks are sufficient to answer the query.
    Can trigger re-retrieval if quality threshold not met.
    
    Attributes:
        llm: Language model for validation assessment
        threshold: Minimum sufficiency score (0.0-1.0)
        max_retries: Maximum retrieval retry attempts
        
    Example:
        >>> from langchain_anthropic import ChatAnthropic
        >>> llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
        >>> validator = ValidatorAgent(llm=llm)
        >>> 
        >>> state = AgentState(query="What is Python?", chunks=[...])
        >>> result = validator.run(state)
        >>> 
        >>> print(result.validation_status)  # "PROCEED" or "RETRIEVE_MORE"
        >>> print(result.validation_score)   # 0.85
    """
    
    def __init__(
        self,
        llm: ChatAnthropic,
        threshold: float = None,
        max_retries: int = None
    ):
        """
        Initialize Validator Agent.
        
        Args:
            llm: ChatAnthropic instance for validation
            threshold: Minimum sufficiency score (default: from config)
            max_retries: Maximum retry attempts (default: from config)
        
        Example:
            >>> llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
            >>> validator = ValidatorAgent(llm=llm, threshold=0.75)
        """
        super().__init__(name="validator", version="1.0.0")
        
        self.llm = llm
        
        # Load settings from config
        settings = get_settings()
        self.threshold = (
            threshold 
            if threshold is not None 
            else settings.validator_threshold
        )
        self.max_retries = (
            max_retries 
            if max_retries is not None 
            else settings.validator_max_retries
        )
        
        self.log(
            f"Initialized with threshold={self.threshold}, "
            f"max_retries={self.max_retries}",
            level="debug"
        )
    
    def execute(self, state: AgentState) -> AgentState:
        """
        Execute validation logic: check chunk sufficiency.
        
        Args:
            state: Current agent state with query and chunks
        
        Returns:
            Updated state with validation_status and validation_score
        
        Raises:
            ValidationError: If validation process fails
        
        Example:
            >>> state = AgentState(query="What is X?", chunks=[...])
            >>> result = validator.execute(state)
            >>> if result.validation_status == "PROCEED":
            ...     # Continue to answer generation
            >>> else:
            ...     # Retrieve more chunks
        """
        try:
            query = state.query
            chunks = state.chunks
            current_round = state.retrieval_round
            
            self.log(
                f"Validating {len(chunks)} chunks (round {current_round})",
                level="info"
            )
            
            # Step 1: Calculate sufficiency score
            score = self._calculate_sufficiency(query, chunks)
            state.validation_score = score
            
            self.log(f"Sufficiency score: {score:.3f}", level="info")
            
            # Step 2: Make decision
            decision = self._make_decision(score, current_round)
            state.validation_status = decision
            
            self.log(
                f"Decision: {decision} (score: {score:.3f}, "
                f"round: {current_round}/{self.max_retries})",
                level="info"
            )
            
            # Step 3: Add metadata
            state.metadata["validator"] = {
                "score": score,
                "decision": decision,
                "retrieval_round": current_round,
                "threshold": self.threshold,
                "max_retries": self.max_retries
            }
            
            return state
            
        except Exception as e:
            self.log(f"Validation failed: {str(e)}", level="error")
            raise ValidationError(
                message=f"Chunk validation failed: {str(e)}",
                validation_type="sufficiency",
                details={
                    "query": state.query,
                    "chunk_count": len(state.chunks)
                }
            ) from e
    
    def _calculate_sufficiency(self, query: str, chunks: List[Chunk]) -> float:
        """
        Calculate if chunks are sufficient to answer query.
        
        Combines three factors:
        - Relevance: 50% weight
        - Coverage: 30% weight
        - Confidence: 20% weight
        
        Args:
            query: User query string
            chunks: List of retrieved chunks
        
        Returns:
            Sufficiency score (0.0-1.0)
        
        Example:
            >>> score = validator._calculate_sufficiency("What is X?", chunks)
            >>> print(score)  # 0.82
        """
        if not chunks:
            self.log("No chunks to validate", level="warning")
            return 0.0
        
        try:
            # Factor 1: Relevance (50%)
            relevance_score = self._check_relevance(query, chunks)
            
            # Factor 2: Coverage (30%)
            coverage_score = self._check_coverage(query, chunks)
            
            # Factor 3: Confidence (20%)
            confidence_score = self._check_confidence(chunks)
            
            self.log(
                f"Validation factors: relevance={relevance_score:.2f}, "
                f"coverage={coverage_score:.2f}, confidence={confidence_score:.2f}",
                level="debug"
            )
            
            # Weighted combination
            final_score = (
                relevance_score * 0.5 +
                coverage_score * 0.3 +
                confidence_score * 0.2
            )
            
            return max(0.0, min(final_score, 1.0))
            
        except Exception as e:
            self.log(f"Error calculating sufficiency: {str(e)}", level="warning")
            # Return conservative score on error
            return 0.5
    
    def _check_relevance(self, query: str, chunks: List[Chunk]) -> float:
        """
        Check if chunks are relevant to query using LLM.
        
        Asks Claude to assess how well chunks relate to the query.
        
        Args:
            query: User query string
            chunks: List of retrieved chunks
        
        Returns:
            Relevance score (0.0-1.0)
        """
        # Prepare context from top chunks
        top_chunks = sorted(chunks, key=lambda c: c.score or 0.0, reverse=True)[:5]
        context = "\n\n".join([
            f"Chunk {i+1} (score: {chunk.score:.2f}):\n{chunk.text[:200]}..."
            for i, chunk in enumerate(top_chunks)
        ])
        
        prompt = f"""Assess how relevant these retrieved chunks are to the query.

Query: "{query}"

Retrieved Chunks:
{context}

Rate relevance on scale 0.0-1.0:
- 0.0-0.3: Not relevant, chunks don't relate to query
- 0.4-0.6: Partially relevant, some connection but missing key aspects
- 0.7-0.9: Relevant, chunks address the query
- 1.0: Highly relevant, chunks directly answer the query

Respond with ONLY a number between 0.0 and 1.0."""

        try:
            response = self.llm.invoke(prompt)
            score_text = response.content.strip()
            
            # Extract number
            import re
            numbers = re.findall(r'0\.\d+|1\.0|0|1', score_text)
            
            if numbers:
                score = float(numbers[0])
                return max(0.0, min(score, 1.0))
            else:
                self.log(
                    f"Could not parse relevance score: {score_text}, using fallback",
                    level="warning"
                )
                return self._fallback_relevance_score(query, chunks)
                
        except Exception as e:
            self.log(
                f"LLM relevance check failed: {str(e)}, using fallback",
                level="warning"
            )
            return self._fallback_relevance_score(query, chunks)
    
    def _fallback_relevance_score(self, query: str, chunks: List[Chunk]) -> float:
        """
        Fallback relevance scoring without LLM.
        
        Uses average chunk scores as proxy for relevance.
        
        Args:
            query: User query string
            chunks: List of retrieved chunks
        
        Returns:
            Fallback relevance score
        """
        if not chunks:
            return 0.0
        
        # Use average of existing chunk scores
        scores = [c.score for c in chunks if c.score is not None]
        
        if scores:
            return sum(scores) / len(scores)
        else:
            # Default to moderate if no scores
            return 0.5
    
    def _check_coverage(self, query: str, chunks: List[Chunk]) -> float:
        """
        Check if chunks cover all aspects of the query.
        
        For simple queries, needs basic coverage.
        For complex queries, needs multiple aspects covered.
        
        Args:
            query: User query string
            chunks: List of retrieved chunks
        
        Returns:
            Coverage score (0.0-1.0)
        """
        if not chunks:
            return 0.0
        
        # Extract query aspects
        query_lower = query.lower()
        
        # Count questions/sub-parts
        question_marks = query.count("?")
        and_or_count = query_lower.count(" and ") + query_lower.count(" or ")
        
        # Estimate number of aspects
        num_aspects = max(1, question_marks + and_or_count)
        
        # Check chunk count relative to aspects
        chunk_count = len(chunks)
        
        # Simple heuristic: need at least 2 chunks per aspect
        ideal_chunks = num_aspects * 2
        coverage_ratio = min(chunk_count / ideal_chunks, 1.0)
        
        # Also consider chunk diversity (different sources)
        unique_docs = len(set(c.doc_id for c in chunks if c.doc_id))
        diversity_score = min(unique_docs / 3, 1.0) if unique_docs > 0 else 0.3
        
        # Combine ratio and diversity
        coverage_score = (coverage_ratio * 0.7) + (diversity_score * 0.3)
        
        return coverage_score
    
    def _check_confidence(self, chunks: List[Chunk]) -> float:
        """
        Check confidence in chunk quality.
        
        Based on:
        - Average chunk scores (higher = better)
        - Score consistency (less variance = better)
        - Minimum score threshold
        
        Args:
            chunks: List of retrieved chunks
        
        Returns:
            Confidence score (0.0-1.0)
        """
        if not chunks:
            return 0.0
        
        scores = [c.score for c in chunks if c.score is not None]
        
        if not scores:
            return 0.5  # Moderate confidence if no scores
        
        # Average score
        avg_score = sum(scores) / len(scores)
        
        # Minimum score (worst chunk)
        min_score = min(scores)
        
        # Score variance (consistency)
        if len(scores) > 1:
            variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
            consistency = max(0.0, 1.0 - variance)
        else:
            consistency = 1.0
        
        # Weighted combination
        confidence = (
            avg_score * 0.5 +
            min_score * 0.3 +
            consistency * 0.2
        )
        
        return confidence
    
    def _make_decision(self, score: float, current_round: int) -> str:
        """
        Make validation decision based on score and retry count.
        
        Decision Logic:
        1. If score >= threshold → PROCEED
        2. If score < threshold AND retries available → RETRIEVE_MORE
        3. If max retries reached → PROCEED (force)
        
        Args:
            score: Sufficiency score
            current_round: Current retrieval round (0-indexed)
        
        Returns:
            "PROCEED" or "RETRIEVE_MORE"
        
        Example:
            >>> decision = validator._make_decision(0.85, 0)
            >>> print(decision)  # "PROCEED"
            
            >>> decision = validator._make_decision(0.45, 0)
            >>> print(decision)  # "RETRIEVE_MORE"
            
            >>> decision = validator._make_decision(0.45, 2)  # max retries
            >>> print(decision)  # "PROCEED"
        """
        # Check if score meets threshold
        if score >= self.threshold:
            return "PROCEED"
        
        # Check if we can retry
        if current_round < self.max_retries:
            self.log(
                f"Score {score:.3f} below threshold {self.threshold}, "
                f"triggering retry (round {current_round + 1}/{self.max_retries})",
                level="warning"
            )
            return "RETRIEVE_MORE"
        
        # Max retries reached, force proceed
        self.log(
            f"Max retries reached ({self.max_retries}), "
            f"proceeding with score {score:.3f}",
            level="warning"
        )
        return "PROCEED"
    
    def validate_chunks_detailed(
        self,
        query: str,
        chunks: List[Chunk]
    ) -> dict:
        """
        Detailed validation analysis for debugging.
        
        Returns all intermediate scores and decision factors.
        
        Args:
            query: User query string
            chunks: List of retrieved chunks
        
        Returns:
            Dictionary with detailed validation analysis
        
        Example:
            >>> details = validator.validate_chunks_detailed("What is X?", chunks)
            >>> print(details["relevance_score"])
            >>> print(details["coverage_score"])
            >>> print(details["final_decision"])
        """
        relevance = self._check_relevance(query, chunks)
        coverage = self._check_coverage(query, chunks)
        confidence = self._check_confidence(chunks)
        
        final_score = (
            relevance * 0.5 +
            coverage * 0.3 +
            confidence * 0.2
        )
        
        decision = self._make_decision(final_score, 0)
        
        return {
            "query": query,
            "chunk_count": len(chunks),
            "relevance_score": relevance,
            "coverage_score": coverage,
            "confidence_score": confidence,
            "final_score": final_score,
            "threshold": self.threshold,
            "final_decision": decision,
            "would_retry": decision == "RETRIEVE_MORE"
        }