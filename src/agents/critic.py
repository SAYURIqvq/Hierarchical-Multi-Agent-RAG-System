"""
Critic Agent - Tactical Level 2 Agent.

Reviews generated answers for quality and provides improvement feedback.
Triggers regeneration if quality is below threshold.
"""

from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

from langchain_anthropic import ChatAnthropic

from src.agents.base_agent import BaseAgent
from src.models.agent_state import AgentState
from src.config import get_settings
from src.utils.logger import setup_logger
from src.utils.exceptions import AgenticRAGException


class CriticDecision(Enum):
    """Critic's decision on answer quality."""
    APPROVED = "approved"
    REGENERATE = "regenerate"
    INSUFFICIENT_INFO = "insufficient_info"


class CriticError(AgenticRAGException):
    """Error during answer critique."""
    pass


class CriticAgent(BaseAgent):
    """
    Critic Agent - Answer quality review.
    
    Reviews generated answers for:
    - Accuracy (based on sources)
    - Completeness (addresses all parts of query)
    - Citation quality (proper attribution)
    - Clarity (well-structured and readable)
    - Relevance (directly answers question)
    
    Makes decisions:
    - APPROVED: Answer is good quality
    - REGENERATE: Answer needs improvement
    - INSUFFICIENT_INFO: Not enough context to answer
    
    Attributes:
        llm: Language model for critique
        quality_threshold: Minimum score to approve (0.0-1.0)
        max_iterations: Maximum regeneration attempts
        
    Example:
        >>> agent = CriticAgent(llm=llm, quality_threshold=0.7)
        >>> state = AgentState(query="...", answer="...", chunks=[...])
        >>> result = agent.run(state)
        >>> print(result.critic_decision)  # APPROVED or REGENERATE
    """
    
    def __init__(
        self,
        llm: Optional[ChatAnthropic] = None,
        quality_threshold: float = 0.7,
        max_iterations: int = 3
    ):
        """
        Initialize Critic Agent.
        
        Args:
            llm: ChatAnthropic instance (creates if None)
            quality_threshold: Minimum quality score (0.0-1.0)
            max_iterations: Max regeneration attempts
        
        Example:
            >>> from langchain_anthropic import ChatAnthropic
            >>> llm = ChatAnthropic(model="claude-3-haiku-20240307")
            >>> agent = CriticAgent(llm=llm, quality_threshold=0.8)
        """
        super().__init__(name="critic", version="1.0.0")
        
        settings = get_settings()
        
        # Initialize LLM
        if llm is None:
            self.llm = ChatAnthropic(
                model=settings.llm_model,
                temperature=0.0,  # Deterministic for consistency
                max_tokens=2000,
                api_key=settings.anthropic_api_key
            )
        else:
            self.llm = llm
        
        self.quality_threshold = quality_threshold
        self.max_iterations = max_iterations
        
        self.log(
            f"Initialized with threshold={quality_threshold:.2f}, "
            f"max_iterations={max_iterations}",
            level="info"
        )
    
    def execute(self, state: AgentState) -> AgentState:
        """
        Execute answer critique.
        
        Args:
            state: Current state with query, answer, and chunks
        
        Returns:
            Updated state with critique results
        
        Raises:
            CriticError: If critique fails
        
        Example:
            >>> state = AgentState(query="...", answer="...", chunks=[...])
            >>> result = agent.execute(state)
            >>> print(result.critic_score)  # 0.0-1.0
        """
        try:
            query = state.query
            answer = state.answer
            chunks = state.chunks
            
            if not answer:
                self.log("No answer to critique", level="warning")
                state.critic_decision = CriticDecision.INSUFFICIENT_INFO
                state.critic_score = 0.0
                return state
            
            self.log(
                f"Critiquing answer for: {query[:50]}...",
                level="info"
            )
            
            # Perform critique
            critique_result = self._critique_answer(query, answer, chunks)
            
            # Update state
            state.critic_score = critique_result['overall_score']
            state.critic_feedback = critique_result['feedback']
            state.critic_scores = critique_result['scores']
            
            # Make decision
            decision = self._make_decision(
                critique_result['overall_score'],
                critique_result
            )
            state.critic_decision = decision
            
            # Add metadata
            state.metadata["critic"] = {
                "overall_score": critique_result['overall_score'],
                "decision": decision.value,
                "scores": critique_result['scores'],
                "iteration": state.metadata.get("regeneration_count", 0)
            }
            
            self.log(
                f"Critique complete: score={critique_result['overall_score']:.3f}, "
                f"decision={decision.value}",
                level="info"
            )
            
            return state
            
        except Exception as e:
            self.log(f"Critique failed: {str(e)}", level="error")
            raise CriticError(
                message=f"Failed to critique answer: {str(e)}",
                details={"query": state.query}
            ) from e
    
    def _critique_answer(
        self,
        query: str,
        answer: str,
        chunks: List
    ) -> Dict[str, Any]:
        """
        Critique answer using LLM.
        
        Args:
            query: User query
            answer: Generated answer
            chunks: Source chunks
        
        Returns:
            Dictionary with scores and feedback
        """
        # Prepare context
        context_parts = []
        for i, chunk in enumerate(chunks[:5], 1):  # Use top 5
            context_parts.append(f"[{i}] {chunk.text[:200]}...")
        
        context = "\n".join(context_parts)
        
        # Create critique prompt
        prompt = f"""You are a quality reviewer for AI-generated answers. Your job is to critique an answer based on provided context.

User Query: {query}

Available Context:
{context}

Generated Answer:
{answer}

Evaluate the answer on these criteria (score each 0.0-1.0):

1. ACCURACY: Does the answer provide correct information based on the context?
2. COMPLETENESS: Does it fully address all parts of the query?
3. CITATIONS: Are citations properly used and relevant?
4. CLARITY: Is the answer well-structured and easy to understand?
5. RELEVANCE: Does it directly answer the question asked?

Respond in this exact format:

SCORES:
accuracy: [0.0-1.0]
completeness: [0.0-1.0]
citations: [0.0-1.0]
clarity: [0.0-1.0]
relevance: [0.0-1.0]

FEEDBACK:
[Specific suggestions for improvement, or "No improvements needed" if excellent]

RECOMMENDATION:
[Either "APPROVED" or "REGENERATE"]"""
        
        # Get critique from LLM
        try:
            response = self.llm.invoke(prompt)
            critique_text = response.content
            
            # Parse response
            parsed = self._parse_critique(critique_text)
            
            return parsed
            
        except Exception as e:
            raise CriticError(
                message=f"LLM critique failed: {str(e)}",
                details={"query": query}
            ) from e
    
    def _parse_critique(self, critique_text: str) -> Dict[str, Any]:
        """
        Parse LLM critique response.
        
        Args:
            critique_text: Raw LLM response
        
        Returns:
            Parsed scores and feedback
        """
        import re
        
        scores = {}
        feedback = ""
        
        # Extract scores
        score_patterns = {
            'accuracy': r'accuracy:\s*([0-9.]+)',
            'completeness': r'completeness:\s*([0-9.]+)',
            'citations': r'citations:\s*([0-9.]+)',
            'clarity': r'clarity:\s*([0-9.]+)',
            'relevance': r'relevance:\s*([0-9.]+)'
        }
        
        for criterion, pattern in score_patterns.items():
            match = re.search(pattern, critique_text, re.IGNORECASE)
            if match:
                scores[criterion] = float(match.group(1))
            else:
                scores[criterion] = 0.5  # Default if not found
        
        # Extract feedback
        feedback_match = re.search(
            r'FEEDBACK:\s*(.+?)(?=RECOMMENDATION:|$)',
            critique_text,
            re.DOTALL | re.IGNORECASE
        )
        if feedback_match:
            feedback = feedback_match.group(1).strip()
        
        # Calculate overall score (weighted average)
        weights = {
            'accuracy': 0.3,
            'completeness': 0.25,
            'citations': 0.15,
            'clarity': 0.15,
            'relevance': 0.15
        }
        
        overall_score = sum(
            scores.get(criterion, 0.5) * weight
            for criterion, weight in weights.items()
        )
        
        return {
            'scores': scores,
            'overall_score': overall_score,
            'feedback': feedback
        }
    
    def _make_decision(
        self,
        overall_score: float,
        critique_result: Dict[str, Any]
    ) -> CriticDecision:
        """
        Make decision based on critique.
        
        Args:
            overall_score: Overall quality score
            critique_result: Full critique results
        
        Returns:
            CriticDecision (APPROVED or REGENERATE)
        """
        # Check if score meets threshold
        if overall_score >= self.quality_threshold:
            return CriticDecision.APPROVED
        
        # Check for critical failures
        scores = critique_result['scores']
        if scores.get('accuracy', 0) < 0.4:
            return CriticDecision.REGENERATE
        
        if scores.get('relevance', 0) < 0.4:
            return CriticDecision.REGENERATE
        
        # Below threshold
        return CriticDecision.REGENERATE
    
    def should_regenerate(self, state: AgentState) -> bool:
        """
        Check if answer should be regenerated.
        
        Args:
            state: AgentState with critic results
        
        Returns:
            True if regeneration needed
        
        Example:
            >>> if agent.should_regenerate(state):
            ...     # Regenerate answer
        """
        if not hasattr(state, 'critic_decision'):
            return False
        
        # Check decision
        if state.critic_decision != CriticDecision.REGENERATE:
            return False
        
        # Check iteration limit
        regeneration_count = state.metadata.get("regeneration_count", 0)
        if regeneration_count >= self.max_iterations:
            self.log(
                f"Max iterations ({self.max_iterations}) reached, stopping",
                level="warning"
            )
            return False
        
        return True