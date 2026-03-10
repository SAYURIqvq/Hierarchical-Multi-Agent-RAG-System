"""
Self-Reflection Loop - Coordinate Writer and Critic agents.

Implements iterative improvement:
Writer → Critic → (if needed) Writer → Critic → ...
"""

from typing import Optional

from src.agents.writer import WriterAgent
from src.agents.critic import CriticAgent, CriticDecision
from src.models.agent_state import AgentState
from src.utils.logger import setup_logger


class SelfReflectionLoop:
    """
    Self-reflection loop for answer improvement.
    
    Coordinates Writer and Critic agents in iterative loop:
    1. Writer generates answer
    2. Critic reviews quality
    3. If REGENERATE → Writer improves with feedback
    4. Repeat until APPROVED or max iterations
    
    Example:
        >>> loop = SelfReflectionLoop(writer=writer, critic=critic)
        >>> state = AgentState(query="...", chunks=[...])
        >>> final_state = loop.run(state)
        >>> print(final_state.answer)  # Improved answer
    """
    
    def __init__(
        self,
        writer: WriterAgent,
        critic: CriticAgent,
        max_iterations: int = 3
    ):
        """
        Initialize self-reflection loop.
        
        Args:
            writer: WriterAgent instance
            critic: CriticAgent instance
            max_iterations: Maximum improvement iterations
        """
        self.writer = writer
        self.critic = critic
        self.max_iterations = max_iterations
        self.logger = setup_logger("self_reflection")
        
        self.logger.info(
            f"Initialized self-reflection loop (max_iterations={max_iterations})"
        )
    
    def run(self, state: AgentState) -> AgentState:
        """
        Run self-reflection loop.
        
        Args:
            state: State with query and chunks
        
        Returns:
            State with final improved answer
        
        Example:
            >>> state = AgentState(query="What is ML?", chunks=[...])
            >>> result = loop.run(state)
            >>> print(result.metadata['regeneration_count'])  # 0-3
        """
        iteration = 0
        
        self.logger.info(f"Starting self-reflection loop for: {state.query[:50]}...")
        
        # Initial answer generation
        self.logger.info("Iteration 0: Initial answer generation")
        state = self.writer.run(state)
        
        # Critique loop
        while iteration < self.max_iterations:
            iteration += 1
            
            self.logger.info(f"Iteration {iteration}: Critique")
            state = self.critic.run(state)
            
            # Check decision
            decision = state.critic_decision
            score = state.critic_score
            
            self.logger.info(
                f"Critic decision: {decision.value} (score: {score:.3f})"
            )
            
            if decision == CriticDecision.APPROVED:
                self.logger.info("✅ Answer approved!")
                break
            
            elif decision == CriticDecision.REGENERATE:
                if iteration >= self.max_iterations:
                    self.logger.warning(
                        f"Max iterations ({self.max_iterations}) reached. "
                        "Accepting current answer."
                    )
                    break
                
                # Regenerate with feedback
                self.logger.info(f"🔄 Regenerating answer (iteration {iteration})")
                self.logger.debug(f"Feedback: {state.critic_feedback[:100]}...")
                
                # Use Writer's regeneration method
                improved_answer = self.writer.generate_with_feedback(
                    query=state.query,
                    chunks=state.chunks,
                    feedback=state.critic_feedback
                )
                
                state.answer = improved_answer
                state.metadata["regeneration_count"] = iteration
            
            else:  # INSUFFICIENT_INFO
                self.logger.warning("Insufficient information to answer query")
                break
        
        # Final metadata
        state.metadata["self_reflection"] = {
            "iterations": iteration,
            "final_score": state.critic_score,
            "final_decision": state.critic_decision.value,
            "improved": iteration > 0
        }
        
        self.logger.info(
            f"Self-reflection complete: {iteration} iterations, "
            f"final score: {state.critic_score:.3f}"
        )
        
        return state
    
    def get_stats(self, state: AgentState) -> dict:
        """
        Get self-reflection statistics.
        
        Args:
            state: AgentState after self-reflection
        
        Returns:
            Dictionary with stats
        """
        return state.metadata.get("self_reflection", {})