"""
Planner Agent - Strategic Level 1 Agent.

Analyzes query complexity and selects appropriate execution strategy.
This is the first agent in the workflow that determines how the system
should process the query.

Complexity Factors:
- Query length (30%)
- Number of questions/sub-questions (20%)
- Entity and relationship indicators (20%)
- Semantic complexity via LLM (30%)

Strategies:
- SIMPLE: Fast path for straightforward queries (< 0.3)
- MULTIHOP: Multi-step reasoning for complex queries (0.3-0.7)
- GRAPH: Relationship-based reasoning (> 0.7)
"""

import re
from typing import Dict, Tuple
from langchain_anthropic import ChatAnthropic

from src.agents.base_agent import BaseAgent
from src.models.agent_state import AgentState, Strategy
from src.utils.exceptions import AgentExecutionError
from src.config import get_settings


class PlannerAgent(BaseAgent):
    """
    Planner Agent - Analyzes query complexity and selects strategy.
    
    This is the strategic decision-maker that determines how the system
    should approach answering a query. It analyzes various factors to
    assign a complexity score and map that to an execution strategy.
    
    Attributes:
        llm: Language model for semantic complexity analysis
        simple_threshold: Complexity threshold for simple strategy
        multihop_threshold: Complexity threshold for multihop strategy
        
    Example:
        >>> from langchain_anthropic import ChatAnthropic
        >>> llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
        >>> planner = PlannerAgent(llm=llm)
        >>> 
        >>> state = AgentState(query="What is Python?")
        >>> result = planner.run(state)
        >>> 
        >>> print(result.complexity)  # 0.2
        >>> print(result.strategy)    # Strategy.SIMPLE
    """
    
    def __init__(
        self,
        llm: ChatAnthropic,
        simple_threshold: float = None,
        multihop_threshold: float = None
    ):
        """
        Initialize Planner Agent.
        
        Args:
            llm: ChatAnthropic instance for semantic analysis
            simple_threshold: Threshold for simple strategy (default: from config)
            multihop_threshold: Threshold for multihop strategy (default: from config)
        
        Example:
            >>> llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
            >>> planner = PlannerAgent(llm=llm)
        """
        super().__init__(name="planner", version="1.0.0")
        
        self.llm = llm
        
        # Load thresholds from config
        settings = get_settings()
        self.simple_threshold = (
            simple_threshold 
            if simple_threshold is not None 
            else settings.planner_complexity_threshold_simple
        )
        self.multihop_threshold = (
            multihop_threshold 
            if multihop_threshold is not None 
            else settings.planner_complexity_threshold_multihop
        )
        
        self.log(
            f"Initialized with thresholds: simple={self.simple_threshold}, "
            f"multihop={self.multihop_threshold}",
            level="debug"
        )
    
    def execute(self, state: AgentState) -> AgentState:
        """
        Execute planning logic: analyze complexity and select strategy.
        
        Args:
            state: Current agent state with query
        
        Returns:
            Updated state with complexity and strategy
        
        Raises:
            AgentExecutionError: If planning fails
        
        Example:
            >>> state = AgentState(query="Compare Python and Java")
            >>> result = planner.execute(state)
            >>> print(result.complexity)  # 0.65
            >>> print(result.strategy)    # Strategy.MULTIHOP
        """
        try:
            query = state.query
            
            self.log(f"Analyzing query: {query[:50]}...", level="info")
            
            # Step 1: Analyze complexity
            complexity = self._analyze_complexity(query)
            state.complexity = complexity
            
            self.log(f"Complexity score: {complexity:.3f}", level="info")
            
            # Step 2: Select strategy
            strategy = self._select_strategy(complexity)
            state.strategy = strategy
            
            self.log(
                f"Selected strategy: {strategy.value} (complexity: {complexity:.3f})",
                level="info"
            )
            
            # Step 3: Add metadata
            state.metadata["planner"] = {
                "complexity": complexity,
                "strategy": strategy.value,
                "thresholds": {
                    "simple": self.simple_threshold,
                    "multihop": self.multihop_threshold
                }
            }
            
            return state
            
        except Exception as e:
            self.log(f"Planning failed: {str(e)}", level="error")
            raise AgentExecutionError(
                agent_name=self.name,
                message=f"Failed to analyze query complexity: {str(e)}",
                details={"query": state.query}
            ) from e
    
    def _analyze_complexity(self, query: str) -> float:
        """
        Analyze query complexity using multiple factors.
        
        Combines heuristic features (60%) and semantic analysis (40%)
        to produce final complexity score.
        
        Args:
            query: User query string
        
        Returns:
            Complexity score between 0.0 and 1.0
        
        Example:
            >>> score = planner._analyze_complexity("What is Python?")
            >>> print(score)  # 0.15
        """
        # Extract heuristic features
        features = self._extract_features(query)
        
        # Calculate heuristic score (60% weight)
        heuristic_score = (
            features["length_score"] * 0.3 +
            features["question_score"] * 0.2 +
            features["entity_score"] * 0.2 +
            features["relationship_score"] * 0.3
        )
        
        self.log(
            f"Heuristic features: length={features['length_score']:.2f}, "
            f"questions={features['question_score']:.2f}, "
            f"entities={features['entity_score']:.2f}, "
            f"relationships={features['relationship_score']:.2f}",
            level="debug"
        )
        
        # Get semantic complexity from LLM (40% weight)
        semantic_score = self._semantic_complexity(query)
        
        self.log(
            f"Scores: heuristic={heuristic_score:.3f}, semantic={semantic_score:.3f}",
            level="debug"
        )
        
        # Weighted combination
        final_score = (heuristic_score * 0.6) + (semantic_score * 0.4)
        
        # Clamp to [0, 1]
        return max(0.0, min(final_score, 1.0))
    
    def _extract_features(self, query: str) -> Dict[str, float]:
        """
        Extract heuristic features from query.
        
        Features:
        - length_score: Based on word count
        - question_score: Number of question marks and sub-questions
        - entity_score: Presence of entity indicators
        - relationship_score: Presence of relationship words
        
        Args:
            query: User query string
        
        Returns:
            Dictionary of feature scores (0.0-1.0)
        
        Example:
            >>> features = planner._extract_features("Compare X and Y")
            >>> print(features["relationship_score"])  # High score
        """
        query_lower = query.lower()
        words = query.split()
        word_count = len(words)
        
        # 1. Length score (longer queries are more complex)
        # 0-5 words: 0.0-0.2, 6-15 words: 0.3-0.6, 16+ words: 0.7-1.0
        if word_count <= 5:
            length_score = word_count / 25  # Max 0.2
        elif word_count <= 15:
            length_score = 0.2 + ((word_count - 5) / 15)  # 0.2-0.87
        else:
            length_score = min(0.7 + ((word_count - 15) / 30), 1.0)
        
        # 2. Question score (multiple questions = complex)
        question_marks = query.count("?")
        question_words = sum(1 for w in ["what", "why", "how", "when", "where", "which"] if w in query_lower)
        and_or_count = query_lower.count(" and ") + query_lower.count(" or ")
        
        question_score = min((question_marks * 0.3 + question_words * 0.1 + and_or_count * 0.15), 1.0)
        
        # 3. Entity score (mentions of specific things)
        entity_indicators = [
            "compare", "difference", "similar", "contrast",
            "relationship", "between", "among",
            "example", "list", "types of", "kinds of"
        ]
        entity_count = sum(1 for indicator in entity_indicators if indicator in query_lower)
        entity_score = min(entity_count / 5, 1.0)
        
        # 4. Relationship score (requires understanding connections)
        relationship_indicators = [
            "how does", "why does", "what causes", "impact of",
            "effect of", "related to", "connected to", "leads to",
            "results in", "because of", "due to", "depends on"
        ]
        relationship_count = sum(1 for indicator in relationship_indicators if indicator in query_lower)
        relationship_score = min(relationship_count / 3, 1.0)
        
        return {
            "length_score": length_score,
            "question_score": question_score,
            "entity_score": entity_score,
            "relationship_score": relationship_score
        }
    
    def _semantic_complexity(self, query: str) -> float:
        """
        Use LLM to assess semantic complexity.
        
        Asks Claude to rate the query complexity based on:
        - Number of concepts involved
        - Depth of reasoning required
        - Need for multi-step thinking
        
        Args:
            query: User query string
        
        Returns:
            Semantic complexity score (0.0-1.0)
        
        Example:
            >>> score = planner._semantic_complexity("Explain quantum computing")
            >>> print(score)  # 0.75 (complex topic)
        """
        prompt = f"""Analyze the complexity of this query on a scale of 0.0 to 1.0.

Query: "{query}"

Consider:
- Number of concepts/topics involved
- Depth of reasoning required
- Whether it needs multi-step thinking
- Whether it involves comparisons or relationships

Complexity scale:
- 0.0-0.3: Simple, single-concept, factual question
- 0.4-0.6: Moderate, requires some reasoning or multiple facts
- 0.7-1.0: Complex, multi-step reasoning, comparisons, or deep analysis

Respond with ONLY a number between 0.0 and 1.0, nothing else."""

        try:
            response = self.llm.invoke(prompt)
            score_text = response.content.strip()
            
            # Extract number from response
            # Handle cases like "0.5" or "The complexity is 0.5"
            numbers = re.findall(r'0\.\d+|1\.0|0|1', score_text)
            
            if numbers:
                score = float(numbers[0])
                # Ensure in valid range
                return max(0.0, min(score, 1.0))
            else:
                self.log(
                    f"Could not parse LLM response: {score_text}, using fallback",
                    level="warning"
                )
                return self._fallback_semantic_score(query)
                
        except Exception as e:
            self.log(
                f"LLM semantic analysis failed: {str(e)}, using fallback",
                level="warning"
            )
            return self._fallback_semantic_score(query)
    
    def _fallback_semantic_score(self, query: str) -> float:
        """
        Fallback semantic scoring without LLM.
        
        Uses simple heuristics when LLM is unavailable.
        
        Args:
            query: User query string
        
        Returns:
            Fallback complexity score (0.0-1.0)
        """
        query_lower = query.lower()
        
        # Complex keywords
        complex_keywords = [
            "explain", "analyze", "compare", "evaluate", "discuss",
            "why", "how come", "relationship", "impact", "effect"
        ]
        
        # Simple keywords
        simple_keywords = [
            "what is", "define", "who is", "when", "where"
        ]
        
        has_complex = any(keyword in query_lower for keyword in complex_keywords)
        has_simple = any(keyword in query_lower for keyword in simple_keywords)
        
        if has_simple and not has_complex:
            return 0.2
        elif has_complex:
            return 0.6
        else:
            # Moderate by default
            return 0.4
    
    def _select_strategy(self, complexity: float) -> Strategy:
        """
        Select execution strategy based on complexity score.
        
        Mapping:
        - [0.0, simple_threshold): SIMPLE
        - [simple_threshold, multihop_threshold): MULTIHOP
        - [multihop_threshold, 1.0]: GRAPH
        
        Args:
            complexity: Complexity score (0.0-1.0)
        
        Returns:
            Selected strategy
        
        Example:
            >>> strategy = planner._select_strategy(0.25)
            >>> print(strategy)  # Strategy.SIMPLE
        """
        if complexity < self.simple_threshold:
            return Strategy.SIMPLE
        elif complexity < self.multihop_threshold:
            return Strategy.MULTIHOP
        else:
            return Strategy.GRAPH
    
    def analyze_query_details(self, query: str) -> Dict[str, any]:
        """
        Detailed analysis of query for debugging/visualization.
        
        Returns all intermediate scores and features for inspection.
        
        Args:
            query: User query string
        
        Returns:
            Dictionary with detailed analysis
        
        Example:
            >>> details = planner.analyze_query_details("Compare X and Y")
            >>> print(details["features"])
            >>> print(details["final_complexity"])
        """
        features = self._extract_features(query)
        semantic = self._semantic_complexity(query)
        
        heuristic = (
            features["length_score"] * 0.3 +
            features["question_score"] * 0.2 +
            features["entity_score"] * 0.2 +
            features["relationship_score"] * 0.3
        )
        
        final_complexity = (heuristic * 0.6) + (semantic * 0.4)
        strategy = self._select_strategy(final_complexity)
        
        return {
            "query": query,
            "features": features,
            "heuristic_score": heuristic,
            "semantic_score": semantic,
            "final_complexity": final_complexity,
            "selected_strategy": strategy.value,
            "thresholds": {
                "simple": self.simple_threshold,
                "multihop": self.multihop_threshold
            }
        }