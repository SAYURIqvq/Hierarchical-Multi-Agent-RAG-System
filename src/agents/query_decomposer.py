"""
Query Decomposer Agent - Tactical Level 2
Breaks complex queries into manageable sub-questions based on strategy.
"""

from typing import List, Dict, Any
from langchain_anthropic import ChatAnthropic

from src.agents.base_agent import BaseAgent
from src.models.agent_state import AgentState, Strategy
from src.config import get_settings
from src.utils.exceptions import AgenticRAGException


class QueryDecomposerError(AgenticRAGException):
    """Error during query decomposition."""
    pass


class QueryDecomposer(BaseAgent):
    """Query Decomposer with strategy-aware logic."""
    
    def __init__(self, llm: ChatAnthropic = None):
        super().__init__(name="query_decomposer", version="1.0.0")
        # ... (sama seperti sebelumnya)
    
    def execute(self, state: AgentState) -> AgentState:
        """
        Decompose query based on Planner's strategy.
        
        Strategy handling:
        - SIMPLE: No decomposition
        - MULTIHOP: Sequential decomposition
        - GRAPH: Entity + relationship decomposition
        """
        query = state.query
        strategy = state.strategy  # ← BACA dari Planner

        if strategy is None:
            self.log("WARNING: Strategy not set! Using SIMPLE as default", level="warning")
            strategy = Strategy.SIMPLE
                    
        self.log(f"Using strategy: {strategy.value}", level="info")
        
        # ========================================
        # STRATEGY-BASED LOGIC
        # ========================================
        if strategy == Strategy.SIMPLE:
            # No decomposition for simple queries
            self.log("Simple strategy - no decomposition", level="info")
            state.sub_queries = [query]
        
        elif strategy == Strategy.MULTIHOP:
            # Sequential decomposition
            self.log("Multihop strategy - sequential decomposition", level="info")
            sub_queries = self._decompose_multihop(query)
            state.sub_queries = sub_queries
        
        elif strategy == Strategy.GRAPH:
            # Entity + relationship decomposition
            self.log("Graph strategy - entity decomposition", level="info")
            sub_queries = self._decompose_graph(query)
            state.sub_queries = sub_queries
        
        else:
            # Fallback
            self.log(f"Unknown strategy: {strategy}, using original query", level="warning")
            state.sub_queries = [query]
        
        # Update metadata
        state.metadata["decomposition"] = {
            "original_query": query,
            "strategy": strategy.value,
            "sub_query_count": len(state.sub_queries),
            "sub_queries": state.sub_queries
        }
        
        self.log(f"Generated {len(state.sub_queries)} sub-queries", level="info")
        
        return state
    
    # ========================================
    # STRATEGY-SPECIFIC METHODS
    # ========================================
    
    def _decompose_multihop(self, query: str) -> List[str]:
        """
        Decompose for MULTIHOP strategy (sequential steps).
        
        Example:
        Query: "What is the capital of the country where Eiffel Tower is?"
        Sub-queries:
        1. "Where is the Eiffel Tower located?"
        2. "What is the capital of France?"
        """
        prompt = f"""You are decomposing a multi-hop question into sequential sub-questions.

Original Query: {query}

Instructions:
1. Identify the reasoning chain needed
2. Create 2-4 sub-questions that build on each other:
   - First question gets initial information
   - Each next question uses previous answer
   - Questions are in logical order
3. Keep questions focused and specific

Respond with ONLY a numbered list of sub-questions.

Sub-questions:"""
        
        try:
            response = self.llm.invoke(prompt)
            sub_queries = self._parse_sub_queries(response.content)
            return sub_queries
        except Exception as e:
            self.log(f"Multihop decomposition failed: {e}", level="error")
            return [query]
    
    def _decompose_graph(self, query: str) -> List[str]:
        """
        Decompose for GRAPH strategy (entities + relationships).
        
        Example:
        Query: "Compare Python and Java"
        Sub-queries:
        1. "Python programming language characteristics"
        2. "Java programming language characteristics"
        3. "Python vs Java comparison"
        """
        prompt = f"""You are decomposing a comparison/relationship question into entity-focused sub-questions.

Original Query: {query}

Instructions:
1. Identify the main entities being discussed
2. Create sub-questions:
   - One question per entity (to gather information)
   - One question for pairwise comparisons/relationships
3. Total 3-6 sub-questions

Respond with ONLY a numbered list of sub-questions.

Sub-questions:"""
        
        try:
            response = self.llm.invoke(prompt)
            sub_queries = self._parse_sub_queries(response.content)
            return sub_queries
        except Exception as e:
            self.log(f"Graph decomposition failed: {e}", level="error")
            return [query]
    
    def _parse_sub_queries(self, text: str) -> List[str]:
        """Parse LLM response into list of sub-queries."""
        import re
        
        lines = text.strip().split('\n')
        sub_queries = []
        
        for line in lines:
            # Match "1. Question" or "1) Question"
            match = re.match(r'^\s*\d+[\.\)]\s*(.+)$', line)
            if match:
                sub_queries.append(match.group(1).strip())
        
        # Fallback
        if not sub_queries:
            sub_queries = [l.strip() for l in lines if l.strip()]
        
        return sub_queries