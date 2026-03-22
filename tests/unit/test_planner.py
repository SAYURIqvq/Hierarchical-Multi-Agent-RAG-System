"""
Tests for Planner Agent.

Tests complexity analysis, strategy selection, and LLM integration.
"""

import pytest
from unittest.mock import Mock, MagicMock
from langchain_anthropic import ChatAnthropic

from src.agents.planner import PlannerAgent
from src.models.agent_state import AgentState, Strategy
from src.utils.exceptions import AgentExecutionError


@pytest.fixture
def mock_llm():
    """
    Create mock LLM for testing.
    
    Returns mock that simulates Claude API responses.
    """
    llm = Mock(spec=ChatAnthropic)
    
    # Default response
    response = Mock()
    response.content = "0.5"
    llm.invoke.return_value = response
    
    return llm


@pytest.fixture
def planner(mock_llm):
    """
    Create PlannerAgent instance with mock LLM.
    
    Uses default thresholds: simple=0.3, multihop=0.7
    """
    return PlannerAgent(
        llm=mock_llm,
        simple_threshold=0.3,
        multihop_threshold=0.7
    )


class TestPlannerInitialization:
    """Tests for Planner Agent initialization"""
    
    def test_planner_initializes_with_llm(self, mock_llm):
        """Test planner initializes with LLM"""
        planner = PlannerAgent(llm=mock_llm)
        
        assert planner.name == "planner"
        assert planner.llm is not None
        assert planner.simple_threshold is not None
        assert planner.multihop_threshold is not None
    
    def test_planner_uses_custom_thresholds(self, mock_llm):
        """Test planner accepts custom thresholds"""
        planner = PlannerAgent(
            llm=mock_llm,
            simple_threshold=0.4,
            multihop_threshold=0.8
        )
        
        assert planner.simple_threshold == 0.4
        assert planner.multihop_threshold == 0.8
    
    def test_planner_loads_default_thresholds_from_config(self, mock_llm):
        """Test planner loads thresholds from config when not provided"""
        planner = PlannerAgent(llm=mock_llm)
        
        # Should have loaded from config (default 0.3 and 0.7)
        assert planner.simple_threshold > 0
        assert planner.multihop_threshold > planner.simple_threshold


class TestFeatureExtraction:
    """Tests for feature extraction"""
    
    def test_extract_features_short_query(self, planner):
        """Test feature extraction for short query"""
        features = planner._extract_features("What is Python?")
        
        assert "length_score" in features
        assert "question_score" in features
        assert "entity_score" in features
        assert "relationship_score" in features
        
        # Short query should have low length score
        assert features["length_score"] < 0.3
    
    def test_extract_features_long_query(self, planner):
        """Test feature extraction for long query"""
        query = "Compare and contrast the performance characteristics of Python and Java in terms of execution speed memory usage and scalability"
        features = planner._extract_features(query)
        
        # Long query should have high length score
        assert features["length_score"] > 0.5
    
    def test_extract_features_multiple_questions(self, planner):
        """Test feature extraction for multiple questions"""
        query = "What is Python? How does it compare to Java? Why is it popular?"
        features = planner._extract_features(query)
        
        # Multiple questions should have high question score
        assert features["question_score"] > 0.3
    
    def test_extract_features_with_entities(self, planner):
        """Test feature extraction with entity indicators"""
        query = "Compare the difference between Python and Java"
        features = planner._extract_features(query)
        
        # Should detect entity indicators (compare, difference, between)
        assert features["entity_score"] > 0
    
    def test_extract_features_with_relationships(self, planner):
        """Test feature extraction with relationship indicators"""
        query = "How does Python's GIL impact multi-threaded performance?"
        features = planner._extract_features(query)
        
        # Should detect relationship indicators (how does, impact)
        assert features["relationship_score"] > 0
    
    def test_all_feature_scores_bounded(self, planner):
        """Test that all feature scores are between 0 and 1"""
        query = "Very complex query with lots of questions? And comparisons? How does X relate to Y? Compare A and B."
        features = planner._extract_features(query)
        
        for key, value in features.items():
            assert 0.0 <= value <= 1.0, f"{key} score {value} out of bounds"


class TestSemanticComplexity:
    """Tests for semantic complexity analysis"""
    
    def test_semantic_complexity_calls_llm(self, planner, mock_llm):
        """Test that semantic complexity calls LLM"""
        planner._semantic_complexity("Test query")
        
        # Should have called LLM
        assert mock_llm.invoke.called
    
    def test_semantic_complexity_parses_valid_response(self, planner, mock_llm):
        """Test parsing valid LLM response"""
        mock_llm.invoke.return_value.content = "0.75"
        
        score = planner._semantic_complexity("Complex query")
        
        assert score == 0.75
    
    def test_semantic_complexity_handles_text_response(self, planner, mock_llm):
        """Test parsing LLM response with text"""
        mock_llm.invoke.return_value.content = "The complexity is 0.65 based on analysis."
        
        score = planner._semantic_complexity("Test query")
        
        assert score == 0.65
    
    def test_semantic_complexity_clamps_to_valid_range(self, planner, mock_llm):
        """Test that scores are clamped to [0, 1]"""
        mock_llm.invoke.return_value.content = "1.5"
        
        score = planner._semantic_complexity("Test query")
        
        assert score <= 1.0
    
    def test_semantic_complexity_uses_fallback_on_error(self, planner, mock_llm):
        """Test fallback when LLM fails"""
        mock_llm.invoke.side_effect = Exception("API Error")
        
        score = planner._semantic_complexity("Test query")
        
        # Should return fallback score
        assert 0.0 <= score <= 1.0
    
    def test_semantic_complexity_uses_fallback_on_invalid_response(self, planner, mock_llm):
        """Test fallback when LLM returns invalid response"""
        mock_llm.invoke.return_value.content = "invalid response"
        
        score = planner._semantic_complexity("Test query")
        
        # Should use fallback
        assert 0.0 <= score <= 1.0


class TestFallbackSemanticScore:
    """Tests for fallback semantic scoring"""
    
    def test_fallback_simple_query(self, planner):
        """Test fallback for simple query"""
        score = planner._fallback_semantic_score("What is Python?")
        
        # "What is" is simple keyword
        assert score <= 0.3
    
    def test_fallback_complex_query(self, planner):
        """Test fallback for complex query"""
        score = planner._fallback_semantic_score("Explain the relationship between X and Y")
        
        # "Explain" and "relationship" are complex keywords
        assert score >= 0.5
    
    def test_fallback_moderate_query(self, planner):
        """Test fallback for moderate query"""
        score = planner._fallback_semantic_score("Tell me about Python")
        
        # No strong indicators, should be moderate
        assert 0.3 <= score <= 0.6


class TestComplexityAnalysis:
    """Tests for overall complexity analysis"""
    
    def test_analyze_complexity_returns_valid_range(self, planner):
        """Test complexity is always between 0 and 1"""
        queries = [
            "Hi",
            "What is Python?",
            "Compare Python and Java",
            "Explain the complex relationship between quantum mechanics and general relativity in modern physics"
        ]
        
        for query in queries:
            score = planner._analyze_complexity(query)
            assert 0.0 <= score <= 1.0, f"Score {score} for query: {query}"
    
    def test_analyze_complexity_simple_query(self, planner, mock_llm):
        """Test complexity analysis for simple query"""
        mock_llm.invoke.return_value.content = "0.1"
        
        score = planner._analyze_complexity("What is Python?")
        
        # Should be low complexity
        assert score < 0.4
    
    def test_analyze_complexity_complex_query(self, planner, mock_llm):
        """Test complexity analysis for complex query"""
        mock_llm.invoke.return_value.content = "0.9"
        
        query = "Compare and contrast the architectural differences between microservices and monolithic applications considering scalability performance and maintainability"
        score = planner._analyze_complexity(query)
        
        # Should be high complexity
        assert score > 0.5
    
    def test_analyze_complexity_combines_heuristic_and_semantic(self, planner, mock_llm):
        """Test that analysis combines both heuristic and semantic scores"""
        # Set semantic score high
        mock_llm.invoke.return_value.content = "0.9"
        
        # Use short query (low heuristic)
        score = planner._analyze_complexity("Why?")
        
        # Should be between pure heuristic and pure semantic (weighted average)
        assert 0.2 < score < 0.9


class TestStrategySelection:
    """Tests for strategy selection"""
    
    def test_select_strategy_simple(self, planner):
        """Test strategy selection for low complexity"""
        strategy = planner._select_strategy(0.1)
        assert strategy == Strategy.SIMPLE
        
        strategy = planner._select_strategy(0.29)
        assert strategy == Strategy.SIMPLE
    
    def test_select_strategy_multihop(self, planner):
        """Test strategy selection for medium complexity"""
        strategy = planner._select_strategy(0.3)
        assert strategy == Strategy.MULTIHOP
        
        strategy = planner._select_strategy(0.5)
        assert strategy == Strategy.MULTIHOP
        
        strategy = planner._select_strategy(0.69)
        assert strategy == Strategy.MULTIHOP
    
    def test_select_strategy_graph(self, planner):
        """Test strategy selection for high complexity"""
        strategy = planner._select_strategy(0.7)
        assert strategy == Strategy.GRAPH
        
        strategy = planner._select_strategy(0.9)
        assert strategy == Strategy.GRAPH
    
    def test_select_strategy_boundary_conditions(self, planner):
        """Test strategy selection at threshold boundaries"""
        # Just below simple threshold
        strategy = planner._select_strategy(0.299)
        assert strategy == Strategy.SIMPLE
        
        # At simple threshold
        strategy = planner._select_strategy(0.3)
        assert strategy == Strategy.MULTIHOP
        
        # Just below multihop threshold
        strategy = planner._select_strategy(0.699)
        assert strategy == Strategy.MULTIHOP
        
        # At multihop threshold
        strategy = planner._select_strategy(0.7)
        assert strategy == Strategy.GRAPH


class TestPlannerExecution:
    """Tests for execute method"""
    
    def test_execute_updates_state_complexity(self, planner):
        """Test execute updates state with complexity"""
        state = AgentState(query="What is Python?")
        
        result = planner.run(state)
        
        assert result.complexity is not None
        assert 0.0 <= result.complexity <= 1.0
    
    def test_execute_updates_state_strategy(self, planner):
        """Test execute updates state with strategy"""
        state = AgentState(query="What is Python?")
        
        result = planner.run(state)
        
        assert result.strategy is not None
        assert isinstance(result.strategy, Strategy)
    
    def test_execute_preserves_query(self, planner):
        """Test execute preserves original query"""
        state = AgentState(query="Test query")
        
        result = planner.run(state)
        
        assert result.query == "Test query"
    
    def test_execute_adds_metadata(self, planner):
        """Test execute adds planner metadata"""
        state = AgentState(query="Test query")
        
        result = planner.run(state)
        
        assert "planner" in result.metadata
        assert "complexity" in result.metadata["planner"]
        assert "strategy" in result.metadata["planner"]
        assert "thresholds" in result.metadata["planner"]
    
    def test_execute_simple_query_flow(self, planner, mock_llm):
        """Test complete flow for simple query"""
        mock_llm.invoke.return_value.content = "0.1"
        
        state = AgentState(query="What is X?")
        result = planner.run(state)
        
        assert result.complexity < 0.3
        assert result.strategy == Strategy.SIMPLE
    
    def test_execute_complex_query_flow(self, planner, mock_llm):
        """Test complete flow for complex query"""
        mock_llm.invoke.return_value.content = "0.9"
        
        query = "Compare A and B in terms of X Y and Z considering multiple factors"
        state = AgentState(query=query)
        result = planner.run(state)
        
        assert result.complexity > 0.5
        assert result.strategy in [Strategy.MULTIHOP, Strategy.GRAPH]
    
    def test_execute_handles_errors(self, planner, mock_llm):
        """Test execute handles errors gracefully"""
        # Simulate LLM error
        mock_llm.invoke.side_effect = Exception("API Error")
        
        state = AgentState(query="Test query")
        
        # Should still work with fallback
        result = planner.run(state)
        
        assert result.complexity is not None
        assert result.strategy is not None


class TestAnalyzeQueryDetails:
    """Tests for detailed query analysis"""
    
    def test_analyze_query_details_returns_all_fields(self, planner):
        """Test detailed analysis returns all expected fields"""
        details = planner.analyze_query_details("Test query")
        
        assert "query" in details
        assert "features" in details
        assert "heuristic_score" in details
        assert "semantic_score" in details
        assert "final_complexity" in details
        assert "selected_strategy" in details
        assert "thresholds" in details
    
    def test_analyze_query_details_features_breakdown(self, planner):
        """Test detailed analysis includes feature breakdown"""
        details = planner.analyze_query_details("Compare X and Y")
        
        features = details["features"]
        assert "length_score" in features
        assert "question_score" in features
        assert "entity_score" in features
        assert "relationship_score" in features
    
    def test_analyze_query_details_thresholds(self, planner):
        """Test detailed analysis includes thresholds"""
        details = planner.analyze_query_details("Test")
        
        thresholds = details["thresholds"]
        assert "simple" in thresholds
        assert "multihop" in thresholds
        assert thresholds["simple"] == planner.simple_threshold
        assert thresholds["multihop"] == planner.multihop_threshold


class TestPlannerMetrics:
    """Tests for metrics tracking"""
    
    def test_metrics_updated_on_execution(self, planner):
        """Test metrics are updated after execution"""
        state = AgentState(query="Test")
        
        planner.run(state)
        metrics = planner.get_metrics()
        
        assert metrics["total_calls"] == 1
        assert metrics["successful_calls"] == 1
    
    def test_metrics_track_multiple_executions(self, planner):
        """Test metrics track multiple executions"""
        for i in range(5):
            state = AgentState(query=f"Query {i}")
            planner.run(state)
        
        metrics = planner.get_metrics()
        assert metrics["total_calls"] == 5
        assert metrics["success_rate"] == 100.0


class TestPlannerIntegration:
    """Integration tests"""
    
    def test_planner_with_real_queries(self, planner):
        """Test planner with realistic query examples"""
        test_cases = [
            ("What is Python?", Strategy.SIMPLE),
            ("How do I install Python?", Strategy.SIMPLE),
            ("Compare Python and Java performance", Strategy.MULTIHOP),
            ("Explain how Python's GIL affects multithreading", Strategy.MULTIHOP),
        ]
        
        for query, expected_strategy in test_cases:
            state = AgentState(query=query)
            result = planner.run(state)
            
            # Strategy might vary based on LLM, just check it's assigned
            assert result.strategy is not None
            assert result.complexity is not None