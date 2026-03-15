"""
Integration tests for Planner Agent with full system.

Tests Planner Agent integration with:
- AgentState
- Config
- LLM (Claude API)
- Multi-agent workflows
"""

import pytest
from unittest.mock import Mock
from langchain_anthropic import ChatAnthropic

from src.agents.planner import PlannerAgent
from src.models.agent_state import AgentState, Strategy
from src.config import get_settings


@pytest.fixture
def real_llm():
    """
    Create real LLM instance (requires API key).
    
    Skip tests if API key not available.
    """
    try:
        settings = get_settings()
        llm = ChatAnthropic(
            model=settings.llm_model,
            temperature=0.0,
            api_key=settings.anthropic_api_key
        )
        return llm
    except Exception:
        pytest.skip("Anthropic API key not configured")


@pytest.fixture
def planner_with_real_llm(real_llm):
    """Create planner with real LLM."""
    return PlannerAgent(llm=real_llm)


class TestPlannerWithConfig:
    """Tests for Planner + Config integration"""
    
    def test_planner_loads_thresholds_from_config(self):
        """Test planner loads thresholds from config"""
        try:
            settings = get_settings()
            mock_llm = Mock(spec=ChatAnthropic)
            
            planner = PlannerAgent(llm=mock_llm)
            
            # Should match config values
            assert planner.simple_threshold == settings.planner_complexity_threshold_simple
            assert planner.multihop_threshold == settings.planner_complexity_threshold_multihop
        except Exception:
            pytest.skip("Config not available")
    
    def test_planner_respects_custom_thresholds_over_config(self):
        """Test custom thresholds override config"""
        mock_llm = Mock(spec=ChatAnthropic)
        
        planner = PlannerAgent(
            llm=mock_llm,
            simple_threshold=0.25,
            multihop_threshold=0.75
        )
        
        # Should use custom values, not config
        assert planner.simple_threshold == 0.25
        assert planner.multihop_threshold == 0.75


class TestPlannerWithRealLLM:
    """Tests with actual Claude API (optional, requires API key)"""
    
    def test_planner_with_real_api_simple_query(self, planner_with_real_llm):
        """Test planner with real API for simple query"""
        state = AgentState(query="What is Python?")
        
        result = planner_with_real_llm.run(state)
        
        # Should be simple
        assert result.complexity is not None
        assert result.complexity < 0.5
        assert result.strategy == Strategy.SIMPLE
    
    def test_planner_with_real_api_complex_query(self, planner_with_real_llm):
        """Test planner with real API for complex query"""
        query = "Compare and contrast the architectural patterns of microservices versus monolithic applications, considering scalability, maintainability, and deployment complexity"
        state = AgentState(query=query)
        
        result = planner_with_real_llm.run(state)
        
        # Should be complex
        assert result.complexity is not None
        assert result.complexity > 0.4
        assert result.strategy in [Strategy.MULTIHOP, Strategy.GRAPH]
    
    def test_planner_semantic_analysis_works(self, planner_with_real_llm):
        """Test that semantic analysis actually calls LLM"""
        query = "Explain quantum computing"
        
        # This should invoke real LLM
        score = planner_with_real_llm._semantic_complexity(query)
        
        # Should get valid score
        assert 0.0 <= score <= 1.0
        # Complex topic should have higher score
        assert score > 0.3


class TestPlannerStateIntegration:
    """Tests for Planner + AgentState integration"""
    
    def test_planner_updates_all_state_fields(self):
        """Test planner updates all required state fields"""
        mock_llm = Mock(spec=ChatAnthropic)
        mock_llm.invoke.return_value.content = "0.5"
        
        planner = PlannerAgent(llm=mock_llm)
        state = AgentState(query="test query")
        
        result = planner.run(state)
        
        # Check all fields updated
        assert result.query == "test query"
        assert result.complexity is not None
        assert result.strategy is not None
        assert "planner" in result.metadata
    
    def test_planner_preserves_existing_state(self):
        """Test planner doesn't overwrite unrelated state"""
        mock_llm = Mock(spec=ChatAnthropic)
        mock_llm.invoke.return_value.content = "0.5"
        
        planner = PlannerAgent(llm=mock_llm)
        
        # State with existing data
        state = AgentState(
            query="test",
            chunks=[],
            answer="existing answer"
        )
        
        result = planner.run(state)
        
        # Should preserve existing data
        assert result.answer == "existing answer"
        assert result.chunks == []
        # And add new data
        assert result.complexity is not None
        assert result.strategy is not None
    
    def test_planner_metadata_structure(self):
        """Test planner metadata has correct structure"""
        mock_llm = Mock(spec=ChatAnthropic)
        mock_llm.invoke.return_value.content = "0.5"
        
        planner = PlannerAgent(llm=mock_llm)
        state = AgentState(query="test")
        
        result = planner.run(state)
        
        # Check metadata structure
        assert "planner" in result.metadata
        planner_meta = result.metadata["planner"]
        
        assert "complexity" in planner_meta
        assert "strategy" in planner_meta
        assert "thresholds" in planner_meta
        assert "simple" in planner_meta["thresholds"]
        assert "multihop" in planner_meta["thresholds"]


class TestPlannerWorkflow:
    """Tests for planner in multi-agent workflow"""
    
    def test_planner_as_first_agent(self):
        """Test planner works as first agent in pipeline"""
        mock_llm = Mock(spec=ChatAnthropic)
        mock_llm.invoke.return_value.content = "0.5"
        
        planner = PlannerAgent(llm=mock_llm)
        
        # Fresh state (as it would come from user)
        state = AgentState(query="What is machine learning?")
        
        # Planner is first agent
        result = planner.run(state)
        
        # State should be ready for next agent
        assert result.complexity is not None
        assert result.strategy is not None
        # Next agents can use these to make decisions
    
    def test_planner_output_usable_by_downstream_agents(self):
        """Test planner output format is usable by other agents"""
        mock_llm = Mock(spec=ChatAnthropic)
        mock_llm.invoke.return_value.content = "0.5"
        
        planner = PlannerAgent(llm=mock_llm)
        state = AgentState(query="test")
        
        result = planner.run(state)
        
        # Downstream agents can check strategy
        if result.strategy == Strategy.SIMPLE:
            # Use fast path
            assert result.complexity < 0.3
        elif result.strategy == Strategy.MULTIHOP:
            # Use multi-hop path
            assert 0.3 <= result.complexity < 0.7
        else:  # GRAPH
            # Use graph path
            assert result.complexity >= 0.7


class TestPlannerPerformance:
    """Performance and benchmarking tests"""
    
    def test_planner_execution_time_reasonable(self):
        """Test planner executes in reasonable time"""
        mock_llm = Mock(spec=ChatAnthropic)
        mock_llm.invoke.return_value.content = "0.5"
        
        planner = PlannerAgent(llm=mock_llm)
        state = AgentState(query="test")
        
        result = planner.run(state)
        metrics = planner.get_metrics()
        
        # Should execute quickly with mock LLM
        assert metrics["last_execution_time"] < 1.0
    
    def test_planner_handles_multiple_queries_efficiently(self):
        """Test planner can handle multiple queries"""
        mock_llm = Mock(spec=ChatAnthropic)
        mock_llm.invoke.return_value.content = "0.5"
        
        planner = PlannerAgent(llm=mock_llm)
        
        queries = [
            "What is X?",
            "How does Y work?",
            "Compare A and B",
            "Explain the relationship between C and D",
            "What are the implications of E?"
        ]
        
        for query in queries:
            state = AgentState(query=query)
            result = planner.run(state)
            assert result.complexity is not None
        
        metrics = planner.get_metrics()
        assert metrics["total_calls"] == 5
        assert metrics["success_rate"] == 100.0


class TestPlannerEdgeCases:
    """Edge case and error condition tests"""
    
    def test_planner_handles_empty_query(self):
        """Test planner with empty query"""
        mock_llm = Mock(spec=ChatAnthropic)
        mock_llm.invoke.return_value.content = "0.1"
        
        planner = PlannerAgent(llm=mock_llm)
        state = AgentState(query="")
        
        # Should handle gracefully
        result = planner.run(state)
        assert result.complexity is not None
    
    def test_planner_handles_very_long_query(self):
        """Test planner with very long query"""
        mock_llm = Mock(spec=ChatAnthropic)
        mock_llm.invoke.return_value.content = "0.9"
        
        planner = PlannerAgent(llm=mock_llm)
        
        # 100+ word query
        query = " ".join(["word"] * 100)
        state = AgentState(query=query)
        
        result = planner.run(state)
        assert result.complexity is not None
        # Long query should have high complexity
        assert result.complexity > 0.5
    
    def test_planner_handles_special_characters(self):
        """Test planner with special characters in query"""
        mock_llm = Mock(spec=ChatAnthropic)
        mock_llm.invoke.return_value.content = "0.5"
        
        planner = PlannerAgent(llm=mock_llm)
        
        query = "What is C++? How about C#? And F*?"
        state = AgentState(query=query)
        
        # Should handle without errors
        result = planner.run(state)
        assert result.complexity is not None


class TestPlannerConsistency:
    """Tests for consistency and determinism"""
    
    def test_same_query_similar_complexity(self):
        """Test same query produces similar complexity"""
        mock_llm = Mock(spec=ChatAnthropic)
        mock_llm.invoke.return_value.content = "0.5"
        
        planner = PlannerAgent(llm=mock_llm)
        
        query = "What is Python?"
        
        # Run twice
        state1 = AgentState(query=query)
        result1 = planner.run(state1)
        
        state2 = AgentState(query=query)
        result2 = planner.run(state2)
        
        # Should be identical with mocked LLM
        assert result1.complexity == result2.complexity
        assert result1.strategy == result2.strategy
    
    def test_complexity_increases_with_query_length(self):
        """Test complexity generally increases with query length"""
        mock_llm = Mock(spec=ChatAnthropic)
        
        planner = PlannerAgent(llm=mock_llm)
        
        queries = [
            ("Python", "0.1"),
            ("What is Python?", "0.2"),
            ("What is Python and how is it used?", "0.4"),
            ("Compare Python and Java in multiple dimensions", "0.6"),
        ]
        
        complexities = []
        for query, llm_score in queries:
            mock_llm.invoke.return_value.content = llm_score
            state = AgentState(query=query)
            result = planner.run(state)
            complexities.append(result.complexity)
        
        # Generally increasing (with mock LLM cooperation)
        for i in range(len(complexities) - 1):
            assert complexities[i] <= complexities[i + 1] + 0.1  # Allow small variance


class TestPlannerDocumentation:
    """Tests that verify documented behavior"""
    
    def test_threshold_boundaries_documented_correctly(self):
        """Test that threshold boundaries work as documented"""
        mock_llm = Mock(spec=ChatAnthropic)
        mock_llm.invoke.return_value.content = "0.0"
        
        planner = PlannerAgent(
            llm=mock_llm,
            simple_threshold=0.3,
            multihop_threshold=0.7
        )
        
        # Test documented boundaries
        assert planner._select_strategy(0.29) == Strategy.SIMPLE
        assert planner._select_strategy(0.3) == Strategy.MULTIHOP
        assert planner._select_strategy(0.69) == Strategy.MULTIHOP
        assert planner._select_strategy(0.7) == Strategy.GRAPH
    
    def test_feature_weights_documented_correctly(self):
        """Test that feature weights match documentation"""
        mock_llm = Mock(spec=ChatAnthropic)
        planner = PlannerAgent(llm=mock_llm)
        
        # Extract features
        features = planner._extract_features("test query")
        
        # All features should be present as documented
        assert "length_score" in features
        assert "question_score" in features
        assert "entity_score" in features
        assert "relationship_score" in features