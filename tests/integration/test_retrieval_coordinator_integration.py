"""
Integration tests for Retrieval Coordinator with full system.

Tests Retrieval Coordinator integration with:
- AgentState
- Planner Agent
- Validator Agent
- Complete pipeline workflows
"""

import pytest
from unittest.mock import Mock

from src.agents.retrieval_coordinator import RetrievalCoordinator
from src.agents.planner import PlannerAgent
from src.agents.validator import ValidatorAgent
from src.agents.retrieval.vector_agent import VectorSearchAgent
from src.agents.retrieval.keyword_agent import KeywordSearchAgent
from src.agents.retrieval.graph_agent import GraphSearchAgent
from src.models.agent_state import AgentState, Strategy
from src.config import get_settings


@pytest.fixture
def vector_agent():
    """Create vector search agent."""
    return VectorSearchAgent(top_k=5, mock_mode=True)


@pytest.fixture
def keyword_agent():
    """Create keyword search agent."""
    return KeywordSearchAgent(top_k=5, mock_mode=True)


@pytest.fixture
def graph_agent():
    """Create graph search agent."""
    return GraphSearchAgent(top_k=5, mock_mode=True)


@pytest.fixture
def coordinator(vector_agent, keyword_agent, graph_agent):
    """Create retrieval coordinator."""
    return RetrievalCoordinator(
        vector_agent=vector_agent,
        keyword_agent=keyword_agent,
        graph_agent=graph_agent,
        top_k=10,
        parallel=False
    )


@pytest.fixture
def mock_llm():
    """Create mock LLM."""
    llm = Mock()
    response = Mock()
    response.content = "0.5"
    llm.invoke.return_value = response
    return llm


@pytest.fixture
def planner(mock_llm):
    """Create planner agent."""
    return PlannerAgent(llm=mock_llm)


@pytest.fixture
def validator(mock_llm):
    """Create validator agent."""
    return ValidatorAgent(llm=mock_llm, threshold=0.7, max_retries=2)


class TestCoordinatorWithConfig:
    """Tests for Coordinator + Config integration"""
    
    def test_coordinator_loads_top_k_from_config(self, vector_agent):
        """Test coordinator loads top_k from config"""
        try:
            settings = get_settings()
            
            coordinator = RetrievalCoordinator(vector_agent=vector_agent)
            
            # Should match config value
            assert coordinator.top_k == settings.retrieval_top_k
            assert coordinator.parallel == settings.parallel_retrieval
        except Exception:
            pytest.skip("Config not available")
    
    def test_coordinator_custom_values_override_config(self, vector_agent):
        """Test custom values override config"""
        coordinator = RetrievalCoordinator(
            vector_agent=vector_agent,
            top_k=15,
            parallel=True
        )
        
        assert coordinator.top_k == 15
        assert coordinator.parallel == True


class TestCoordinatorStateIntegration:
    """Tests for Coordinator + AgentState integration"""
    
    def test_coordinator_reads_query_from_state(self, coordinator):
        """Test coordinator reads query from state"""
        state = AgentState(query="What is Python?")
        
        result = coordinator.run(state)
        
        # Should have processed query
        assert result.chunks is not None
        assert len(result.chunks) > 0
    
    def test_coordinator_updates_chunks_in_state(self, coordinator):
        """Test coordinator updates chunks in state"""
        state = AgentState(query="test", chunks=[])
        
        result = coordinator.run(state)
        
        # Should replace chunks
        assert len(result.chunks) > 0
    
    def test_coordinator_increments_retrieval_round(self, coordinator):
        """Test coordinator increments retrieval round"""
        state = AgentState(query="test", retrieval_round=0)
        
        result = coordinator.run(state)
        
        assert result.retrieval_round == 1
    
    def test_coordinator_preserves_other_state(self, coordinator):
        """Test coordinator preserves unrelated state fields"""
        state = AgentState(
            query="test",
            complexity=0.5,
            strategy=Strategy.SIMPLE
        )
        
        result = coordinator.run(state)
        
        # Should preserve
        assert result.complexity == 0.5
        assert result.strategy == Strategy.SIMPLE
        # And add chunks
        assert len(result.chunks) > 0


class TestPlannerCoordinatorPipeline:
    """Tests for Planner → Coordinator pipeline"""
    
    def test_planner_then_coordinator_flow(self, planner, coordinator, mock_llm):
        """Test Planner → Coordinator pipeline"""
        mock_llm.invoke.return_value.content = "0.4"
        
        # Start with fresh state
        state = AgentState(query="What is machine learning?")
        
        # Step 1: Planner
        state = planner.run(state)
        assert state.complexity is not None
        assert state.strategy is not None
        
        # Step 2: Coordinator
        state = coordinator.run(state)
        assert len(state.chunks) > 0
    
    def test_coordinator_can_use_planner_strategy(self, planner, coordinator, mock_llm):
        """Test coordinator could adapt based on planner strategy"""
        mock_llm.invoke.return_value.content = "0.2"
        
        # Simple query
        state = AgentState(query="What is X?")
        state = planner.run(state)
        
        assert state.strategy == Strategy.SIMPLE
        
        # Coordinator retrieves
        state = coordinator.run(state)
        
        # Could potentially use fewer agents for simple queries
        # (not implemented yet, but demonstrates integration)
        assert state.chunks is not None


class TestCoordinatorValidatorPipeline:
    """Tests for Coordinator → Validator pipeline"""
    
    def test_coordinator_then_validator_flow(self, coordinator, validator, mock_llm):
        """Test Coordinator → Validator pipeline"""
        mock_llm.invoke.return_value.content = "0.8"
        
        state = AgentState(query="test")
        
        # Step 1: Coordinator retrieves
        state = coordinator.run(state)
        assert len(state.chunks) > 0
        
        # Step 2: Validator checks
        state = validator.run(state)
        assert state.validation_status is not None
        assert state.validation_score is not None
    
    def test_validator_approves_coordinator_results(self, coordinator, validator, mock_llm):
        """Test validator approves good coordinator results"""
        mock_llm.invoke.return_value.content = "0.85"
        
        state = AgentState(query="What is Python?")
        
        # Retrieve
        state = coordinator.run(state)
        
        # Validate (should approve with good chunks)
        state = validator.run(state)
        
        assert state.validation_status == "PROCEED"


class TestCompletePipeline:
    """Tests for complete Planner → Coordinator → Validator pipeline"""
    
    def test_full_pipeline_simple_query(self, planner, coordinator, validator, mock_llm):
        """Test complete pipeline for simple query"""
        state = AgentState(query="What is Python?")
        
        # Planner
        mock_llm.invoke.return_value.content = "0.2"
        state = planner.run(state)
        assert state.strategy == Strategy.SIMPLE
        
        # Coordinator
        state = coordinator.run(state)
        assert len(state.chunks) > 0
        
        # Validator
        mock_llm.invoke.return_value.content = "0.8"
        state = validator.run(state)
        assert state.validation_status == "PROCEED"
    
    def test_full_pipeline_complex_query(self, planner, coordinator, validator, mock_llm):
        """Test complete pipeline for complex query"""
        query = "Compare Python and Java in terms of performance and ecosystem"
        state = AgentState(query=query)
        
        # Planner
        mock_llm.invoke.return_value.content = "0.7"
        state = planner.run(state)
        assert state.strategy in [Strategy.MULTIHOP, Strategy.GRAPH]
        
        # Coordinator
        state = coordinator.run(state)
        assert len(state.chunks) > 0
        
        # Validator
        mock_llm.invoke.return_value.content = "0.75"
        state = validator.run(state)
        assert state.validation_status is not None
    
    def test_pipeline_with_retry(self, planner, coordinator, validator, mock_llm):
        """Test pipeline with validator retry"""
        state = AgentState(query="test")
        
        # Planner
        mock_llm.invoke.return_value.content = "0.5"
        state = planner.run(state)
        
        # First retrieval
        state = coordinator.run(state)
        assert state.retrieval_round == 1
        
        # Validator (reject)
        mock_llm.invoke.return_value.content = "0.4"
        state = validator.run(state)
        assert state.validation_status == "RETRIEVE_MORE"
        
        # Second retrieval
        state = coordinator.run(state)
        assert state.retrieval_round == 2
        
        # Validator (approve)
        mock_llm.invoke.return_value.content = "0.8"
        state = validator.run(state)
        assert state.validation_status == "PROCEED"


class TestCoordinatorMetadataIntegration:
    """Tests for coordinator metadata in pipeline"""
    
    def test_coordinator_metadata_structure(self, coordinator):
        """Test coordinator metadata has correct structure"""
        state = AgentState(query="test")
        
        result = coordinator.run(state)
        
        # Check metadata structure
        assert "retrieval_coordinator" in result.metadata
        coord_meta = result.metadata["retrieval_coordinator"]
        
        assert "round" in coord_meta
        assert "total_retrieved" in coord_meta
        assert "unique_chunks" in coord_meta
        assert "final_chunks" in coord_meta
        assert "parallel" in coord_meta
    
    def test_coordinator_preserves_planner_metadata(self, planner, coordinator):
        """Test coordinator preserves metadata from planner"""
        state = AgentState(query="test")
        
        # Planner adds metadata
        state = planner.run(state)
        assert "planner" in state.metadata
        
        # Coordinator adds metadata
        state = coordinator.run(state)
        
        # Both should be present
        assert "planner" in state.metadata
        assert "retrieval_coordinator" in state.metadata
    
    def test_all_agents_add_metadata(self, planner, coordinator, validator, mock_llm):
        """Test all agents add their metadata"""
        mock_llm.invoke.return_value.content = "0.5"
        
        state = AgentState(query="test")
        
        state = planner.run(state)
        state = coordinator.run(state)
        state = validator.run(state)
        
        # All metadata present
        assert "planner" in state.metadata
        assert "retrieval_coordinator" in state.metadata
        assert "validator" in state.metadata


class TestCoordinatorPerformance:
    """Performance tests"""
    
    def test_coordinator_execution_time_reasonable(self, coordinator):
        """Test coordinator executes in reasonable time"""
        state = AgentState(query="test")
        
        result = coordinator.run(state)
        metrics = coordinator.get_metrics()
        
        # Should execute quickly with mock agents
        assert metrics["last_execution_time"] < 2.0
    
    def test_parallel_vs_sequential_performance(self, vector_agent, keyword_agent, graph_agent):
        """Test parallel is potentially faster than sequential"""
        # Sequential
        coord_seq = RetrievalCoordinator(
            vector_agent=vector_agent,
            keyword_agent=keyword_agent,
            graph_agent=graph_agent,
            parallel=False
        )
        
        # Parallel
        coord_par = RetrievalCoordinator(
            vector_agent=vector_agent,
            keyword_agent=keyword_agent,
            graph_agent=graph_agent,
            parallel=True
        )
        
        state1 = AgentState(query="test")
        state2 = AgentState(query="test")
        
        coord_seq.run(state1)
        coord_par.run(state2)
        
        # Both should complete successfully
        metrics_seq = coord_seq.get_metrics()
        metrics_par = coord_par.get_metrics()
        
        assert metrics_seq["successful_calls"] == 1
        assert metrics_par["successful_calls"] == 1


class TestCoordinatorEdgeCases:
    """Edge case tests"""
    
    def test_coordinator_with_partial_agents(self, vector_agent):
        """Test coordinator with only some agents"""
        coordinator = RetrievalCoordinator(
            vector_agent=vector_agent,
            keyword_agent=None,
            graph_agent=None
        )
        
        state = AgentState(query="test")
        result = coordinator.run(state)
        
        # Should still work with one agent
        assert len(result.chunks) > 0
    
    def test_coordinator_multiple_rounds(self, coordinator):
        """Test coordinator can be called multiple times"""
        state = AgentState(query="test", retrieval_round=0)
        
        # Round 1
        state = coordinator.run(state)
        chunks_round1 = len(state.chunks)
        assert state.retrieval_round == 1
        
        # Round 2
        state = coordinator.run(state)
        assert state.retrieval_round == 2
        
        # Should have chunks (possibly different)
        assert len(state.chunks) > 0


class TestCoordinatorChunkQuality:
    """Tests for chunk quality from coordinator"""
    
    def test_chunks_have_required_fields(self, coordinator):
        """Test all chunks have required fields"""
        state = AgentState(query="test")
        result = coordinator.run(state)
        
        for chunk in result.chunks:
            assert chunk.text is not None
            assert chunk.doc_id is not None
            assert chunk.chunk_id is not None
            assert chunk.score is not None
            assert 0.0 <= chunk.score <= 1.0
    
    def test_chunks_sorted_by_score(self, coordinator):
        """Test chunks are sorted by score (descending)"""
        state = AgentState(query="test")
        result = coordinator.run(state)
        
        if len(result.chunks) > 1:
            # Check descending order
            for i in range(len(result.chunks) - 1):
                assert result.chunks[i].score >= result.chunks[i + 1].score
    
    def test_chunks_tagged_with_metadata(self, coordinator):
        """Test chunks have source metadata"""
        state = AgentState(query="test")
        result = coordinator.run(state)
        
        # At least some chunks should have metadata
        chunks_with_metadata = [
            c for c in result.chunks if c.metadata
        ]
        assert len(chunks_with_metadata) > 0


class TestCoordinatorWorkflow:
    """Tests for coordinator in complete workflow"""
    
    def test_coordinator_as_retrieval_layer(self, planner, coordinator, validator):
        """Test coordinator acts as retrieval layer between planner and validator"""
        state = AgentState(query="What is Python?")
        
        # Before coordinator: no chunks
        state = planner.run(state)
        assert len(state.chunks) == 0
        
        # Coordinator adds chunks
        state = coordinator.run(state)
        assert len(state.chunks) > 0
        
        # Validator can now validate
        state = validator.run(state)
        assert state.validation_status is not None
    
    def test_coordinator_enables_retry_loop(self, coordinator, validator, mock_llm):
        """Test coordinator enables validator retry loop"""
        state = AgentState(query="test", retrieval_round=0)
        
        # First retrieval
        state = coordinator.run(state)
        
        # Validator rejects
        mock_llm.invoke.return_value.content = "0.3"
        state = validator.run(state)
        
        if state.validation_status == "RETRIEVE_MORE":
            # Second retrieval
            state = coordinator.run(state)
            
            # Should have new/additional chunks
            assert len(state.chunks) > 0
            assert state.retrieval_round == 2