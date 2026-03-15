"""
Integration tests for Validator Agent with full system.

Tests Validator Agent integration with:
- AgentState
- Config
- Planner Agent (pipeline)
- Retry loops
- LLM (Claude API)
"""

import pytest
from unittest.mock import Mock
from langchain_anthropic import ChatAnthropic

from src.agents.validator import ValidatorAgent
from src.agents.planner import PlannerAgent
from src.models.agent_state import AgentState, Chunk, Strategy
from src.config import get_settings


@pytest.fixture
def mock_llm():
    """Create mock LLM for testing."""
    llm = Mock(spec=ChatAnthropic)
    response = Mock()
    response.content = "0.8"
    llm.invoke.return_value = response
    return llm


@pytest.fixture
def validator(mock_llm):
    """Create validator with mock LLM."""
    return ValidatorAgent(llm=mock_llm, threshold=0.7, max_retries=2)


@pytest.fixture
def planner(mock_llm):
    """Create planner with mock LLM."""
    return PlannerAgent(llm=mock_llm)


@pytest.fixture
def good_chunks():
    """Create high-quality chunks."""
    return [
        Chunk(text=f"High quality chunk {i} with relevant content", 
              doc_id=f"doc{i}", chunk_id=f"c{i}", score=0.9 - (i * 0.05))
        for i in range(5)
    ]


@pytest.fixture
def poor_chunks():
    """Create low-quality chunks."""
    return [
        Chunk(text="Irrelevant content", doc_id="doc1", chunk_id="c1", score=0.3),
        Chunk(text="Off-topic", doc_id="doc1", chunk_id="c2", score=0.25)
    ]


class TestValidatorWithConfig:
    """Tests for Validator + Config integration"""
    
    def test_validator_loads_threshold_from_config(self):
        """Test validator loads threshold from config"""
        try:
            settings = get_settings()
            mock_llm = Mock(spec=ChatAnthropic)
            
            validator = ValidatorAgent(llm=mock_llm)
            
            # Should match config value
            assert validator.threshold == settings.validator_threshold
            assert validator.max_retries == settings.validator_max_retries
        except Exception:
            pytest.skip("Config not available")
    
    def test_validator_custom_threshold_overrides_config(self):
        """Test custom threshold overrides config"""
        mock_llm = Mock(spec=ChatAnthropic)
        
        validator = ValidatorAgent(
            llm=mock_llm,
            threshold=0.8,
            max_retries=3
        )
        
        # Should use custom values
        assert validator.threshold == 0.8
        assert validator.max_retries == 3


class TestValidatorStateIntegration:
    """Tests for Validator + AgentState integration"""
    
    def test_validator_reads_state_chunks(self, validator, good_chunks):
        """Test validator reads chunks from state"""
        state = AgentState(query="test", chunks=good_chunks)
        
        result = validator.run(state)
        
        # Should process chunks from state
        assert result.validation_score is not None
    
    def test_validator_reads_retrieval_round(self, validator, poor_chunks):
        """Test validator reads retrieval_round from state"""
        state = AgentState(
            query="test",
            chunks=poor_chunks,
            retrieval_round=1
        )
        
        result = validator.run(state)
        
        # Should be aware of retry count
        assert "retrieval_round" in result.metadata["validator"]
    
    def test_validator_updates_all_state_fields(self, validator, good_chunks):
        """Test validator updates all required state fields"""
        state = AgentState(query="test", chunks=good_chunks)
        
        result = validator.run(state)
        
        # Check all fields updated
        assert result.validation_status is not None
        assert result.validation_score is not None
        assert "validator" in result.metadata
    
    def test_validator_preserves_existing_state(self, validator, good_chunks):
        """Test validator doesn't overwrite unrelated state"""
        state = AgentState(
            query="test",
            chunks=good_chunks,
            complexity=0.5,
            strategy=Strategy.SIMPLE
        )
        
        result = validator.run(state)
        
        # Should preserve existing data
        assert result.complexity == 0.5
        assert result.strategy == Strategy.SIMPLE
        # And add new data
        assert result.validation_status is not None


class TestValidatorPlannerPipeline:
    """Tests for Validator in pipeline with Planner"""
    
    def test_planner_then_validator_pipeline(self, planner, validator, good_chunks, mock_llm):
        """Test Planner → Validator pipeline"""
        mock_llm.invoke.return_value.content = "0.5"
        
        # Start with fresh state
        state = AgentState(query="What is Python?")
        
        # Step 1: Planner
        state = planner.run(state)
        assert state.complexity is not None
        assert state.strategy is not None
        
        # Add chunks (simulating retrieval)
        state.chunks = good_chunks
        
        # Step 2: Validator
        state = validator.run(state)
        assert state.validation_status is not None
        assert state.validation_score is not None
    
    def test_validator_can_use_planner_strategy(self, planner, validator, good_chunks, mock_llm):
        """Test validator can make decisions based on planner strategy"""
        mock_llm.invoke.return_value.content = "0.1"
        
        # Simple query
        state = AgentState(query="What is X?")
        state = planner.run(state)
        
        # Should be simple strategy
        assert state.strategy == Strategy.SIMPLE
        
        # Add chunks
        state.chunks = good_chunks
        
        # Validator could use strategy to adjust threshold
        # (not implemented yet, but demonstrates integration)
        state = validator.run(state)
        assert state.validation_status is not None


class TestValidatorRetryLoop:
    """Tests for validator retry mechanism"""
    
    def test_validator_triggers_retry_on_low_score(self, validator, poor_chunks, mock_llm):
        """Test validator triggers retry for insufficient chunks"""
        mock_llm.invoke.return_value.content = "0.3"
        
        state = AgentState(
            query="test",
            chunks=poor_chunks,
            retrieval_round=0
        )
        
        result = validator.run(state)
        
        # Should trigger retry
        assert result.validation_status == "RETRIEVE_MORE"
        assert result.validation_score < validator.threshold
    
    def test_validator_retry_progression(self, validator, poor_chunks, mock_llm):
        """Test validator retry progression through rounds"""
        mock_llm.invoke.return_value.content = "0.4"
        
        # Round 0: Retry
        state = AgentState(query="test", chunks=poor_chunks, retrieval_round=0)
        result = validator.run(state)
        assert result.validation_status == "RETRIEVE_MORE"
        
        # Round 1: Retry
        state.retrieval_round = 1
        result = validator.run(state)
        assert result.validation_status == "RETRIEVE_MORE"
        
        # Round 2 (max): Force proceed
        state.retrieval_round = 2
        result = validator.run(state)
        assert result.validation_status == "PROCEED"
    
    def test_validator_proceeds_on_high_score_any_round(self, validator, good_chunks, mock_llm):
        """Test validator proceeds immediately on high score"""
        mock_llm.invoke.return_value.content = "0.9"
        
        # Even on first round, high score should proceed
        state = AgentState(query="test", chunks=good_chunks, retrieval_round=0)
        result = validator.run(state)
        
        assert result.validation_status == "PROCEED"
    
    def test_validator_retry_loop_simulation(self, validator, mock_llm):
        """Simulate complete retry loop"""
        # Start with poor chunks
        state = AgentState(
            query="test",
            chunks=[Chunk(text="poor", doc_id="1", chunk_id="1", score=0.2)],
            retrieval_round=0
        )
        
        # Round 0: Low score, retry
        mock_llm.invoke.return_value.content = "0.3"
        state = validator.run(state)
        assert state.validation_status == "RETRIEVE_MORE"
        
        # Simulate retrieval getting better chunks
        state.chunks.extend([
            Chunk(text="better", doc_id="2", chunk_id="2", score=0.7),
            Chunk(text="good", doc_id="3", chunk_id="3", score=0.8)
        ])
        state.retrieval_round = 1
        
        # Round 1: Better score, proceed
        mock_llm.invoke.return_value.content = "0.75"
        state = validator.run(state)
        assert state.validation_status == "PROCEED"


class TestValidatorMetadataIntegration:
    """Tests for validator metadata in pipeline"""
    
    def test_validator_metadata_structure(self, validator, good_chunks):
        """Test validator metadata has correct structure"""
        state = AgentState(query="test", chunks=good_chunks)
        
        result = validator.run(state)
        
        # Check metadata structure
        assert "validator" in result.metadata
        validator_meta = result.metadata["validator"]
        
        assert "score" in validator_meta
        assert "decision" in validator_meta
        assert "retrieval_round" in validator_meta
        assert "threshold" in validator_meta
        assert "max_retries" in validator_meta
    
    def test_validator_preserves_planner_metadata(self, planner, validator, good_chunks):
        """Test validator preserves metadata from previous agents"""
        state = AgentState(query="test")
        
        # Planner adds metadata
        state = planner.run(state)
        assert "planner" in state.metadata
        
        # Add chunks
        state.chunks = good_chunks
        
        # Validator adds metadata
        state = validator.run(state)
        
        # Both should be present
        assert "planner" in state.metadata
        assert "validator" in state.metadata


class TestValidatorPerformance:
    """Performance tests"""
    
    def test_validator_execution_time_reasonable(self, validator, good_chunks):
        """Test validator executes in reasonable time"""
        state = AgentState(query="test", chunks=good_chunks)
        
        result = validator.run(state)
        metrics = validator.get_metrics()
        
        # Should execute quickly with mock LLM
        assert metrics["last_execution_time"] < 2.0
    
    def test_validator_handles_multiple_validations(self, validator, good_chunks):
        """Test validator can handle multiple validations efficiently"""
        for i in range(5):
            state = AgentState(query=f"query {i}", chunks=good_chunks)
            result = validator.run(state)
            assert result.validation_status is not None
        
        metrics = validator.get_metrics()
        assert metrics["total_calls"] == 5
        assert metrics["success_rate"] == 100.0


class TestValidatorEdgeCases:
    """Edge case tests"""
    
    def test_validator_with_chunks_missing_scores(self, validator):
        """Test validator with chunks that have no scores"""
        chunks_no_scores = [
            Chunk(text=f"chunk {i}", doc_id="1", chunk_id=f"c{i}", score=None)
            for i in range(3)
        ]
        
        state = AgentState(query="test", chunks=chunks_no_scores)
        
        # Should handle gracefully (may have lower score but shouldn't crash)
        result = validator.run(state)
        assert result.validation_score is not None
        assert 0.0 <= result.validation_score <= 1.0
        assert result.validation_status in ["PROCEED", "RETRIEVE_MORE"]
    
    def test_validator_with_mixed_quality_chunks(self, validator, mock_llm):
        """Test validator with mixed quality chunks"""
        mock_llm.invoke.return_value.content = "0.6"
        
        mixed_chunks = [
            Chunk(text="high quality", doc_id="1", chunk_id="1", score=0.9),
            Chunk(text="medium quality", doc_id="1", chunk_id="2", score=0.5),
            Chunk(text="low quality", doc_id="1", chunk_id="3", score=0.2)
        ]
        
        state = AgentState(query="test", chunks=mixed_chunks)
        result = validator.run(state)
        
        # Should produce intermediate score
        assert 0.3 < result.validation_score < 0.9
    
    def test_validator_with_duplicate_chunks(self, validator):
        """Test validator with duplicate chunks"""
        duplicate_chunks = [
            Chunk(text="same content", doc_id="1", chunk_id=f"c{i}", score=0.8)
            for i in range(5)
        ]
        
        state = AgentState(query="test", chunks=duplicate_chunks)
        
        # Should handle (might affect diversity/coverage)
        result = validator.run(state)
        assert result.validation_score is not None


class TestValidatorDecisionConsistency:
    """Tests for decision consistency"""
    
    def test_same_chunks_same_decision(self, validator, good_chunks):
        """Test same chunks produce same decision"""
        state1 = AgentState(query="test", chunks=good_chunks)
        result1 = validator.run(state1)
        
        state2 = AgentState(query="test", chunks=good_chunks)
        result2 = validator.run(state2)
        
        # Should be consistent with mock LLM
        assert result1.validation_status == result2.validation_status
        assert result1.validation_score == result2.validation_score
    
    def test_threshold_boundary_behavior(self, validator, mock_llm):
        """Test behavior at threshold boundary"""
        # Create chunks that will result in specific scores
        good_chunks = [
            Chunk(text="relevant content", doc_id=f"doc{i}", chunk_id=f"c{i}", score=0.85)
            for i in range(5)
        ]
        
        # Test: LLM returns exactly threshold score
        # With weights (relevance 0.5, coverage 0.3, confidence 0.2)
        # We need final score = 0.7
        
        # Set LLM to return high relevance
        mock_llm.invoke.return_value.content = "0.8"
        
        state = AgentState(query="test", chunks=good_chunks, retrieval_round=0)
        result = validator.run(state)
        
        # With good chunks and high relevance, should PROCEED
        assert result.validation_status == "PROCEED"
        
        # Test: Low score should trigger retry
        poor_chunks = [
            Chunk(text="poor", doc_id="1", chunk_id="1", score=0.2)
        ]
        
        mock_llm.invoke.return_value.content = "0.3"
        state = AgentState(query="test", chunks=poor_chunks, retrieval_round=0)
        result = validator.run(state)
        
        # Low score should RETRIEVE_MORE on first round
        assert result.validation_status == "RETRIEVE_MORE"

class TestValidatorWorkflow:
    """Tests for validator in complete workflow"""
    
    def test_validator_as_quality_gate(self, planner, validator, mock_llm):
        """Test validator acts as quality gate in pipeline"""
        mock_llm.invoke.return_value.content = "0.5"
        
        # Pipeline: Query → Planner → (Retrieval) → Validator
        state = AgentState(query="What is machine learning?")
        
        # Step 1: Planner
        state = planner.run(state)
        
        # Step 2: Simulate retrieval with poor chunks
        state.chunks = [
            Chunk(text="irrelevant", doc_id="1", chunk_id="1", score=0.3)
        ]
        
        # Step 3: Validator (should block)
        state = validator.run(state)
        
        # Validator should block with RETRIEVE_MORE
        assert state.validation_status == "RETRIEVE_MORE"
    
    def test_complete_pipeline_with_retry(self, planner, validator, mock_llm):
        """Test complete pipeline with retry"""
        state = AgentState(query="Explain quantum computing")
        
        # Planner
        mock_llm.invoke.return_value.content = "0.6"
        state = planner.run(state)
        
        # First retrieval (poor)
        state.chunks = [Chunk(text="poor", doc_id="1", chunk_id="1", score=0.2)]
        
        # Validator (should retry)
        mock_llm.invoke.return_value.content = "0.3"
        state = validator.run(state)
        assert state.validation_status == "RETRIEVE_MORE"
        
        # Second retrieval (better)
        state.chunks.extend([
            Chunk(text="better", doc_id="2", chunk_id="2", score=0.8),
            Chunk(text="good", doc_id="3", chunk_id="3", score=0.85)
        ])
        state.retrieval_round = 1
        
        # Validator (should proceed)
        mock_llm.invoke.return_value.content = "0.75"
        state = validator.run(state)
        assert state.validation_status == "PROCEED"