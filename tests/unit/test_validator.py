"""
Tests for Validator Agent.

Tests chunk validation, sufficiency scoring, and decision logic.
"""

import pytest
from unittest.mock import Mock
from langchain_anthropic import ChatAnthropic

from src.agents.validator import ValidatorAgent
from src.models.agent_state import AgentState, Chunk
from src.utils.exceptions import ValidationError


@pytest.fixture
def mock_llm():
    """Create mock LLM for testing."""
    llm = Mock(spec=ChatAnthropic)
    
    # Default response for relevance check
    response = Mock()
    response.content = "0.8"
    llm.invoke.return_value = response
    
    return llm


@pytest.fixture
def validator(mock_llm):
    """Create ValidatorAgent instance with mock LLM."""
    return ValidatorAgent(
        llm=mock_llm,
        threshold=0.7,
        max_retries=2
    )


@pytest.fixture
def sample_chunks():
    """Create sample chunks for testing."""
    return [
        Chunk(
            text="Python is a high-level programming language.",
            doc_id="doc1",
            chunk_id="chunk1",
            score=0.9
        ),
        Chunk(
            text="Python was created by Guido van Rossum.",
            doc_id="doc1",
            chunk_id="chunk2",
            score=0.85
        ),
        Chunk(
            text="Python is widely used for web development.",
            doc_id="doc2",
            chunk_id="chunk3",
            score=0.8
        )
    ]


class TestValidatorInitialization:
    """Tests for Validator Agent initialization"""
    
    def test_validator_initializes_with_llm(self, mock_llm):
        """Test validator initializes with LLM"""
        validator = ValidatorAgent(llm=mock_llm)
        
        assert validator.name == "validator"
        assert validator.llm is not None
        assert validator.threshold is not None
        assert validator.max_retries is not None
    
    def test_validator_uses_custom_threshold(self, mock_llm):
        """Test validator accepts custom threshold"""
        validator = ValidatorAgent(
            llm=mock_llm,
            threshold=0.8,
            max_retries=3
        )
        
        assert validator.threshold == 0.8
        assert validator.max_retries == 3
    
    def test_validator_loads_defaults_from_config(self, mock_llm):
        """Test validator loads defaults from config"""
        validator = ValidatorAgent(llm=mock_llm)
        
        # Should have loaded from config
        assert validator.threshold > 0
        assert validator.max_retries >= 0


class TestRelevanceChecking:
    """Tests for relevance checking"""
    
    def test_check_relevance_calls_llm(self, validator, mock_llm, sample_chunks):
        """Test relevance check calls LLM"""
        validator._check_relevance("What is Python?", sample_chunks)
        
        assert mock_llm.invoke.called
    
    def test_check_relevance_parses_valid_response(self, validator, mock_llm, sample_chunks):
        """Test parsing valid LLM response"""
        mock_llm.invoke.return_value.content = "0.85"
        
        score = validator._check_relevance("What is Python?", sample_chunks)
        
        assert score == 0.85
    
    def test_check_relevance_handles_text_response(self, validator, mock_llm, sample_chunks):
        """Test parsing LLM response with text"""
        mock_llm.invoke.return_value.content = "The relevance is 0.75 based on analysis."
        
        score = validator._check_relevance("What is Python?", sample_chunks)
        
        assert score == 0.75
    
    def test_check_relevance_clamps_to_valid_range(self, validator, mock_llm, sample_chunks):
        """Test scores are clamped to [0, 1]"""
        mock_llm.invoke.return_value.content = "1.5"
        
        score = validator._check_relevance("What is Python?", sample_chunks)
        
        assert score <= 1.0
    
    def test_check_relevance_uses_fallback_on_error(self, validator, mock_llm, sample_chunks):
        """Test fallback when LLM fails"""
        mock_llm.invoke.side_effect = Exception("API Error")
        
        score = validator._check_relevance("What is Python?", sample_chunks)
        
        # Should return fallback score
        assert 0.0 <= score <= 1.0
    
    def test_check_relevance_empty_chunks(self, validator):
        """Test relevance check with empty chunks"""
        mock_llm = Mock()
        validator_instance = ValidatorAgent(llm=mock_llm)
        
        score = validator_instance._check_relevance("test", [])
        
        # Should handle gracefully
        assert score == 0.0


class TestFallbackRelevanceScore:
    """Tests for fallback relevance scoring"""
    
    def test_fallback_uses_chunk_scores(self, validator, sample_chunks):
        """Test fallback uses average chunk scores"""
        score = validator._fallback_relevance_score("test", sample_chunks)
        
        # Should be average of chunk scores (0.9, 0.85, 0.8)
        expected = (0.9 + 0.85 + 0.8) / 3
        assert abs(score - expected) < 0.01
    
    def test_fallback_empty_chunks(self, validator):
        """Test fallback with empty chunks"""
        score = validator._fallback_relevance_score("test", [])
        
        assert score == 0.0
    
    def test_fallback_chunks_without_scores(self, validator):
        """Test fallback with chunks that have no scores"""
        chunks = [
            Chunk(text="test", doc_id="1", chunk_id="1", score=None),
            Chunk(text="test", doc_id="1", chunk_id="2", score=None)
        ]
        
        score = validator._fallback_relevance_score("test", chunks)
        
        # Should return default moderate score
        assert score == 0.5


class TestCoverageChecking:
    """Tests for coverage checking"""
    
    def test_check_coverage_simple_query(self, validator, sample_chunks):
        """Test coverage for simple query"""
        score = validator._check_coverage("What is Python?", sample_chunks)
        
        # 3 chunks for 1 aspect (1 question)
        assert score > 0.5
    
    def test_check_coverage_complex_query(self, validator, sample_chunks):
        """Test coverage for complex query"""
        query = "What is Python? How is it used? Why is it popular?"
        score = validator._check_coverage(query, sample_chunks)
        
        # 3 chunks for 3 aspects (3 questions) - marginal
        assert 0.0 <= score <= 1.0
    
    def test_check_coverage_empty_chunks(self, validator):
        """Test coverage with empty chunks"""
        score = validator._check_coverage("test", [])
        
        assert score == 0.0
    
    def test_check_coverage_considers_diversity(self, validator):
        """Test coverage considers document diversity"""
        # All chunks from same document
        same_doc_chunks = [
            Chunk(text=f"chunk{i}", doc_id="doc1", chunk_id=f"c{i}")
            for i in range(5)
        ]
        
        # Chunks from different documents
        diverse_chunks = [
            Chunk(text=f"chunk{i}", doc_id=f"doc{i}", chunk_id=f"c{i}")
            for i in range(5)
        ]
        
        score_same = validator._check_coverage("test", same_doc_chunks)
        score_diverse = validator._check_coverage("test", diverse_chunks)
        
        # Diverse sources should have better coverage
        assert score_diverse > score_same


class TestConfidenceChecking:
    """Tests for confidence checking"""
    
    def test_check_confidence_high_scores(self, validator, sample_chunks):
        """Test confidence with high-quality chunks"""
        score = validator._check_confidence(sample_chunks)
        
        # All chunks have good scores (0.8+)
        assert score > 0.7
    
    def test_check_confidence_low_scores(self, validator):
        """Test confidence with low-quality chunks"""
        low_quality_chunks = [
            Chunk(text="test", doc_id="1", chunk_id="1", score=0.3),
            Chunk(text="test", doc_id="1", chunk_id="2", score=0.2)
        ]
        
        score = validator._check_confidence(low_quality_chunks)
        
        # Low scores should give low confidence
        assert score < 0.5
    
    def test_check_confidence_empty_chunks(self, validator):
        """Test confidence with empty chunks"""
        score = validator._check_confidence([])
        
        assert score == 0.0
    
    def test_check_confidence_no_scores(self, validator):
        """Test confidence when chunks have no scores"""
        chunks = [
            Chunk(text="test", doc_id="1", chunk_id="1", score=None),
            Chunk(text="test", doc_id="1", chunk_id="2", score=None)
        ]
        
        score = validator._check_confidence(chunks)
        
        # Should return moderate default
        assert score == 0.5
    
    def test_check_confidence_considers_consistency(self, validator):
        """Test confidence considers score variance"""
        # Consistent scores
        consistent_chunks = [
            Chunk(text="test", doc_id="1", chunk_id=f"c{i}", score=0.8)
            for i in range(3)
        ]
        
        # Variable scores
        variable_chunks = [
            Chunk(text="test", doc_id="1", chunk_id="c1", score=0.9),
            Chunk(text="test", doc_id="1", chunk_id="c2", score=0.5),
            Chunk(text="test", doc_id="1", chunk_id="c3", score=0.3)
        ]
        
        score_consistent = validator._check_confidence(consistent_chunks)
        score_variable = validator._check_confidence(variable_chunks)
        
        # Consistent scores should have higher confidence
        assert score_consistent > score_variable


class TestSufficiencyCalculation:
    """Tests for overall sufficiency calculation"""
    
    def test_calculate_sufficiency_combines_factors(self, validator, mock_llm, sample_chunks):
        """Test sufficiency combines all factors"""
        mock_llm.invoke.return_value.content = "0.9"
        
        score = validator._calculate_sufficiency("What is Python?", sample_chunks)
        
        # Should be weighted average of relevance, coverage, confidence
        assert 0.0 <= score <= 1.0
    
    def test_calculate_sufficiency_empty_chunks(self, validator):
        """Test sufficiency with empty chunks"""
        score = validator._calculate_sufficiency("test", [])
        
        assert score == 0.0
    
    def test_calculate_sufficiency_weights(self, validator, mock_llm):
        """Test that weights are applied correctly"""
        # High relevance (0.9), low coverage/confidence
        mock_llm.invoke.return_value.content = "0.9"
        
        chunks = [
            Chunk(text="test", doc_id="1", chunk_id="1", score=0.3)
        ]
        
        score = validator._calculate_sufficiency("complex query?", chunks)
        
        # Relevance is 50% weight, so high relevance should dominate
        assert score > 0.4


class TestDecisionMaking:
    """Tests for validation decision logic"""
    
    def test_make_decision_proceed_high_score(self, validator):
        """Test PROCEED decision for high score"""
        decision = validator._make_decision(score=0.85, current_round=0)
        
        assert decision == "PROCEED"
    
    def test_make_decision_retrieve_more_low_score(self, validator):
        """Test RETRIEVE_MORE for low score"""
        decision = validator._make_decision(score=0.45, current_round=0)
        
        assert decision == "RETRIEVE_MORE"
    
    def test_make_decision_at_threshold(self, validator):
        """Test decision at exact threshold"""
        decision = validator._make_decision(score=0.7, current_round=0)
        
        # At threshold should PROCEED
        assert decision == "PROCEED"
    
    def test_make_decision_just_below_threshold(self, validator):
        """Test decision just below threshold"""
        decision = validator._make_decision(score=0.69, current_round=0)
        
        # Below threshold should RETRIEVE_MORE
        assert decision == "RETRIEVE_MORE"
    
    def test_make_decision_max_retries_reached(self, validator):
        """Test PROCEED when max retries reached"""
        decision = validator._make_decision(score=0.3, current_round=2)
        
        # Max retries (2), should force PROCEED
        assert decision == "PROCEED"
    
    def test_make_decision_retry_progression(self, validator):
        """Test decision changes as retries increase"""
        low_score = 0.4
        
        # Round 0: should retry
        decision_0 = validator._make_decision(low_score, 0)
        assert decision_0 == "RETRIEVE_MORE"
        
        # Round 1: should retry
        decision_1 = validator._make_decision(low_score, 1)
        assert decision_1 == "RETRIEVE_MORE"
        
        # Round 2 (max): should proceed
        decision_2 = validator._make_decision(low_score, 2)
        assert decision_2 == "PROCEED"


class TestValidatorExecution:
    """Tests for execute method"""
    
    def test_execute_updates_state_with_score(self, validator, sample_chunks):
        """Test execute updates state with validation score"""
        state = AgentState(query="What is Python?", chunks=sample_chunks)
        
        result = validator.run(state)
        
        assert result.validation_score is not None
        assert 0.0 <= result.validation_score <= 1.0
    
    def test_execute_updates_state_with_status(self, validator, sample_chunks):
        """Test execute updates state with validation status"""
        state = AgentState(query="What is Python?", chunks=sample_chunks)
        
        result = validator.run(state)
        
        assert result.validation_status is not None
        assert result.validation_status in ["PROCEED", "RETRIEVE_MORE"]
    
    def test_execute_preserves_query(self, validator, sample_chunks):
        """Test execute preserves original query"""
        state = AgentState(query="Test query", chunks=sample_chunks)
        
        result = validator.run(state)
        
        assert result.query == "Test query"
    
    def test_execute_preserves_chunks(self, validator, sample_chunks):
        """Test execute preserves chunks"""
        state = AgentState(query="test", chunks=sample_chunks)
        
        result = validator.run(state)
        
        assert len(result.chunks) == len(sample_chunks)
    
    def test_execute_adds_metadata(self, validator, sample_chunks):
        """Test execute adds validator metadata"""
        state = AgentState(query="test", chunks=sample_chunks)
        
        result = validator.run(state)
        
        assert "validator" in result.metadata
        assert "score" in result.metadata["validator"]
        assert "decision" in result.metadata["validator"]
        assert "threshold" in result.metadata["validator"]
    
    def test_execute_sufficient_chunks(self, validator, mock_llm):
        """Test execute with sufficient chunks"""
        mock_llm.invoke.return_value.content = "0.9"
        
        good_chunks = [
            Chunk(text=f"relevant chunk {i}", doc_id="1", chunk_id=f"c{i}", score=0.9)
            for i in range(5)
        ]
        
        state = AgentState(query="test", chunks=good_chunks)
        result = validator.run(state)
        
        assert result.validation_status == "PROCEED"
        assert result.validation_score >= validator.threshold
    
    def test_execute_insufficient_chunks(self, validator, mock_llm):
        """Test execute with insufficient chunks"""
        mock_llm.invoke.return_value.content = "0.3"
        
        poor_chunks = [
            Chunk(text="irrelevant", doc_id="1", chunk_id="c1", score=0.2)
        ]
        
        state = AgentState(query="test", chunks=poor_chunks, retrieval_round=0)
        result = validator.run(state)
        
        assert result.validation_status == "RETRIEVE_MORE"
        assert result.validation_score < validator.threshold


class TestValidateChunksDetailed:
    """Tests for detailed validation analysis"""
    
    def test_validate_chunks_detailed_returns_all_fields(self, validator, sample_chunks):
        """Test detailed validation returns all expected fields"""
        details = validator.validate_chunks_detailed("What is Python?", sample_chunks)
        
        assert "query" in details
        assert "chunk_count" in details
        assert "relevance_score" in details
        assert "coverage_score" in details
        assert "confidence_score" in details
        assert "final_score" in details
        assert "threshold" in details
        assert "final_decision" in details
        assert "would_retry" in details
    
    def test_validate_chunks_detailed_scores_breakdown(self, validator, sample_chunks):
        """Test detailed validation includes score breakdown"""
        details = validator.validate_chunks_detailed("test", sample_chunks)
        
        assert 0.0 <= details["relevance_score"] <= 1.0
        assert 0.0 <= details["coverage_score"] <= 1.0
        assert 0.0 <= details["confidence_score"] <= 1.0
        assert 0.0 <= details["final_score"] <= 1.0


class TestValidatorMetrics:
    """Tests for metrics tracking"""
    
    def test_metrics_updated_on_execution(self, validator, sample_chunks):
        """Test metrics are updated after execution"""
        state = AgentState(query="test", chunks=sample_chunks)
        
        validator.run(state)
        metrics = validator.get_metrics()
        
        assert metrics["total_calls"] == 1
        assert metrics["successful_calls"] == 1
    
    def test_metrics_track_multiple_validations(self, validator, sample_chunks):
        """Test metrics track multiple validations"""
        for i in range(3):
            state = AgentState(query=f"query {i}", chunks=sample_chunks)
            validator.run(state)
        
        metrics = validator.get_metrics()
        assert metrics["total_calls"] == 3
        assert metrics["success_rate"] == 100.0


class TestValidatorEdgeCases:
    """Edge case tests"""
    
    def test_validator_handles_no_chunks(self, validator):
        """Test validator with no chunks"""
        state = AgentState(query="test", chunks=[])
        
        result = validator.run(state)
        
        assert result.validation_score == 0.0
        assert result.validation_status == "RETRIEVE_MORE"
    
    def test_validator_handles_single_chunk(self, validator):
        """Test validator with single chunk"""
        chunk = Chunk(text="test", doc_id="1", chunk_id="1", score=0.9)
        state = AgentState(query="test", chunks=[chunk])
        
        result = validator.run(state)
        
        assert result.validation_score is not None
    
    def test_validator_handles_many_chunks(self, validator):
        """Test validator with many chunks"""
        many_chunks = [
            Chunk(text=f"chunk {i}", doc_id=f"doc{i%3}", chunk_id=f"c{i}", score=0.8)
            for i in range(50)
        ]
        
        state = AgentState(query="test", chunks=many_chunks)
        result = validator.run(state)
        
        # Should handle without issues
        assert result.validation_score is not None