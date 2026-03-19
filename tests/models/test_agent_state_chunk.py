"""
Integration tests for AgentState with unified Chunk model.
"""

import pytest
from src.models.agent_state import AgentState, Strategy
from src.models.chunk import Chunk


class TestAgentStateWithChunk:
    """Test AgentState compatibility with Chunk."""
    
    def test_agent_state_accepts_chunks(self):
        """Test AgentState can hold chunks."""
        chunks = [
            Chunk(chunk_id="c1", text="Chunk 1", doc_id="doc_123", score=0.9),
            Chunk(chunk_id="c2", text="Chunk 2", doc_id="doc_123", score=0.8),
        ]
        
        state = AgentState(
            query="Test query",
            chunks=chunks
        )
        
        assert len(state.chunks) == 2
        assert state.chunks[0].chunk_id == "c1"
        assert state.chunks[1].score == 0.8
    
    def test_agent_state_with_full_pipeline(self):
        """Test AgentState through full pipeline."""
        # Create chunks
        chunks = [
            Chunk(
                chunk_id="c1",
                text="Python is a programming language",
                doc_id="doc_123",
                chunk_type="child",
                parent_id="p1",
                embedding=[0.1, 0.2, 0.3],
                score=0.95,
                metadata={"filename": "intro.pdf"}
            )
        ]
        
        # Create state
        state = AgentState(
            query="What is Python?",
            complexity=0.3,
            strategy=Strategy.SIMPLE,
            chunks=chunks,
            validation_status="PROCEED",
            validation_score=0.85,
            answer="Python is a programming language.",
            critic_score=0.9
        )
        
        # Verify all fields work
        assert state.query == "What is Python?"
        assert state.strategy == Strategy.SIMPLE
        assert len(state.chunks) == 1
        assert state.chunks[0].has_embedding()
        assert state.answer is not None
    
    def test_empty_chunks(self):
        """Test AgentState with no chunks."""
        state = AgentState(query="Test")
        
        assert len(state.chunks) == 0
        assert state.chunks == []
    
    def test_chunk_modification(self):
        """Test modifying chunks in AgentState."""
        chunk = Chunk(chunk_id="c1", text="Original", score=0.5)
        state = AgentState(query="Test", chunks=[chunk])
        
        # Modify chunk
        state.chunks[0].score = 0.9
        
        assert state.chunks[0].score == 0.9


class TestAgentStateSerialization:
    """Test AgentState serialization with chunks."""
    
    def test_state_to_dict(self):
        """Test converting state to dictionary."""
        chunk = Chunk(
            chunk_id="c1",
            text="Test",
            doc_id="doc_123",
            metadata={"page": 1}
        )
        
        state = AgentState(
            query="Test query",
            chunks=[chunk]
        )
        
        # Convert to dict (Pydantic)
        state_dict = state.dict()
        
        assert 'query' in state_dict
        assert 'chunks' in state_dict
        assert len(state_dict['chunks']) == 1
    
    def test_state_json_serialization(self):
        """Test JSON serialization."""
        chunk = Chunk(chunk_id="c1", text="Test")
        state = AgentState(query="Test", chunks=[chunk])
        
        # Pydantic's json() method
        json_str = state.json()
        
        assert isinstance(json_str, str)
        assert "c1" in json_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])