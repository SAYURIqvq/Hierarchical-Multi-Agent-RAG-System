import pytest
from src.models.agent_state import AgentState, Chunk, Strategy

def test_agent_state_creation():
    """Test basic state creation"""
    state = AgentState(query="What is Python?")
    assert state.query == "What is Python?"
    assert state.complexity is None
    assert len(state.chunks) == 0

def test_chunk_validation():
    """Test chunk score validation"""
    # Valid score
    chunk = Chunk(text="test", doc_id="1", chunk_id="1", score=0.8)
    assert chunk.score == 0.8
    
    # Invalid score
    with pytest.raises(ValueError):
        Chunk(text="test", doc_id="1", chunk_id="1", score=1.5)

def test_add_chunk():
    """Test adding chunks"""
    state = AgentState(query="test")
    chunk = Chunk(text="info", doc_id="1", chunk_id="1", score=0.9)
    
    state.add_chunk(chunk)
    assert len(state.chunks) == 1
    assert state.chunks[0].text == "info"

def test_get_top_chunks():
    """Test getting top chunks"""
    state = AgentState(query="test")
    
    # Add chunks with different scores
    state.add_chunk(Chunk(text="c1", doc_id="1", chunk_id="1", score=0.5))
    state.add_chunk(Chunk(text="c2", doc_id="1", chunk_id="2", score=0.9))
    state.add_chunk(Chunk(text="c3", doc_id="1", chunk_id="3", score=0.7))
    
    top_2 = state.get_top_chunks(k=2)
    assert len(top_2) == 2
    assert top_2[0].score == 0.9  # Highest first
    assert top_2[1].score == 0.7

def test_strategy_enum():
    """Test strategy enum"""
    state = AgentState(query="test", strategy=Strategy.SIMPLE)
    assert state.strategy == "simple"  # Auto-converted to string