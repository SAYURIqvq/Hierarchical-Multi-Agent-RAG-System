"""
Unit tests for unified Chunk model.
"""

import pytest
from src.models.chunk import (
    Chunk,
    generate_chunk_id,
    get_parent_chunks,
    get_child_chunks,
    get_chunk_with_parent,
    find_chunk_by_id
)


class TestChunkCreation:
    """Test basic chunk creation."""
    
    def test_minimal_chunk(self):
        """Test creating chunk with minimal fields."""
        chunk = Chunk(
            chunk_id="test_001",
            text="Test chunk text",
            doc_id="doc_123"
        )
        
        assert chunk.chunk_id == "test_001"
        assert chunk.text == "Test chunk text"
        assert chunk.doc_id == "doc_123"
        assert chunk.chunk_type == "child"  # Default
    
    def test_chunk_with_all_fields(self):
        """Test creating chunk with all fields."""
        chunk = Chunk(
            chunk_id="test_001",
            text="Test text",
            doc_id="doc_123",
            parent_id="parent_001",
            children_ids=["child_001", "child_002"],
            tokens=[1, 2, 3, 4],
            token_count=4,
            start_idx=0,
            end_idx=100,
            chunk_type="parent",
            embedding=[0.1, 0.2, 0.3],
            metadata={"page": 1},
            score=0.95
        )
        
        assert chunk.parent_id == "parent_001"
        assert len(chunk.children_ids) == 2
        assert chunk.token_count == 4
        assert chunk.has_embedding()
        assert chunk.score == 0.95


class TestChunkMethods:
    """Test chunk helper methods."""
    
    def test_is_parent(self):
        """Test parent detection."""
        parent = Chunk(chunk_id="p1", text="Parent", chunk_type="parent")
        child = Chunk(chunk_id="c1", text="Child", chunk_type="child", parent_id="p1")
        
        assert parent.is_parent()
        assert not child.is_parent()
    
    def test_is_child(self):
        """Test child detection."""
        parent = Chunk(chunk_id="p1", text="Parent", chunk_type="parent")
        child = Chunk(chunk_id="c1", text="Child", chunk_type="child", parent_id="p1")
        
        assert child.is_child()
        assert not parent.is_child()
    
    def test_has_embedding(self):
        """Test embedding detection."""
        chunk_no_emb = Chunk(chunk_id="c1", text="Test")
        chunk_with_emb = Chunk(chunk_id="c2", text="Test", embedding=[0.1, 0.2])
        
        assert not chunk_no_emb.has_embedding()
        assert chunk_with_emb.has_embedding()
    
    def test_get_embedding_dimension(self):
        """Test embedding dimension."""
        chunk = Chunk(chunk_id="c1", text="Test", embedding=[0.1, 0.2, 0.3])
        
        assert chunk.get_embedding_dimension() == 3


class TestChunkSerialization:
    """Test serialization methods."""
    
    def test_to_dict(self):
        """Test converting chunk to dictionary."""
        chunk = Chunk(
            chunk_id="test_001",
            text="Test",
            doc_id="doc_123",
            metadata={"page": 1}
        )
        
        chunk_dict = chunk.to_dict()
        
        assert isinstance(chunk_dict, dict)
        assert chunk_dict['chunk_id'] == "test_001"
        assert chunk_dict['text'] == "Test"
        assert chunk_dict['metadata'] == {"page": 1}
    
    def test_from_dict(self):
        """Test creating chunk from dictionary."""
        data = {
            'chunk_id': "test_001",
            'text': "Test text",
            'doc_id': "doc_123",
            'chunk_type': "child",
            'metadata': {"page": 1}
        }
        
        chunk = Chunk.from_dict(data)
        
        assert chunk.chunk_id == "test_001"
        assert chunk.text == "Test text"
        assert chunk.metadata == {"page": 1}
    
    def test_clone(self):
        """Test cloning chunk."""
        original = Chunk(
            chunk_id="test_001",
            text="Test",
            metadata={"page": 1}
        )
        
        cloned = original.clone()
        
        assert cloned.chunk_id == original.chunk_id
        assert cloned.text == original.text
        assert cloned.metadata == original.metadata
        assert cloned is not original  # Different objects


class TestHelperFunctions:
    """Test helper functions."""
    
    def test_generate_chunk_id(self):
        """Test chunk ID generation."""
        chunk_id = generate_chunk_id("doc_123", "parent_0")
        
        assert isinstance(chunk_id, str)
        assert len(chunk_id) == 32  # MD5 hash length
    
    def test_get_parent_chunks(self):
        """Test filtering parent chunks."""
        chunks = [
            Chunk(chunk_id="p1", text="Parent 1", chunk_type="parent"),
            Chunk(chunk_id="c1", text="Child 1", chunk_type="child"),
            Chunk(chunk_id="p2", text="Parent 2", chunk_type="parent"),
        ]
        
        parents = get_parent_chunks(chunks)
        
        assert len(parents) == 2
        assert all(c.is_parent() for c in parents)
    
    def test_get_child_chunks(self):
        """Test filtering child chunks."""
        chunks = [
            Chunk(chunk_id="p1", text="Parent 1", chunk_type="parent"),
            Chunk(chunk_id="c1", text="Child 1", chunk_type="child"),
            Chunk(chunk_id="c2", text="Child 2", chunk_type="child"),
        ]
        
        children = get_child_chunks(chunks)
        
        assert len(children) == 2
        assert all(c.is_child() for c in children)
    
    def test_find_chunk_by_id(self):
        """Test finding chunk by ID."""
        chunks = [
            Chunk(chunk_id="c1", text="Chunk 1"),
            Chunk(chunk_id="c2", text="Chunk 2"),
            Chunk(chunk_id="c3", text="Chunk 3"),
        ]
        
        chunk = find_chunk_by_id(chunks, "c2")
        
        assert chunk is not None
        assert chunk.chunk_id == "c2"
        assert chunk.text == "Chunk 2"
    
    def test_get_chunk_with_parent(self):
        """Test getting chunk with parent."""
        chunks = [
            Chunk(chunk_id="p1", text="Parent", chunk_type="parent"),
            Chunk(chunk_id="c1", text="Child", chunk_type="child", parent_id="p1"),
        ]
        
        result = get_chunk_with_parent("c1", chunks)
        
        assert result['chunk'].chunk_id == "c1"
        assert result['parent'].chunk_id == "p1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])