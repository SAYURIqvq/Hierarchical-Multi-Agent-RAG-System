"""
Tests for Retrieval Coordinator Agent.

Tests swarm spawning, parallel execution, aggregation, and deduplication.
"""

import pytest
from unittest.mock import Mock, patch

from src.agents.retrieval_coordinator import RetrievalCoordinator
from src.agents.retrieval.vector_agent import VectorSearchAgent
from src.agents.retrieval.keyword_agent import KeywordSearchAgent
from src.agents.retrieval.graph_agent import GraphSearchAgent
from src.models.agent_state import AgentState, Chunk


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
    """Create retrieval coordinator with all agents."""
    return RetrievalCoordinator(
        vector_agent=vector_agent,
        keyword_agent=keyword_agent,
        graph_agent=graph_agent,
        top_k=10,
        parallel=False  # Sequential for easier testing
    )


@pytest.fixture
def parallel_coordinator(vector_agent, keyword_agent, graph_agent):
    """Create coordinator with parallel execution."""
    return RetrievalCoordinator(
        vector_agent=vector_agent,
        keyword_agent=keyword_agent,
        graph_agent=graph_agent,
        top_k=10,
        parallel=True
    )


class TestCoordinatorInitialization:
    """Tests for coordinator initialization"""
    
    def test_coordinator_initializes_with_agents(self, vector_agent, keyword_agent, graph_agent):
        """Test coordinator initializes with all agents"""
        coordinator = RetrievalCoordinator(
            vector_agent=vector_agent,
            keyword_agent=keyword_agent,
            graph_agent=graph_agent
        )
        
        assert coordinator.name == "retrieval_coordinator"
        assert coordinator.vector_agent is not None
        assert coordinator.keyword_agent is not None
        assert coordinator.graph_agent is not None
    
    def test_coordinator_accepts_custom_top_k(self, vector_agent):
        """Test coordinator accepts custom top_k"""
        coordinator = RetrievalCoordinator(
            vector_agent=vector_agent,
            top_k=15
        )
        
        assert coordinator.top_k == 15
    
    def test_coordinator_loads_defaults_from_config(self, vector_agent):
        """Test coordinator loads defaults from config"""
        coordinator = RetrievalCoordinator(vector_agent=vector_agent)
        
        # Should have loaded from config
        assert coordinator.top_k > 0
        assert coordinator.parallel is not None


class TestSwarmSpawning:
    """Tests for swarm spawning"""
    
    def test_spawn_swarm_sequential(self, coordinator):
        """Test swarm execution in sequential mode"""
        chunks = coordinator._spawn_swarm("What is Python?")
        
        # Should get chunks from all agents
        assert len(chunks) > 0
        # 3 agents × 5 chunks each = 15 total
        assert len(chunks) == 15
    
    def test_spawn_swarm_parallel(self, parallel_coordinator):
        """Test swarm execution in parallel mode"""
        chunks = parallel_coordinator._spawn_swarm("What is Python?")
        
        # Should get chunks from all agents
        assert len(chunks) > 0
        assert len(chunks) == 15
    
    def test_spawn_swarm_with_single_agent(self, vector_agent):
        """Test swarm with only one agent"""
        coordinator = RetrievalCoordinator(
            vector_agent=vector_agent,
            top_k=10
        )
        
        chunks = coordinator._spawn_swarm("test")
        
        # Should only get chunks from vector agent
        assert len(chunks) == 5


class TestResultAggregation:
    """Tests for result aggregation"""
    
    def test_aggregate_chunks_from_multiple_agents(self, coordinator):
        """Test aggregation combines all chunks"""
        state = AgentState(query="What is machine learning?")
        
        result = coordinator.run(state)
        
        # Should have chunks from all agents
        assert len(result.chunks) > 0
    
    def test_chunks_tagged_with_source(self, coordinator):
        """Test chunks are tagged with source metadata"""
        state = AgentState(query="test")
        result = coordinator.run(state)
        
        # Check source tags
        sources = set()
        for chunk in result.chunks:
            if "source" in chunk.metadata:
                sources.add(chunk.metadata["source"])
        
        # Should have multiple sources (though dedup may reduce)
        assert len(sources) > 0


class TestDeduplication:
    """Tests for deduplication logic"""
    
    def test_deduplicate_removes_exact_duplicates(self, coordinator):
        """Test deduplication removes exact duplicates"""
        # Create duplicate chunks
        chunk1 = Chunk(text="Same content", doc_id="1", chunk_id="1", score=0.9, metadata={})
        chunk2 = Chunk(text="Same content", doc_id="1", chunk_id="2", score=0.8, metadata={})
        chunk3 = Chunk(text="Different content", doc_id="2", chunk_id="3", score=0.7, metadata={})
        
        duplicates = [chunk1, chunk2, chunk3]
        unique = coordinator._deduplicate(duplicates)
        
        # Should keep only 2 unique
        assert len(unique) == 2
    
    def test_deduplicate_keeps_highest_score(self, coordinator):
        """Test deduplication keeps chunk with highest score"""
        chunk1 = Chunk(text="Same content", doc_id="1", chunk_id="1", score=0.9, metadata={})
        chunk2 = Chunk(text="Same content", doc_id="1", chunk_id="2", score=0.95, metadata={})
        
        duplicates = [chunk1, chunk2]
        unique = coordinator._deduplicate(duplicates)
        
        # Should keep chunk with score 0.95
        assert len(unique) == 1
        assert unique[0].score == 0.95
    
    def test_deduplicate_case_insensitive(self, coordinator):
        """Test deduplication is case insensitive"""
        chunk1 = Chunk(text="Python Programming", doc_id="1", chunk_id="1", score=0.9, metadata={})
        chunk2 = Chunk(text="python programming", doc_id="1", chunk_id="2", score=0.8, metadata={})
        
        duplicates = [chunk1, chunk2]
        unique = coordinator._deduplicate(duplicates)
        
        # Should treat as duplicates
        assert len(unique) == 1
    
    def test_deduplicate_normalizes_whitespace(self, coordinator):
        """Test deduplication normalizes whitespace"""
        chunk1 = Chunk(text="Python  is   great", doc_id="1", chunk_id="1", score=0.9, metadata={})
        chunk2 = Chunk(text="Python is great", doc_id="1", chunk_id="2", score=0.8, metadata={})
        
        duplicates = [chunk1, chunk2]
        unique = coordinator._deduplicate(duplicates)
        
        # Should treat as duplicates
        assert len(unique) == 1
    
    def test_deduplicate_empty_list(self, coordinator):
        """Test deduplication with empty list"""
        unique = coordinator._deduplicate([])
        
        assert len(unique) == 0


class TestTopKSelection:
    """Tests for top-k selection"""
    
    def test_select_top_k_returns_correct_count(self, coordinator):
        """Test top-k returns requested number"""
        chunks = [
            Chunk(
                text=f"chunk {i}",
                doc_id="doc1",
                chunk_id=f"c{i}",
                score=max(0.1, 0.9 - (i * 0.04)),  # Keep scores in valid range
                metadata={}
            )
            for i in range(20)
        ]
        
        top_5 = coordinator._select_top_k(chunks, 5)
        
        assert len(top_5) == 5
    
    def test_select_top_k_sorted_by_score(self, coordinator):
        """Test top-k chunks are sorted by score"""
        chunks = [
            Chunk(text="low", doc_id="1", chunk_id="1", score=0.5, metadata={}),
            Chunk(text="high", doc_id="1", chunk_id="2", score=0.9, metadata={}),
            Chunk(text="medium", doc_id="1", chunk_id="3", score=0.7, metadata={})
        ]
        
        top_3 = coordinator._select_top_k(chunks, 3)
        
        # Should be in descending order
        assert top_3[0].score == 0.9
        assert top_3[1].score == 0.7
        assert top_3[2].score == 0.5
    
    def test_select_top_k_handles_fewer_chunks(self, coordinator):
        """Test top-k when fewer chunks available"""
        chunks = [
            Chunk(text="chunk1", doc_id="1", chunk_id="1", score=0.9, metadata={}),
            Chunk(text="chunk2", doc_id="1", chunk_id="2", score=0.8, metadata={})
        ]
        
        top_10 = coordinator._select_top_k(chunks, 10)
        
        # Should return all available
        assert len(top_10) == 2
    
    def test_select_top_k_empty_list(self, coordinator):
        """Test top-k with empty list"""
        top_k = coordinator._select_top_k([], 10)
        
        assert len(top_k) == 0


class TestCoordinatorExecution:
    """Tests for execute method"""
    
    def test_execute_updates_state_with_chunks(self, coordinator):
        """Test execute updates state with chunks"""
        state = AgentState(query="What is Python?")
        
        result = coordinator.run(state)
        
        assert len(result.chunks) > 0
        assert len(result.chunks) <= coordinator.top_k
    
    def test_execute_increments_retrieval_round(self, coordinator):
        """Test execute increments retrieval round"""
        state = AgentState(query="test", retrieval_round=0)
        
        result = coordinator.run(state)
        
        assert result.retrieval_round == 1
    
    def test_execute_preserves_query(self, coordinator):
        """Test execute preserves original query"""
        state = AgentState(query="original query")
        
        result = coordinator.run(state)
        
        assert result.query == "original query"
    
    def test_execute_adds_metadata(self, coordinator):
        """Test execute adds coordinator metadata"""
        state = AgentState(query="test")
        
        result = coordinator.run(state)
        
        assert "retrieval_coordinator" in result.metadata
        meta = result.metadata["retrieval_coordinator"]
        
        assert "round" in meta
        assert "total_retrieved" in meta
        assert "unique_chunks" in meta
        assert "final_chunks" in meta
    
    def test_execute_multiple_rounds(self, coordinator):
        """Test multiple retrieval rounds"""
        state = AgentState(query="test", retrieval_round=0)
        
        # Round 1
        state = coordinator.run(state)
        assert state.retrieval_round == 1
        
        # Round 2
        state = coordinator.run(state)
        assert state.retrieval_round == 2


class TestHashContent:
    """Tests for content hashing"""
    
    def test_hash_content_same_text(self, coordinator):
        """Test same text produces same hash"""
        hash1 = coordinator._hash_content("Python is great")
        hash2 = coordinator._hash_content("Python is great")
        
        assert hash1 == hash2
    
    def test_hash_content_different_text(self, coordinator):
        """Test different text produces different hash"""
        hash1 = coordinator._hash_content("Python is great")
        hash2 = coordinator._hash_content("Java is great")
        
        assert hash1 != hash2
    
    def test_hash_content_case_insensitive(self, coordinator):
        """Test hashing is case insensitive"""
        hash1 = coordinator._hash_content("Python")
        hash2 = coordinator._hash_content("python")
        
        assert hash1 == hash2
    
    def test_hash_content_whitespace_normalized(self, coordinator):
        """Test whitespace is normalized"""
        hash1 = coordinator._hash_content("Python  is   great")
        hash2 = coordinator._hash_content("Python is great")
        
        assert hash1 == hash2


class TestRetrieveWithDetails:
    """Tests for detailed retrieval analysis"""
    
    def test_retrieve_with_details_returns_all_fields(self, coordinator):
        """Test detailed retrieval returns all expected fields"""
        details = coordinator.retrieve_with_details("What is Python?")
        
        assert "query" in details
        assert "total_retrieved" in details
        assert "source_counts" in details
        assert "unique_chunks" in details
        assert "duplicates_removed" in details
        assert "final_chunks" in details
        assert "top_k" in details
        assert "parallel" in details
        assert "chunks" in details
    
    def test_retrieve_with_details_source_breakdown(self, coordinator):
        """Test detailed retrieval includes source breakdown"""
        details = coordinator.retrieve_with_details("test")
        
        # Should have source counts
        assert isinstance(details["source_counts"], dict)
    
    def test_retrieve_with_details_dedup_stats(self, coordinator):
        """Test detailed retrieval includes deduplication stats"""
        details = coordinator.retrieve_with_details("test")
        
        total = details["total_retrieved"]
        unique = details["unique_chunks"]
        removed = details["duplicates_removed"]
        
        # Math should be consistent
        assert total == unique + removed


class TestCoordinatorMetrics:
    """Tests for metrics tracking"""
    
    def test_metrics_updated_on_execution(self, coordinator):
        """Test metrics are updated after execution"""
        state = AgentState(query="test")
        
        coordinator.run(state)
        metrics = coordinator.get_metrics()
        
        assert metrics["total_calls"] == 1
        assert metrics["successful_calls"] == 1
    
    def test_metrics_track_multiple_retrievals(self, coordinator):
        """Test metrics track multiple retrievals"""
        for i in range(3):
            state = AgentState(query=f"query {i}")
            coordinator.run(state)
        
        metrics = coordinator.get_metrics()
        assert metrics["total_calls"] == 3
        assert metrics["success_rate"] == 100.0


class TestCoordinatorEdgeCases:
    """Edge case tests"""
    
    def test_coordinator_with_no_agents(self):
        """Test coordinator with no agents"""
        coordinator = RetrievalCoordinator(top_k=10)
        
        chunks = coordinator._spawn_swarm("test")
        
        # Should return empty list
        assert len(chunks) == 0
    
    def test_coordinator_handles_agent_failure(self, vector_agent):
        """Test coordinator handles agent failure gracefully"""
        # Create mock agent that fails
        failing_agent = Mock()
        failing_agent.run.side_effect = Exception("Agent failed")
        
        coordinator = RetrievalCoordinator(
            vector_agent=vector_agent,
            keyword_agent=failing_agent,
            parallel=False
        )
        
        # Should still work with remaining agents
        state = AgentState(query="test")
        result = coordinator.run(state)
        
        # Should have chunks from vector agent
        assert len(result.chunks) > 0
    
    def test_coordinator_empty_query(self, coordinator):
        """Test coordinator with empty query"""
        state = AgentState(query="")
        
        result = coordinator.run(state)
        
        # Should handle gracefully
        assert result.chunks is not None


class TestParallelVsSequential:
    """Tests comparing parallel and sequential execution"""
    
    def test_parallel_and_sequential_same_results(
        self,
        coordinator,
        parallel_coordinator
    ):
        """Test parallel and sequential produce same chunk count"""
        query = "What is Python?"
        
        # Sequential
        state1 = AgentState(query=query)
        result1 = coordinator.run(state1)
        
        # Parallel
        state2 = AgentState(query=query)
        result2 = parallel_coordinator.run(state2)
        
        # Should have same number of final chunks
        assert len(result1.chunks) == len(result2.chunks)