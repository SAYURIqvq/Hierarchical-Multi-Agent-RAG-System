"""
Tests for LangGraph Workflow.

Tests workflow construction, node execution, and routing logic.
"""

import pytest
from unittest.mock import Mock

from src.orchestration.langgraph_workflow import AgenticRAGWorkflow
from src.agents.planner import PlannerAgent
from src.agents.validator import ValidatorAgent
from src.agents.retrieval_coordinator import RetrievalCoordinator
from src.agents.retrieval.vector_agent import VectorSearchAgent
from src.models.agent_state import AgentState, Strategy
from src.utils.exceptions import OrchestrationError


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


@pytest.fixture
def coordinator():
    """Create retrieval coordinator with mock agents."""
    vector_agent = VectorSearchAgent(top_k=5, mock_mode=True)
    return RetrievalCoordinator(
        vector_agent=vector_agent,
        top_k=10,
        parallel=False
    )


@pytest.fixture
def workflow(planner, coordinator, validator):
    """Create workflow instance."""
    return AgenticRAGWorkflow(planner, coordinator, validator)


class TestWorkflowInitialization:
    """Tests for workflow initialization"""
    
    def test_workflow_initializes_with_agents(self, planner, coordinator, validator):
        """Test workflow initializes with all agents"""
        workflow = AgenticRAGWorkflow(planner, coordinator, validator)
        
        assert workflow.planner is not None
        assert workflow.coordinator is not None
        assert workflow.validator is not None
        assert workflow.workflow is not None
    
    def test_workflow_builds_graph(self, workflow):
        """Test workflow builds LangGraph successfully"""
        assert workflow.workflow is not None
        assert workflow.logger is not None


class TestWorkflowNodes:
    """Tests for individual workflow nodes"""
    
    def test_planner_node_executes(self, workflow):
        """Test planner node executes successfully"""
        state = AgentState(query="What is Python?")
        
        result = workflow._planner_node(state)
        
        assert result.complexity is not None
        assert result.strategy is not None
    
    def test_retrieval_node_executes(self, workflow):
        """Test retrieval node executes successfully"""
        state = AgentState(query="test", retrieval_round=0)
        
        result = workflow._retrieval_node(state)
        
        assert len(result.chunks) > 0
        assert result.retrieval_round == 1
    
    def test_validator_node_executes(self, workflow):
        """Test validator node executes successfully"""
        # Create state with chunks
        vector_agent = VectorSearchAgent(top_k=5, mock_mode=True)
        temp_state = AgentState(query="test")
        temp_state = vector_agent.run(temp_state)
        
        result = workflow._validator_node(temp_state)
        
        assert result.validation_status is not None
        assert result.validation_score is not None


class TestConditionalRouting:
    """Tests for conditional routing logic"""
    
    def test_should_continue_with_proceed(self, workflow):
        """Test routing when validation passes"""
        state = AgentState(query="test", validation_status="PROCEED")
        
        decision = workflow._should_continue(state)
        
        assert decision == "end"
    
    def test_should_continue_with_retrieve_more(self, workflow):
        """Test routing when validation fails"""
        state = AgentState(query="test", validation_status="RETRIEVE_MORE")
        
        decision = workflow._should_continue(state)
        
        assert decision == "continue"
    
    def test_should_continue_with_unknown_status(self, workflow):
        """Test routing with unknown validation status"""
        state = AgentState(query="test", validation_status="UNKNOWN")
        
        decision = workflow._should_continue(state)
        
        # Should default to end
        assert decision == "end"


class TestWorkflowExecution:
    """Tests for complete workflow execution"""
    
    def test_run_executes_workflow(self, workflow):
        """Test run method executes complete workflow"""
        result = workflow.run("What is Python?")
        
        # Should have results from all agents
        assert result.complexity is not None
        assert result.strategy is not None
        assert len(result.chunks) > 0
        assert result.validation_status is not None
    
    def test_run_returns_final_state(self, workflow):
        """Test run returns AgentState"""
        result = workflow.run("test")
        
        assert isinstance(result, AgentState)
        assert result.query == "test"
    
    def test_run_with_simple_query(self, workflow, mock_llm):
        """Test workflow with simple query"""
        mock_llm.invoke.return_value.content = "0.2"
        
        result = workflow.run("What is X?")
        
        # Should complete with simple strategy
        assert result.complexity < 0.3
        assert result.strategy == Strategy.SIMPLE
    
    def test_run_with_complex_query(self, workflow, mock_llm):
        """Test workflow with complex query"""
        mock_llm.invoke.return_value.content = "0.8"
        
        result = workflow.run("Compare A and B in terms of X, Y, and Z")
        
        # Should complete with complex strategy
        assert result.complexity >= 0.4
        assert result.strategy in [Strategy.MULTIHOP, Strategy.GRAPH]


class TestWorkflowWithTrace:
    """Tests for traced workflow execution"""
    
    def test_run_with_trace_returns_all_fields(self, workflow):
        """Test traced run returns all expected fields"""
        trace = workflow.run_with_trace("What is Python?")
        
        assert "final_state" in trace
        assert "execution_path" in trace
        assert "node_outputs" in trace
        assert "total_nodes_executed" in trace
    
    def test_run_with_trace_execution_path(self, workflow):
        """Test execution path is recorded"""
        trace = workflow.run_with_trace("test")
        
        path = trace["execution_path"]
        
        # Should have executed these nodes
        assert "planner" in path
        assert "retrieval" in path
        assert "validator" in path
    
    def test_run_with_trace_node_outputs(self, workflow):
        """Test node outputs are captured"""
        trace = workflow.run_with_trace("test")
        
        outputs = trace["node_outputs"]
        
        assert "planner" in outputs
        assert "retrieval" in outputs
        assert "validator" in outputs
    
    def test_run_with_trace_planner_output(self, workflow):
        """Test planner output is captured correctly"""
        trace = workflow.run_with_trace("test")
        
        planner_output = trace["node_outputs"]["planner"]
        
        assert "complexity" in planner_output
        assert "strategy" in planner_output
    
    def test_run_with_trace_retrieval_attempts(self, workflow):
        """Test retrieval attempts are recorded"""
        trace = workflow.run_with_trace("test")
        
        retrieval_attempts = trace["node_outputs"]["retrieval"]
        
        assert isinstance(retrieval_attempts, list)
        assert len(retrieval_attempts) > 0
        assert "round" in retrieval_attempts[0]
        assert "chunk_count" in retrieval_attempts[0]


class TestWorkflowRetryLogic:
    """Tests for retry loop functionality"""
    
    def test_workflow_retries_on_low_validation(self, workflow, mock_llm):
        """Test workflow retries when validation fails"""
        # First validation fails
        mock_llm.invoke.side_effect = [
            Mock(content="0.5"),  # Planner
            Mock(content="0.4"),  # First validation (fail)
            Mock(content="0.8"),  # Second validation (pass)
        ]
        
        trace = workflow.run_with_trace("test")
        
        # Should have multiple retrieval attempts
        retrieval_attempts = trace["node_outputs"]["retrieval"]
        assert len(retrieval_attempts) >= 1
    
    def test_workflow_stops_after_max_retries(self, planner, coordinator):
        """Test workflow stops after max retries"""
        # Create validator with max_retries=1
        mock_llm = Mock()
        validator = ValidatorAgent(llm=mock_llm, threshold=0.9, max_retries=1)
        
        workflow = AgenticRAGWorkflow(planner, coordinator, validator)
        
        # Always return low validation score
        mock_llm.invoke.return_value.content = "0.3"
        
        trace = workflow.run_with_trace("test")
        
        # Should stop after max retries
        retrieval_attempts = trace["node_outputs"]["retrieval"]
        assert len(retrieval_attempts) <= 2  # Initial + 1 retry


class TestWorkflowInfo:
    """Tests for workflow introspection"""
    
    def test_get_workflow_info_returns_structure(self, workflow):
        """Test get_workflow_info returns workflow structure"""
        info = workflow.get_workflow_info()
        
        assert "nodes" in info
        assert "edges" in info
        assert "retry_logic" in info
        assert "max_retries" in info
    
    def test_get_workflow_info_nodes(self, workflow):
        """Test workflow info includes all nodes"""
        info = workflow.get_workflow_info()
        
        nodes = info["nodes"]
        assert "planner" in nodes
        assert "retrieval" in nodes
        assert "validator" in nodes
    
    def test_get_workflow_info_edges(self, workflow):
        """Test workflow info includes edges"""
        info = workflow.get_workflow_info()
        
        edges = info["edges"]
        assert "fixed" in edges
        assert "conditional" in edges


class TestWorkflowErrorHandling:
    """Tests for error handling"""
    
    def test_workflow_handles_planner_error(self, coordinator, validator):
        """Test workflow handles planner failure"""
        # Create failing planner
        failing_planner = Mock()
        failing_planner.run.side_effect = Exception("Planner failed")
        
        workflow = AgenticRAGWorkflow(failing_planner, coordinator, validator)
        
        with pytest.raises(OrchestrationError) as exc_info:
            workflow.run("test")
        
        assert exc_info.value.node_name == "planner"
    
    def test_workflow_handles_retrieval_error(self, planner, validator):
        """Test workflow handles retrieval failure"""
        # Create failing coordinator
        failing_coordinator = Mock()
        failing_coordinator.run.side_effect = Exception("Retrieval failed")
        
        workflow = AgenticRAGWorkflow(planner, failing_coordinator, validator)
        
        with pytest.raises(OrchestrationError) as exc_info:
            workflow.run("test")
        
        assert exc_info.value.node_name == "retrieval"
    
    def test_workflow_handles_validator_error(self, planner, coordinator):
        """Test workflow handles validator failure"""
        # Create failing validator
        failing_validator = Mock()
        failing_validator.run.side_effect = Exception("Validator failed")
        
        workflow = AgenticRAGWorkflow(planner, coordinator, failing_validator)
        
        with pytest.raises(OrchestrationError) as exc_info:
            workflow.run("test")
        
        assert exc_info.value.node_name == "validator"


class TestWorkflowStateManagement:
    """Tests for state management through workflow"""
    
    def test_state_preserved_through_workflow(self, workflow):
        """Test state is preserved and updated through workflow"""
        result = workflow.run("test query")
        
        # Original query preserved
        assert result.query == "test query"
        
        # All fields populated
        assert result.complexity is not None
        assert result.strategy is not None
        assert len(result.chunks) > 0
        assert result.validation_status is not None
    
    def test_metadata_accumulated(self, workflow):
        """Test metadata accumulates from all agents"""
        result = workflow.run("test")
        
        # Should have metadata from all agents
        assert "planner" in result.metadata
        assert "retrieval_coordinator" in result.metadata
        assert "validator" in result.metadata