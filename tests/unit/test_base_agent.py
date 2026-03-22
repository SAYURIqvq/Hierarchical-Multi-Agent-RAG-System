"""
Tests for BaseAgent abstract class.

This module tests the base agent functionality including:
- Abstract method enforcement
- Metrics tracking
- Error handling
- Logging integration
"""

import pytest
import time
from src.agents.base_agent import BaseAgent
from src.models.agent_state import AgentState
from src.utils.exceptions import AgentExecutionError


class DummyAgent(BaseAgent):
    """
    Concrete implementation of BaseAgent for testing.
    
    This simple agent just adds metadata to the state to demonstrate
    the agent execution pattern.
    """
    
    def __init__(self, name: str = "dummy", should_fail: bool = False):
        """
        Initialize dummy agent.
        
        Args:
            name: Agent name
            should_fail: If True, execute() will raise an error
        """
        super().__init__(name=name, version="1.0.0")
        self.should_fail = should_fail
        self.execution_count = 0
    
    def execute(self, state: AgentState) -> AgentState:
        """
        Execute dummy agent logic.
        
        Args:
            state: Current state
        
        Returns:
            Updated state with metadata
        
        Raises:
            ValueError: If should_fail is True
        """
        self.execution_count += 1
        
        if self.should_fail:
            raise ValueError("Intentional failure for testing")
        
        # Add metadata to demonstrate state modification
        state.metadata["processed_by"] = self.name
        state.metadata["execution_count"] = self.execution_count
        
        return state


class SlowAgent(BaseAgent):
    """
    Agent that simulates slow processing for timing tests.
    """
    
    def __init__(self, delay: float = 0.1):
        """
        Initialize slow agent.
        
        Args:
            delay: Processing delay in seconds
        """
        super().__init__(name="slow_agent", version="1.0.0")
        self.delay = delay
    
    def execute(self, state: AgentState) -> AgentState:
        """Execute with artificial delay."""
        time.sleep(self.delay)
        state.metadata["processed"] = True
        return state


class TestBaseAgentAbstraction:
    """Tests for abstract class behavior"""
    
    def test_cannot_instantiate_base_agent_directly(self):
        """Test that BaseAgent cannot be instantiated directly"""
        with pytest.raises(TypeError) as exc_info:
            BaseAgent(name="test")
        
        assert "abstract" in str(exc_info.value).lower()
    
    def test_must_implement_execute_method(self):
        """Test that subclasses must implement execute()"""
        
        class IncompleteAgent(BaseAgent):
            """Agent without execute() implementation"""
            pass
        
        with pytest.raises(TypeError) as exc_info:
            IncompleteAgent(name="incomplete")
        
        assert "abstract" in str(exc_info.value).lower()
    
    def test_concrete_implementation_works(self):
        """Test that proper implementation can be instantiated"""
        agent = DummyAgent(name="test_agent")
        
        assert agent.name == "test_agent"
        assert agent.version == "1.0.0"
        assert hasattr(agent, 'execute')


class TestBaseAgentInitialization:
    """Tests for agent initialization"""
    
    def test_agent_initialization_with_defaults(self):
        """Test agent initializes with default values"""
        agent = DummyAgent(name="test")
        
        assert agent.name == "test"
        assert agent.version == "1.0.0"
        assert agent.logger is not None
        assert agent.metrics is not None
    
    def test_agent_initialization_with_custom_version(self):
        """Test agent initialization with custom version"""
        
        class CustomAgent(BaseAgent):
            def __init__(self):
                super().__init__(name="custom", version="2.5.1")
            
            def execute(self, state: AgentState) -> AgentState:
                return state
        
        agent = CustomAgent()
        assert agent.version == "2.5.1"
    
    def test_initial_metrics_are_zero(self):
        """Test that metrics start at zero"""
        agent = DummyAgent()
        metrics = agent.get_metrics()
        
        assert metrics["total_calls"] == 0
        assert metrics["successful_calls"] == 0
        assert metrics["failed_calls"] == 0
        assert metrics["total_time_seconds"] == 0.0
        assert metrics["average_time_seconds"] == 0.0
    
    def test_agent_has_created_at_timestamp(self):
        """Test that agent tracks creation time"""
        agent = DummyAgent()
        metrics = agent.get_metrics()
        
        assert "created_at" in metrics
        assert metrics["created_at"] is not None


class TestBaseAgentExecution:
    """Tests for agent execution"""
    
    def test_execute_method_is_called(self):
        """Test that execute() is called during run()"""
        agent = DummyAgent(name="test")
        state = AgentState(query="test query")
        
        result = agent.run(state)
        
        assert agent.execution_count == 1
        assert result.metadata["processed_by"] == "test"
    
    def test_execute_returns_updated_state(self):
        """Test that execute() can modify state"""
        agent = DummyAgent(name="test")
        state = AgentState(query="test query")
        
        result = agent.run(state)
        
        assert result.query == "test query"  # Original preserved
        assert "processed_by" in result.metadata  # New data added
    
    def test_multiple_executions(self):
        """Test multiple executions work correctly"""
        agent = DummyAgent(name="test")
        
        for i in range(3):
            state = AgentState(query=f"query {i}")
            result = agent.run(state)
            assert result.metadata["execution_count"] == i + 1
        
        assert agent.execution_count == 3


class TestBaseAgentMetrics:
    """Tests for metrics tracking"""
    
    def test_metrics_updated_on_success(self):
        """Test that metrics are updated on successful execution"""
        agent = DummyAgent(name="test")
        state = AgentState(query="test")
        
        agent.run(state)
        metrics = agent.get_metrics()
        
        assert metrics["total_calls"] == 1
        assert metrics["successful_calls"] == 1
        assert metrics["failed_calls"] == 0
        assert metrics["success_rate"] == 100.0
    
    def test_metrics_updated_on_failure(self):
        """Test that metrics are updated on failed execution"""
        agent = DummyAgent(name="test", should_fail=True)
        state = AgentState(query="test")
        
        with pytest.raises(AgentExecutionError):
            agent.run(state)
        
        metrics = agent.get_metrics()
        assert metrics["total_calls"] == 1
        assert metrics["successful_calls"] == 0
        assert metrics["failed_calls"] == 1
        assert metrics["success_rate"] == 0.0
    
    def test_success_rate_calculation(self):
        """Test success rate is calculated correctly"""
        agent = DummyAgent(name="test")
        
        # 3 successful
        for _ in range(3):
            state = AgentState(query="test")
            agent.run(state)
        
        # 1 failed
        agent.should_fail = True
        state = AgentState(query="test")
        with pytest.raises(AgentExecutionError):
            agent.run(state)
        
        metrics = agent.get_metrics()
        assert metrics["total_calls"] == 4
        assert metrics["successful_calls"] == 3
        assert metrics["failed_calls"] == 1
        assert metrics["success_rate"] == 75.0
    
    def test_timing_metrics(self):
        """Test that execution time is tracked"""
        agent = SlowAgent(delay=0.1)
        state = AgentState(query="test")
        
        agent.run(state)
        metrics = agent.get_metrics()
        
        assert metrics["total_time_seconds"] >= 0.1
        assert metrics["average_time_seconds"] >= 0.1
        assert metrics["last_execution_time"] >= 0.1
    
    def test_average_time_calculation(self):
        """Test average time is calculated correctly"""
        agent = DummyAgent(name="test")
        
        # Run multiple times
        for _ in range(3):
            state = AgentState(query="test")
            agent.run(state)
        
        metrics = agent.get_metrics()
        assert metrics["average_time_seconds"] > 0
        assert metrics["total_calls"] == 3
    
    def test_reset_metrics(self):
        """Test that metrics can be reset"""
        agent = DummyAgent(name="test")
        state = AgentState(query="test")
        
        # Execute a few times
        agent.run(state)
        agent.run(state)
        
        # Reset
        agent.reset_metrics()
        metrics = agent.get_metrics()
        
        assert metrics["total_calls"] == 0
        assert metrics["successful_calls"] == 0
        assert metrics["failed_calls"] == 0
        assert metrics["total_time_seconds"] == 0.0
        assert "created_at" in metrics  # Preserved


class TestBaseAgentErrorHandling:
    """Tests for error handling"""
    
    def test_error_wrapped_in_agent_execution_error(self):
        """Test that errors are wrapped in AgentExecutionError"""
        agent = DummyAgent(name="test", should_fail=True)
        state = AgentState(query="test")
        
        with pytest.raises(AgentExecutionError) as exc_info:
            agent.run(state)
        
        error = exc_info.value
        assert error.agent_name == "test"
        assert "Intentional failure" in str(error)
    
    def test_error_contains_details(self):
        """Test that error contains execution details"""
        agent = DummyAgent(name="test", should_fail=True)
        state = AgentState(query="test query")
        
        with pytest.raises(AgentExecutionError) as exc_info:
            agent.run(state)
        
        error = exc_info.value
        assert error.details is not None
        assert "execution_time" in error.details
        assert "query" in error.details
        assert error.details["query"] == "test query"
    
    def test_agent_execution_error_not_double_wrapped(self):
        """Test that AgentExecutionError is not wrapped again"""
        
        class ErrorAgent(BaseAgent):
            def execute(self, state: AgentState) -> AgentState:
                raise AgentExecutionError(
                    agent_name="error_agent",
                    message="Already wrapped"
                )
        
        agent = ErrorAgent(name="error_agent")
        state = AgentState(query="test")
        
        with pytest.raises(AgentExecutionError) as exc_info:
            agent.run(state)
        
        # Should be the same error, not wrapped again
        assert exc_info.value.agent_name == "error_agent"
        assert "Already wrapped" in str(exc_info.value)


class TestBaseAgentLogging:
    """Tests for logging functionality"""
    
    def test_agent_has_logger(self):
        """Test that agent has logger instance"""
        agent = DummyAgent(name="test")
        assert agent.logger is not None
    
    def test_log_method_works(self):
        """Test that log() method doesn't raise errors"""
        agent = DummyAgent(name="test")
        
        # Should not raise
        agent.log("Test message", level="info")
        agent.log("Debug message", level="debug")
        agent.log("Warning message", level="warning")
    
    def test_logger_name_includes_agent_name(self):
        """Test that logger name includes agent name"""
        agent = DummyAgent(name="test_agent")
        assert "test_agent" in agent.logger.name


class TestBaseAgentHelperMethods:
    """Tests for helper methods"""
    
    def test_get_info(self):
        """Test get_info() returns agent information"""
        agent = DummyAgent(name="test")
        info = agent.get_info()
        
        assert info["name"] == "test"
        assert info["version"] == "1.0.0"
        assert info["class"] == "DummyAgent"
    
    def test_repr(self):
        """Test __repr__ returns meaningful string"""
        agent = DummyAgent(name="test")
        repr_str = repr(agent)
        
        assert "DummyAgent" in repr_str
        assert "test" in repr_str
        assert "1.0.0" in repr_str
    
    def test_str(self):
        """Test __str__ returns user-friendly string"""
        agent = DummyAgent(name="test")
        str_repr = str(agent)
        
        assert "test" in str_repr
        assert "1.0.0" in str_repr


class TestBaseAgentIntegration:
    """Integration tests with other components"""
    
    def test_agent_state_integration(self):
        """Test that agent works with AgentState"""
        agent = DummyAgent(name="test")
        state = AgentState(
            query="test query",
            complexity=0.5,
            chunks=[]
        )
        
        result = agent.run(state)
        
        assert isinstance(result, AgentState)
        assert result.query == "test query"
        assert result.complexity == 0.5
    
    def test_multiple_agents_chain(self):
        """Test multiple agents can process same state"""
        agent1 = DummyAgent(name="agent1")
        agent2 = DummyAgent(name="agent2")
        
        state = AgentState(query="test")
        
        # Chain execution
        state = agent1.run(state)
        state = agent2.run(state)
        
        # Both agents processed the state
        assert state.metadata.get("processed_by") == "agent2"
        assert agent1.execution_count == 1
        assert agent2.execution_count == 1
    
    def test_metrics_independent_between_agents(self):
        """Test that each agent has independent metrics"""
        agent1 = DummyAgent(name="agent1")
        agent2 = DummyAgent(name="agent2")
        state = AgentState(query="test")
        
        # Execute agent1 twice
        agent1.run(state)
        agent1.run(state)
        
        # Execute agent2 once
        agent2.run(state)
        
        metrics1 = agent1.get_metrics()
        metrics2 = agent2.get_metrics()
        
        assert metrics1["total_calls"] == 2
        assert metrics2["total_calls"] == 1