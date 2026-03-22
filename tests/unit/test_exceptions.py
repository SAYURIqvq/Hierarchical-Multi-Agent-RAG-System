"""
Tests for custom exceptions.

This module tests all custom exception classes to ensure they:
- Can be instantiated correctly
- Store attributes properly
- Display error messages correctly
- Maintain exception hierarchy
"""

import pytest
from src.utils.exceptions import (
    AgenticRAGException,
    AgentExecutionError,
    RetrievalError,
    ValidationError,
    GenerationError,
    OrchestrationError,
    ConfigurationError
)


class TestAgenticRAGException:
    """Tests for base AgenticRAGException"""
    
    def test_basic_exception(self):
        """Test basic exception creation"""
        exc = AgenticRAGException("Test error")
        assert str(exc) == "Test error"
        assert exc.message == "Test error"
        assert exc.details == {}
    
    def test_exception_with_details(self):
        """Test exception with details dictionary"""
        details = {"code": 500, "component": "test"}
        exc = AgenticRAGException("Test error", details=details)
        
        assert exc.message == "Test error"
        assert exc.details == details
        assert "Details:" in str(exc)
    
    def test_exception_can_be_raised(self):
        """Test exception can be raised and caught"""
        with pytest.raises(AgenticRAGException) as exc_info:
            raise AgenticRAGException("Test error")
        
        assert "Test error" in str(exc_info.value)


class TestAgentExecutionError:
    """Tests for AgentExecutionError"""
    
    def test_agent_execution_error_basic(self):
        """Test basic agent execution error"""
        exc = AgentExecutionError(
            agent_name="planner",
            message="Failed to analyze"
        )
        
        assert exc.agent_name == "planner"
        assert "planner" in str(exc)
        assert "Failed to analyze" in str(exc)
    
    def test_agent_execution_error_with_details(self):
        """Test agent execution error with details"""
        details = {"query": "test", "complexity": 0.5}
        exc = AgentExecutionError(
            agent_name="validator",
            message="Validation failed",
            details=details
        )
        
        assert exc.details == details
        assert exc.agent_name == "validator"
    
    def test_inheritance_from_base(self):
        """Test that AgentExecutionError inherits from base"""
        exc = AgentExecutionError("test_agent", "test message")
        assert isinstance(exc, AgenticRAGException)


class TestRetrievalError:
    """Tests for RetrievalError"""
    
    def test_retrieval_error_basic(self):
        """Test basic retrieval error"""
        exc = RetrievalError("Connection failed")
        assert "Connection failed" in str(exc)
        assert exc.retrieval_type is None
    
    def test_retrieval_error_with_type(self):
        """Test retrieval error with retrieval type"""
        exc = RetrievalError(
            message="Timeout",
            retrieval_type="vector"
        )
        
        assert exc.retrieval_type == "vector"
        assert "vector" in str(exc)
        assert "Timeout" in str(exc)
    
    def test_retrieval_error_with_details(self):
        """Test retrieval error with full details"""
        details = {"host": "localhost", "timeout": 30}
        exc = RetrievalError(
            message="Connection timeout",
            retrieval_type="keyword",
            details=details
        )
        
        assert exc.retrieval_type == "keyword"
        assert exc.details == details


class TestValidationError:
    """Tests for ValidationError"""
    
    def test_validation_error_basic(self):
        """Test basic validation error"""
        exc = ValidationError("Insufficient quality")
        assert "Insufficient quality" in str(exc)
    
    def test_validation_error_with_score(self):
        """Test validation error with score"""
        exc = ValidationError(
            message="Low quality",
            validation_type="sufficiency",
            score=0.45
        )
        
        assert exc.validation_type == "sufficiency"
        assert exc.score == 0.45
        assert "0.45" in str(exc)
    
    def test_validation_error_full(self):
        """Test validation error with all parameters"""
        details = {"threshold": 0.7, "chunks": 3}
        exc = ValidationError(
            message="Below threshold",
            validation_type="quality",
            score=0.5,
            details=details
        )
        
        assert exc.validation_type == "quality"
        assert exc.score == 0.5
        assert exc.details == details


class TestGenerationError:
    """Tests for GenerationError"""
    
    def test_generation_error_basic(self):
        """Test basic generation error"""
        exc = GenerationError("Failed to generate")
        assert "Failed to generate" in str(exc)
        assert exc.llm_error is None
    
    def test_generation_error_with_llm_error(self):
        """Test generation error with LLM error"""
        exc = GenerationError(
            message="Generation failed",
            llm_error="Rate limit exceeded"
        )
        
        assert exc.llm_error == "Rate limit exceeded"
        assert "Rate limit exceeded" in str(exc)
    
    def test_generation_error_full(self):
        """Test generation error with all parameters"""
        details = {"model": "claude-3-5-sonnet", "tokens": 100000}
        exc = GenerationError(
            message="Token limit exceeded",
            llm_error="Context too long",
            details=details
        )
        
        assert exc.llm_error == "Context too long"
        assert exc.details == details


class TestOrchestrationError:
    """Tests for OrchestrationError"""
    
    def test_orchestration_error_basic(self):
        """Test basic orchestration error"""
        exc = OrchestrationError("Workflow failed")
        assert "Workflow failed" in str(exc)
        assert exc.node_name is None
    
    def test_orchestration_error_with_node(self):
        """Test orchestration error with node name"""
        exc = OrchestrationError(
            message="State transition failed",
            node_name="validator"
        )
        
        assert exc.node_name == "validator"
        assert "validator" in str(exc)
    
    def test_orchestration_error_full(self):
        """Test orchestration error with all parameters"""
        details = {"from": "retrieval", "to": "generator"}
        exc = OrchestrationError(
            message="Invalid transition",
            node_name="validator",
            details=details
        )
        
        assert exc.node_name == "validator"
        assert exc.details == details


class TestConfigurationError:
    """Tests for ConfigurationError"""
    
    def test_configuration_error_basic(self):
        """Test basic configuration error"""
        exc = ConfigurationError("Invalid config")
        assert "Invalid config" in str(exc)
        assert exc.config_key is None
    
    def test_configuration_error_with_key(self):
        """Test configuration error with config key"""
        exc = ConfigurationError(
            message="Missing API key",
            config_key="ANTHROPIC_API_KEY"
        )
        
        assert exc.config_key == "ANTHROPIC_API_KEY"
        assert "ANTHROPIC_API_KEY" in str(exc)
    
    def test_configuration_error_full(self):
        """Test configuration error with all parameters"""
        details = {"required": True, "found": False}
        exc = ConfigurationError(
            message="Key not found",
            config_key="VOYAGE_API_KEY",
            details=details
        )
        
        assert exc.config_key == "VOYAGE_API_KEY"
        assert exc.details == details


class TestExceptionHierarchy:
    """Tests for exception inheritance"""
    
    def test_all_inherit_from_base(self):
        """Test that all custom exceptions inherit from base"""
        exceptions = [
            AgentExecutionError("test", "msg"),
            RetrievalError("msg"),
            ValidationError("msg"),
            GenerationError("msg"),
            OrchestrationError("msg"),
            ConfigurationError("msg")
        ]
        
        for exc in exceptions:
            assert isinstance(exc, AgenticRAGException)
    
    def test_can_catch_with_base_exception(self):
        """Test that all exceptions can be caught with base class"""
        with pytest.raises(AgenticRAGException):
            raise RetrievalError("Test error")
        
        with pytest.raises(AgenticRAGException):
            raise ValidationError("Test error")