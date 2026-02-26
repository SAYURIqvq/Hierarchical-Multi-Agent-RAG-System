"""
Custom exceptions for Agentic RAG System.

This module defines all custom exceptions used throughout the system.
Each exception type represents a specific error category for better
error handling and debugging.

Exception Hierarchy:
    AgenticRAGException (base)
    ├── AgentExecutionError
    ├── RetrievalError
    ├── ValidationError
    ├── GenerationError
    └── OrchestrationError
"""


class AgenticRAGException(Exception):
    """
    Base exception for all Agentic RAG errors.
    
    All custom exceptions in the system should inherit from this class.
    This allows catching all system-specific errors with a single except clause.
    
    Args:
        message: Error message describing what went wrong
        details: Optional dictionary with additional error context
    
    Attributes:
        message: The error message
        details: Additional error details (optional)
    
    Example:
        >>> raise AgenticRAGException("Something went wrong", details={"code": 500})
    """
    
    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)
    
    def __str__(self):
        """String representation of the exception"""
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class AgentExecutionError(AgenticRAGException):
    """
    Raised when an agent fails to execute its task.
    
    This exception indicates that an agent encountered an error during
    its execution phase. The error could be due to invalid state,
    processing failures, or internal logic errors.
    
    Args:
        agent_name: Name of the agent that failed
        message: Description of what went wrong
        details: Optional error context (state, stack trace, etc)
    
    Attributes:
        agent_name: The agent that encountered the error
        message: Error description
        details: Additional context
    
    Example:
        >>> raise AgentExecutionError(
        ...     agent_name="planner",
        ...     message="Failed to analyze complexity",
        ...     details={"query": "...", "error": "..."}
        ... )
    """
    
    def __init__(self, agent_name: str, message: str, details: dict = None):
        self.agent_name = agent_name
        full_message = f"Agent '{agent_name}' execution failed: {message}"
        super().__init__(full_message, details)


class RetrievalError(AgenticRAGException):
    """
    Raised when document retrieval fails.
    
    This exception indicates failures in the retrieval phase, such as:
    - Vector store connection errors
    - Search failures
    - Empty results when content was expected
    - Timeout errors
    
    Args:
        message: Description of the retrieval failure
        retrieval_type: Type of retrieval that failed (vector/keyword/graph)
        details: Optional error context
    
    Attributes:
        retrieval_type: The type of retrieval that failed
        message: Error description
        details: Additional context
    
    Example:
        >>> raise RetrievalError(
        ...     message="Vector store connection timeout",
        ...     retrieval_type="vector",
        ...     details={"timeout": 30, "host": "localhost"}
        ... )
    """
    
    def __init__(self, message: str, retrieval_type: str = None, details: dict = None):
        self.retrieval_type = retrieval_type
        if retrieval_type:
            full_message = f"Retrieval failed ({retrieval_type}): {message}"
        else:
            full_message = f"Retrieval failed: {message}"
        super().__init__(full_message, details)


class ValidationError(AgenticRAGException):
    """
    Raised when validation checks fail.
    
    This exception indicates that retrieved content or generated answers
    failed validation checks. Could be due to:
    - Insufficient retrieved chunks
    - Low quality scores
    - Missing required information
    - Content that doesn't match query
    
    Args:
        message: Description of validation failure
        validation_type: Type of validation that failed
        score: Optional validation score that failed threshold
        details: Optional error context
    
    Attributes:
        validation_type: What kind of validation failed
        score: The validation score (if applicable)
        message: Error description
        details: Additional context
    
    Example:
        >>> raise ValidationError(
        ...     message="Insufficient chunk quality",
        ...     validation_type="sufficiency",
        ...     score=0.45,
        ...     details={"threshold": 0.7, "chunks": 3}
        ... )
    """
    
    def __init__(
        self, 
        message: str, 
        validation_type: str = None, 
        score: float = None,
        details: dict = None
    ):
        self.validation_type = validation_type
        self.score = score
        
        full_message = f"Validation failed: {message}"
        if validation_type:
            full_message = f"Validation failed ({validation_type}): {message}"
        if score is not None:
            full_message += f" [score: {score:.2f}]"
        
        super().__init__(full_message, details)


class GenerationError(AgenticRAGException):
    """
    Raised when answer generation fails.
    
    This exception indicates failures during the answer generation phase:
    - LLM API errors
    - Invalid responses
    - Token limit exceeded
    - Generation quality issues
    
    Args:
        message: Description of generation failure
        llm_error: Optional underlying LLM error message
        details: Optional error context
    
    Attributes:
        llm_error: The underlying LLM error (if any)
        message: Error description
        details: Additional context
    
    Example:
        >>> raise GenerationError(
        ...     message="Failed to generate answer",
        ...     llm_error="Rate limit exceeded",
        ...     details={"model": "claude-3-5-sonnet", "tokens": 100000}
        ... )
    """
    
    def __init__(self, message: str, llm_error: str = None, details: dict = None):
        self.llm_error = llm_error
        
        full_message = f"Generation failed: {message}"
        if llm_error:
            full_message += f" | LLM Error: {llm_error}"
        
        super().__init__(full_message, details)


class OrchestrationError(AgenticRAGException):
    """
    Raised when workflow orchestration fails.
    
    This exception indicates failures in the LangGraph workflow:
    - State transition errors
    - Invalid workflow configuration
    - Node execution failures
    - Circular dependencies
    
    Args:
        message: Description of orchestration failure
        node_name: Optional name of the failing node
        details: Optional error context
    
    Attributes:
        node_name: The workflow node that failed (if applicable)
        message: Error description
        details: Additional context
    
    Example:
        >>> raise OrchestrationError(
        ...     message="Invalid state transition",
        ...     node_name="validator",
        ...     details={"from": "retrieval", "to": "generator"}
        ... )
    """
    
    def __init__(self, message: str, node_name: str = None, details: dict = None):
        self.node_name = node_name
        
        full_message = f"Orchestration failed: {message}"
        if node_name:
            full_message = f"Orchestration failed at node '{node_name}': {message}"
        
        super().__init__(full_message, details)


class ConfigurationError(AgenticRAGException):
    """
    Raised when system configuration is invalid.
    
    This exception indicates problems with system configuration:
    - Missing required environment variables
    - Invalid API keys
    - Missing required dependencies
    - Invalid configuration values
    
    Args:
        message: Description of configuration issue
        config_key: Optional configuration key that's problematic
        details: Optional error context
    
    Attributes:
        config_key: The configuration key that's invalid
        message: Error description
        details: Additional context
    
    Example:
        >>> raise ConfigurationError(
        ...     message="Missing API key",
        ...     config_key="ANTHROPIC_API_KEY",
        ...     details={"required": True, "found": False}
        ... )
    """
    
    def __init__(self, message: str, config_key: str = None, details: dict = None):
        self.config_key = config_key
        
        full_message = f"Configuration error: {message}"
        if config_key:
            full_message = f"Configuration error for '{config_key}': {message}"
        
        super().__init__(full_message, details)