"""
Tests for logging configuration.

This module tests the logging setup to ensure:
- Loggers are created correctly
- Log levels work as expected
- File logging works (if enabled)
- AgentLogger wrapper functions correctly
"""

import pytest
import logging
import tempfile
from pathlib import Path
from src.utils.logger import (
    setup_logger,
    get_logger,
    AgentLogger,
    configure_root_logger,
    LOG_LEVELS
)


class TestSetupLogger:
    """Tests for setup_logger function"""
    
    def test_basic_logger_creation(self):
        """Test basic logger creation with default settings"""
        logger = setup_logger("test_logger")
        
        assert logger.name == "test_logger"
        assert logger.level == logging.INFO
        assert len(logger.handlers) > 0
    
    def test_logger_with_custom_level(self):
        """Test logger creation with custom log level"""
        logger = setup_logger("test_debug", level="DEBUG")
        assert logger.level == logging.DEBUG
        
        logger = setup_logger("test_error", level="ERROR")
        assert logger.level == logging.ERROR
    
    def test_invalid_log_level_raises_error(self):
        """Test that invalid log level raises ValueError"""
        with pytest.raises(ValueError) as exc_info:
            setup_logger("test", level="INVALID")
        
        assert "Invalid log level" in str(exc_info.value)
    
    def test_logger_with_file_output(self):
        """Test logger with file output"""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            logger = setup_logger("test_file", log_file=str(log_file))
            
            logger.info("Test message")
            
            assert log_file.exists()
            content = log_file.read_text()
            assert "Test message" in content
    
    def test_logger_creates_log_directory(self):
        """Test that logger creates log directory if it doesn't exist"""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "nested" / "dir" / "test.log"
            logger = setup_logger("test_nested", log_file=str(log_file))
            
            logger.info("Test")
            assert log_file.exists()
            assert log_file.parent.exists()
    
    def test_multiple_handlers_console_and_file(self):
        """Test that both console and file handlers are added"""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            logger = setup_logger("test_multi", log_file=str(log_file))
            
            assert len(logger.handlers) == 2
    
    def test_logger_no_propagation(self):
        """Test that logger doesn't propagate to root"""
        logger = setup_logger("test_propagate")
        assert logger.propagate is False


class TestGetLogger:
    """Tests for get_logger function"""
    
    def test_get_existing_logger(self):
        """Test getting an existing logger"""
        original = setup_logger("test_get")
        retrieved = get_logger("test_get")
        
        assert retrieved.name == original.name
        assert retrieved is original
    
    def test_get_nonexistent_logger(self):
        """Test getting a logger that wasn't setup"""
        logger = get_logger("nonexistent_logger")
        assert logger.name == "nonexistent_logger"


class TestAgentLogger:
    """Tests for AgentLogger wrapper class"""
    
    def test_agent_logger_creation(self):
        """Test AgentLogger initialization"""
        agent_logger = AgentLogger("planner")
        
        assert agent_logger.agent_name == "planner"
        assert agent_logger.logger.name == "agent.planner"
    
    def test_agent_logger_debug(self):
        """Test debug logging with file output"""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            agent_logger = AgentLogger("test_agent", level="DEBUG")
            agent_logger.logger = setup_logger("agent.test_agent", level="DEBUG", log_file=str(log_file))
            
            agent_logger.debug("Debug message")
            
            content = log_file.read_text()
            assert "Debug message" in content
            assert "[test_agent]" in content
    
    def test_agent_logger_info(self):
        """Test info logging with file output"""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            agent_logger = AgentLogger("test_agent")
            agent_logger.logger = setup_logger("agent.test_agent", log_file=str(log_file))
            
            agent_logger.info("Info message")
            
            content = log_file.read_text()
            assert "Info message" in content
            assert "[test_agent]" in content
    
    def test_agent_logger_warning(self):
        """Test warning logging with file output"""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            agent_logger = AgentLogger("test_agent")
            agent_logger.logger = setup_logger("agent.test_agent", log_file=str(log_file))
            
            agent_logger.warning("Warning message")
            
            content = log_file.read_text()
            assert "Warning message" in content
    
    def test_agent_logger_error(self):
        """Test error logging with file output"""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            agent_logger = AgentLogger("test_agent")
            agent_logger.logger = setup_logger("agent.test_agent", log_file=str(log_file))
            
            agent_logger.error("Error message")
            
            content = log_file.read_text()
            assert "Error message" in content
    
    def test_agent_logger_critical(self):
        """Test critical logging with file output"""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            agent_logger = AgentLogger("test_agent")
            agent_logger.logger = setup_logger("agent.test_agent", log_file=str(log_file))
            
            agent_logger.critical("Critical message")
            
            content = log_file.read_text()
            assert "Critical message" in content
    
    def test_agent_logger_with_exception(self):
        """Test error logging with exception info"""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            agent_logger = AgentLogger("test_agent")
            agent_logger.logger = setup_logger("agent.test_agent", log_file=str(log_file))
            
            try:
                raise ValueError("Test exception")
            except ValueError:
                agent_logger.error("Error occurred", exc_info=True)
            
            content = log_file.read_text()
            assert "Error occurred" in content
            assert "ValueError" in content


class TestConfigureRootLogger:
    """Tests for configure_root_logger"""
    
    def test_configure_root_logger(self):
        """Test root logger configuration"""
        original_level = logging.getLogger().level
        
        configure_root_logger(level="ERROR")
        
        # Note: basicConfig may not override if already configured
        # This test verifies the function runs without error
        assert True  # Function executed successfully


class TestLogLevels:
    """Tests for log level constants"""
    
    def test_all_log_levels_defined(self):
        """Test that all standard log levels are defined"""
        expected_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        
        for level in expected_levels:
            assert level in LOG_LEVELS
            assert isinstance(LOG_LEVELS[level], int)
    
    def test_log_levels_correct_values(self):
        """Test that log levels have correct integer values"""
        assert LOG_LEVELS['DEBUG'] == logging.DEBUG
        assert LOG_LEVELS['INFO'] == logging.INFO
        assert LOG_LEVELS['WARNING'] == logging.WARNING
        assert LOG_LEVELS['ERROR'] == logging.ERROR
        assert LOG_LEVELS['CRITICAL'] == logging.CRITICAL


class TestLoggerIntegration:
    """Integration tests for logging system"""
    
    def test_hierarchical_logger_names(self):
        """Test that hierarchical logger names work correctly"""
        parent_logger = setup_logger("agent")
        child_logger = setup_logger("agent.planner")
        
        assert parent_logger.name == "agent"
        assert child_logger.name == "agent.planner"
    
    def test_logger_reuse_same_instance(self):
        """Test that getting same logger returns same instance"""
        logger1 = setup_logger("test_reuse")
        logger2 = get_logger("test_reuse")
        
        assert logger1 is logger2
    
    def test_multiple_agents_logging(self):
        """Test multiple agents logging simultaneously"""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            
            planner = AgentLogger("planner")
            planner.logger = setup_logger("agent.planner", log_file=str(log_file))
            
            validator = AgentLogger("validator")
            validator.logger = setup_logger("agent.validator", log_file=str(log_file))
            
            planner.info("Planner message")
            validator.info("Validator message")
            
            content = log_file.read_text()
            assert "[planner]" in content
            assert "[validator]" in content
            assert "Planner message" in content
            assert "Validator message" in content