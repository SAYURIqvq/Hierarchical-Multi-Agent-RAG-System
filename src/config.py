"""
Configuration management for Agentic RAG System.
"""

import os
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator, ConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # ===== API Keys =====
    anthropic_api_key: str = Field(..., description="Anthropic API key for Claude", min_length=10)
    voyage_api_key: str = Field(..., description="Voyage AI API key for embeddings", min_length=10)
    cohere_api_key: Optional[str] = Field(None, description="Cohere API key for reranking (optional)")
    
    # ===== Database Configuration =====
    database_url: str = Field(default="postgresql://localhost:5432/agentic_rag", description="PostgreSQL connection string")
    redis_url: str = Field(default="redis://localhost:6379", description="Redis connection string")
    chroma_persist_dir: str = Field(default="./data/chroma", description="ChromaDB persistence directory")
    collection_name: str = Field(default="documents", description="ChromaDB collection name")
    vector_search_top_k: int = Field(default=10, description="Number of chunks for vector search", gt=0)
    
    # ===== Model Configuration =====
    llm_model: str = Field(default="claude-3-haiku-20240307", description="Claude model to use")
    llm_temperature: float = Field(default=0.0, description="LLM temperature (0.0-1.0)", ge=0.0, le=1.0)
    llm_max_tokens: int = Field(default=4096, description="Maximum tokens for LLM response", gt=0)
    embedding_model: str = Field(default="voyage-large-2", description="Voyage AI embedding model")
    embedding_dimension: int = Field(default=1536, description="Embedding vector dimension", gt=0)
    
    # ===== Chunking Configuration =====
    chunk_size: int = Field(default=500, description="Document chunk size in tokens", gt=0)
    chunk_overlap: int = Field(default=50, description="Overlap between chunks in tokens", ge=0)
    parent_chunk_size: int = Field(default=2000, description="Parent chunk size for hierarchical chunking", gt=0)
    
    # ===== Retrieval Configuration =====
    retrieval_top_k: int = Field(default=10, description="Number of chunks to retrieve", gt=0)
    vector_search_weight: float = Field(default=0.7, description="Weight for vector search in hybrid ranking", ge=0.0, le=1.0)
    keyword_search_weight: float = Field(default=0.3, description="Weight for keyword search in hybrid ranking", ge=0.0, le=1.0)
    
    # ===== Agent Configuration =====
    planner_complexity_threshold_simple: float = Field(default=0.3, description="Threshold for simple query classification", ge=0.0, le=1.0)
    planner_complexity_threshold_multihop: float = Field(default=0.7, description="Threshold for multihop query classification", ge=0.0, le=1.0)
    validator_threshold: float = Field(default=0.7, description="Validation sufficiency threshold", ge=0.0, le=1.0)
    validator_max_retries: int = Field(default=2, description="Maximum retrieval retry attempts", ge=0)
    critic_max_iterations: int = Field(default=3, description="Maximum critic regeneration iterations", ge=1)
    
    # ===== Cache Configuration =====
    cache_enabled: bool = Field(default=True, description="Enable Redis caching")
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds (1 hour default)", gt=0)
    
    # ===== System Configuration =====
    log_level: str = Field(default="INFO", description="Logging level")
    environment: str = Field(default="development", description="Runtime environment")
    max_file_size_mb: int = Field(default=50, description="Maximum upload file size in MB", gt=0)
    allowed_file_types: str = Field(default="pdf,docx,txt", description="Allowed upload file types (comma-separated)")
    
    # ===== Performance Configuration =====
    batch_size: int = Field(default=128, description="Batch size for embedding generation", gt=0)
    parallel_retrieval: bool = Field(default=True, description="Enable parallel retrieval agents")
    request_timeout: int = Field(default=30, description="Request timeout in seconds", gt=0)
    
    @field_validator("keyword_search_weight")
    @classmethod
    def validate_weights_sum(cls, v, info):
        """Validate that vector and keyword weights sum to 1.0."""
        if info.data.get("vector_search_weight") is not None:
            vector_weight = info.data["vector_search_weight"]
            total = vector_weight + v
            if abs(total - 1.0) > 0.01:
                raise ValueError(
                    f"vector_search_weight ({vector_weight}) + "
                    f"keyword_search_weight ({v}) must sum to 1.0"
                )
        return v
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level is valid."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"log_level must be one of: {valid_levels}")
        return v_upper
    
    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v):
        """Validate environment is valid."""
        valid_envs = ["development", "production", "test"]
        v_lower = v.lower()
        if v_lower not in valid_envs:
            raise ValueError(f"environment must be one of: {valid_envs}")
        return v_lower
    
    def get_allowed_file_types_list(self) -> List[str]:
        """Get allowed file types as list."""
        return [ext.strip() for ext in self.allowed_file_types.split(",")]
    
    def get_database_config(self) -> dict:
        """Get database configuration as dictionary."""
        return {
            "url": self.database_url,
            "pool_size": 10,
            "max_overflow": 20,
            "pool_timeout": 30
        }
    
    def get_redis_config(self) -> dict:
        """Get Redis configuration as dictionary."""
        return {
            "url": self.redis_url,
            "decode_responses": True,
            "socket_timeout": 5,
            "socket_connect_timeout": 5
        }
    
    def get_chroma_config(self) -> dict:
        """Get ChromaDB configuration as dictionary."""
        return {
            "persist_directory": self.chroma_persist_dir,
            "embedding_dimension": self.embedding_dimension
        }
    
    def get_llm_config(self) -> dict:
        """Get LLM configuration as dictionary."""
        return {
            "model": self.llm_model,
            "temperature": self.llm_temperature,
            "max_tokens": self.llm_max_tokens,
            "api_key": self.anthropic_api_key
        }
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get global settings instance."""
    return settings


def reload_settings() -> Settings:
    """Reload settings from environment."""
    global settings
    settings = Settings()
    return settings