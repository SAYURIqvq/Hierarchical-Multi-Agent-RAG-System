"""
Tests for configuration management.
"""

import pytest
from pydantic import ValidationError
from src.config import Settings, get_settings, reload_settings


class TestSettingsBasic:
    """Tests for Settings class basic functionality"""
    
    def test_production_settings_loads(self):
        """Test that production Settings class can load from .env"""
        try:
            settings = Settings()
            assert settings.anthropic_api_key is not None
            assert settings.voyage_api_key is not None
        except ValidationError as e:
            pytest.skip(f"Missing required API keys in .env: {e}")
    
    def test_settings_has_required_fields(self):
        """Test that Settings has all required fields"""
        try:
            settings = Settings()
            assert hasattr(settings, 'anthropic_api_key')
            assert hasattr(settings, 'voyage_api_key')
            assert hasattr(settings, 'chunk_size')
            assert hasattr(settings, 'log_level')
        except ValidationError:
            pytest.skip("Missing API keys in .env")
    
    def test_settings_default_values(self):
        """Test default values are set correctly"""
        try:
            settings = Settings()
            assert settings.chunk_size == 500
            assert settings.chunk_overlap == 50
            assert settings.log_level == "INFO"
            assert settings.environment == "development"
        except ValidationError:
            pytest.skip("Missing API keys in .env")


class TestValidation:
    """Tests for settings validation logic"""
    
    def test_temperature_field_constraints(self):
        """Test temperature field has correct constraints"""
        temp_field = Settings.model_fields['llm_temperature']
        
        # Check constraints exist in metadata
        constraints = temp_field.metadata
        has_ge = any(hasattr(c, 'ge') and c.ge == 0.0 for c in constraints)
        has_le = any(hasattr(c, 'le') and c.le == 1.0 for c in constraints)
        
        assert has_ge, "Temperature should have ge=0.0 constraint"
        assert has_le, "Temperature should have le=1.0 constraint"
    
    def test_chunk_size_field_constraints(self):
        """Test chunk_size must be positive"""
        chunk_field = Settings.model_fields['chunk_size']
        
        # Check gt constraint exists
        constraints = chunk_field.metadata
        has_gt = any(hasattr(c, 'gt') and c.gt == 0 for c in constraints)
        
        assert has_gt, "Chunk size should have gt=0 constraint"
    
    def test_validator_threshold_constraints(self):
        """Test validator_threshold is between 0 and 1"""
        field = Settings.model_fields['validator_threshold']
        
        # Check constraints exist
        constraints = field.metadata
        has_ge = any(hasattr(c, 'ge') and c.ge == 0.0 for c in constraints)
        has_le = any(hasattr(c, 'le') and c.le == 1.0 for c in constraints)
        
        assert has_ge, "Validator threshold should have ge=0.0 constraint"
        assert has_le, "Validator threshold should have le=1.0 constraint"


class TestConfigHelpers:
    """Tests for configuration helper methods"""
    
    def test_get_database_config(self):
        """Test get_database_config method exists and works"""
        try:
            settings = Settings()
            db_config = settings.get_database_config()
            assert isinstance(db_config, dict)
            assert 'url' in db_config
            assert 'pool_size' in db_config
        except ValidationError:
            pytest.skip("Missing API keys in .env")
    
    def test_get_redis_config(self):
        """Test get_redis_config method exists and works"""
        try:
            settings = Settings()
            redis_config = settings.get_redis_config()
            assert isinstance(redis_config, dict)
            assert 'url' in redis_config
            assert 'decode_responses' in redis_config
        except ValidationError:
            pytest.skip("Missing API keys in .env")
    
    def test_get_chroma_config(self):
        """Test get_chroma_config method exists and works"""
        try:
            settings = Settings()
            chroma_config = settings.get_chroma_config()
            assert isinstance(chroma_config, dict)
            assert 'persist_directory' in chroma_config
            assert 'embedding_dimension' in chroma_config
        except ValidationError:
            pytest.skip("Missing API keys in .env")
    
    def test_get_llm_config(self):
        """Test get_llm_config method exists and works"""
        try:
            settings = Settings()
            llm_config = settings.get_llm_config()
            assert isinstance(llm_config, dict)
            assert 'model' in llm_config
            assert 'api_key' in llm_config
            assert 'temperature' in llm_config
            assert 'max_tokens' in llm_config
        except ValidationError:
            pytest.skip("Missing API keys in .env")
    
    def test_get_allowed_file_types_list(self):
        """Test get_allowed_file_types_list converts comma-separated to list"""
        try:
            settings = Settings()
            file_types = settings.get_allowed_file_types_list()
            assert isinstance(file_types, list)
            assert len(file_types) > 0
        except ValidationError:
            pytest.skip("Missing API keys in .env")
    
    def test_is_production(self):
        """Test is_production method"""
        try:
            settings = Settings()
            result = settings.is_production()
            assert isinstance(result, bool)
        except ValidationError:
            pytest.skip("Missing API keys in .env")
    
    def test_is_development(self):
        """Test is_development method"""
        try:
            settings = Settings()
            result = settings.is_development()
            assert isinstance(result, bool)
        except ValidationError:
            pytest.skip("Missing API keys in .env")


class TestGlobalSettings:
    """Tests for global settings functions"""
    
    def test_get_settings_returns_settings_instance(self):
        """Test get_settings returns Settings instance"""
        settings = get_settings()
        assert isinstance(settings, Settings)
    
    def test_reload_settings_returns_settings_instance(self):
        """Test reload_settings returns Settings instance"""
        settings = reload_settings()
        assert isinstance(settings, Settings)
    
    def test_settings_singleton(self):
        """Test that get_settings returns same instance"""
        from src.config import settings
        s1 = get_settings()
        assert s1 is settings


class TestFieldTypes:
    """Test that fields have correct types"""
    
    def test_string_fields(self):
        """Test string field types"""
        try:
            settings = Settings()
            assert isinstance(settings.anthropic_api_key, str)
            assert isinstance(settings.voyage_api_key, str)
            assert isinstance(settings.llm_model, str)
            assert isinstance(settings.log_level, str)
        except ValidationError:
            pytest.skip("Missing API keys in .env")
    
    def test_integer_fields(self):
        """Test integer field types"""
        try:
            settings = Settings()
            assert isinstance(settings.chunk_size, int)
            assert isinstance(settings.chunk_overlap, int)
            assert isinstance(settings.llm_max_tokens, int)
        except ValidationError:
            pytest.skip("Missing API keys in .env")
    
    def test_float_fields(self):
        """Test float field types"""
        try:
            settings = Settings()
            assert isinstance(settings.llm_temperature, float)
            assert isinstance(settings.vector_search_weight, float)
            assert isinstance(settings.validator_threshold, float)
        except ValidationError:
            pytest.skip("Missing API keys in .env")
    
    def test_boolean_fields(self):
        """Test boolean field types"""
        try:
            settings = Settings()
            assert isinstance(settings.cache_enabled, bool)
            assert isinstance(settings.parallel_retrieval, bool)
        except ValidationError:
            pytest.skip("Missing API keys in .env")