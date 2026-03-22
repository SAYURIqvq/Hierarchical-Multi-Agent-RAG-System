"""
Tests for helper utility functions.
"""

import pytest
import time
from datetime import datetime
from src.utils.helpers import (
    clean_text,
    truncate_text,
    generate_hash,
    timer,
    measure_time,
    safe_divide,
    flatten_dict,
    merge_dicts,
    chunk_list,
    remove_duplicates,
    format_timestamp,
    parse_bool,
    clamp,
    percentage,
    retry,
    extract_numbers,
    count_words
)


class TestTextProcessing:
    """Tests for text processing functions"""
    
    def test_clean_text_removes_extra_spaces(self):
        """Test cleaning multiple spaces"""
        text = "Hello   World"
        result = clean_text(text)
        assert result == "Hello World"
    
    def test_clean_text_fixes_line_breaks(self):
        """Test fixing excessive line breaks"""
        text = "Hello\n\n\n\nWorld"
        result = clean_text(text)
        assert result == "Hello\n\nWorld"
    
    def test_clean_text_strips_whitespace(self):
        """Test stripping leading/trailing whitespace"""
        text = "  Hello World  "
        result = clean_text(text)
        assert result == "Hello World"
    
    def test_clean_text_empty_string(self):
        """Test cleaning empty string"""
        assert clean_text("") == ""
        assert clean_text(None) == ""
    
    def test_truncate_text_within_limit(self):
        """Test truncate when text is within limit"""
        text = "Short text"
        result = truncate_text(text, max_length=100)
        assert result == "Short text"
    
    def test_truncate_text_exceeds_limit(self):
        """Test truncate when text exceeds limit"""
        text = "This is a very long sentence that needs truncating"
        result = truncate_text(text, max_length=20)
        assert len(result) <= 20
        assert result.endswith("...")
    
    def test_truncate_text_custom_suffix(self):
        """Test truncate with custom suffix"""
        text = "Long text here"
        result = truncate_text(text, max_length=10, suffix=">>")
        assert result.endswith(">>")
    
    def test_generate_hash_consistent(self):
        """Test that same text generates same hash"""
        hash1 = generate_hash("test")
        hash2 = generate_hash("test")
        assert hash1 == hash2
    
    def test_generate_hash_different_texts(self):
        """Test that different texts generate different hashes"""
        hash1 = generate_hash("test1")
        hash2 = generate_hash("test2")
        assert hash1 != hash2
    
    def test_generate_hash_custom_length(self):
        """Test hash with custom length"""
        hash_result = generate_hash("test", length=16)
        assert len(hash_result) == 16


class TestTiming:
    """Tests for timing functions"""
    
    def test_timer_decorator(self, capsys):
        """Test timer decorator"""
        @timer
        def slow_function():
            time.sleep(0.1)
            return "done"
        
        result = slow_function()
        captured = capsys.readouterr()
        
        assert result == "done"
        assert "slow_function" in captured.out
        assert "took" in captured.out
    
    def test_measure_time(self):
        """Test measure_time function"""
        def test_func():
            time.sleep(0.1)
            return "result"
        
        result, elapsed = measure_time(test_func)
        
        assert result == "result"
        assert elapsed >= 0.1
        assert elapsed < 0.2


class TestMathHelpers:
    """Tests for math helper functions"""
    
    def test_safe_divide_normal(self):
        """Test normal division"""
        result = safe_divide(10, 2)
        assert result == 5.0
    
    def test_safe_divide_by_zero(self):
        """Test division by zero returns default"""
        result = safe_divide(10, 0, default=0.0)
        assert result == 0.0
    
    def test_safe_divide_custom_default(self):
        """Test division by zero with custom default"""
        result = safe_divide(10, 0, default=-1.0)
        assert result == -1.0
    
    def test_clamp_within_range(self):
        """Test clamp when value is within range"""
        result = clamp(5, 0, 10)
        assert result == 5
    
    def test_clamp_below_min(self):
        """Test clamp when value is below minimum"""
        result = clamp(-5, 0, 10)
        assert result == 0
    
    def test_clamp_above_max(self):
        """Test clamp when value is above maximum"""
        result = clamp(15, 0, 10)
        assert result == 10
    
    def test_percentage_normal(self):
        """Test percentage calculation"""
        result = percentage(25, 100)
        assert result == 25.0
    
    def test_percentage_with_decimals(self):
        """Test percentage with rounding"""
        result = percentage(1, 3, decimals=2)
        assert result == 33.33
    
    def test_percentage_zero_total(self):
        """Test percentage with zero total"""
        result = percentage(10, 0)
        assert result == 0.0


class TestDataStructures:
    """Tests for data structure helpers"""
    
    def test_flatten_dict_simple(self):
        """Test flattening simple nested dict"""
        nested = {"a": {"b": 1}}
        result = flatten_dict(nested)
        assert result == {"a.b": 1}
    
    def test_flatten_dict_deep(self):
        """Test flattening deeply nested dict"""
        nested = {"a": {"b": {"c": 1}}, "d": 2}
        result = flatten_dict(nested)
        assert result == {"a.b.c": 1, "d": 2}
    
    def test_flatten_dict_custom_separator(self):
        """Test flattening with custom separator"""
        nested = {"a": {"b": 1}}
        result = flatten_dict(nested, sep="_")
        assert result == {"a_b": 1}
    
    def test_merge_dicts(self):
        """Test merging multiple dictionaries"""
        result = merge_dicts({"a": 1}, {"b": 2}, {"c": 3})
        assert result == {"a": 1, "b": 2, "c": 3}
    
    def test_merge_dicts_override(self):
        """Test that later dicts override earlier ones"""
        result = merge_dicts({"a": 1}, {"a": 2})
        assert result == {"a": 2}
    
    def test_chunk_list(self):
        """Test splitting list into chunks"""
        items = [1, 2, 3, 4, 5]
        result = chunk_list(items, chunk_size=2)
        assert result == [[1, 2], [3, 4], [5]]
    
    def test_chunk_list_exact_division(self):
        """Test chunking with exact division"""
        items = [1, 2, 3, 4]
        result = chunk_list(items, chunk_size=2)
        assert result == [[1, 2], [3, 4]]
    
    def test_remove_duplicates_simple(self):
        """Test removing duplicates from simple list"""
        items = [1, 2, 2, 3, 1]
        result = remove_duplicates(items)
        assert result == [1, 2, 3]
    
    def test_remove_duplicates_preserves_order(self):
        """Test that order is preserved"""
        items = [3, 1, 2, 1, 3]
        result = remove_duplicates(items)
        assert result == [3, 1, 2]
    
    def test_remove_duplicates_with_key(self):
        """Test removing duplicates with key function"""
        items = [{"id": 1}, {"id": 2}, {"id": 1}]
        result = remove_duplicates(items, key=lambda x: x["id"])
        assert len(result) == 2
        assert result[0]["id"] == 1
        assert result[1]["id"] == 2


class TestFormatting:
    """Tests for formatting functions"""
    
    def test_format_timestamp_current(self):
        """Test formatting current timestamp"""
        result = format_timestamp()
        assert len(result) > 0
        # Should be parseable back to datetime
        datetime.strptime(result, "%Y-%m-%d %H:%M:%S")
    
    def test_format_timestamp_custom(self):
        """Test formatting with custom format"""
        dt = datetime(2024, 12, 23, 10, 30, 0)
        result = format_timestamp(dt, format_str="%Y-%m-%d")
        assert result == "2024-12-23"
    
    def test_parse_bool_true_values(self):
        """Test parsing true boolean values"""
        assert parse_bool(True) is True
        assert parse_bool("true") is True
        assert parse_bool("yes") is True
        assert parse_bool("1") is True
        assert parse_bool(1) is True
    
    def test_parse_bool_false_values(self):
        """Test parsing false boolean values"""
        assert parse_bool(False) is False
        assert parse_bool("false") is False
        assert parse_bool("no") is False
        assert parse_bool("0") is False
        assert parse_bool(0) is False


class TestRetry:
    """Tests for retry decorator"""
    
    def test_retry_success_first_attempt(self):
        """Test retry when function succeeds on first attempt"""
        call_count = [0]
        
        @retry(max_attempts=3)
        def success_func():
            call_count[0] += 1
            return "success"
        
        result = success_func()
        assert result == "success"
        assert call_count[0] == 1
    
    def test_retry_success_after_failures(self):
        """Test retry when function succeeds after failures"""
        call_count = [0]
        
        @retry(max_attempts=3, delay=0.01)
        def flaky_func():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ValueError("Temporary error")
            return "success"
        
        result = flaky_func()
        assert result == "success"
        assert call_count[0] == 3
    
    def test_retry_max_attempts_exceeded(self):
        """Test retry when max attempts exceeded"""
        call_count = [0]
        
        @retry(max_attempts=3, delay=0.01)
        def always_fails():
            call_count[0] += 1
            raise ValueError("Always fails")
        
        with pytest.raises(ValueError):
            always_fails()
        
        assert call_count[0] == 3


class TestTextExtraction:
    """Tests for text extraction functions"""
    
    def test_extract_numbers(self):
        """Test extracting numbers from text"""
        text = "I have 3 apples and 2.5 oranges"
        result = extract_numbers(text)
        assert result == [3.0, 2.5]
    
    def test_extract_numbers_negative(self):
        """Test extracting negative numbers"""
        text = "Temperature is -5.5 degrees"
        result = extract_numbers(text)
        assert result == [-5.5]
    
    def test_extract_numbers_none(self):
        """Test when no numbers present"""
        text = "No numbers here"
        result = extract_numbers(text)
        assert result == []
    
    def test_count_words(self):
        """Test counting words"""
        text = "Hello world this is a test"
        result = count_words(text)
        assert result == 6
    
    def test_count_words_empty(self):
        """Test counting words in empty string"""
        result = count_words("")
        assert result == 0  # split("") returns ['']