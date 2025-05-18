"""Unit tests for settings module."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from settings import Settings


def test_validate_api_key_valid():
    """Test validate_api_key with valid API key."""
    assert Settings.validate_api_key("sk-test123") == "sk-test123"
    assert Settings.validate_api_key("OPENAI_API_KEY_test") == "OPENAI_API_KEY_test"


def test_validate_api_key_invalid():
    """Test validate_api_key with invalid API key."""
    with pytest.raises(ValueError, match="OpenAI API key cannot be empty"):
        Settings.validate_api_key("")
    
    with pytest.raises(ValueError, match="OpenAI API key should start with 'sk-'"):
        Settings.validate_api_key("invalid-key")


def test_validate_directory_exists_existing():
    """Test validate_directory_exists with existing directory."""
    with patch("pathlib.Path.exists", return_value=True):
        test_path = Path("/test/dir")
        result = Settings.validate_directory_exists(test_path)
        assert result == test_path


def test_validate_directory_creates_missing():
    """Test validate_directory_exists creates missing directory."""
    with patch("pathlib.Path.exists", return_value=False), \
         patch("pathlib.Path.mkdir"), \
         patch("settings.logger.warning") as mock_logger:
        
        test_path = Path("/test/dir")
        result = Settings.validate_directory_exists(test_path)
        
        test_path.mkdir.assert_called_once_with(parents=True, exist_ok=True)
        
        mock_logger.assert_called_once()
        
        assert result == test_path
