"""Unit tests for the main module."""

from unittest.mock import MagicMock, patch

import pytest

import main


@pytest.fixture
def mock_demo():
    """Mock the demo instance."""
    with patch("main.create_voice_chat_interface") as mock_create:
        mock_demo = MagicMock()
        mock_create.return_value = mock_demo
        yield mock_demo


def test_main_function(mock_demo):
    """Test the main function."""
    with patch("main.settings") as mock_settings:
        mock_settings.debug = True
        mock_settings.host = "test_host"
        
        mock_demo.launch.return_value = None
        
        main.main()
        
        mock_demo.launch.assert_called_once_with(
            server_name="test_host",
            server_port=7861,
            debug=True,
            share=False,
            max_threads=40,
        )
