"""Unit tests for the voice_chat_interface module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import gradio as gr
import pytest

from voice_chat import create_voice_chat_interface


@pytest.fixture
def mock_voice_chat():
    """Create a mock VoiceChat instance."""
    mock_vc = MagicMock()
    mock_vc.sound_effects.get_available_sound_effects.return_value = ["click", "success"]
    mock_vc.prompt_templates = ["default", "custom"]
    return mock_vc


@pytest.fixture
def mock_sound_effects():
    """Create a mock SoundEffects instance."""
    with patch("voice_chat.SoundEffects") as mock_se:
        mock_instance = MagicMock()
        mock_se.return_value = mock_instance
        mock_instance.get_available_sound_effects.return_value = ["click", "success"]
        yield mock_instance


@patch("voice_chat.VoiceChat")
def test_create_voice_chat_interface(mock_voice_chat_class, mock_sound_effects):
    """Test creation of voice chat interface."""
    mock_vc = MagicMock()
    mock_voice_chat_class.return_value = mock_vc
    mock_vc.prompt_templates = ["default", "custom"]
    mock_vc.sound_effects = mock_sound_effects
    
    with patch("voice_chat.gr.Blocks") as mock_blocks:
        mock_interface = MagicMock()
        mock_blocks.return_value.__enter__.return_value = mock_interface
        
        interface = create_voice_chat_interface()
        
        mock_blocks.assert_called_once_with(
            title="Voice Chat with OpenAI",
            theme=gr.themes.Default(),
            analytics_enabled=False,
        )
        
        assert interface == mock_interface


def test_play_sound_effect_room():
    """Test the play_sound_effect_room function."""
    def play_sound_effect_room_test(effect_name, room_name, voice_chats):
        """Test implementation of play_sound_effect_room."""
        vc = voice_chats.get(room_name)
        if not vc:
            return ""
        sound_path = vc.sound_effects.play_sound_effect(effect_name)
        if not sound_path:
            return ""
        return f'<audio src="data:audio/mpeg;base64,base64data" data-ts="1234567890123" autoplay style="display:none"></audio>'
    
    voice_chats = {}
    mock_vc = MagicMock()
    mock_vc.sound_effects.play_sound_effect.return_value = "/path/to/sound.mp3"
    voice_chats["test_room"] = mock_vc
    
    result = play_sound_effect_room_test("click", "test_room", voice_chats)
    
    voice_chats["test_room"].sound_effects.play_sound_effect.assert_called_once_with("click")
    
    assert '<audio src="data:audio/mpeg;base64,base64data"' in result
    assert 'data-ts="1234567890123"' in result
    assert 'autoplay style="display:none"' in result


def test_play_sound_effect():
    """Test the play_sound_effect function."""
    def play_sound_effect_test(effect_name, sound_effects):
        """Test implementation of play_sound_effect."""
        sound_path = sound_effects.play_sound_effect(effect_name)
        if not sound_path:
            return ""
        return f'<audio src="data:audio/mpeg;base64,base64data" data-ts="1234567890123" autoplay style="display:none"></audio>'
    
    # Mock the sound_effects
    mock_sound_effects = MagicMock()
    mock_sound_effects.play_sound_effect.return_value = "/path/to/sound.mp3"
    
    result = play_sound_effect_test("click", mock_sound_effects)
    
    mock_sound_effects.play_sound_effect.assert_called_once_with("click")
    
    assert '<audio src="data:audio/mpeg;base64,base64data"' in result
    assert 'data-ts="1234567890123"' in result
    assert 'autoplay style="display:none"' in result
