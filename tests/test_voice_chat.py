"""Unit tests for the voice_chat module."""

import os
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from openai import OpenAI

from models import ChatMessage, VoiceChatState
from voice_chat import VoiceChat


@pytest.fixture
def mock_openai_client() -> Generator[MagicMock, None, None]:
    """Create a mock OpenAI client for testing."""
    with patch("openai.OpenAI") as mock_client:
        mock_transcription = MagicMock()
        mock_transcription.text = "Hello, how are you?"

        mock_choice = MagicMock()
        mock_choice.message.content = "I'm doing well, thank you for asking!"
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]

        mock_speech = MagicMock()
        mock_speech.stream_to_file = MagicMock()

        client_instance = mock_client.return_value
        client_instance.audio.transcriptions.create.return_value = mock_transcription
        client_instance.chat.completions.create.return_value = mock_completion
        client_instance.audio.speech.create.return_value = mock_speech

        yield client_instance


@pytest.fixture
def voice_chat(mock_openai_client: MagicMock) -> VoiceChat:
    """Create a VoiceChat instance with a mock OpenAI client."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        return VoiceChat()


@pytest.fixture
def temp_audio_file() -> Generator[str, None, None]:
    """Create a temporary audio file for testing."""
    temp_dir = Path(tempfile.gettempdir())
    audio_path = temp_dir / "test_audio.mp3"

    audio_path.touch()

    yield str(audio_path)

    if audio_path.exists():
        audio_path.unlink()


class TestVoiceChat:
    """Test cases for the VoiceChat class."""

    def test_init(self, voice_chat: VoiceChat) -> None:
        """Test VoiceChat initialization."""
        assert voice_chat.client is not None
        assert isinstance(voice_chat.state, VoiceChatState)
        assert voice_chat.state.conversation_history == []

    def test_transcribe_audio(
        self, voice_chat: VoiceChat, mock_openai_client: MagicMock, temp_audio_file: str
    ) -> None:
        """Test audio transcription."""
        result = voice_chat.transcribe_audio(temp_audio_file)

        mock_openai_client.audio.transcriptions.create.assert_called_once()

        assert result == "Hello, how are you?"

    def test_generate_response(
        self, voice_chat: VoiceChat, mock_openai_client: MagicMock
    ) -> None:
        """Test response generation."""
        user_message = "Hello, how are you?"
        result = voice_chat.generate_response(user_message)

        mock_openai_client.chat.completions.create.assert_called_once()

        assert result == "I'm doing well, thank you for asking!"

        assert len(voice_chat.state.conversation_history) == 2
        assert voice_chat.state.conversation_history[0].role == "user"
        assert voice_chat.state.conversation_history[0].content == user_message
        assert voice_chat.state.conversation_history[1].role == "assistant"
        assert voice_chat.state.conversation_history[1].content == result

    def test_synthesize_speech(
        self, voice_chat: VoiceChat, mock_openai_client: MagicMock
    ) -> None:
        """Test speech synthesis."""
        text = "I'm doing well, thank you for asking!"
        result = voice_chat.synthesize_speech(text)

        mock_openai_client.audio.speech.create.assert_called_once()

        assert isinstance(result, str)
        assert "response.mp3" in result

    def test_process_voice_input(
        self, voice_chat: VoiceChat, temp_audio_file: str
    ) -> None:
        """Test voice input processing."""
        with (
            patch.object(voice_chat, "transcribe_audio") as mock_transcribe,
            patch.object(voice_chat, "process_input") as mock_process,
        ):
            mock_transcribe.return_value = "Hello, how are you?"
            mock_process.return_value = ("Hello, how are you?", "/tmp/response.mp3", [])

            result = voice_chat.process_voice_input(temp_audio_file)

            mock_transcribe.assert_called_once_with(temp_audio_file)
            mock_process.assert_called_once_with("Hello, how are you?")

            assert result == ("Hello, how are you?", "/tmp/response.mp3", [])

    def test_process_text_input(self, voice_chat: VoiceChat) -> None:
        """Test text input processing."""
        with patch.object(voice_chat, "process_input") as mock_process:
            mock_process.return_value = ("Hello, how are you?", "/tmp/response.mp3", [])

            result = voice_chat.process_text_input("Hello, how are you?")

            mock_process.assert_called_once_with("Hello, how are you?")

            assert result == ("Hello, how are you?", "/tmp/response.mp3", [])

    def test_process_input(self, voice_chat: VoiceChat) -> None:
        """Test input processing."""
        voice_chat.state.conversation_history = [
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi there"),
        ]

        with (
            patch.object(voice_chat, "generate_response") as mock_generate,
            patch.object(voice_chat, "synthesize_speech") as mock_synthesize,
        ):
            mock_generate.return_value = "I'm doing well, thank you for asking!"
            mock_synthesize.return_value = "/tmp/response.mp3"

            result = voice_chat.process_input("How are you?")

            mock_generate.assert_called_once_with("How are you?")
            mock_synthesize.assert_called_once_with(
                "I'm doing well, thank you for asking!"
            )

            assert result[0] == "How are you?"
            assert result[1] == "/tmp/response.mp3"
            # Expect list of message dicts for user and assistant
            assert result[2] == [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ]
