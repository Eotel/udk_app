# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Development
- Run the application: `uv run -- gradio main.py`
- Run tests: `uv run -- pytest -xvs`
- Run linter and formatter: `uv run -- ruff check --fix . && uv run -- ruff format .`
- Run a specific test: `uv run -- pytest -xvs tests/test_name.py::TestClass::test_method`

## Architecture Overview

This is a voice chat application that uses OpenAI's API for:
1. Speech-to-text (audio transcription)
2. Text completion (chat responses)
3. Text-to-speech (speech synthesis)

### Core Components

1. **VoiceChat** - Main class in `voice_chat.py` that:
   - Transcribes audio to text
   - Generates chat responses using OpenAI
   - Synthesizes speech from text
   - Manages conversation state and prompt templates

2. **SoundEffects** - Class in `sound_effects.py` that:
   - Loads sound effects from the assets directory
   - Provides methods to play sounds in the UI

3. **Pydantic Models** - Defined in `models.py`:
   - Request/response models for API interactions
   - Data models for the application state

4. **Prompt Templates** - Jinja2 templates in the `prompts/` directory:
   - Define system prompts for different chat personalities
   - Allow switching between different prompt styles in the UI

5. **Gradio Interface** - The web UI built with Gradio that:
   - Handles voice and text input
   - Displays chat history
   - Provides sound effect buttons
   - Allows selecting different prompt templates

### Data Flow

1. User inputs voice or text in the Gradio interface
2. If voice: Audio is transcribed to text via OpenAI's Whisper model
3. Text is passed to the chat system, which:
   - Adds the message to conversation history
   - Sends the conversation to OpenAI's chat completion API
   - Receives and processes the response
4. Response text is converted to speech using OpenAI's TTS model
5. Audio response is played in the UI
6. Chat history is updated with the new conversation turn

### Dependencies

- **gradio**: Web interface framework
- **openai**: OpenAI API client
- **jinja2**: Template engine for prompt templates
- **loguru**: Logging utility
- **pydantic**: Data validation and settings management

## Testing

Tests are located in the `tests/` directory and use pytest. There are two main test files:
- `test_sound_effects.py`: Tests for the sound effects functionality
- `test_voice_chat.py`: Tests for the voice chat functionality, with mocked OpenAI API calls