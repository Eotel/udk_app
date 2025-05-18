# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

UDK App is a voice chat application using OpenAI's API for transcription, chat, and text-to-speech synthesis:

1. **Voice Chat**: Core functionality for speech-to-text, chat completions, and text-to-speech
2. **Gradio Interface**: User interface for voice/text input and AI responses
3. **Sound Effects**: Background music and sound effects for user interactions

The system provides an interactive voice interface for conversing with OpenAI's language models.

## Build & Development Commands

```bash
# Setup environment
uv sync

# Run development server
uv run -- gradio main.py

# Run tests
uv run -- pytest -xvs

# Run tests with coverage
uv run -- pytest --cov=. --cov-report=term-missing -xvs

# Run linter and formatter
uv run -- ruff check --fix . && uv run -- ruff format .
```

## Architecture Overview

### Core Components

- **VoiceChat**: Main class handling OpenAI API interactions
  - Transcription (speech-to-text)
  - Chat response generation
  - Text-to-speech synthesis
  - Chat history management

- **SoundEffects**: Class for managing sound effects and background music
  - Loading sound effects from directory
  - Playing sound effects and background music

- **Settings**: Application configuration using Pydantic
  - OpenAI API configuration
  - Model settings
  - Directory paths
  - Server settings

### Data Models

- **Pydantic Models** in models.py
  - TranscriptionRequest/Response
  - ChatMessage and ChatRequest
  - SpeechRequest/Response
  - VoiceChatState

## Data Flow

1. User inputs voice through microphone or text through textbox
2. Voice input is transcribed using OpenAI's Whisper API
3. Text is sent to OpenAI's chat completion API
4. Response is generated and displayed in the chat history
5. Response is synthesized to speech using OpenAI's TTS API
6. Sound effects play during interactions

## Development Guidelines

### Python

- Use proper typing for all functions
- Follow PEP 8 style guidelines
- Use Pydantic for data validation and schemas
- **REQUIRED**: Run linter, formatter, and tests (`uv run -- ruff check --fix . && uv run -- ruff format . && uv run -- pytest -xvs`) before committing any code changes
- **REQUIRED**: Code that does not pass linting **MUST NOT** be added to git
- **REQUIRED**: Code that does not pass tests **MUST NOT** be added to git

### Gradio Interface

- Use gr.update() for responsive UI updates
- Split functionality into separate service modules
- Avoid adding complex logic to main.py

### Testing

- Use pytest for implementing unit tests
- Create tests in a separate branch and PR
- Focus test coverage on service modules
- Run tests using: `uv run -- pytest -xvs`

### Linting and Formatting

- The project uses Ruff for both linting and formatting
- The COM812 rule may cause conflicts with the formatter and should be added to the ignore configuration
- Run linting and formatting using: `uv run -- ruff check --fix . && uv run -- ruff format .`
