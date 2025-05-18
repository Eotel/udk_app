# UDK App - Voice Chat with OpenAI

An interactive voice chat application using OpenAI's API for transcription, chat, and text-to-speech synthesis.

## Features

- Voice input using microphone
- Text input for chat
- AI responses with voice synthesis using OpenAI's TTS API
- Multiple chat rooms with different prompt templates
- Sound effects and background music
- Streaming responses for real-time interaction

## Requirements

- Python 3.13+
- OpenAI API Key
- Sound files (MP3 format) in the assets/sounds directory

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Eotel/udk_app.git
   cd udk_app
   ```

2. Install dependencies using uv:
   ```bash
   uv sync
   ```

3. Set up environment variables:
   ```bash
   # Create a .env file with your OpenAI API key
   echo "OPENAI_API_KEY=your_api_key_here" > .env
   ```

## Usage

Run the application:

```bash
uv run -- gradio main.py
```

The application will be available at http://127.0.0.1:7861 (or the port specified in settings.py).

## Project Structure

- `main.py`: Entry point for the application
- `voice_chat.py`: Core voice chat functionality and Gradio interface
- `sound_effects.py`: Sound effects manager
- `models.py`: Pydantic models for request/response data
- `settings.py`: Application settings
- `tests/`: Test directory

## Development

### Testing

Run tests:

```bash
uv run -- pytest -xvs
```

Run tests with coverage:

```bash
uv run -- pytest --cov=. --cov-report=term-missing -xvs
```

### Linting and Formatting

```bash
uv run -- ruff check --fix . && uv run -- ruff format .
```

## License

[MIT License](LICENSE)
